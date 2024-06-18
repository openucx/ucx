/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_perf.h"
#include "proto_debug.h"
#include "proto_init.h"

#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>


struct ucp_proto_perf_segment {
    /* List element */
    ucs_list_link_t          list;

    /* Start value of this segment (inclusive) */
    size_t                   start;

    /* End value of this segment (inclusive) */
    size_t                   end;

    /* Associacted performance node */
    ucp_proto_perf_node_t    *node;

    /* Linear function representing value of each contributing factor */
    ucp_proto_perf_factors_t perf_factors;
};

/*
 * Protocol performance structure.
 */
struct ucp_proto_perf {
    /* Name of the protocol */
    char            name[UCP_PROTO_DESC_STR_MAX];

    /* List of segments */
    ucs_list_link_t segments;
};

/*
 * Iterate over the segments of the protocol perf structure function.
 */
#define ucp_proto_perf_segment_foreach(_seg, _perf) \
    ucs_list_for_each((_seg), &(_perf)->segments, list)

static const char *ucp_proto_perf_factor_names[] = {
    [UCP_PROTO_PERF_FACTOR_LOCAL_CPU]  = "cpu",
    [UCP_PROTO_PERF_FACTOR_REMOTE_CPU] = "cpu-remote",
    [UCP_PROTO_PERF_FACTOR_LOCAL_TL]   = "tl",
    [UCP_PROTO_PERF_FACTOR_REMOTE_TL]  = "tl-remote",
    [UCP_PROTO_PERF_FACTOR_LATENCY]    = "lat",
    [UCP_PROTO_PERF_FACTOR_LAST]       = NULL
};

#if ENABLE_ASSERT
static const char*
ucp_proto_perf_log(const ucp_proto_perf_t *perf, ucs_log_level_t log_level)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ucp_proto_perf_str(perf, &strb);
    ucs_log(log_level, "%s %s", perf->name, ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
    return "";
}
#endif

static void ucp_proto_perf_check(const ucp_proto_perf_t *perf)
{
#if ENABLE_ASSERT
    const ucp_proto_perf_segment_t *seg;
    size_t min_start;

    min_start = 0;
    ucp_proto_perf_segment_foreach(seg, perf) {
        ucs_assertv((seg->start >= min_start) && (seg->start <= seg->end),
                    "perf=%p seg->start=%zu seg->end=%zu min_start=%zu %s",
                    perf, seg->start, seg->end, min_start,
                    ucp_proto_perf_log(perf, UCS_LOG_LEVEL_ERROR));
        if (seg->end == SIZE_MAX) {
            ucs_assertv(ucs_list_is_last(&perf->segments, &seg->list),
                        "perf=%p seg->start=%zu seg->end=%zu %s",
                        perf, seg->start, seg->end,
                        ucp_proto_perf_log(perf, UCS_LOG_LEVEL_ERROR));
        } else {
            min_start = seg->end + 1;
        }
    }
#endif
}

static ucs_status_t ucp_proto_perf_segment_new(const ucp_proto_perf_t *perf,
                                               size_t start, size_t end,
                                               ucp_proto_perf_segment_t **seg_p)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucp_proto_perf_segment_t *seg;

    ucs_assertv(start <= end, "perf=%p start=%zu end=%zu", perf, start, end);

    seg = ucs_malloc(sizeof(*seg), "ucp_proto_perf_segment");
    if (seg == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        seg->perf_factors[factor_id] = UCS_LINEAR_FUNC_ZERO;
    }

    seg->start = start;
    seg->end   = end;
    seg->node  = NULL;

    *seg_p = seg;
    return UCS_OK;
}

static ucs_status_t ucp_proto_perf_segment_split(const ucp_proto_perf_t *perf,
                                                 ucp_proto_perf_segment_t *seg,
                                                 size_t seg_end)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucp_proto_perf_segment_t *new_seg;
    ucs_status_t status;

    ucs_assertv(seg_end < seg->end, "seg=%p seg->end=%zu seg_end=%zu", seg,
                seg->end, seg_end);

    status = ucp_proto_perf_segment_new(perf, seg_end + 1, seg->end, &new_seg);
    if (status != UCS_OK) {
        return status;
    }

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        new_seg->perf_factors[factor_id] = seg->perf_factors[factor_id];
    }
    new_seg->node = ucp_proto_perf_node_dup(seg->node);

    seg->end = seg_end;
    ucs_list_insert_after(&seg->list, &new_seg->list);

    return UCS_OK;
}

static void
ucp_proto_perf_node_update_factors(ucp_proto_perf_node_t *perf_node,
                                   const ucp_proto_perf_factors_t perf_factors)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucs_linear_func_t perf_factor;

    /* Add the functions to the segment and the performance node */
    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
        perf_factor = perf_factors[factor_id];
        if (ucs_linear_func_is_zero(perf_factor, UCP_PROTO_PERF_EPSILON)) {
            continue;
        }

        ucp_proto_perf_node_update_data(perf_node,
                                        ucp_proto_perf_factor_names[factor_id],
                                        perf_factors[factor_id]);
    }
}

static void
ucp_proto_perf_segment_add_funcs(ucp_proto_perf_t *perf,
                                 ucp_proto_perf_segment_t *seg,
                                 const ucp_proto_perf_factors_t perf_factors,
                                 ucp_proto_perf_node_t *perf_node)
{
    ucp_proto_perf_factor_id_t factor_id;

    if (seg->node == NULL) {
        seg->node = ucp_proto_perf_node_new_data(perf->name, "");
    }

    /* Add the functions to the segment and the performance node */
    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
        ucs_linear_func_add_inplace(&seg->perf_factors[factor_id],
                                    perf_factors[factor_id]);
    }

    ucp_proto_perf_node_update_factors(seg->node, seg->perf_factors);
    ucp_proto_perf_node_add_child(seg->node, perf_node);
}

ucs_status_t ucp_proto_perf_create(const char *name, ucp_proto_perf_t **perf_p)
{
    ucp_proto_perf_t *perf;

    perf = ucs_malloc(sizeof(*perf), "ucp_proto_perf");
    if (perf == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_strncpy_zero(perf->name, name, sizeof(perf->name));
    ucs_list_head_init(&perf->segments);
    *perf_p = perf;
    return UCS_OK;
}

void ucp_proto_perf_destroy(ucp_proto_perf_t *perf)
{
    ucp_proto_perf_segment_t *seg, *tmp;

    ucs_list_for_each_safe(seg, tmp, &perf->segments, list) {
        ucp_proto_perf_node_deref(&seg->node);
        ucs_free(seg);
    }
    ucs_free(perf);
}

int ucp_proto_perf_is_empty(const ucp_proto_perf_t *perf)
{
    return ucs_list_is_empty(&perf->segments);
}

ucs_status_t
ucp_proto_perf_add_funcs(ucp_proto_perf_t *perf, size_t start, size_t end,
                         const ucp_proto_perf_factors_t perf_factors,
                         ucp_proto_perf_node_t *child_perf_node,
                         const char *title, const char *desc_fmt, ...)
{
    ucp_proto_perf_segment_t *seg, *new_seg;
    ucp_proto_perf_node_t *perf_node;
    ucs_status_t status;
    size_t seg_end;
    va_list ap;

    ucp_proto_perf_check(perf);

    va_start(ap, desc_fmt);
    perf_node = ucp_proto_perf_node_new(UCP_PROTO_PERF_NODE_TYPE_DATA,
                                        0, title, desc_fmt, ap);
    va_end(ap);

    ucp_proto_perf_node_update_factors(perf_node, perf_factors);
    ucp_proto_perf_node_add_child(perf_node, child_perf_node);

    /*                   __________         _________________
     * perf before:     |__________|       |_________________|
     *                __________________
     * range to add: |__________________|
     *                __________________    _________________
     * perf after:   |__|__________|____|  |_________________|
     */
    seg = ucs_list_head(&perf->segments, ucp_proto_perf_segment_t, list);
    while ((&seg->list != &perf->segments) && (start <= end)) {
        if (start > seg->end) {
            /* The current segment ends before the requested range */
            seg = ucs_list_next(&seg->list, ucp_proto_perf_segment_t, list);
            continue;
        }

        if (start < seg->start) {
            /* Insert an empty segment before the current segment */
            seg_end = ucs_min(end, seg->start - 1);
            status  = ucp_proto_perf_segment_new(perf, start, seg_end, &new_seg);
            if (status != UCS_OK) {
                goto out;
            }

            ucs_list_insert_before(&seg->list, &new_seg->list);
            seg = new_seg;
        } else {
            /* Split the first affected segment */
            if (start > seg->start) {
                status = ucp_proto_perf_segment_split(perf, seg, start - 1);
                if (status != UCS_OK) {
                    goto out;
                }

                /* Move to the first segment affected by the specified range */
                seg = ucs_list_next(&seg->list, ucp_proto_perf_segment_t, list);
            }

            /* Split the last affected segment */
            if (end < seg->end) {
                status = ucp_proto_perf_segment_split(perf, seg, end);
                if (status != UCS_OK) {
                    goto out;
                }
            }
        }

        ucs_assertv(start <= seg->start, "start=%zu seg->start=%zu", start,
                    seg->start);
        ucs_assertv(end >= seg->end, "end=%zu seg->end=%zu", end, seg->end);

        ucp_proto_perf_segment_add_funcs(perf, seg, perf_factors, perf_node);
        if (seg->end == SIZE_MAX) {
            goto out_ok; /*Avoid wraparound */
        }

        start = seg->end + 1;
        seg   = ucs_list_next(&seg->list, ucp_proto_perf_segment_t, list);
    }

    /* Add the remainder */
    if (start <= end) {
        status = ucp_proto_perf_segment_new(perf, start, end, &seg);
        if (status != UCS_OK) {
            goto out;
        }

        ucs_list_add_tail(&perf->segments, &seg->list);
        ucp_proto_perf_segment_add_funcs(perf, seg, perf_factors, perf_node);
    }

out_ok:
    status = UCS_OK;
out:
    ucp_proto_perf_node_deref(&perf_node);
    ucp_proto_perf_check(perf);
    return status;
}

ucs_status_t ucp_proto_perf_aggregate(const char *name,
                                      const ucp_proto_perf_t *const *perf_elems,
                                      unsigned num_elems,
                                      ucp_proto_perf_t **perf_p)
{
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_perf_segment_t *new_seg;
    ucp_proto_perf_t *perf;
    ucs_list_link_t **pos;
    unsigned i, curr_elem;
    ucs_status_t status;
    size_t start, end;

    status = ucp_proto_perf_create(name, &perf);
    if (status != UCS_OK) {
        goto err;
    }

    if (num_elems == 0) {
        goto out;
    }

    pos = ucs_alloca(sizeof(*pos) * num_elems);
    for (i = 0; i < num_elems; ++i) {
        ucp_proto_perf_check(perf_elems[i]);
        pos[i] = perf_elems[i]->segments.next;
    }

    curr_elem = 0;
    start     = 0;
    end       = SIZE_MAX;
    while (pos[curr_elem] != &perf_elems[curr_elem]->segments) {
        seg = ucs_container_of(pos[curr_elem], ucp_proto_perf_segment_t, list);
        if (seg->end < start) {
            /* Find the next segment that may contain the start point */
            pos[curr_elem] = pos[curr_elem]->next;
            continue;
        }

        if (start < seg->start) {
            /* Segment does not contain start point, start over from the
               beginning of the segment */
            start = seg->start;
        } else {
            end = ucs_min(seg->end, end);
            ++curr_elem;
            if (curr_elem < num_elems) {
                /* Have more elements to intersect with */
                continue;
            }

            /* Finished intersecting all perf elements - create a new segment */
            status = ucp_proto_perf_segment_new(perf, start, end, &new_seg);
            if (status != UCS_OK) {
                goto err_destroy;
            }

            ucs_list_add_tail(&perf->segments, &new_seg->list);
            for (i = 0; i < num_elems; ++i) {
                seg = ucs_container_of(pos[i], ucp_proto_perf_segment_t, list);
                ucp_proto_perf_segment_add_funcs(perf, new_seg,
                                                 seg->perf_factors, seg->node);
            }

            if (end == SIZE_MAX) {
                goto out;
            }

            start = end + 1;
        }

        end       = SIZE_MAX;
        curr_elem = 0;
    }

out:
    ucp_proto_perf_check(perf);
    *perf_p = perf;
    return UCS_OK;

err_destroy:
    ucp_proto_perf_destroy(perf);
err:
    return status;
}

ucs_status_t ucp_proto_perf_aggregate2(const char *name,
                                       const ucp_proto_perf_t *perf1,
                                       const ucp_proto_perf_t *perf2,
                                       ucp_proto_perf_t **perf_p)
{
    const ucp_proto_perf_t *perf_elems[2] = {perf1, perf2};

    return ucp_proto_perf_aggregate(name, perf_elems, 2, perf_p);
}

ucs_status_t ucp_proto_perf_turn_remote(const ucp_proto_perf_t *remote_perf,
                                        ucp_proto_perf_t **perf_p)
{
    ucp_proto_perf_factors_t turned_factors;
    ucp_proto_perf_segment_t *remote_seg;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    ucp_proto_perf_check(remote_perf);

    status = ucp_proto_perf_create("turned", &perf);
    if (status != UCS_OK) {
        return status;
    }

    /* Turn local factors to remote and vice versa */
    ucp_proto_perf_segment_foreach(remote_seg, remote_perf) {
        turned_factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU] =
                remote_seg->perf_factors[UCP_PROTO_PERF_FACTOR_REMOTE_CPU];
        turned_factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] =
                remote_seg->perf_factors[UCP_PROTO_PERF_FACTOR_REMOTE_TL];
        turned_factors[UCP_PROTO_PERF_FACTOR_LATENCY] =
                remote_seg->perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY];
        turned_factors[UCP_PROTO_PERF_FACTOR_REMOTE_TL] =
                remote_seg->perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL];
        turned_factors[UCP_PROTO_PERF_FACTOR_REMOTE_CPU] =
                remote_seg->perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU];

        status = ucp_proto_perf_add_funcs(perf, remote_seg->start,
                                          remote_seg->end, turned_factors,
                                          remote_seg->node, perf->name, "");
        if (status != UCS_OK) {
            ucp_proto_perf_destroy(perf);
            return status;
        }
    }

    *perf_p = perf;
    return UCS_OK;
}

ucs_status_t
ucp_proto_perf_envelope(const ucp_proto_perf_t *perf, int convex,
                        ucp_proto_flat_perf_t *flat_perf)
{
    ucp_proto_perf_envelope_elem_t *envelope_elem;
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_envelope_t envelope;
    ucs_status_t status;
    size_t range_start;

    ucp_proto_perf_check(perf);

    ucs_array_init_dynamic(&envelope);
    ucp_proto_perf_segment_foreach(seg, perf) {
        status = ucp_proto_perf_envelope_make(seg->perf_factors,
                                              UCP_PROTO_PERF_FACTOR_LAST,
                                              seg->start, seg->end, convex,
                                              &envelope);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Find range start */
    seg = ucp_proto_perf_find_segment_lb(perf, 0);
    if (seg == NULL) {
        /* Empty perf */
        return UCS_OK;
    }
    range_start = seg->start;

    ucs_array_for_each(envelope_elem, &envelope) {
        while (envelope_elem->max_length > seg->end) {
            /* Find segment for current envelope*/
            seg = ucs_list_next(&seg->list, ucp_proto_perf_segment_t, list);
        }

        range        = ucs_array_append(flat_perf, return UCS_ERR_NO_MEMORY);
        range->start = range_start;
        range->end   = envelope_elem->max_length;
        range->value = seg->perf_factors[envelope_elem->index];
        range_start  = envelope_elem->max_length + 1;
    }

    return UCS_OK;
}

ucs_status_t ucp_proto_perf_sum(const ucp_proto_perf_t *perf,
                                ucp_proto_flat_perf_t *flat_perf)
{
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_factor_id_t factor_id;

    ucp_proto_perf_segment_foreach(seg, perf) {
        range        = ucs_array_append(flat_perf, return UCS_ERR_NO_MEMORY);
        range->start = seg->start;
        range->end   = seg->end;
        range->value = UCS_LINEAR_FUNC_ZERO;

        for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             factor_id++) {
            ucs_linear_func_add_inplace(&range->value,
                                        seg->perf_factors[factor_id]);
        }
    }

    return UCS_OK;
}

ucp_proto_perf_segment_t *
ucp_proto_perf_find_segment_lb(const ucp_proto_perf_t *perf, size_t lb)

{
    ucp_proto_perf_segment_t *seg;

    ucp_proto_perf_segment_foreach(seg, perf) {
        if (lb <= seg->end) {
            return seg;
        }
    }

    return NULL;
}

ucs_linear_func_t
ucp_proto_perf_segment_func(const ucp_proto_perf_segment_t *seg,
                            ucp_proto_perf_factor_id_t factor_id)
{
    return seg->perf_factors[factor_id];
}

size_t ucp_proto_perf_segment_start(const ucp_proto_perf_segment_t *seg)
{
    return seg->start;
}

size_t ucp_proto_perf_segment_end(const ucp_proto_perf_segment_t *seg)
{
    return seg->end;
}

ucp_proto_perf_node_t *
ucp_proto_perf_segment_node(const ucp_proto_perf_segment_t *seg)
{
    return seg->node;
}

const ucp_proto_perf_segment_t *
ucp_proto_perf_segment_next(const ucp_proto_perf_t *perf,
                            const ucp_proto_perf_segment_t *seg)
{
    if (ucs_list_is_last(&perf->segments, &seg->list)) {
        return NULL;
    }

    return ucs_list_next(&seg->list, ucp_proto_perf_segment_t, list);
}

const ucp_proto_perf_segment_t *
ucp_proto_perf_segment_last(const ucp_proto_perf_t *perf)
{
    if (ucs_list_is_empty(&perf->segments)) {
        return NULL;
    }

    return ucs_list_tail(&perf->segments, ucp_proto_perf_segment_t, list);
}

void ucp_proto_perf_segment_str(const ucp_proto_perf_segment_t *seg,
                                ucs_string_buffer_t *strb)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucs_linear_func_t perf_factor;

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        perf_factor = ucp_proto_perf_segment_func(seg, factor_id);
        if (ucs_linear_func_is_zero(perf_factor, UCP_PROTO_PERF_EPSILON)) {
            continue;
        }

        ucs_string_buffer_appendf(strb, "%s: " UCP_PROTO_PERF_FUNC_FMT " ",
                                  ucp_proto_perf_factor_names[factor_id],
                                  UCP_PROTO_PERF_FUNC_ARG(&perf_factor));
    }
    ucs_string_buffer_rtrim(strb, NULL);
}

void ucp_proto_perf_str(const ucp_proto_perf_t *perf, ucs_string_buffer_t *strb)
{
    ucp_proto_perf_segment_t *seg;
    char range_str[64];

    ucp_proto_perf_segment_foreach(seg, perf) {
        ucs_memunits_range_str(seg->start, seg->end, range_str,
                               sizeof(range_str));
        ucs_string_buffer_appendf(strb, "%s {", range_str);
        ucp_proto_perf_segment_str(seg, strb);
        ucs_string_buffer_appendf(strb, "} ");
    }
    ucs_string_buffer_rtrim(strb, NULL);
}

const ucp_proto_flat_perf_range_t *
ucp_proto_flat_perf_find_lb(const ucp_proto_flat_perf_t *flat_perf, size_t lb)
{
    ucp_proto_flat_perf_range_t *range;

    ucs_array_for_each(range, flat_perf) {
        if (lb <= range->end) {
            return range;
        }
    }
    return NULL;
}
