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

#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>


struct ucp_proto_perf_segment {
    /* List element */
    ucs_list_link_t       list;

    /* Start value of this segment (inclusive) */
    size_t                start;

    /* End value of this segment (inclusive) */
    size_t                end;

    /* Associacted performance node */
    ucp_proto_perf_node_t *node;

    /* Linear function representing value of each contributing factor */
    ucs_linear_func_t     factors[UCP_PROTO_PERF_FACTOR_LAST];
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
    [UCP_PROTO_PERF_FACTOR_REMOTE_CPU] = "rem_cpu",
    [UCP_PROTO_PERF_FACTOR_LOCAL_TL]   = "tl",
    [UCP_PROTO_PERF_FACTOR_REMOTE_TL]  = "rem_tl",
    [UCP_PROTO_PERF_FACTOR_LATENCY]    = "lat",
    [UCP_PROTO_PERF_FACTOR_SINGLE]     = "sngl",
    [UCP_PROTO_PERF_FACTOR_MULTI]      = "mult",
    [UCP_PROTO_PERF_FACTOR_LAST]       = NULL
};

static void ucp_proto_perf_check(const ucp_proto_perf_t *perf)
{
#if ENABLE_ASSERT
    const ucp_proto_perf_segment_t *seg;
    size_t min_start;

    min_start = 0;
    ucp_proto_perf_segment_foreach(seg, perf) {
        ucs_assertv((seg->start >= min_start) && (seg->start <= seg->end),
                    "perf=%p seg->start=%zu seg->end=%zu min_start=%zu", perf,
                    seg->start, seg->end, min_start);
        if (seg->end == SIZE_MAX) {
            ucs_assertv(ucs_list_is_last(&perf->segments, &seg->list),
                        "perf=%p seg->start=%zu seg->end=%zu", perf, seg->start,
                        seg->end);
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

    seg->start = start;
    seg->end   = end;
    seg->node  = NULL;
    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        seg->factors[factor_id] = UCS_LINEAR_FUNC_ZERO;
    }

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
        new_seg->factors[factor_id] = seg->factors[factor_id];
    }
    new_seg->node = ucp_proto_perf_node_dup(seg->node);

    seg->end = seg_end;
    ucs_list_insert_after(&seg->list, &new_seg->list);

    return UCS_OK;
}

static void ucp_proto_perf_segment_add_funcs(
        ucp_proto_perf_t *perf, ucp_proto_perf_segment_t *seg,
        const ucs_linear_func_t funcs[UCP_PROTO_PERF_FACTOR_LAST],
        uint64_t factors_bitmap, ucp_proto_perf_node_t *perf_node)
{
    unsigned factor_id;

    /* Create a performance node for this segment if it does not exist yet */
    if (seg->node == NULL) {
        seg->node = ucp_proto_perf_node_new_data(perf->name, "");
    }

    /* Add the functions to the segment and the performance node */
    ucs_for_each_bit(factor_id, factors_bitmap) {
        ucs_assert(factor_id < UCP_PROTO_PERF_FACTOR_LAST); /* For Coverity */
        ucs_linear_func_add_inplace(&seg->factors[factor_id], funcs[factor_id]);
        ucp_proto_perf_node_update_data(seg->node,
                                        ucp_proto_perf_factor_names[factor_id],
                                        seg->factors[factor_id]);
    }

    /* Add the child performance node to the segment performance node */
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

ucs_status_t ucp_proto_perf_from_caps(const char *name,
                                      const ucp_proto_caps_t *proto_caps,
                                      ucp_proto_perf_t **perf_p)
{
    ucs_linear_func_t funcs[UCP_PROTO_PERF_FACTOR_LAST];
    const ucp_proto_perf_range_t *range;
    ucp_proto_perf_segment_t *seg;
    ucp_proto_perf_t *perf;
    ucs_status_t status;
    size_t range_start;

    status = ucp_proto_perf_create(name, &perf);
    if (status != UCS_OK) {
        goto err;
    }

    range_start = proto_caps->min_length;
    ucs_carray_for_each(range, proto_caps->ranges, proto_caps->num_ranges) {
        if (range->max_length < range_start) {
            /* Skip empty ranges */
            continue;
        }

        status = ucp_proto_perf_segment_new(perf, range_start,
                                            range->max_length, &seg);
        if (status != UCS_OK) {
            goto err_destroy;
        }

        seg->node = range->node;
        ucp_proto_perf_node_ref(seg->node);
        ucs_list_add_tail(&perf->segments, &seg->list);

        funcs[UCP_PROTO_PERF_FACTOR_LOCAL_CPU] =
                range->perf[UCP_PROTO_PERF_TYPE_CPU];
        funcs[UCP_PROTO_PERF_FACTOR_SINGLE] =
                range->perf[UCP_PROTO_PERF_TYPE_SINGLE];
        funcs[UCP_PROTO_PERF_FACTOR_MULTI] =
                range->perf[UCP_PROTO_PERF_TYPE_MULTI];
        ucp_proto_perf_segment_add_funcs(
                perf, seg, funcs,
                UCS_BIT(UCP_PROTO_PERF_FACTOR_LOCAL_CPU) |
                UCS_BIT(UCP_PROTO_PERF_FACTOR_SINGLE) |
                UCS_BIT(UCP_PROTO_PERF_FACTOR_MULTI),
                NULL);

        range_start = range->max_length + 1;
    }

    ucp_proto_perf_check(perf);
    *perf_p = perf;
    return UCS_OK;

err_destroy:
    ucp_proto_perf_destroy(perf);
err:
    return status;
}

ucs_status_t ucp_proto_perf_add_funcs(
        ucp_proto_perf_t *perf, size_t start, size_t end,
        const ucs_linear_func_t funcs[UCP_PROTO_PERF_FACTOR_LAST],
        uint64_t factors_bitmap, ucp_proto_perf_node_t *perf_node)
{
    ucp_proto_perf_segment_t *seg, *new_seg;
    ucs_status_t status;
    size_t seg_end;

    ucp_proto_perf_check(perf);

    /*                   __________         _________________
     * perf before:     |__________|       |_________________|
     *                __________________
     * range to add: |__________________|
     *                __________________    _________________
     * perf after:   |__|__________|____|  |_____|__________|
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

        ucp_proto_perf_segment_add_funcs(perf, seg, funcs, factors_bitmap,
                                         perf_node);
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
        ucp_proto_perf_segment_add_funcs(perf, seg, funcs, factors_bitmap,
                                         perf_node);
    }

    status = UCS_OK;

out:
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
                ucp_proto_perf_segment_add_funcs(
                        perf, new_seg, seg->factors,
                        UCS_MASK(UCP_PROTO_PERF_FACTOR_LAST), seg->node);
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
    return seg->factors[factor_id];
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

static void ucp_proto_perf_segment_dump(const ucp_proto_perf_segment_t *seg,
                                        ucs_string_buffer_t *strb)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucs_linear_func_t func;

    ucs_string_buffer_appendf(strb, "{%zu..%zu", seg->start, seg->end);

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        func = seg->factors[factor_id];
        if (ucs_linear_func_is_zero(func, UCP_PROTO_PERF_EPSILON)) {
            continue;
        }

        ucs_string_buffer_appendf(strb, " %s:%.2f+%.2fx",
                                  ucp_proto_perf_factor_names[factor_id],
                                  func.c, func.m);
    }

    ucs_string_buffer_appendf(strb, "} ");
}

void ucp_proto_perf_dump(const ucp_proto_perf_t *perf,
                         ucs_string_buffer_t *strb)
{
    ucp_proto_perf_segment_t *seg;

    ucp_proto_perf_segment_foreach(seg, perf) {
        ucp_proto_perf_segment_dump(seg, strb);
    }
    ucs_string_buffer_rtrim(strb, NULL);
}
