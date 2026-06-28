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
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>

#include <float.h>

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
    [UCP_PROTO_PERF_FACTOR_LOCAL_CPU]         = "cpu",
    [UCP_PROTO_PERF_FACTOR_REMOTE_CPU]        = "cpu-remote",
    [UCP_PROTO_PERF_FACTOR_LOCAL_TL]          = "tl",
    [UCP_PROTO_PERF_FACTOR_REMOTE_TL]         = "tl-remote",
    [UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY]  = "mtcopy",
    [UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY] = "mtcopy-remote",
    [UCP_PROTO_PERF_FACTOR_LATENCY]           = "lat",
    [UCP_PROTO_PERF_FACTOR_LAST]              = NULL
};

static void ucp_proto_perf_check(const ucp_proto_perf_t *perf)
{
#if ENABLE_ASSERT
    ucs_string_buffer_t funcb = UCS_STRING_BUFFER_INITIALIZER;
    const char *reason        = NULL;
    const ucp_proto_perf_segment_t *seg;
    size_t min_start;

    ucs_assert(perf != NULL);

    min_start = 0;
    ucp_proto_perf_segment_foreach(seg, perf) {
        if (seg->start < min_start) {
            reason = "seg->start < min_start";
            break;
        }
        if (seg->start > seg->end) {
            reason = "seg->start > seg->end";
            break;
        }
        if (seg->end < SIZE_MAX) {
            min_start = seg->end + 1;
        } else if (!ucs_list_is_last(&perf->segments, &seg->list)) {
            reason = "!ucs_list_is_last(&perf->segments, &seg->list)";
            break;
        }
    }

    if (reason != NULL) {
        ucp_proto_perf_str(perf, &funcb);
        ucs_fatal("%s seg=[%zu, %zu] min_start=%zu\nperf=%p name=%s %s",
                  reason, seg->start, seg->end, min_start, perf, perf->name,
                  ucs_string_buffer_cstr(&funcb));
        ucs_string_buffer_cleanup(&funcb);
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

static void ucp_proto_perf_node_update_factor(ucp_proto_perf_node_t *perf_node,
                                              const char *perf_factor_name,
                                              ucs_linear_func_t perf_factor)
{
    if (ucs_linear_func_is_zero(perf_factor, UCP_PROTO_PERF_EPSILON)) {
        return;
    }

    ucp_proto_perf_node_update_data(perf_node, perf_factor_name, perf_factor);
}

static void
ucp_proto_perf_node_update_factors(ucp_proto_perf_node_t *perf_node,
                                   const ucp_proto_perf_factors_t perf_factors)
{
    ucp_proto_perf_factor_id_t factor_id;

    /* Add the functions to the segment and the performance node */
    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
        ucp_proto_perf_node_update_factor(perf_node,
                                          ucp_proto_perf_factor_names[factor_id],
                                          perf_factors[factor_id]);
    }
}

static void
ucp_proto_perf_segment_update_factor(ucp_proto_perf_segment_t *seg,
                                     ucp_proto_perf_factor_id_t factor_id,
                                     ucs_linear_func_t perf_factor)
{
    seg->perf_factors[factor_id] = perf_factor;
    ucp_proto_perf_node_update_factor(seg->node,
                                      ucp_proto_perf_factor_names[factor_id],
                                      perf_factor);
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
        ucp_proto_perf_segment_update_factor(
                seg, factor_id,
                ucs_linear_func_add(seg->perf_factors[factor_id],
                                    perf_factors[factor_id]));
    }

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

const char *ucp_proto_perf_name(const ucp_proto_perf_t *perf)
{
    return perf->name;
}

ucs_status_t
ucp_proto_perf_add_funcs(ucp_proto_perf_t *perf, size_t start, size_t end,
                         const ucp_proto_perf_factors_t perf_factors,
                         ucp_proto_perf_node_t *perf_node,
                         ucp_proto_perf_node_t *child_perf_node)
{
    ucp_proto_perf_segment_t *seg, *new_seg;
    ucs_status_t status;
    size_t seg_end;

    ucp_proto_perf_check(perf);
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
            goto out_ok; /* Avoid wraparound */
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

void ucp_proto_perf_apply_func(ucp_proto_perf_t *perf, ucs_linear_func_t func,
                               const char *name, const char *desc_fmt, ...)
{
    ucp_proto_perf_segment_t *seg;
    ucp_proto_perf_factor_id_t factor_id;
    va_list ap;
    ucp_proto_perf_node_t *func_node;

    ucp_proto_perf_segment_foreach(seg, perf) {
        for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             ++factor_id) {
            ucp_proto_perf_segment_update_factor(
                    seg, factor_id,
                    ucs_linear_func_compose(func,
                                            seg->perf_factors[factor_id]));
        }

        va_start(ap, desc_fmt);
        func_node = ucp_proto_perf_node_new(UCP_PROTO_PERF_NODE_TYPE_DATA, 0,
                                            name, desc_fmt, ap);
        va_end(ap);

        ucp_proto_perf_node_own_child(seg->node, &func_node);
    }
}

void ucp_proto_perf_stages_apply_func(ucp_proto_perf_stage_t *stages,
                                      unsigned num_stages,
                                      ucs_linear_func_t func)
{
    ucp_proto_perf_factor_id_t factor_id;
    unsigned i;

    for (i = 0; i < num_stages; ++i) {
        for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             ++factor_id) {
            if (ucs_linear_func_is_zero(stages[i].factors[factor_id],
                                        UCP_PROTO_PERF_EPSILON)) {
                continue;
            }

            stages[i].factors[factor_id] = ucs_linear_func_compose(
                    func, stages[i].factors[factor_id]);
        }
    }
}


/* TODO:
 * Reconsider correctness of PPLN perf estimation logic since in case of async
 * operations it seems wrong to choose the longest factor without paying
 * attention to actions that performed simultaneously but aren't a part of the
 * pipeline.
 * 
 * E.g. RTS + rndv/get ppln, in case of async operations simultaneous
 * RTS sends can potentially make another factor the longest one due
 * to different factors overheads in RTS.
 * 
 * That can be potentially solved by calculating all the pipeline logic in the
 * end during initialization of `flat_perf` structure but it is unclear how to
 * distinguish which parts of the factors are part of pipeline and which aren't
 * at that moment.
 */
const ucp_proto_perf_segment_t *
ucp_proto_perf_add_ppln(const ucp_proto_perf_t *perf,
                        ucp_proto_perf_t *ppln_perf, size_t max_length)
{
    ucp_proto_perf_factors_t factors   = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_proto_perf_segment_t *frag_seg = ucs_list_tail(&perf->segments,
                                                       ucp_proto_perf_segment_t,
                                                       list);
    size_t frag_size                   = ucp_proto_perf_segment_end(frag_seg);
    ucp_proto_perf_factor_id_t factor_id, max_factor_id;
    ucs_linear_func_t factor_func;
    ucs_status_t status;
    char frag_str[64];
    ucp_proto_perf_node_t *perf_node;

    if (frag_size >= max_length) {
        return NULL;
    }

    /* Turn all factors overheads to constant and choose longest one */
    max_factor_id = 0;
    ucs_assert(max_factor_id != UCP_PROTO_PERF_FACTOR_LATENCY);
    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        factor_func          = ucp_proto_perf_segment_func(frag_seg, factor_id);
        factors[factor_id].c = ucs_linear_func_apply(factor_func, frag_size);
        if ((factors[factor_id].c > factors[max_factor_id].c) &&
            (factor_id != UCP_PROTO_PERF_FACTOR_LATENCY)) {
            max_factor_id = factor_id;
        }
    }

    /* Longest factor still has linear part */
    factors[max_factor_id]    = ucp_proto_perf_segment_func(frag_seg,
                                                            max_factor_id);
    /* Apply the fragment overhead to the performance function linear part
     * since this overhead exists for each fragment */
    factors[max_factor_id].m += factors[max_factor_id].c / frag_size;

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));
    perf_node = ucp_proto_perf_node_new_data("pipeline", "frag size: %s",
                                             frag_str);
    status    = ucp_proto_perf_add_funcs(ppln_perf, frag_size + 1, max_length,
                                         factors, perf_node,
                                         ucp_proto_perf_segment_node(frag_seg));
    if (status != UCS_OK) {
        return NULL;
    }

    return frag_seg;
}

static size_t
ucp_proto_perf_stage_num_frags(size_t msg_size, size_t frag_size)
{
    return (msg_size == 0) ? 0 : ucs_div_round_up(msg_size, frag_size);
}

static size_t
ucp_proto_perf_stage_frag_range_end(size_t num_frags, size_t frag_size)
{
    if (num_frags == 0) {
        return 0;
    }

    if (num_frags > (SIZE_MAX / frag_size)) {
        return SIZE_MAX;
    }

    return num_frags * frag_size;
}

static int
ucp_proto_perf_stage_role_is_valid(ucp_proto_perf_stage_role_t role)
{
    return (role == UCP_PROTO_PERF_STAGE_ROLE_SETUP) ||
           (role == UCP_PROTO_PERF_STAGE_ROLE_RECURRING) ||
           (role == UCP_PROTO_PERF_STAGE_ROLE_DRAIN) ||
           (role == UCP_PROTO_PERF_STAGE_ROLE_CONTROL);
}

static int
ucp_proto_perf_stage_overlap_is_valid(ucp_proto_perf_stage_overlap_t overlap)
{
    return (overlap == UCP_PROTO_PERF_STAGE_OVERLAP_SERIAL) ||
           (overlap == UCP_PROTO_PERF_STAGE_OVERLAP_PARALLEL) ||
           (overlap == UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL);
}

static int
ucp_proto_perf_stage_is_recurring_serial(const ucp_proto_perf_stage_t *stage)
{
    return (stage->role == UCP_PROTO_PERF_STAGE_ROLE_RECURRING) &&
           ((stage->overlap == UCP_PROTO_PERF_STAGE_OVERLAP_SERIAL) ||
            (stage->overlap == UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL));
}

static int
ucp_proto_perf_stage_same_serial_resource(const ucp_proto_perf_stage_t *stage1,
                                          const ucp_proto_perf_stage_t *stage2)
{
    if (!ucp_proto_perf_stage_is_recurring_serial(stage1) ||
        !ucp_proto_perf_stage_is_recurring_serial(stage2)) {
        return 0;
    }

    if ((stage1->overlap == UCP_PROTO_PERF_STAGE_OVERLAP_SERIAL) ||
        (stage2->overlap == UCP_PROTO_PERF_STAGE_OVERLAP_SERIAL)) {
        return stage1->overlap == stage2->overlap;
    }

    return stage1->resource_id == stage2->resource_id;
}

static ucs_linear_func_t
ucp_proto_perf_stage_factor(const ucp_proto_perf_stage_t *stage,
                            ucp_proto_perf_factor_id_t factor_id,
                            size_t num_frags)
{
    ucs_linear_func_t func = stage->factors[factor_id];

    if (stage->role == UCP_PROTO_PERF_STAGE_ROLE_RECURRING) {
        func.c *= num_frags;
    }

    return func;
}

static void
ucp_proto_perf_stage_add_factor(ucp_proto_perf_factors_t factors,
                                ucp_proto_perf_factor_id_t factor_id,
                                ucs_linear_func_t func)
{
    if (ucs_linear_func_is_zero(func, UCP_PROTO_PERF_EPSILON)) {
        return;
    }

    ucs_linear_func_add_inplace(&factors[factor_id], func);
}

static ucp_proto_perf_factor_id_t
ucp_proto_perf_stage_first_factor(const ucp_proto_perf_stage_t *stage)
{
    ucp_proto_perf_factor_id_t factor_id;

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST_WO_LATENCY;
         ++factor_id) {
        if (!ucs_linear_func_is_zero(stage->factors[factor_id],
                                     UCP_PROTO_PERF_EPSILON)) {
            return factor_id;
        }
    }

    return UCP_PROTO_PERF_FACTOR_LAST;
}

static ucp_proto_perf_factor_id_t
ucp_proto_perf_stage_serial_group_factor(const ucp_proto_perf_stage_t *stages,
                                         unsigned num_stages,
                                         unsigned stage_index)
{
    ucp_proto_perf_factor_id_t factor_id;
    unsigned i;

    for (i = stage_index; i < num_stages; ++i) {
        if (!ucp_proto_perf_stage_same_serial_resource(&stages[stage_index],
                                                       &stages[i])) {
            continue;
        }

        factor_id = ucp_proto_perf_stage_first_factor(&stages[i]);
        if (factor_id != UCP_PROTO_PERF_FACTOR_LAST) {
            return factor_id;
        }
    }

    return UCP_PROTO_PERF_FACTOR_LAST;
}

static void
ucp_proto_perf_stage_add_parallel(ucp_proto_perf_factors_t factors,
                                  const ucp_proto_perf_stage_t *stage,
                                  size_t num_frags)
{
    ucp_proto_perf_factor_id_t factor_id;

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
        ucp_proto_perf_stage_add_factor(
                factors, factor_id,
                ucp_proto_perf_stage_factor(stage, factor_id, num_frags));
    }
}

static void
ucp_proto_perf_stage_add_serial_group(ucp_proto_perf_factors_t factors,
                                      const ucp_proto_perf_stage_t *stages,
                                      unsigned num_stages,
                                      unsigned stage_index,
                                      size_t num_frags)
{
    ucp_proto_perf_factor_id_t group_factor_id, factor_id;
    ucs_linear_func_t func;
    unsigned i;

    group_factor_id = ucp_proto_perf_stage_serial_group_factor(
            stages, num_stages, stage_index);
    if (group_factor_id == UCP_PROTO_PERF_FACTOR_LAST) {
        group_factor_id = UCP_PROTO_PERF_FACTOR_LOCAL_CPU;
    }

    for (i = stage_index; i < num_stages; ++i) {
        if (!ucp_proto_perf_stage_same_serial_resource(&stages[stage_index],
                                                       &stages[i])) {
            continue;
        }

        for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             ++factor_id) {
            func = ucp_proto_perf_stage_factor(&stages[i], factor_id,
                                               num_frags);
            if (factor_id == UCP_PROTO_PERF_FACTOR_LATENCY) {
                ucp_proto_perf_stage_add_factor(factors, factor_id, func);
            } else {
                ucp_proto_perf_stage_add_factor(factors, group_factor_id,
                                                func);
            }
        }
    }
}

static int
ucp_proto_perf_stage_serial_group_was_added(const ucp_proto_perf_stage_t *stages,
                                            unsigned stage_index)
{
    unsigned i;

    for (i = 0; i < stage_index; ++i) {
        if (ucp_proto_perf_stage_same_serial_resource(&stages[i],
                                                      &stages[stage_index])) {
            return 1;
        }
    }

    return 0;
}

static void
ucp_proto_perf_staged_pipeline_factors(ucp_proto_perf_factors_t factors,
                                       const ucp_proto_perf_stage_t *stages,
                                       unsigned num_stages,
                                       size_t num_frags)
{
    unsigned i;

    for (i = 0; i < num_stages; ++i) {
        if (ucp_proto_perf_stage_is_recurring_serial(&stages[i])) {
            if (!ucp_proto_perf_stage_serial_group_was_added(stages, i)) {
                ucp_proto_perf_stage_add_serial_group(factors, stages,
                                                      num_stages, i,
                                                      num_frags);
            }
        } else {
            ucp_proto_perf_stage_add_parallel(factors, &stages[i], num_frags);
        }
    }
}

static void
ucp_proto_perf_stage_add_recurring_fixed(ucp_proto_perf_factors_t factors,
                                         ucp_proto_perf_factor_id_t factor_id,
                                         ucs_linear_func_t func)
{
    func.m = 0;
    ucp_proto_perf_stage_add_factor(factors, factor_id, func);
}

static void
ucp_proto_perf_stage_add_parallel_recurring_fixed(
        ucp_proto_perf_factors_t factors, const ucp_proto_perf_stage_t *stage)
{
    ucp_proto_perf_factor_id_t factor_id;

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
         ++factor_id) {
        ucp_proto_perf_stage_add_recurring_fixed(factors, factor_id,
                                                 stage->factors[factor_id]);
    }
}

static void
ucp_proto_perf_stage_add_serial_group_recurring_fixed(
        ucp_proto_perf_factors_t factors, const ucp_proto_perf_stage_t *stages,
        unsigned num_stages, unsigned stage_index)
{
    ucp_proto_perf_factor_id_t group_factor_id, factor_id;
    unsigned i;

    group_factor_id = ucp_proto_perf_stage_serial_group_factor(
            stages, num_stages, stage_index);
    if (group_factor_id == UCP_PROTO_PERF_FACTOR_LAST) {
        group_factor_id = UCP_PROTO_PERF_FACTOR_LOCAL_CPU;
    }

    for (i = stage_index; i < num_stages; ++i) {
        if (!ucp_proto_perf_stage_same_serial_resource(&stages[stage_index],
                                                       &stages[i])) {
            continue;
        }

        for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             ++factor_id) {
            ucp_proto_perf_stage_add_recurring_fixed(
                    factors, group_factor_id, stages[i].factors[factor_id]);
        }
    }
}

static void
ucp_proto_perf_staged_pipeline_recurring_fixed(
        ucp_proto_perf_factors_t factors, const ucp_proto_perf_stage_t *stages,
        unsigned num_stages)
{
    unsigned i;

    for (i = 0; i < num_stages; ++i) {
        if (stages[i].role != UCP_PROTO_PERF_STAGE_ROLE_RECURRING) {
            continue;
        }

        if (ucp_proto_perf_stage_is_recurring_serial(&stages[i])) {
            if (!ucp_proto_perf_stage_serial_group_was_added(stages, i)) {
                ucp_proto_perf_stage_add_serial_group_recurring_fixed(
                        factors, stages, num_stages, i);
            }
        } else {
            ucp_proto_perf_stage_add_parallel_recurring_fixed(factors,
                                                              &stages[i]);
        }
    }
}

static void
ucp_proto_perf_staged_pipeline_make_tail_factors(
        ucp_proto_perf_factors_t factors, const ucp_proto_perf_stage_t *stages,
        unsigned num_stages, size_t frag_size, size_t range_start,
        size_t num_frags)
{
    ucp_proto_perf_factors_t fixed_factors;
    ucp_proto_perf_factor_id_t factor_id;
    double fixed_slope;

    ucp_proto_perf_staged_pipeline_factors(factors, stages, num_stages,
                                           num_frags);

    memset(fixed_factors, 0, sizeof(fixed_factors));
    ucp_proto_perf_staged_pipeline_recurring_fixed(fixed_factors, stages,
                                                   num_stages);

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
        fixed_slope = fixed_factors[factor_id].c / frag_size;
        factors[factor_id].m += fixed_slope;
        factors[factor_id].c -= fixed_slope * range_start;
    }
}

static ucs_status_t
ucp_proto_perf_staged_pipeline_check_params(size_t range_start,
                                            size_t range_end,
                                            const ucp_proto_perf_stage_t *stages,
                                            unsigned num_stages,
                                            size_t frag_size)
{
    unsigned i;

    if ((num_stages == 0) || (stages == NULL) || (frag_size == 0) ||
        (range_start > range_end)) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < num_stages; ++i) {
        if (!ucp_proto_perf_stage_role_is_valid(stages[i].role) ||
            !ucp_proto_perf_stage_overlap_is_valid(stages[i].overlap) ||
            ((stages[i].frag_size != 0) &&
             (stages[i].frag_size != frag_size))) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}

static ucs_status_t
ucp_proto_perf_add_staged_pipeline_tail(
        ucp_proto_perf_t *ppln_perf, size_t range_start, size_t range_end,
        const ucp_proto_perf_stage_t *stages, unsigned num_stages,
        size_t frag_size, ucp_proto_perf_node_t *child_perf_node)
{
    ucp_proto_perf_factors_t factors;
    ucp_proto_perf_node_t *perf_node;
    size_t num_frags;
    unsigned i;
    char frag_str[64];

    memset(factors, 0, sizeof(factors));
    num_frags = ucp_proto_perf_stage_num_frags(range_start, frag_size);

    ucp_proto_perf_staged_pipeline_make_tail_factors(factors, stages,
                                                     num_stages, frag_size,
                                                     range_start, num_frags);

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));
    perf_node = ucp_proto_perf_node_new_data(
            "staged pipeline", "frag size: %s, fragments: %zu+", frag_str,
            num_frags);
    for (i = 0; i < num_stages; ++i) {
        ucp_proto_perf_node_add_child(perf_node, stages[i].perf_node);
    }

    /*
     * Exact per-fragment ranges preserve small boundary behavior. The tail
     * keeps the exact value at range_start and amortizes recurring fixed
     * fragment cost into the slope, avoiding unbounded range enumeration.
     */
    return ucp_proto_perf_add_funcs(ppln_perf, range_start, range_end,
                                    factors, perf_node, child_perf_node);
}

ucs_status_t
ucp_proto_perf_add_staged_pipeline(ucp_proto_perf_t *ppln_perf,
                                   size_t range_start, size_t range_end,
                                   const ucp_proto_perf_stage_t *stages,
                                   unsigned num_stages, size_t frag_size,
                                   ucp_proto_perf_node_t *child_perf_node)
{
    ucp_proto_perf_factors_t factors;
    ucp_proto_perf_node_t *perf_node;
    ucs_status_t status;
    size_t range_iter, range_iter_end, num_frags;
    unsigned i;
    char frag_str[64];

    status = ucp_proto_perf_staged_pipeline_check_params(
            range_start, range_end, stages, num_stages, frag_size);
    if (status != UCS_OK) {
        return status;
    }

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));

    range_iter = range_start;
    while (range_iter <= range_end) {
        memset(factors, 0, sizeof(factors));
        num_frags      = ucp_proto_perf_stage_num_frags(range_iter,
                                                        frag_size);
        if (num_frags >
            UCP_PROTO_PERF_STAGED_PIPELINE_MAX_EXACT_FRAGS) {
            return ucp_proto_perf_add_staged_pipeline_tail(
                    ppln_perf, range_iter, range_end, stages, num_stages,
                    frag_size, child_perf_node);
        }

        range_iter_end = ucp_proto_perf_stage_frag_range_end(num_frags,
                                                             frag_size);
        range_iter_end = ucs_min(range_iter_end, range_end);

        ucp_proto_perf_staged_pipeline_factors(factors, stages, num_stages,
                                               num_frags);

        perf_node = ucp_proto_perf_node_new_data(
                "staged pipeline", "frag size: %s, fragments: %zu", frag_str,
                num_frags);
        for (i = 0; i < num_stages; ++i) {
            ucp_proto_perf_node_add_child(perf_node, stages[i].perf_node);
        }

        status = ucp_proto_perf_add_funcs(ppln_perf, range_iter,
                                          range_iter_end, factors, perf_node,
                                          child_perf_node);
        if (status != UCS_OK) {
            return status;
        }

        if (range_iter_end == SIZE_MAX) {
            break;
        }

        range_iter = range_iter_end + 1;
    }

    return UCS_OK;
}

static int
ucp_proto_perf_factor_stage_resource(ucp_proto_perf_factor_id_t factor_id,
                                     uint64_t *resource_id_p)
{
    switch (factor_id) {
    case UCP_PROTO_PERF_FACTOR_LOCAL_CPU:
    case UCP_PROTO_PERF_FACTOR_LOCAL_TL:
    case UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY:
        *resource_id_p = UCP_PROTO_PERF_STAGE_RESOURCE_LOCAL;
        return 1;
    case UCP_PROTO_PERF_FACTOR_REMOTE_CPU:
    case UCP_PROTO_PERF_FACTOR_REMOTE_TL:
    case UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY:
        *resource_id_p = UCP_PROTO_PERF_STAGE_RESOURCE_REMOTE;
        return 1;
    default:
        return 0;
    }
}

ucs_status_t
ucp_proto_perf_segment_make_stages(const ucp_proto_perf_segment_t *seg,
                                   size_t frag_size,
                                   ucp_proto_perf_stage_t *stages,
                                   unsigned max_stages,
                                   unsigned *num_stages_p)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucs_linear_func_t factor;
    uint64_t resource_id;
    unsigned num_stages = 0;

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
        factor = ucp_proto_perf_segment_func(seg, factor_id);
        if (ucs_linear_func_is_zero(factor, UCP_PROTO_PERF_EPSILON) ||
            !ucp_proto_perf_factor_stage_resource(factor_id, &resource_id)) {
            continue;
        }

        if (num_stages == max_stages) {
            return UCS_ERR_EXCEEDS_LIMIT;
        }

        memset(&stages[num_stages], 0, sizeof(stages[num_stages]));
        stages[num_stages].name      = ucp_proto_perf_factor_names[factor_id];
        stages[num_stages].role      = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
        stages[num_stages].overlap   = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
        stages[num_stages].frag_size = frag_size;
        stages[num_stages].resource_id = resource_id;
        stages[num_stages].factors[factor_id] = factor;
        ++num_stages;
    }

    *num_stages_p = num_stages;
    return UCS_OK;
}

const ucp_proto_perf_segment_t *
ucp_proto_perf_add_ppln_staged(const ucp_proto_perf_t *frag_perf,
                              ucp_proto_perf_t *ppln_perf,
                              size_t max_length,
                              const ucp_proto_perf_stage_t *stages,
                              unsigned num_stages)
{
    ucp_proto_perf_segment_t *frag_seg;
    size_t frag_size;
    ucs_status_t status;

    if (num_stages == 0) {
        return ucp_proto_perf_add_ppln(frag_perf, ppln_perf, max_length);
    }

    frag_seg  = ucs_list_tail(&frag_perf->segments, ucp_proto_perf_segment_t,
                              list);
    frag_size = ucp_proto_perf_segment_end(frag_seg);
    if (frag_size >= max_length) {
        return NULL;
    }

    status = ucp_proto_perf_add_staged_pipeline(
            ppln_perf, frag_size + 1, max_length, stages, num_stages,
            frag_size, ucp_proto_perf_segment_node(frag_seg));
    if (status != UCS_OK) {
        return NULL;
    }

    return frag_seg;
}

ucs_status_t ucp_proto_perf_remote(const ucp_proto_perf_t *remote_perf,
                                   ucp_proto_perf_t **perf_p)
{
    ucp_proto_perf_factor_id_t convert_map[][2] = {
        {UCP_PROTO_PERF_FACTOR_LOCAL_CPU, UCP_PROTO_PERF_FACTOR_REMOTE_CPU},
        {UCP_PROTO_PERF_FACTOR_LOCAL_TL, UCP_PROTO_PERF_FACTOR_REMOTE_TL},
        {UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY,
         UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY}
    };
    ucp_proto_perf_factor_id_t(*convert_pair)[2];
    ucp_proto_perf_segment_t *remote_seg, *new_seg;
    ucp_proto_perf_factors_t perf_factors;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    ucp_proto_perf_check(remote_perf);

    status = ucp_proto_perf_create("remote", &perf);
    if (status != UCS_OK) {
        return status;
    }

    /* Convert local factors to remote and vice versa */
    ucp_proto_perf_segment_foreach(remote_seg, remote_perf) {
        ucs_carray_for_each(convert_pair, convert_map,
                            ucs_static_array_size(convert_map)) {
            perf_factors[(*convert_pair)[0]] =
                    remote_seg->perf_factors[(*convert_pair)[1]];
            perf_factors[(*convert_pair)[1]] =
                    remote_seg->perf_factors[(*convert_pair)[0]];
        }
        perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY] =
                remote_seg->perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY];

        status = ucp_proto_perf_segment_new(perf, remote_seg->start,
                                            remote_seg->end, &new_seg);
        if (status != UCS_OK) {
            goto err_cleanup_perf;
        }

        ucs_list_add_tail(&perf->segments, &new_seg->list);
        ucp_proto_perf_segment_add_funcs(perf, new_seg, perf_factors,
                                         remote_seg->node);
    }

    *perf_p = perf;
    return UCS_OK;

err_cleanup_perf:
    ucp_proto_perf_destroy(perf);
    return status;
}

static ucs_status_t
ucp_proto_flat_perf_alloc(ucp_proto_flat_perf_t **flat_perf_p)
{
    ucp_proto_flat_perf_t *flat_perf;

    flat_perf = ucs_malloc(sizeof(*flat_perf), "flat_perf");
    if (flat_perf == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *flat_perf_p = flat_perf;
    return UCS_OK;
}

static ucs_linear_func_t
ucp_proto_perf_factors_sum(const ucs_linear_func_t *factors)
{
    ucp_proto_perf_factor_id_t factor_id;
    ucs_linear_func_t sum = UCS_LINEAR_FUNC_ZERO;

    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
         ++factor_id) {
        ucs_linear_func_add_inplace(&sum, factors[factor_id]);
    }

    return sum;
}

static void
ucp_proto_perf_staged_pipeline_segment_factors(
        ucp_proto_perf_factors_t factors,
        const ucp_proto_perf_stage_t *stages, unsigned num_stages,
        size_t frag_size, size_t range_start)
{
    size_t num_frags;

    memset(factors, 0, sizeof(ucp_proto_perf_factors_t));
    num_frags = ucp_proto_perf_stage_num_frags(range_start, frag_size);
    if (num_frags > UCP_PROTO_PERF_STAGED_PIPELINE_MAX_EXACT_FRAGS) {
        ucp_proto_perf_staged_pipeline_make_tail_factors(
                factors, stages, num_stages, frag_size, range_start,
                num_frags);
    } else {
        ucp_proto_perf_staged_pipeline_factors(factors, stages, num_stages,
                                               num_frags);
    }
}

ucs_status_t
ucp_proto_perf_staged_pipeline_flat(const ucp_proto_perf_t *perf,
                                    const ucp_proto_perf_stage_t *stages,
                                    unsigned num_stages,
                                    ucp_proto_flat_perf_t **flat_perf_ptr)
{
    ucp_proto_perf_envelope_elem_t *envelope_elem;
    ucp_proto_perf_factors_t stage_factors;
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_envelope_t envelope;
    ucp_proto_flat_perf_t *flat_perf;
    ucs_linear_func_t non_stage_func;
    ucs_linear_func_t stage_latency;
    ucs_linear_func_t stage_sum;
    ucs_status_t status;
    size_t range_start;
    size_t frag_size;

    ucs_assert(num_stages > 0);
    frag_size = stages[0].frag_size;

    status = ucp_proto_flat_perf_alloc(&flat_perf);
    if (status != UCS_OK) {
        return status;
    }

    ucp_proto_perf_check(perf);

    ucs_array_init_dynamic(flat_perf);
    ucs_array_init_dynamic(&envelope);
    ucp_proto_perf_segment_foreach(seg, perf) {
        ucs_array_clear(&envelope);
        ucp_proto_perf_staged_pipeline_segment_factors(
                stage_factors, stages, num_stages, frag_size, seg->start);

        stage_latency = stage_factors[UCP_PROTO_PERF_FACTOR_LATENCY];
        stage_sum     = ucp_proto_perf_factors_sum(stage_factors);
        non_stage_func = ucs_linear_func_sub(
                ucp_proto_perf_factors_sum(seg->perf_factors), stage_sum);
        stage_factors[UCP_PROTO_PERF_FACTOR_LATENCY] = UCS_LINEAR_FUNC_ZERO;

        status = ucp_proto_perf_envelope_make(
                stage_factors, UCP_PROTO_PERF_FACTOR_LAST_WO_LATENCY,
                seg->start, seg->end, 0, &envelope);
        if (status != UCS_OK) {
            goto err_cleanup;
        }

        range_start = seg->start;
        ucs_array_for_each(envelope_elem, &envelope) {
            range        = ucs_array_append(flat_perf,
                                            status = UCS_ERR_NO_MEMORY;
                                            goto err_cleanup);
            range->start = range_start;
            range->end   = envelope_elem->max_length;
            range->value = non_stage_func;
            ucs_linear_func_add_inplace(&range->value, stage_latency);
            ucs_linear_func_add_inplace(
                    &range->value, stage_factors[envelope_elem->index]);
            range->node  = ucp_proto_perf_node_new_data(perf->name,
                                                        "staged flat perf");
            ucp_proto_perf_node_add_child(range->node, seg->node);
            ucp_proto_perf_node_add_data(range->node, "total", range->value);

            range_start = envelope_elem->max_length + 1;
        }
    }

    *flat_perf_ptr = flat_perf;
    ucs_array_cleanup_dynamic(&envelope);
    return UCS_OK;

err_cleanup:
    ucp_proto_flat_perf_destroy(flat_perf);
    ucs_array_cleanup_dynamic(&envelope);
    return status;
}

ucs_status_t ucp_proto_perf_envelope(const ucp_proto_perf_t *perf, int convex,
                                     ucp_proto_flat_perf_t **flat_perf_ptr)
{
    ucp_proto_perf_envelope_elem_t *envelope_elem;
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_envelope_t envelope;
    ucp_proto_flat_perf_t *flat_perf;
    ucs_status_t status;
    size_t range_start;

    status = ucp_proto_flat_perf_alloc(&flat_perf);
    if (status != UCS_OK) {
        return status;
    }

    ucp_proto_perf_check(perf);

    ucs_array_init_dynamic(flat_perf);
    ucs_array_init_dynamic(&envelope);
    ucp_proto_perf_segment_foreach(seg, perf) {
        ucs_array_clear(&envelope);
        status = ucp_proto_perf_envelope_make(
                seg->perf_factors, UCP_PROTO_PERF_FACTOR_LAST_WO_LATENCY,
                seg->start, seg->end, convex, &envelope);
        if (status != UCS_OK) {
            goto err_cleanup;
        }

        range_start = seg->start;
        ucs_array_for_each(envelope_elem, &envelope) {
            range        = ucs_array_append(flat_perf,
                                            status = UCS_ERR_NO_MEMORY;
                                            goto err_cleanup);
            range->start = range_start;
            range->end   = envelope_elem->max_length;
            range->value = seg->perf_factors[envelope_elem->index];
            range->node  = ucp_proto_perf_node_new_data(
                    perf->name, ucp_envelope_convex_names[convex]);
            ucp_proto_perf_node_add_child(range->node, seg->node);
            ucp_proto_perf_node_add_data(range->node, "total", range->value);

            range_start = envelope_elem->max_length + 1;
        }
    }

    *flat_perf_ptr = flat_perf;
    ucs_array_cleanup_dynamic(&envelope);
    return UCS_OK;

err_cleanup:
    ucp_proto_flat_perf_destroy(flat_perf);
    ucs_array_cleanup_dynamic(&envelope);
    return status;
}

ucs_status_t ucp_proto_perf_sum(const ucp_proto_perf_t *perf,
                                ucp_proto_flat_perf_t **flat_perf_ptr)
{
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_factor_id_t factor_id;
    ucp_proto_flat_perf_t *flat_perf;
    ucs_status_t status;

    status = ucp_proto_flat_perf_alloc(&flat_perf);
    if (status != UCS_OK) {
        return status;
    }

    ucs_array_init_dynamic(flat_perf);
    ucp_proto_perf_segment_foreach(seg, perf) {
        range        = ucs_array_append(flat_perf, status = UCS_ERR_NO_MEMORY;
                                        goto err_cleanup);
        range->start = seg->start;
        range->end   = seg->end;
        range->value = UCS_LINEAR_FUNC_ZERO;
        range->node  = ucp_proto_perf_node_new_data(perf->name, "flat perf");

        for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             factor_id++) {
            ucs_linear_func_add_inplace(&range->value,
                                        seg->perf_factors[factor_id]);
        }

        ucp_proto_perf_node_add_child(range->node, seg->node);
        ucp_proto_perf_node_add_data(range->node, "sum", range->value);
    }

    *flat_perf_ptr = flat_perf;
    return UCS_OK;

err_cleanup:
    ucp_proto_flat_perf_destroy(flat_perf);
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

ucp_proto_perf_segment_t *
ucp_proto_perf_find_segment_tail(const ucp_proto_perf_t *perf)
{
    if (ucs_list_is_empty(&perf->segments)) {
        return NULL;
    }

    return ucs_list_tail(&perf->segments, ucp_proto_perf_segment_t, list);
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

void ucp_proto_flat_perf_str(const ucp_proto_flat_perf_t *flat_perf,
                             ucs_string_buffer_t *strb)
{
    ucp_proto_flat_perf_range_t *range;
    char range_str[64];

    ucs_array_for_each(range, flat_perf) {
        ucs_memunits_range_str(range->start, range->end, range_str,
                               sizeof(range_str));
        ucs_string_buffer_appendf(strb, "%s {", range_str);
        ucs_string_buffer_appendf(strb, UCP_PROTO_PERF_FUNC_FMT,
                                  UCP_PROTO_PERF_FUNC_ARG(&range->value));
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

void ucp_proto_flat_perf_destroy(ucp_proto_flat_perf_t *flat_perf)
{
    ucp_proto_flat_perf_range_t *range;

    ucs_array_for_each(range, flat_perf) {
        ucp_proto_perf_node_deref(&range->node);
    }

    ucs_array_cleanup_dynamic(flat_perf);
    ucs_free(flat_perf);
}
