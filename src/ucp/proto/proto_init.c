/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_init.h"
#include "proto_debug.h"
#include "proto_select.inl"

#include <ucs/datastruct/array.inl>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/debug/log.h>
#include <float.h>


/* Compare two protocols which intersect at point X, by examining their value
 * at point (X + UCP_PROTO_MSGLEN_EPSILON)
 */
#define UCP_PROTO_MSGLEN_EPSILON   0.5

UCS_ARRAY_IMPL(ucp_proto_perf_envelope, unsigned,
               ucp_proto_perf_envelope_elem_t,);


void ucp_proto_common_add_ppln_range(const ucp_proto_init_params_t *init_params,
                                     const ucp_proto_perf_range_t *frag_range,
                                     size_t max_length)
{
    ucp_proto_caps_t *caps = init_params->caps;
    ucp_proto_perf_range_t *ppln_range;
    double frag_overhead;

    /* Add pipelined range */
    ppln_range = &caps->ranges[caps->num_ranges++];

    /* Overhead of sending one fragment before starting the pipeline */
    frag_overhead =
            ucs_linear_func_apply(frag_range->perf[UCP_PROTO_PERF_TYPE_SINGLE],
                                  frag_range->max_length) -
            ucs_linear_func_apply(frag_range->perf[UCP_PROTO_PERF_TYPE_MULTI],
                                  frag_range->max_length);

    ppln_range->max_length = max_length;

    /* Apply the pipelining effect when sending multiple fragments */
    ppln_range->perf[UCP_PROTO_PERF_TYPE_SINGLE] =
            ucs_linear_func_add(frag_range->perf[UCP_PROTO_PERF_TYPE_MULTI],
                                ucs_linear_func_make(frag_overhead, 0));

    /* Multiple send performance is the same */
    ppln_range->perf[UCP_PROTO_PERF_TYPE_MULTI] =
            frag_range->perf[UCP_PROTO_PERF_TYPE_MULTI];

    ppln_range->name = "ppln";

    ucs_trace("frag-size: %zd" UCP_PROTO_TIME_FMT(frag_overhead),
              frag_range->max_length, UCP_PROTO_TIME_ARG(frag_overhead));
}

void ucp_proto_common_init_base_caps(
        const ucp_proto_common_init_params_t *params, size_t min_length)
{
    ucp_proto_caps_t *caps = params->super.caps;

    caps->cfg_thresh   = params->cfg_thresh;
    caps->cfg_priority = params->cfg_priority;
    caps->min_length   = ucs_max(params->min_length, min_length);
    caps->num_ranges   = 0;
}

void ucp_proto_perf_envelope_append(ucp_proto_perf_envelope_t *list,
                                    const char *name,
                                    const ucp_proto_perf_range_t *range,
                                    size_t frag_size, ucs_linear_func_t bias)
{
    ucs_linear_func_t base, frag_overhead;
    ucp_proto_perf_envelope_elem_t *elem;

    elem                   = ucs_array_append_fixed(ucp_proto_perf_envelope,
                                                    list);
    elem->id               = 0;
    elem->range.super.name = name;
    elem->range.super.perf[UCP_PROTO_PERF_TYPE_SINGLE]
        = range->perf[UCP_PROTO_PERF_TYPE_SINGLE];

    base = range->perf[UCP_PROTO_PERF_TYPE_MULTI];
    /* account for the overhead of each fragment of a multi-fragment message */
    frag_overhead = ucs_linear_func_make(base.c,
            base.m + (base.c / frag_size));

    elem->range.super.perf[UCP_PROTO_PERF_TYPE_MULTI]
        = ucs_linear_func_compose(bias, frag_overhead);
    ucs_trace("%s"
            UCP_PROTO_PERF_FUNC_FMT(single)
            UCP_PROTO_PERF_FUNC_FMT(base)
            UCP_PROTO_PERF_FUNC_FMT(frag_overhead)
            UCP_PROTO_PERF_FUNC_FMT(multi), name,
            UCP_PROTO_PERF_FUNC_ARG(
                &elem->range.super.perf[UCP_PROTO_PERF_TYPE_SINGLE]),
            UCP_PROTO_PERF_FUNC_ARG(&base),
            UCP_PROTO_PERF_FUNC_ARG(&frag_overhead),
            UCP_PROTO_PERF_FUNC_ARG(
                &elem->range.super.perf[UCP_PROTO_PERF_TYPE_MULTI]));
}

ucs_status_t
ucp_proto_perf_envelope_make(const ucp_proto_perf_envelope_t *list,
                             ucp_proto_perf_envelope_t *envelope_list,
                             size_t min_length, size_t max_length,
                             ucp_proto_perf_type_t perf_type, int convex)
{
    size_t start = min_length;
    char num_str[64];
    struct {
        int                                  idx;
        double                               result;
        const ucp_proto_perf_envelope_elem_t *elem;
    } curr, best;
    ucp_proto_perf_envelope_elem_t *new_elem;
    ucp_proto_perf_type_t tmp_perf_type;
    ucs_status_t status;
    size_t midpoint;
    double x_intersect;
    uint64_t mask;

    ucs_assert(ucs_array_length(list) < 64);
    mask = UCS_MASK(ucs_array_length(list));

    do {
        ucs_assert(mask != 0);

        /* Find best trend at the 'start' point */
        best.idx    = -1;
        best.result = DBL_MAX;
        best.elem   = NULL;
        ucs_for_each_bit(curr.idx, mask) {
            curr.elem   = &ucs_array_elem(list, curr.idx);
            curr.result = ucs_linear_func_apply(
                    curr.elem->range.super.perf[perf_type],
                    start + UCP_PROTO_MSGLEN_EPSILON);
            ucs_assert(curr.result != DBL_MAX);
            if ((best.elem == NULL) ||
                ((curr.result < best.result) == convex)) {
                best = curr;
            }
        }

        /* Since mask != 0, we should find at least one trend */
        ucs_assert(best.elem != NULL);
        ucs_trace("at %s: selected %s",
                  ucs_memunits_to_str(start, num_str, sizeof(num_str)),
                  best.elem->range.super.name);
        ucs_log_indent(1);

        /* Find first (smallest) intersection point between the current best
         * trend and any other trend. This would be the point where that
         * other trend becomes the best one.
         */
        midpoint = max_length;
        mask    &= ~UCS_BIT(best.idx);
        ucs_for_each_bit(curr.idx, mask) {
            curr.elem = &ucs_array_elem(list, curr.idx);
            status = ucs_linear_func_intersect(
                    curr.elem->range.super.perf[perf_type],
                    best.elem->range.super.perf[perf_type],
                    &x_intersect);
            if ((status == UCS_OK) && (x_intersect > start)) {
                /* We care only if the intersection is after 'start', since
                 * otherwise 'best' is better than 'curr' at
                 * 'end' as well as at 'start'.
                 */
                midpoint = ucs_min(ucs_double_to_sizet(x_intersect, SIZE_MAX),
                                   midpoint);
                ucs_memunits_to_str(midpoint, num_str, sizeof(num_str));
                ucs_trace("intersects with %s at %.2f, midpoint is %s",
                          curr.elem->range.super.name, x_intersect, num_str);
            } else {
                ucs_trace("intersects with %s out of range",
                          curr.elem->range.super.name);
            }
        }
        ucs_log_indent(-1);

        status = ucs_array_append(ucp_proto_perf_envelope, envelope_list);
        if (status != UCS_OK) {
            return status;
        }

        new_elem                          = ucs_array_last(envelope_list);
        new_elem->id                      = best.elem->id;
        new_elem->range.cfg_thresh        = best.elem->range.cfg_thresh;
        new_elem->range.super.name        = best.elem->range.super.name;
        new_elem->range.super.max_length  = midpoint;
        for (tmp_perf_type = 0; tmp_perf_type < UCP_PROTO_PERF_TYPE_LAST;
             tmp_perf_type++) {
            new_elem->range.super.perf[tmp_perf_type]
                = best.elem->range.super.perf[tmp_perf_type];
        }

        start = midpoint + 1;
    } while (midpoint < max_length);

    return UCS_OK;
}

ucs_status_t ucp_proto_common_add_perf_ranges(
        const ucp_proto_common_init_params_t *params,
        size_t min_length, size_t max_length,
        const ucp_proto_perf_envelope_t *parallel_stage_list)
{
    ucs_linear_func_t sum_perf = ucs_linear_func_make(0, 0);
    ucp_proto_caps_t *caps     = params->super.caps;
    ucp_proto_perf_range_t *range;
    UCS_ARRAY_DEFINE_ONSTACK(concave, ucp_proto_perf_envelope, 4);
    ucp_proto_perf_envelope_elem_t *elem;
    ucs_status_t status;
    char num_str[64];

    ucs_trace("range[%d..] %s", caps->num_ranges,
              ucs_memunits_range_str(min_length, max_length, num_str,
                                     sizeof(num_str)));

    /* Single-fragment is adding overheads and transfer time */
    ucs_array_for_each(elem, parallel_stage_list) {
        ucs_linear_func_add_inplace(&sum_perf,
                elem->range.super.perf[UCP_PROTO_PERF_TYPE_SINGLE]);
    }

    /* Multi-fragment is pipelining overheads and network transfer */
    status = ucp_proto_perf_envelope_make(parallel_stage_list, &concave,
                                          min_length, max_length,
                                          UCP_PROTO_PERF_TYPE_MULTI, 0);
    if (status != UCS_OK) {
        return status;
    }

    ucs_array_for_each(elem, &concave) {
        range = &caps->ranges[caps->num_ranges++];
        range->name                             = elem->range.super.name;
        range->max_length                       = elem->range.super.max_length;
        /* "single" perfomance estimation is sum of "stages" */
        range->perf[UCP_PROTO_PERF_TYPE_SINGLE] = sum_perf;
        /* "multiple" perfomance estimation is concave envelope of "stages" */
        range->perf[UCP_PROTO_PERF_TYPE_MULTI]
            = elem->range.super.perf[UCP_PROTO_PERF_TYPE_MULTI];
    }

    return UCS_OK;
}

ucs_status_t
ucp_proto_common_init_caps(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *perf,
                           ucp_md_map_t reg_md_map)
{
    UCS_ARRAY_DEFINE_ONSTACK(list, ucp_proto_perf_envelope, 4);
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    ucp_proto_perf_range_t send_perf, xfer_perf, recv_perf;
    ucs_linear_func_t send_overhead, xfer_time, recv_overhead;
    ucs_memory_type_t recv_mem_type;
    uint32_t op_attr_mask;
    ucs_status_t status;
    size_t frag_size;

    ucs_trace("caps" UCP_PROTO_TIME_FMT(send_pre_overhead)
              UCP_PROTO_TIME_FMT(send_post_overhead)
              UCP_PROTO_TIME_FMT(recv_overhead) UCP_PROTO_TIME_FMT(latency),
              UCP_PROTO_TIME_ARG(perf->send_pre_overhead),
              UCP_PROTO_TIME_ARG(perf->send_post_overhead),
              UCP_PROTO_TIME_ARG(perf->recv_overhead),
              UCP_PROTO_TIME_ARG(perf->latency));

    /* Remote access implies zero copy on receiver */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assert(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY);
    }

    op_attr_mask = ucp_proto_select_op_attr_from_flags(select_param->op_flags);

    /* Calculate sender overhead */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        send_overhead = ucp_proto_common_memreg_time(params, reg_md_map);
    } else if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR) {
        send_overhead = ucs_linear_func_make(0, 0);
    } else {
        ucs_assert(reg_md_map == 0);
        status = ucp_proto_common_buffer_copy_time(
                params->super.worker, "send-copy", UCS_MEMORY_TYPE_HOST,
                select_param->mem_type, params->memtype_op, &send_overhead);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Add constant CPU overhead */
    send_overhead.c += perf->send_pre_overhead;

    send_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE]   = send_overhead;
    send_perf.perf[UCP_PROTO_PERF_TYPE_MULTI]    = send_overhead;
    send_perf.perf[UCP_PROTO_PERF_TYPE_MULTI].c += perf->send_post_overhead;

    /* Calculate transport time */
    if ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        /* If we care only about time to start sending the message, ignore
           the transport time */
        xfer_time = ucs_linear_func_make(0, 0);
    } else {
        xfer_time = ucs_linear_func_make(0, 1.0 / perf->bandwidth);
    }

    xfer_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE]    = xfer_time;
    xfer_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE].c += perf->latency +
                                               perf->sys_latency;
    xfer_perf.perf[UCP_PROTO_PERF_TYPE_MULTI]     = xfer_time;

    /*
     * Add the latency of response/ACK back from the receiver.
     */
    if (/* Protocol is waiting for response */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE) ||
        /* Send time is representing request completion, which in case of zcopy
           waits for ACK from remote side. */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
         (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY))) {
        xfer_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE].c += perf->latency;
        send_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE].c += perf->send_post_overhead;
    }

    /* Calculate receiver overhead */
    if (/* Don't care about receiver time for one-sided remote access */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) ||
        /* Count only send completion time without waiting for a response */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
         !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE))) {
        recv_overhead = ucs_linear_func_make(0, 0);
    } else {
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY) {
            /* Receiver has to register its buffer */
            recv_overhead = ucp_proto_common_memreg_time(params, reg_md_map);
        } else {
            if (params->super.rkey_config_key == NULL) {
                /* Assume same memory type as sender */
                recv_mem_type = select_param->mem_type;
            } else {
                recv_mem_type = params->super.rkey_config_key->mem_type;
            }

            /* Receiver has to copy data */
            recv_overhead = ucs_linear_func_make(0, 0); /* silence cppcheck */
            ucp_proto_common_buffer_copy_time(params->super.worker, "recv-copy",
                                              UCS_MEMORY_TYPE_HOST,
                                              recv_mem_type,
                                              UCT_EP_OP_PUT_SHORT,
                                              &recv_overhead);
        }

        /* Receiver has to process the incoming message */
        if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS)) {
            /* latency measure: add remote-side processing time */
            recv_overhead.c += perf->recv_overhead;
        }
    }

    recv_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE] = recv_overhead;
    recv_perf.perf[UCP_PROTO_PERF_TYPE_MULTI]  = recv_overhead;

    /* Get fragment size */
    ucs_assert(perf->max_frag >= params->hdr_size);
    frag_size = ucs_min(params->max_length, perf->max_frag - params->hdr_size);

    /* Initialize capabilities */
    ucp_proto_common_init_base_caps(params, perf->min_length);
    ucp_proto_perf_envelope_append(&list, "send", &send_perf, frag_size,
                                   ucs_linear_func_make(0, 1));
    ucp_proto_perf_envelope_append(&list, "xfer", &xfer_perf, frag_size,
                                   ucs_linear_func_make(0, 1));
    ucp_proto_perf_envelope_append(&list, "recv", &recv_perf, frag_size,
                                   ucs_linear_func_make(0, 1));

    /* Add ranges representing sending single fragment */
    status = ucp_proto_common_add_perf_ranges(params, 0, frag_size, &list);
    if (status != UCS_OK) {
        return status;
    }

    /* Append range representing sending rest of the fragments, if frag_size is
       not the max length and the protocol supports fragmentation */
    if ((frag_size < params->max_length) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG)) {
        ucp_proto_common_add_ppln_range(
                &params->super,
                &params->super.caps->ranges[params->super.caps->num_ranges - 1],
                params->max_length);
    }

    return UCS_OK;
}
