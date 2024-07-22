/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_init.h"
#include "proto_debug.h"
#include "proto_select.inl"

#include <ucp/core/ucp_ep.inl>
#include <ucs/datastruct/array.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/debug/log.h>
#include <float.h>


/* Compare two protocols which intersect at point X, by examining their value
 * at point (X + UCP_PROTO_MSGLEN_EPSILON)
 */
#define UCP_PROTO_MSGLEN_EPSILON   0.5

ucs_status_t
ucp_proto_perf_envelope_make(const ucs_linear_func_t *funcs, uint64_t funcs_num,
                             size_t range_start, size_t range_end, int convex,
                             ucp_proto_perf_envelope_t *envelope_list)
{
    size_t start = range_start;
    char num_str[64];
    struct {
        unsigned index;
        double   result;
    } curr, best;
    ucp_proto_perf_envelope_elem_t *new_elem;
    ucs_status_t status;
    size_t midpoint;
    double x_sample, x_intersect;
    uint64_t mask;

    ucs_assert_always(funcs_num < 64);
    mask = UCS_MASK(funcs_num);

    do {
        /* Find best trend at the 'start' point */
        best.index  = UINT_MAX;
        best.result = DBL_MAX;
        x_sample    = start;
        if (x_sample < range_end) {
            x_sample += UCP_PROTO_MSGLEN_EPSILON;
        }
        ucs_for_each_bit(curr.index, mask) {
            curr.result = ucs_linear_func_apply(funcs[curr.index], x_sample);
            ucs_assertv((curr.result != DBL_MAX) && !isnan(curr.result),
                        UCP_PROTO_PERF_FUNC_FMT " curr.index=%u x_sample=%f",
                        UCP_PROTO_PERF_FUNC_ARG(&funcs[curr.index]), curr.index,
                        x_sample);
            if ((best.index == UINT_MAX) ||
                ((curr.result < best.result) == convex)) {
                best = curr;
            }
        }

        /* Since mask != 0, we should find at least one trend */
        ucs_assert(best.index != UINT_MAX);
        ucs_trace("at %s: selected stage[%d]",
                  ucs_memunits_to_str(start, num_str, sizeof(num_str)),
                  best.index);
        ucs_log_indent(1);

        /* Find first (smallest) intersection point between the current best
         * trend and any other trend. This would be the point where that
         * other trend becomes the best one.
         */
        midpoint = range_end;
        mask    &= ~UCS_BIT(best.index);
        ucs_for_each_bit(curr.index, mask) {
            status = ucs_linear_func_intersect(funcs[curr.index],
                                               funcs[best.index], &x_intersect);
            if ((status == UCS_OK) && (x_intersect > x_sample)) {
                /* We care only if the intersection is after 'x_sample', since
                 * otherwise 'best' is better than 'curr' at 'end' as well as
                 * at 'x_sample'. Since 'x_sample' differs from start only
                 * for 0.5 we make an estimation and set it as 'best' for
                 * 'start' as well.
                 */
                midpoint = ucs_min(ucs_double_to_sizet(x_intersect, SIZE_MAX),
                                   midpoint);
                ucs_memunits_to_str(midpoint, num_str, sizeof(num_str));
                ucs_trace("intersects with stage[%d] at %.2f, midpoint is %s",
                          curr.index, x_intersect, num_str);
            } else {
                ucs_trace("intersects with stage[%d] out of range", curr.index);
            }
        }
        ucs_log_indent(-1);

        new_elem             = ucs_array_append(envelope_list,
                                                return UCS_ERR_NO_MEMORY);
        new_elem->index      = best.index;
        new_elem->max_length = midpoint;

        start = midpoint + 1;
    } while (midpoint < range_end);

    return UCS_OK;
}

ucp_proto_common_init_params_t
ucp_proto_common_params_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_common_init_params_t params = {
        .super         = *init_params,
        .cfg_thresh    = UCS_MEMUNITS_AUTO,
        .min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .send_op       = UCT_EP_OP_LAST,
        .memtype_op    = UCT_EP_OP_LAST
    };
    return params;
}

ucs_status_t
ucp_proto_common_add_ppln_perf(ucp_proto_perf_t *perf,
                               const ucp_proto_perf_segment_t *frag_seg,
                               size_t max_length)
{
    ucp_proto_perf_factors_t factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_proto_perf_factor_id_t factor_id, max_factor_id;
    ucs_linear_func_t factor_func;
    double max_value;
    size_t frag_size;
    char frag_str[64];

    frag_size = ucp_proto_perf_segment_end(frag_seg);
    ucs_assertv(frag_size < max_length, "frag_size=%zu max_length=%zu",
                frag_size, max_length);
    ucs_assertv(ucp_proto_perf_find_segment_lb(perf, frag_size + 1) == NULL,
                "ppln range already contains perf data frag_size=%zu",
                frag_size);

    /*
     * 3-factor 3-msg pipeline:
     * 1 msg: [=1=] [======2======] [=3=]
     * 2 msg:       [=1=]           [======2======] [=3=]
     * 3 msg:             [=1=]                     [======2======] [=3=]
     * Approximation:
     *        [=1=] [======================2======================] [=3=]
     * 
     * All the factors except longest one turn into constant fragment overhead
     * due to overlapping (1 and 3 from example).
     */
    max_factor_id = 0;
    max_value     = -DBL_MAX;
    for (factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST; factor_id++) {
        factor_func          = ucp_proto_perf_segment_func(frag_seg, factor_id);
        factors[factor_id].c = ucs_linear_func_apply(factor_func, frag_size);
        if (factors[factor_id].c > max_value) {
            max_factor_id = factor_id;
            max_value     = factors[factor_id].c;
        }
    }

    /* Longest factor still saves the slope but it's constant part turns
     * to dynamic since it start to depend on number of sent fragments
     * (2 from example).
     */
    factors[max_factor_id]    = ucp_proto_perf_segment_func(frag_seg,
                                                            max_factor_id);
    factors[max_factor_id].m += factors[max_factor_id].c / frag_size;
    factors[max_factor_id].c  = 0;


    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));
    return ucp_proto_perf_add_funcs(perf, frag_size + 1, max_length, factors,
                                    ucp_proto_perf_segment_node(frag_seg),
                                    "pipeline", "frag size: %s", frag_str);
}

static ucs_status_t
ucp_proto_init_add_tl_perf(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *tl_perf,
                           ucp_proto_perf_node_t *const tl_perf_node,
                           size_t range_start, size_t range_end,
                           ucp_proto_perf_t *perf)
{
    ucp_proto_perf_factors_t perf_factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    const double latency       = tl_perf->latency + tl_perf->sys_latency;
    const double send_overhead = tl_perf->send_pre_overhead +
                                 tl_perf->send_post_overhead;
    uint32_t op_attr_mask;

    ucs_trace("caps" UCP_PROTO_TIME_FMT(send_pre_overhead)
              UCP_PROTO_TIME_FMT(send_post_overhead)
              UCP_PROTO_TIME_FMT(recv_overhead) UCP_PROTO_TIME_FMT(latency),
              UCP_PROTO_TIME_ARG(tl_perf->send_pre_overhead),
              UCP_PROTO_TIME_ARG(tl_perf->send_post_overhead),
              UCP_PROTO_TIME_ARG(tl_perf->recv_overhead),
              UCP_PROTO_TIME_ARG(tl_perf->latency));

    op_attr_mask = ucp_proto_select_op_attr_unpack(
            params->super.select_param->op_attr);

    perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU].c += send_overhead;
    perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY].c   += latency;

    if (!(op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS)) {
        perf_factors[UCP_PROTO_PERF_FACTOR_REMOTE_CPU].c +=
                tl_perf->recv_overhead;
    }

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE) {
        perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY].c    += latency;
        perf_factors[UCP_PROTO_PERF_FACTOR_REMOTE_CPU].c += send_overhead;
    }

    /* With fast completion bcopy we don't count transport time */
    if (!(op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) ||
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL].m += 1.0 /
                                                          tl_perf->bandwidth;
    }

    /* Send time is representing request completion, which in case of zcopy
       waits for ACK from remote side. */
    if ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY].c += latency;
    }

    return ucp_proto_perf_add_funcs(perf, range_start, range_end, perf_factors,
                                    tl_perf_node, "transport", "");
}

ucs_status_t
ucp_proto_init_add_memreg_time(const ucp_proto_common_init_params_t *params,
                               ucp_md_map_t reg_md_map,
                               ucp_proto_perf_factor_id_t cpu_factor_id,
                               const char *perf_node_name,
                               size_t range_start, size_t range_end,
                               ucp_proto_perf_t *perf)
{
    ucp_context_h context                 = params->super.worker->context;
    ucp_proto_perf_factors_t perf_factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_proto_perf_node_t *reg_perf_node;
    const uct_md_attr_v2_t *md_attr;
    ucp_md_index_t md_index;
    const char *md_name;

    if (reg_md_map == 0) {
        return UCS_OK;
    }

    if (context->rcache != NULL) {
        perf_factors[cpu_factor_id] =
                ucs_linear_func_make(context->config.ext.rcache_overhead, 0);
        ucp_proto_perf_add_funcs(perf, range_start, range_end, perf_factors,
                                 NULL, "rcache lookup", "");
        return UCS_OK;
    }

    reg_perf_node = ucp_proto_perf_node_new_data("mem reg", "");

    /* Go over all memory domains */
    ucs_for_each_bit(md_index, reg_md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        md_name = context->tl_mds[md_index].rsc.md_name;
        ucs_linear_func_add_inplace(&perf_factors[cpu_factor_id],
                                    md_attr->reg_cost);
        ucs_trace("md %s reg: " UCP_PROTO_PERF_FUNC_FMT, md_name,
                  UCP_PROTO_PERF_FUNC_ARG(&md_attr->reg_cost));
        ucp_proto_perf_node_add_data(reg_perf_node, md_name, md_attr->reg_cost);
    }

    if (!ucs_is_pow2(reg_md_map)) {
        /* Multiple memory domains */
        ucp_proto_perf_node_add_data(reg_perf_node, "total",
                                     perf_factors[cpu_factor_id]);
    }

    return ucp_proto_perf_add_funcs(perf, range_start, range_end, perf_factors,
                                    reg_perf_node, perf_node_name, "%u mds",
                                    ucs_popcount(reg_md_map));
}

ucs_status_t
ucp_proto_init_add_buffer_copy_time(ucp_worker_h worker, const char *title,
                                    ucs_memory_type_t local_mem_type,
                                    ucs_memory_type_t remote_mem_type,
                                    uct_ep_operation_t memtype_op,
                                    size_t range_start, size_t range_end,
                                    ucp_proto_perf_factor_id_t cpu_factor_id,
                                    ucp_proto_perf_t *perf)
{
    ucp_proto_perf_factors_t perf_factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_context_h context                 = worker->context;
    ucs_memory_type_t src_mem_type, dst_mem_type;
    ucp_proto_perf_node_t *tl_perf_node;
    const ucp_ep_config_t *ep_config;
    ucp_worker_iface_t *wiface;
    uct_perf_attr_t perf_attr;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (UCP_MEM_IS_HOST(local_mem_type) && UCP_MEM_IS_HOST(remote_mem_type)) {
        perf_factors[cpu_factor_id] =
                ucs_linear_func_make(0, 1.0 / context->config.ext.bcopy_bw);
        ucp_proto_perf_add_funcs(perf, range_start, range_end, perf_factors,
                                NULL, title, "memcpy");
        return UCS_OK;
    }

    if (worker->mem_type_ep[local_mem_type] != NULL) {
        ep_config = ucp_ep_config(worker->mem_type_ep[local_mem_type]);
    } else if (worker->mem_type_ep[remote_mem_type] != NULL) {
        ep_config = ucp_ep_config(worker->mem_type_ep[remote_mem_type]);
    } else {
        ucs_debug("cannot copy memory between %s and %s",
                  ucs_memory_type_names[local_mem_type],
                  ucs_memory_type_names[remote_mem_type]);
        return UCS_ERR_UNSUPPORTED;
    }

    /* Use the v2 API to query overhead and BW */
    perf_attr.local_memory_type  = local_mem_type;
    perf_attr.remote_memory_type = remote_mem_type;
    perf_attr.operation          = memtype_op;

    switch (memtype_op) {
    case UCT_EP_OP_PUT_SHORT:
    case UCT_EP_OP_GET_SHORT:
        lane = ep_config->key.rma_lanes[0];
        break;
    case UCT_EP_OP_PUT_ZCOPY:
    case UCT_EP_OP_GET_ZCOPY:
        lane = ep_config->key.rma_bw_lanes[0];
        break;
    case UCT_EP_OP_LAST:
        return UCS_ERR_UNSUPPORTED;
    default:
        ucs_fatal("invalid UCT copy operation: %d", memtype_op);
    }

    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                           UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE |
                           UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE |
                           UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_RECV_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH |
                           UCT_PERF_ATTR_FIELD_LATENCY;
    perf_attr.operation  = memtype_op;

    rsc_index = ep_config->key.lanes[lane].rsc_index;
    wiface    = ucp_worker_iface(worker, rsc_index);
    status    = ucp_worker_iface_estimate_perf(wiface, &perf_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* all allowed copy operations are one-sided */
    ucs_assert(perf_attr.recv_overhead < UCP_PROTO_PERF_EPSILON);

    perf_factors[UCP_PROTO_PERF_FACTOR_LATENCY].c =
            ucp_tl_iface_latency(context, &perf_attr.latency);
    perf_factors[cpu_factor_id].c =
            perf_attr.send_pre_overhead + perf_attr.send_post_overhead;
    perf_factors[cpu_factor_id].m =
            1.0 / ucp_tl_iface_bandwidth(context, &perf_attr.bandwidth);

    if ((memtype_op == UCT_EP_OP_GET_SHORT) ||
        (memtype_op == UCT_EP_OP_GET_ZCOPY)) {
        src_mem_type = remote_mem_type;
        dst_mem_type = local_mem_type;
    } else {
        src_mem_type = local_mem_type;
        dst_mem_type = remote_mem_type;
    }

    ucp_proto_common_lane_perf_node(context, rsc_index, &perf_attr,
                                    &tl_perf_node);

    ucp_proto_perf_add_funcs(perf, range_start, range_end, perf_factors,
                             tl_perf_node, title, "%s to %s",
                             ucs_memory_type_names[src_mem_type],
                             ucs_memory_type_names[dst_mem_type]);

    ucp_proto_perf_node_deref(&tl_perf_node);

    return UCS_OK;
}

static ucs_status_t
ucp_proto_init_add_buffer_perf(const ucp_proto_common_init_params_t *params,
                               size_t range_start, size_t range_end,
                               ucp_md_map_t reg_md_map, ucp_proto_perf_t *perf)
{
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    ucs_memory_type_t recv_mem_type;
    uint32_t op_attr_mask;
    ucs_status_t status;

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        status = ucp_proto_init_add_memreg_time(
                params, reg_md_map, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
                "send memory registration", range_start, range_end, perf);
        if (status != UCS_OK) {
            return status;
        }
    } else if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR)) {
        ucs_assert(reg_md_map == 0);
        status = ucp_proto_init_add_buffer_copy_time(
                params->super.worker, "send copy", UCS_MEMORY_TYPE_HOST,
                select_param->mem_type, params->memtype_op, range_start,
                range_end, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, perf);
        if (status != UCS_OK) {
            return status;
        }
    }

    op_attr_mask = ucp_proto_select_op_attr_unpack(select_param->op_attr);
    if (/* Remote access implies zero copy on receiver */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) ||
        /* Count only send completion time without waiting for a response */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE))) {
        return UCS_OK;
    }

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY) {
        /* Receiver has to register its buffer */
        status = ucp_proto_init_add_memreg_time(
                params, reg_md_map, UCP_PROTO_PERF_FACTOR_REMOTE_CPU,
                "send memory registration", range_start, range_end, perf);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Receiver has to copy data.
     * Assume same memory type as sender if no rkey */
    recv_mem_type = (params->super.rkey_config_key == NULL) ? 
            select_param->mem_type : params->super.rkey_config_key->mem_type;
    status        = ucp_proto_init_add_buffer_copy_time(
            params->super.worker, "recv copy", UCS_MEMORY_TYPE_HOST,
            recv_mem_type, UCT_EP_OP_PUT_SHORT, range_start, range_end,
            UCP_PROTO_PERF_FACTOR_REMOTE_CPU, perf);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static int
ucp_proto_common_check_mem_access(const ucp_proto_common_init_params_t *params)
{
    uint8_t mem_type = params->super.select_param->mem_type;

    /*
     * - HDR_ONLY protocols don't need access to payload memory
     * - ZCOPY protocols don't need to copy payload memory
     * - CPU accessible memory doesn't require memtype_op
     * - memtype_op should be defined to valid op if memory is CPU inaccessible
     */
    return (params->flags & UCP_PROTO_COMMON_INIT_FLAG_HDR_ONLY) ||
           (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) ||
           UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type) ||
           (params->memtype_op != UCT_EP_OP_LAST);
}

ucs_status_t
ucp_proto_common_init_perf(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *tl_perf,
                           ucp_proto_perf_node_t *const tl_perf_node,
                           ucp_md_map_t reg_md_map, ucp_proto_perf_t **perf_p)
{
    const char *proto_name = ucp_proto_id_field(params->super.proto_id, name);
    const ucp_proto_perf_segment_t *frag_seg;
    size_t range_start, range_end;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    range_start = ucs_max(params->min_length, tl_perf->min_length);

    if (ucp_proto_common_check_mem_access(params)) {
        range_end = params->max_length;
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG) {
            /* Cap single-fragment protocols by max fragment size */
            ucs_assert(tl_perf->max_frag >= params->hdr_size);
            range_end = ucs_min(range_end,
                                tl_perf->max_frag - params->hdr_size);
        }
    } else {
        /* If memory access is not possible, support only empty message */
        range_end = 0;
    }

    if (range_end < tl_perf->min_length) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_perf_create(proto_name, &perf);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_proto_init_add_tl_perf(params, tl_perf, tl_perf_node,
                                        range_start, range_end, perf);
    if (status != UCS_OK) {
            goto out;
    }

    if (range_end > 0) {
        /* Add buffer copy/register cost for non-empty messages */
        status = ucp_proto_init_add_buffer_perf(params, ucs_max(1, range_start),
                                                range_end, reg_md_map, perf);
        if (status != UCS_OK) {
            goto out;
        }

        /* Add range that represent sending many fragments */
        if ((range_end < params->max_length) &&
            !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG)) {
            frag_seg = ucp_proto_perf_segment_last(perf);
            ucs_assert(frag_seg != NULL);
            status   = ucp_proto_common_add_ppln_perf(perf, frag_seg,
                                                      params->max_length);
        }
    }

out:
    if (status != UCS_OK) {
        ucp_proto_perf_destroy(perf);
    } else {
        *perf_p = perf;
    }

    return status;
}

int ucp_proto_init_check_op(const ucp_proto_init_params_t *init_params,
                            uint64_t op_id_mask)
{
    return ucp_proto_select_check_op(init_params->select_param, op_id_mask);
}
