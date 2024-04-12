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

void ucp_proto_common_add_ppln_range(ucp_proto_caps_t *caps,
                                     const ucp_proto_perf_range_t *frag_range,
                                     size_t max_length)
{
    ucp_proto_perf_range_t *ppln_range = &caps->ranges[caps->num_ranges];
    size_t frag_size                   = frag_range->max_length;
    ucs_linear_func_t *ppln_perf       = ppln_range->perf;
    ucp_proto_perf_type_t perf_type;
    double frag_overhead;
    char frag_str[64];

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));
    ppln_range->node = ucp_proto_perf_node_new_data("pipeline", "frag size: %s",
                                                    frag_str);

    UCP_PROTO_PERF_TYPE_FOREACH(perf_type) {
        /* For multi-fragment protocols, we need to apply the fragment
         * size to the performance function linear factor.
         */
        ppln_perf[perf_type]    = frag_range->perf[perf_type];
        ppln_perf[perf_type].m += frag_range->perf[perf_type].c / frag_size;
    }

    /* Overhead of sending one fragment before starting the pipeline */
    /* Calculation of frag-overhead should be based on frag_range->perf
     * but it causes significant performance degradation in the current perf
     * prediction scheme */
    frag_overhead =
            ucs_linear_func_apply(ppln_perf[UCP_PROTO_PERF_TYPE_SINGLE], frag_size) -
            ucs_linear_func_apply(ppln_perf[UCP_PROTO_PERF_TYPE_MULTI], frag_size);
    ucs_assert(frag_overhead >= 0);

    ucs_trace("frag-size: %zd" UCP_PROTO_TIME_FMT(frag_overhead), frag_size,
              UCP_PROTO_TIME_ARG(frag_overhead));

    /* Apply the pipelining effect when sending multiple fragments */
    ppln_perf[UCP_PROTO_PERF_TYPE_SINGLE] =
            ucs_linear_func_add(ppln_perf[UCP_PROTO_PERF_TYPE_MULTI],
                                ucs_linear_func_make(frag_overhead, 0));

    ppln_range->max_length = max_length;

    ucp_proto_perf_range_add_data(ppln_range);
    ucp_proto_perf_node_add_scalar(ppln_range->node, "frag-ovh", frag_overhead);
    ucp_proto_perf_node_add_child(ppln_range->node, frag_range->node);

    ++caps->num_ranges;
}

void ucp_proto_common_init_base_caps(
        const ucp_proto_common_init_params_t *params, ucp_proto_caps_t *caps,
        size_t min_length)
{
    caps->cfg_thresh   = params->cfg_thresh;
    caps->cfg_priority = params->cfg_priority;
    caps->min_length   = ucs_max(params->min_length, min_length);
    caps->num_ranges   = 0;
}

static int ucp_proto_perf_range_is_zero(const ucp_proto_perf_range_t *range)
{
    return ucs_linear_func_is_zero(range->perf[UCP_PROTO_PERF_TYPE_SINGLE],
                                   UCP_PROTO_PERF_EPSILON) &&
           ucs_linear_func_is_zero(range->perf[UCP_PROTO_PERF_TYPE_MULTI],
                                   UCP_PROTO_PERF_EPSILON);
}

void ucp_proto_perf_range_add_data(const ucp_proto_perf_range_t *range)
{
    ucp_proto_perf_node_add_data(range->node, "sngl",
                                 range->perf[UCP_PROTO_PERF_TYPE_SINGLE]);
    ucp_proto_perf_node_add_data(range->node, "mult",
                                 range->perf[UCP_PROTO_PERF_TYPE_MULTI]);
    ucp_proto_perf_node_add_data(range->node, "cpu",
                                 range->perf[UCP_PROTO_PERF_TYPE_CPU]);
}

ucs_status_t
ucp_proto_perf_envelope_make(const ucp_proto_perf_list_t *perf_list,
                             size_t range_start, size_t range_end, int convex,
                             ucp_proto_perf_envelope_t *envelope_list)
{
    const ucs_linear_func_t *perf_list_ptr = ucs_array_begin(perf_list);
    const unsigned perf_list_length        = ucs_array_length(perf_list);
    size_t start                           = range_start;
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

    ucs_assert_always(perf_list_length < 64);
    mask = UCS_MASK(perf_list_length);

    do {
        ucs_assert(mask != 0);

        /* Find best trend at the 'start' point */
        best.index  = UINT_MAX;
        best.result = DBL_MAX;
        ucs_for_each_bit(curr.index, mask) {
            x_sample    = start + UCP_PROTO_MSGLEN_EPSILON;
            curr.result = ucs_linear_func_apply(perf_list_ptr[curr.index],
                                                x_sample);
            ucs_assert(curr.result != DBL_MAX);
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
            status = ucs_linear_func_intersect(perf_list_ptr[curr.index],
                                               perf_list_ptr[best.index],
                                               &x_intersect);
            if ((status == UCS_OK) && (x_intersect > start)) {
                /* We care only if the intersection is after 'start', since
                 * otherwise 'best' is better than 'curr' at
                 * 'end' as well as at 'start'.
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

ucs_status_t
ucp_proto_init_parallel_stages(const char *proto_name, size_t range_start,
                               size_t range_end, double bias,
                               const ucp_proto_perf_range_t **stages,
                               unsigned num_stages, ucp_proto_caps_t *caps)
{
    ucs_linear_func_t bias_func = ucs_linear_func_make(0.0, 1.0 - bias);
    UCS_ARRAY_DEFINE_ONSTACK(ucp_proto_perf_envelope_t, concave, 16);
    UCS_ARRAY_DEFINE_ONSTACK(ucp_proto_perf_list_t, stage_list, 16);
    ucs_linear_func_t sum_single_perf, sum_cpu_perf;
    const ucp_proto_perf_range_t **stage_elem;
    ucp_proto_perf_envelope_elem_t *elem;
    ucp_proto_perf_node_t *stage_node;
    ucp_proto_perf_range_t *range;
    ucs_linear_func_t *perf_elem;
    ucs_status_t status;
    char range_str[64];

    ucs_memunits_range_str(range_start, range_end, range_str,
                           sizeof(range_str));
    ucs_trace("%s bias %.0f%%", range_str, bias * 100.0);

    ucs_log_indent(1);
    sum_single_perf = UCS_LINEAR_FUNC_ZERO;
    sum_cpu_perf    = UCS_LINEAR_FUNC_ZERO;
    ucs_carray_for_each(stage_elem, stages, num_stages) {
        /* Summarize single and CPU time */
        ucs_linear_func_add_inplace(&sum_single_perf,
                                    (*stage_elem)->perf[UCP_PROTO_PERF_TYPE_SINGLE]);
        ucs_linear_func_add_inplace(&sum_cpu_perf,
                                    (*stage_elem)->perf[UCP_PROTO_PERF_TYPE_CPU]);

        /* Add all multi perf ranges to envelope array */
        perf_elem  = ucs_array_append(&stage_list, status = UCS_ERR_NO_MEMORY;
                                      goto out);
        *perf_elem = (*stage_elem)->perf[UCP_PROTO_PERF_TYPE_MULTI];

        ucs_trace("stage[%zu] %s " UCP_PROTO_PERF_FUNC_TYPES_FMT
                  UCP_PROTO_PERF_FUNC_FMT(perf_elem),
                  stage_elem - stages,
                  ucp_proto_perf_node_name((*stage_elem)->node),
                  UCP_PROTO_PERF_FUNC_TYPES_ARG((*stage_elem)->perf),
                  UCP_PROTO_PERF_FUNC_ARG(perf_elem));
    }

    /* Add CPU time as another parallel stage */
    perf_elem  = ucs_array_append(&stage_list, status = UCS_ERR_NO_MEMORY;
                                 goto out);
    *perf_elem = sum_cpu_perf;

    /* Multi-fragment is pipelining overheads and network transfer */
    status = ucp_proto_perf_envelope_make(&stage_list, range_start, range_end,
                                          0, &concave);
    if (status != UCS_OK) {
        goto out;
    }

    ucs_array_for_each(elem, &concave) {
        range             = &caps->ranges[caps->num_ranges];
        range->max_length = elem->max_length;
        if (fabs(bias) > UCP_PROTO_PERF_EPSILON) {
            range->node = ucp_proto_perf_node_new_data(proto_name, "bias %f",
                                                       bias);
        } else {
            range->node = ucp_proto_perf_node_new_data(proto_name, "");
        }

        /* "single" performance estimation is sum of "stages" with the bias */
        range->perf[UCP_PROTO_PERF_TYPE_SINGLE] =
                ucs_linear_func_compose(bias_func, sum_single_perf);

        /* "multiple" performance estimation is concave envelope of "stages" */
        range->perf[UCP_PROTO_PERF_TYPE_MULTI] = ucs_linear_func_compose(
                bias_func, ucs_array_elem(&stage_list, elem->index));

        /* CPU overhead is the sum of all stages */
        range->perf[UCP_PROTO_PERF_TYPE_CPU] = sum_cpu_perf;

        ucp_proto_perf_range_add_data(range);

        ucs_trace("range[%d] %s" UCP_PROTO_PERF_FUNC_TYPES_FMT,
                  caps->num_ranges,
                  ucs_memunits_range_str(range_start, range->max_length,
                                         range_str, sizeof(range_str)),
                  UCP_PROTO_PERF_FUNC_TYPES_ARG(range->perf));

        stage_node = ucp_proto_perf_node_new_compose("stage", "");

        ucs_carray_for_each(stage_elem, stages, num_stages) {
            if (ucp_proto_perf_range_is_zero(*stage_elem)) {
                continue;
            }

            /* range->node ---> stage_node ---> [ stage{i}->node ... ] */
            ucp_proto_perf_node_add_child(stage_node, (*stage_elem)->node);
        }

        ucp_proto_perf_node_own_child(range->node, &stage_node);

        ++caps->num_ranges;
        range_start = range->max_length + 1;
    }
    ucs_assertv(range_start == (range_end + 1), "range_start=%zu range_end=%zu",
                range_start, range_end);

    status = UCS_OK;

out:
    ucs_log_indent(-1);
    return status;
}

void ucp_proto_init_memreg_time(const ucp_proto_common_init_params_t *params,
                                ucp_md_map_t reg_md_map,
                                ucs_linear_func_t *memreg_time,
                                ucp_proto_perf_node_t **perf_node_p)
{
    ucp_context_h context            = params->super.worker->context;
    ucp_proto_perf_node_t *perf_node = NULL;
    const uct_md_attr_v2_t *md_attr;
    ucp_md_index_t md_index;
    const char *md_name;

    *memreg_time = UCS_LINEAR_FUNC_ZERO;

    if (reg_md_map == 0) {
        goto out;
    }

    if (context->rcache != NULL) {
        perf_node = ucp_proto_perf_node_new_data("rcache lookup", "");

        *memreg_time = UCP_RCACHE_LOOKUP_FUNC;

        ucp_proto_perf_node_add_data(perf_node, "lookup", *memreg_time);

        goto out;
    }

    perf_node = ucp_proto_perf_node_new_data("mem reg", "");

    /* Go over all memory domains */
    ucs_for_each_bit(md_index, reg_md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        md_name = context->tl_mds[md_index].rsc.md_name;
        ucs_linear_func_add_inplace(memreg_time, md_attr->reg_cost);
        ucs_trace("md %s" UCP_PROTO_PERF_FUNC_FMT(reg_cost), md_name,
                  UCP_PROTO_PERF_FUNC_ARG(&md_attr->reg_cost));

        ucp_proto_perf_node_add_data(perf_node, md_name, md_attr->reg_cost);
    }

    if (!ucs_is_pow2(reg_md_map)) {
        /* Multiple memory domains */
        ucp_proto_perf_node_add_data(perf_node, "total", *memreg_time);
    }

out:
    *perf_node_p = perf_node;
}

ucs_status_t
ucp_proto_init_buffer_copy_time(ucp_worker_h worker, const char *title,
                                ucs_memory_type_t local_mem_type,
                                ucs_memory_type_t remote_mem_type,
                                uct_ep_operation_t memtype_op,
                                ucs_linear_func_t *copy_time,
                                ucp_proto_perf_node_t **perf_node_p)
{
    ucp_context_h context = worker->context;
    ucs_memory_type_t src_mem_type, dst_mem_type;
    ucp_proto_perf_node_t *perf_node, *tl_perf_node;
    const ucp_ep_config_t *ep_config;
    ucp_worker_iface_t *wiface;
    uct_perf_attr_t perf_attr;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (UCP_MEM_IS_HOST(local_mem_type) && UCP_MEM_IS_HOST(remote_mem_type)) {
        *copy_time = ucs_linear_func_make(0,
                                          1.0 / context->config.ext.bcopy_bw);

        perf_node = ucp_proto_perf_node_new_data("memcpy", "");
        ucp_proto_perf_node_add_bandwidth(perf_node, "bcopy_bw",
                                          context->config.ext.bcopy_bw);
        *perf_node_p = perf_node;
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
    ucs_assert(perf_attr.recv_overhead < 1e-15);
    copy_time->c = ucp_tl_iface_latency(context, &perf_attr.latency) +
                   perf_attr.send_pre_overhead + perf_attr.send_post_overhead +
                   perf_attr.recv_overhead;
    copy_time->m = 1.0 / ucp_tl_iface_bandwidth(context, &perf_attr.bandwidth);

    if ((memtype_op == UCT_EP_OP_GET_SHORT) ||
        (memtype_op == UCT_EP_OP_GET_ZCOPY)) {
        src_mem_type = remote_mem_type;
        dst_mem_type = local_mem_type;
    } else {
        src_mem_type = local_mem_type;
        dst_mem_type = remote_mem_type;
    }

    perf_node = ucp_proto_perf_node_new_data(
            title, "%s to %s", ucs_memory_type_names[src_mem_type],
            ucs_memory_type_names[dst_mem_type]);

    ucp_proto_perf_node_add_data(perf_node, "", *copy_time);

    ucp_proto_common_lane_perf_node(context, rsc_index, &perf_attr,
                                    &tl_perf_node);
    ucp_proto_perf_node_own_child(perf_node, &tl_perf_node);

    *perf_node_p = perf_node;

    return UCS_OK;
}

static ucs_status_t
ucp_proto_common_init_send_perf(const ucp_proto_common_init_params_t *params,
                                const ucp_proto_common_tl_perf_t *tl_perf,
                                ucp_md_map_t reg_md_map, int empty_msg,
                                ucp_proto_perf_range_t *send_perf)
{
    ucp_proto_perf_node_t *child_perf_node;
    ucs_linear_func_t send_overhead;
    ucs_status_t status;

    send_perf->node = ucp_proto_perf_node_new_data("send-ovrh", "");

    /* Remote access implies zero copy on receiver */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assert(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY);
    }

    /* Calculate sender overhead */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        ucp_proto_init_memreg_time(params, reg_md_map, &send_overhead,
                                   &child_perf_node);
        ucp_proto_perf_node_own_child(send_perf->node, &child_perf_node);
    } else if ((params->flags & UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR) ||
               empty_msg) {
        send_overhead = UCS_LINEAR_FUNC_ZERO;
    } else {
        ucs_assert(reg_md_map == 0);
        status = ucp_proto_init_buffer_copy_time(
                params->super.worker, "send copy", UCS_MEMORY_TYPE_HOST,
                params->super.select_param->mem_type, params->memtype_op,
                &send_overhead, &child_perf_node);
        if (status != UCS_OK) {
            ucp_proto_perf_node_deref(&send_perf->node);
            return status;
        }

        ucp_proto_perf_node_own_child(send_perf->node, &child_perf_node);
    }

    send_overhead.c                            += tl_perf->send_pre_overhead;
    send_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE] = send_overhead;

    send_overhead.c                           += tl_perf->send_post_overhead;
    send_perf->perf[UCP_PROTO_PERF_TYPE_MULTI] = send_overhead;
    send_perf->perf[UCP_PROTO_PERF_TYPE_CPU]   = send_overhead;

    ucp_proto_perf_range_add_data(send_perf);

    return UCS_OK;
}

static void
ucp_proto_common_init_xfer_perf(const ucp_proto_common_init_params_t *params,
                                const ucp_proto_common_tl_perf_t *tl_perf,
                                ucp_proto_perf_node_t *const tl_perf_node,
                                ucp_proto_perf_range_t *xfer_perf)
{
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    ucs_linear_func_t xfer_time;
    uint32_t op_attr_mask;

    xfer_perf->node = ucp_proto_perf_node_new_data("xfer", "");

    op_attr_mask = ucp_proto_select_op_attr_unpack(select_param->op_attr);

    if ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        /* If we care only about time to start sending the message, ignore
           the transport time */
        xfer_time = UCS_LINEAR_FUNC_ZERO;
    } else {
        xfer_time = ucs_linear_func_make(0, 1.0 / tl_perf->bandwidth);
    }

    xfer_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE]    = xfer_time;
    xfer_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE].c += tl_perf->latency +
                                                     tl_perf->sys_latency;
    xfer_perf->perf[UCP_PROTO_PERF_TYPE_MULTI]     = xfer_time;
    xfer_perf->perf[UCP_PROTO_PERF_TYPE_CPU]       = UCS_LINEAR_FUNC_ZERO;

    /*
     * Add the latency of response/ACK back from the receiver.
     */
    if (/* Protocol is waiting for response */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE) ||
        /* Send time is representing request completion, which in case of zcopy
           waits for ACK from remote side. */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
         (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY))) {
        xfer_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE].c += tl_perf->latency;
        xfer_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE].c +=
                tl_perf->send_post_overhead;
    }

    ucp_proto_perf_range_add_data(xfer_perf);
    ucp_proto_perf_node_add_child(xfer_perf->node, tl_perf_node);
}

static ucs_status_t
ucp_proto_common_init_recv_perf(const ucp_proto_common_init_params_t *params,
                                const ucp_proto_common_tl_perf_t *tl_perf,
                                ucp_md_map_t reg_md_map, int empty_msg,
                                ucp_proto_perf_range_t *recv_perf)
{
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    ucp_proto_perf_node_t *child_perf_node;
    ucs_linear_func_t recv_overhead;
    ucs_memory_type_t recv_mem_type;
    uint32_t op_attr_mask;
    ucs_status_t status;

    recv_perf->node = ucp_proto_perf_node_new_data("recv-ovrh", "");

    op_attr_mask = ucp_proto_select_op_attr_unpack(select_param->op_attr);

    if (/* Don't care about receiver time for one-sided remote access */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) ||
        /* Count only send completion time without waiting for a response */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
         !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE)) ||
        empty_msg) {
        recv_overhead = UCS_LINEAR_FUNC_ZERO;
    } else {
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY) {
            /* Receiver has to register its buffer */
            ucp_proto_init_memreg_time(params, reg_md_map, &recv_overhead,
                                       &child_perf_node);
        } else {
            if (params->super.rkey_config_key == NULL) {
                /* Assume same memory type as sender */
                recv_mem_type = select_param->mem_type;
            } else {
                recv_mem_type = params->super.rkey_config_key->mem_type;
            }

            /* Silence cppcheck */
            recv_overhead = UCS_LINEAR_FUNC_ZERO;

            /* Receiver has to copy data */
            status = ucp_proto_init_buffer_copy_time(
                    params->super.worker, "recv copy", UCS_MEMORY_TYPE_HOST,
                    recv_mem_type, UCT_EP_OP_PUT_SHORT, &recv_overhead,
                    &child_perf_node);
            if (status != UCS_OK) {
                ucp_proto_perf_node_deref(&recv_perf->node);
                return status;
            }
        }

        /* Receiver has to process the incoming message */
        if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS)) {
            /* latency measure: add remote-side processing time */
            recv_overhead.c += tl_perf->recv_overhead;
        }

        ucp_proto_perf_node_own_child(recv_perf->node, &child_perf_node);
    }

    recv_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE] = recv_overhead;
    recv_perf->perf[UCP_PROTO_PERF_TYPE_MULTI]  = recv_overhead;
    recv_perf->perf[UCP_PROTO_PERF_TYPE_CPU]    = UCS_LINEAR_FUNC_ZERO;

    ucp_proto_perf_range_add_data(recv_perf);

    return UCS_OK;
}

static ucs_status_t
ucp_proto_init_single_frag_ranges(const ucp_proto_common_init_params_t *params,
                                  const ucp_proto_common_tl_perf_t *tl_perf,
                                  ucp_proto_perf_node_t *const tl_perf_node,
                                  ucp_md_map_t reg_md_map, size_t range_start,
                                  size_t range_end, ucp_proto_caps_t *caps)
{
    const char *proto_name = ucp_proto_id_field(params->super.proto_id, name);
    int empty_msg          = (range_end == 0);
    ucp_proto_perf_range_t xfer_perf, send_perf, recv_perf;
    const ucp_proto_perf_range_t *parallel_stages[3];
    ucs_status_t status;

    /* Network transfer time */
    ucp_proto_common_init_xfer_perf(params, tl_perf, tl_perf_node, &xfer_perf);

    /* Sender overhead */
    status = ucp_proto_common_init_send_perf(params, tl_perf, reg_md_map,
                                             empty_msg, &send_perf);
    if (status != UCS_OK) {
        goto out_deref_xfer_perf;
    }

    /* Receiver overhead */
    status = ucp_proto_common_init_recv_perf(params, tl_perf, reg_md_map,
                                             empty_msg, &recv_perf);
    if (status != UCS_OK) {
        goto out_deref_send_perf;
    }

    parallel_stages[0] = &send_perf;
    parallel_stages[1] = &xfer_perf;
    parallel_stages[2] = &recv_perf;

    /* Add ranges representing sending single fragment */
    status = ucp_proto_init_parallel_stages(proto_name, range_start, range_end,
                                            0.0, parallel_stages, 3, caps);

    ucp_proto_perf_node_deref(&recv_perf.node);
out_deref_send_perf:
    ucp_proto_perf_node_deref(&send_perf.node);
out_deref_xfer_perf:
    ucp_proto_perf_node_deref(&xfer_perf.node);
    return status;
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
ucp_proto_common_init_caps(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *tl_perf,
                           ucp_proto_perf_node_t *const tl_perf_node,
                           ucp_md_map_t reg_md_map, ucp_proto_caps_t *caps)
{
    size_t range_end;
    ucs_status_t status;

    ucs_trace("caps" UCP_PROTO_TIME_FMT(send_pre_overhead)
              UCP_PROTO_TIME_FMT(send_post_overhead)
              UCP_PROTO_TIME_FMT(recv_overhead) UCP_PROTO_TIME_FMT(latency),
              UCP_PROTO_TIME_ARG(tl_perf->send_pre_overhead),
              UCP_PROTO_TIME_ARG(tl_perf->send_post_overhead),
              UCP_PROTO_TIME_ARG(tl_perf->recv_overhead),
              UCP_PROTO_TIME_ARG(tl_perf->latency));

    /* Initialize capabilities */
    ucp_proto_common_init_base_caps(params, caps, tl_perf->min_length);

    /* Add range representing sending empty message */
    if (caps->min_length == 0) {
        status = ucp_proto_init_single_frag_ranges(params, tl_perf,
                                                   tl_perf_node, reg_md_map, 0,
                                                   0, caps);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Get fragment size */
    ucs_assert(tl_perf->max_frag >= params->hdr_size);
    range_end = ucs_min(params->max_length,
                        tl_perf->max_frag - params->hdr_size);
    if ((range_end == 0) || !ucp_proto_common_check_mem_access(params)) {
        /* Return UNSUPPORTED if protocol cannot be used on any range */
        return (caps->min_length == 0) ? UCS_OK : UCS_ERR_UNSUPPORTED;
    }

    ucs_assertv_always(range_end >= caps->min_length,
                       "range_end=%zu caps->min_length=%zu",
                       range_end, caps->min_length);

    /* Add ranges representing sending single fragment */
    status = ucp_proto_init_single_frag_ranges(params, tl_perf, tl_perf_node,
                                               reg_md_map, caps->min_length,
                                               range_end, caps);
    if (status != UCS_OK) {
        return status;
    }

    /* Append range representing sending rest of the fragments, if range_end is
       not the max length and the protocol supports fragmentation */
    if ((range_end < params->max_length) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG)) {
        ucp_proto_common_add_ppln_range(caps,
                                        &caps->ranges[caps->num_ranges - 1],
                                        params->max_length);
    }

    return UCS_OK;
}

int ucp_proto_init_check_op(const ucp_proto_init_params_t *init_params,
                            uint64_t op_id_mask)
{
    return ucp_proto_select_check_op(init_params->select_param, op_id_mask);
}
