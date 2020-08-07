/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_common.h"
#include "proto_select.inl"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_worker.inl>
#include <ucp/dt/dt.h>


static ucp_rsc_index_t
ucp_proto_common_get_rsc_index(const ucp_proto_common_init_params_t *params,
                               ucp_lane_index_t lane)
{
    ucp_rsc_index_t rsc_index;

    ucs_assert(lane < UCP_MAX_LANES);

    rsc_index = params->super.ep_config_key->lanes[lane].rsc_index;
    ucs_assert(rsc_index < UCP_MAX_RESOURCES);

    return rsc_index;
}

ucp_rsc_index_t
ucp_proto_common_get_md_index(const ucp_proto_common_init_params_t *params,
                              ucp_lane_index_t lane)
{
    ucp_context_h     context = params->super.worker->context;
    ucp_rsc_index_t rsc_index = ucp_proto_common_get_rsc_index(params, lane);

    return context->tl_rscs[rsc_index].md_index;
}

static const uct_iface_attr_t *
ucp_proto_common_get_iface_attr(const ucp_proto_common_init_params_t *params,
                                ucp_lane_index_t lane)
{
    return ucp_worker_iface_get_attr(params->super.worker,
                                     ucp_proto_common_get_rsc_index(params, lane));
}

static size_t ucp_proto_get_iface_attr_field(const uct_iface_attr_t *iface_attr,
                                             ptrdiff_t field_offset)
{
    return *(const size_t*)UCS_PTR_BYTE_OFFSET(iface_attr, field_offset);
}

static double
ucp_proto_common_iface_bandwidth(const ucp_proto_common_init_params_t *params,
                                 const uct_iface_attr_t *iface_attr)
{
    return ucp_tl_iface_bandwidth(params->super.worker->context,
                                  &iface_attr->bandwidth);
}

ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_common_init_params_t *params,
                            ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
                            ucp_lane_index_t *lanes, ucp_lane_index_t max_lanes)
{
    ucp_context_h context                        = params->super.worker->context;
    const ucp_ep_config_key_t *ep_config_key     = params->super.ep_config_key;
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    const uct_iface_attr_t *iface_attr;
    ucp_lane_index_t lane, num_lanes;
    const uct_md_attr_t *md_attr;
    ucp_rsc_index_t rsc_index;
    ucs_string_buffer_t strb;
    ucp_md_index_t md_index;

    ucp_proto_select_param_str(select_param, &strb);
    ucs_trace("selecting out of %d lanes for %s %s", ep_config_key->num_lanes,
              params->super.proto_name, ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        if ((select_param->dt_class == UCP_DATATYPE_GENERIC) ||
            (select_param->dt_class == UCP_DATATYPE_IOV)) {
            /* Generic/IOV datatype cannot be used with zero-copy send */
            /* TODO support IOV registration */
            ucs_trace("datatype %s cannot be used with zcopy",
                      ucp_datatype_class_names[select_param->dt_class]);
            return 0;
        }
    } else if ((select_param->dt_class != UCP_DATATYPE_GENERIC) &&
               !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(select_param->mem_type)) {
        /* If zero-copy is off, the memory must be host-accessible for
         * non-generic type (for generic type there is no buffer to access) */
        ucs_trace("memory type %s with datatype %s is not supported",
                  ucs_memory_type_names[select_param->mem_type],
                  ucp_datatype_class_names[select_param->dt_class]);
        return 0;
    }

    num_lanes = 0;
    for (lane = 0; (lane < ep_config_key->num_lanes) && (num_lanes < max_lanes);
         ++lane) {

        /* Check if lane type matches */
        if (!(ep_config_key->lanes[lane].lane_types & UCS_BIT(lane_type))) {
            ucs_trace("lane[%d]: no %s", lane,
                      ucp_lane_type_info[lane_type].short_name);
            continue;
        }

        /* Check iface capabilities */
        iface_attr = ucp_proto_common_get_iface_attr(params, lane);
        if (!ucs_test_all_flags(iface_attr->cap.flags, tl_cap_flags)) {
            ucs_trace("lane[%d]: no cap 0x%lx", lane, tl_cap_flags);
            continue;
        }

        rsc_index = ep_config_key->lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        md_index = context->tl_rscs[rsc_index].md_index;
        md_attr  = &context->tl_mds[md_index].attr;

        /* Check memory registration capabilities for zero-copy case */
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
            if (md_attr->cap.flags & UCT_MD_FLAG_NEED_MEMH) {
                /* Memory domain must support registration on the relevant
                 * memory type */
                if (!(md_attr->cap.flags & UCT_MD_FLAG_REG) ||
                    !(md_attr->cap.reg_mem_types & UCS_BIT(select_param->mem_type))) {
                    ucs_trace("lane[%d]: no reg of mem type %s", lane,
                              ucs_memory_type_names[select_param->mem_type]);
                    continue;
                }
            } else {
                /* Memory domain which does not require a registration for zero
                 * copy operation must be able to access the relevant memory type
                 * TODO UCT should expose a bitmap of accessible memory types
                 */
                if (md_attr->cap.access_mem_type != select_param->mem_type) {
                    ucs_trace("lane[%d]: no access to mem type %s", lane,
                              ucs_memory_type_names[select_param->mem_type]);
                    continue;
                }
            }
        }

        lanes[num_lanes++] = lane;
    }

    ucs_trace("selected %d lanes", num_lanes);
    return num_lanes;
}

static void
ucp_proto_common_add_overheads(const ucp_proto_common_init_params_t *params,
                               double tl_overhead, ucp_md_map_t reg_md_map)
{
    ucp_context_h context = params->super.worker->context;
    ucs_linear_func_t send_overheads;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;

    send_overheads.c = tl_overhead + params->overhead;
    send_overheads.m = 0;

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        /* Go over all memory domains */
        ucs_for_each_bit(md_index, reg_md_map) {
            md_attr = &context->tl_mds[md_index].attr;
            ucs_linear_func_add_inplace(&send_overheads, md_attr->reg_cost);
        }
    }

    ucs_linear_func_add_inplace(&params->super.caps->ranges[0].perf,
                                send_overheads);
}

static void
ucp_proto_common_calc_completion(const ucp_proto_common_init_params_t *params,
                                 ucs_linear_func_t pack_time,
                                 ucs_linear_func_t uct_time,
                                 size_t frag_size, double latency)
{
    ucp_proto_perf_range_t *range = &params->super.caps->ranges[0];

    range->max_length  = SIZE_MAX;
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        range->perf    = uct_time; /* Time to send data */
        range->perf.c += latency;  /* Time to receive an ACK back, which is
                                      needed to release the send buffer */
    } else {
        range->perf    = pack_time; /* Time to pack the data */
    }
}

static void
ucp_proto_common_calc_latency(const ucp_proto_common_init_params_t *params,
                              ucs_linear_func_t pack_time,
                              ucs_linear_func_t uct_time,
                              size_t frag_size, double overhead)
{
    ucp_proto_perf_range_t *range = &params->super.caps->ranges[0];
    ucs_linear_func_t   recv_time = ucs_linear_func_make(overhead, pack_time.m);

    range->max_length = SIZE_MAX;
    range->perf       = ucs_linear_func_add(uct_time, recv_time);
    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        ucs_linear_func_add_inplace(&range->perf, pack_time);
    }
}

void ucp_proto_common_calc_perf(const ucp_proto_common_init_params_t *params,
                                ucp_lane_index_t lane)
{
    ucp_context_h context              = params->super.worker->context;
    ucp_proto_caps_t *caps             = params->super.caps;
    const uct_iface_attr_t *iface_attr = ucp_proto_common_get_iface_attr(params,
                                                                         lane);
    double bandwidth, overhead, latency;
    ucs_linear_func_t pack_time;
    ucs_linear_func_t uct_time;
    uint32_t op_attr_mask;
    ucp_md_map_t md_map;
    size_t frag_size;

    /* TODO
     * - support multiple lanes
     * - consider remote/local system device
     * - consider memory type for pack/unpack
     */

    bandwidth        = ucp_proto_common_iface_bandwidth(params, iface_attr);
    overhead         = iface_attr->overhead;
    latency          = ucp_tl_iface_latency(context, &iface_attr->latency) +
                       params->latency;
    md_map           = UCS_BIT(ucp_proto_common_get_md_index(params, lane));
    frag_size        = ucp_proto_get_iface_attr_field(iface_attr,
                                                      params->fragsz_offset) -
                       params->hdr_size;

    caps->cfg_thresh = params->cfg_thresh;
    caps->min_length = 0;
    caps->num_ranges = 1;

    op_attr_mask = ucp_proto_select_op_attr_from_flags(
                            params->super.select_param->op_flags);
    uct_time     = ucs_linear_func_make(latency, 1.0 / bandwidth);
    pack_time    = ucs_linear_func_make(0, 1.0 / context->config.ext.bcopy_bw);

    if (op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) {
        /* calculate time to complete the send operation locally */
        ucp_proto_common_calc_completion(params, pack_time, uct_time, frag_size,
                                         latency);
    } else {
        /* calculate the time it takes for the message to be received on the
         * remote side */
        ucp_proto_common_calc_latency(params, pack_time, uct_time, frag_size,
                                      overhead);
    }

    ucp_proto_common_add_overheads(params, overhead, md_map);
}
