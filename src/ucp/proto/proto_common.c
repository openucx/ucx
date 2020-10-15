/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_common.inl"


static ucp_rsc_index_t
ucp_proto_common_get_rsc_index(const ucp_proto_init_params_t *params,
                               ucp_lane_index_t lane)
{
    ucp_rsc_index_t rsc_index;

    ucs_assert(lane < UCP_MAX_LANES);

    rsc_index = params->ep_config_key->lanes[lane].rsc_index;
    ucs_assert(rsc_index < UCP_MAX_RESOURCES);

    return rsc_index;
}

void ucp_proto_common_lane_priv_init(const ucp_proto_common_init_params_t *params,
                                     ucp_md_map_t md_map, ucp_lane_index_t lane,
                                     ucp_proto_common_lane_priv_t *lane_priv)
{
    const ucp_rkey_config_key_t *rkey_config_key = params->super.rkey_config_key;
    ucp_md_index_t md_index, dst_md_index;

    md_index     = ucp_proto_common_get_md_index(&params->super, lane);
    dst_md_index = params->super.ep_config_key->lanes[lane].dst_md_index;

    lane_priv->lane = lane;

    /* Local key index */
    if (md_map & UCS_BIT(md_index)) {
        lane_priv->memh_index = ucs_bitmap2idx(md_map, md_index);
    } else {
        lane_priv->memh_index = UCP_NULL_RESOURCE;
    }

    /* Remote key index */
    if ((rkey_config_key != NULL) &&
        (rkey_config_key->md_map & UCS_BIT(dst_md_index))) {
        lane_priv->rkey_index = ucs_bitmap2idx(rkey_config_key->md_map,
                                               dst_md_index);
    } else {
        lane_priv->rkey_index = UCP_NULL_RESOURCE;
    }
}

void ucp_proto_common_lane_priv_str(const ucp_proto_common_lane_priv_t *lpriv,
                                    ucs_string_buffer_t *strb)
{
    ucs_string_buffer_appendf(strb, "ln:%d", lpriv->lane);
    if (lpriv->memh_index != UCP_NULL_RESOURCE) {
        ucs_string_buffer_appendf(strb, "/mh:%d", lpriv->memh_index);
    }
    if (lpriv->rkey_index != UCP_NULL_RESOURCE) {
        ucs_string_buffer_appendf(strb, "/rk:%d", lpriv->rkey_index);
    }
}

ucp_md_index_t
ucp_proto_common_get_md_index(const ucp_proto_init_params_t *params,
                              ucp_lane_index_t lane)
{
    ucp_rsc_index_t rsc_index = ucp_proto_common_get_rsc_index(params, lane);
    return params->worker->context->tl_rscs[rsc_index].md_index;
}

const uct_iface_attr_t *
ucp_proto_common_get_iface_attr(const ucp_proto_init_params_t *params,
                                ucp_lane_index_t lane)
{
    return ucp_worker_iface_get_attr(params->worker,
                                     ucp_proto_common_get_rsc_index(params, lane));
}

size_t ucp_proto_get_iface_attr_field(const uct_iface_attr_t *iface_attr,
                                      ptrdiff_t field_offset)
{
    return *(const size_t*)UCS_PTR_BYTE_OFFSET(iface_attr, field_offset);
}

double
ucp_proto_common_iface_bandwidth(const ucp_proto_common_init_params_t *params,
                                 const uct_iface_attr_t *iface_attr)
{
    return ucp_tl_iface_bandwidth(params->super.worker->context,
                                  &iface_attr->bandwidth);
}

ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_common_init_params_t *params,
                            ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
                            ucp_lane_index_t max_lanes, ucp_lane_map_t exclude_map,
                            ucp_lane_index_t *lanes, ucp_md_map_t *reg_md_map_p)
{
    ucp_context_h context                        = params->super.worker->context;
    const ucp_ep_config_key_t *ep_config_key     = params->super.ep_config_key;
    const ucp_rkey_config_key_t *rkey_config_key = params->super.rkey_config_key;
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    const uct_iface_attr_t *iface_attr;
    ucp_lane_index_t lane, num_lanes;
    const uct_md_attr_t *md_attr;
    ucp_rsc_index_t rsc_index;
    ucs_string_buffer_t strb;
    ucp_md_index_t md_index;
    ucp_lane_map_t lane_map;

    ucp_proto_select_param_str(select_param, &strb);
    ucs_trace("selecting %d out of %d lanes for %s %s", max_lanes,
              ep_config_key->num_lanes, params->super.proto_name,
              ucs_string_buffer_cstr(&strb));
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
    } else if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_MEM_TYPE) &&
               (select_param->dt_class != UCP_DATATYPE_GENERIC) &&
               !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(select_param->mem_type)) {
        /* If zero-copy is off, the memory must be host-accessible for
         * non-generic type (for generic type there is no buffer to access) */
        ucs_trace("memory type %s with datatype %s is not supported",
                  ucs_memory_type_names[select_param->mem_type],
                  ucp_datatype_class_names[select_param->dt_class]);
        return 0;
    }

    lane_map      = UCS_MASK(ep_config_key->num_lanes) & ~exclude_map;
    *reg_md_map_p = 0;
    num_lanes     = 0;
    ucs_for_each_bit(lane, lane_map) {
        if (num_lanes >= max_lanes) {
            break;
        }

        /* Check if lane type matches */
        ucs_assert(lane < UCP_MAX_LANES);
        if (!(ep_config_key->lanes[lane].lane_types & UCS_BIT(lane_type))) {
            ucs_trace("lane[%d]: no %s", lane,
                      ucp_lane_type_info[lane_type].short_name);
            continue;
        }

        rsc_index = ep_config_key->lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        /* Check iface capabilities */
        iface_attr = ucp_proto_common_get_iface_attr(&params->super, lane);
        if (!ucs_test_all_flags(iface_attr->cap.flags, tl_cap_flags)) {
            ucs_trace("lane[%d]: no cap 0x%"PRIx64, lane, tl_cap_flags);
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

                *reg_md_map_p |= UCS_BIT(md_index);
            } else {
                /* Memory domain which does not require a registration for zero
                 * copy operation must be able to access the relevant memory type
                 * TODO UCT should expose a bitmap of accessible memory types
                 */
                if (!(md_attr->cap.access_mem_types & UCS_BIT(select_param->mem_type))) {
                    ucs_trace("lane[%d]: no access to mem type %s", lane,
                              ucs_memory_type_names[select_param->mem_type]);
                    continue;
                }
            }
        }

        /* Check remote access capabilities */
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
            ucs_assert(rkey_config_key != NULL);
            if (md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) {
                if (!(rkey_config_key->md_map &
                    UCS_BIT(ep_config_key->lanes[lane].dst_md_index))) {
                    ucs_trace("lane[%d]: no support of dst md map 0x%lx", lane,
                              rkey_config_key->md_map);
                    continue;
                }
            } else {
                if (md_attr->cap.access_mem_types != rkey_config_key->mem_type) {
                    ucs_trace("lane[%d]: no access to remote mem type %s", lane,
                              ucs_memory_type_names[rkey_config_key->mem_type]);
                    continue;
                }
            }
        }

        lanes[num_lanes++] = lane;
    }

    ucs_trace("selected %d lanes", num_lanes);
    return num_lanes;
}

static ucs_linear_func_t
ucp_proto_common_recv_time(const ucp_proto_common_init_params_t *params,
                           double tl_overhead, ucs_linear_func_t pack_time)
{
    ucs_linear_func_t recv_time = ucs_linear_func_make(0, 0);

    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS)) {
        /* latency measure: add remote-side processing time */
        recv_time.c = tl_overhead;
    }

    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY)) {
        recv_time.m = pack_time.m;
    }

    return recv_time;
}

static void ucp_proto_common_add_perf(const ucp_proto_init_params_t *params,
                                      ucs_linear_func_t func)
{
    ucp_proto_caps_t *caps = params->caps;
    unsigned i;

    for (i = 0; i < caps->num_ranges; ++i) {
        ucs_linear_func_add_inplace(&caps->ranges[i].perf, func);
    }
}

static void
ucp_proto_common_add_overheads(const ucp_proto_common_init_params_t *params,
                               double tl_overhead, ucp_md_map_t reg_md_map)
{
    ucp_context_h context = params->super.worker->context;
    ucs_linear_func_t send_overheads;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;

    send_overheads = ucs_linear_func_make(tl_overhead + params->overhead, 0.0);

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        /* Go over all memory domains */
        ucs_for_each_bit(md_index, reg_md_map) {
            md_attr = &context->tl_mds[md_index].attr;
            ucs_linear_func_add_inplace(&send_overheads, md_attr->reg_cost);
        }
    }

    ucp_proto_common_add_perf(&params->super, send_overheads);
}

static void
ucp_proto_common_calc_completion(const ucp_proto_common_init_params_t *params,
                                 const ucp_proto_common_perf_params_t *perf_params,
                                 ucs_linear_func_t pack_time,
                                 ucs_linear_func_t uct_time,
                                 size_t frag_size, double latency)
{
    ucp_proto_perf_range_t *range =
            &params->super.caps->ranges[params->super.caps->num_ranges++];

    if (perf_params->is_multi) {
        /* Multi fragment protocol has no limit */
        range->max_length = SIZE_MAX;
    } else {
        /* Single fragment protocol can send only one fragment */
        range->max_length = frag_size;
    }

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
                              const ucp_proto_common_perf_params_t *perf_params,
                              ucs_linear_func_t pack_time,
                              ucs_linear_func_t uct_time,
                              size_t frag_size, double overhead)
{
    ucs_linear_func_t piped_size, piped_send_cost, recv_time;
    ucp_proto_perf_range_t *range;

    recv_time         = ucp_proto_common_recv_time(params, overhead, pack_time);

    /* Performance for 0...frag_size */
    range             = &params->super.caps->ranges[params->super.caps->num_ranges++];
    range->max_length = frag_size;
    range->perf       = ucs_linear_func_add(uct_time, recv_time);
    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        ucs_linear_func_add_inplace(&range->perf, pack_time);
    }

    /* If the 1st range already covers up to SIZE_MAX, or the protocol does not
     * support multi-fragment - no more ranges are created.
     */
    if ((range->max_length == SIZE_MAX) || !perf_params->is_multi) {
        return;
    }

    /* Performance for frag_size+1...MAX */
    range             = &params->super.caps->ranges[params->super.caps->num_ranges++];
    range->max_length = SIZE_MAX;
    range->perf       = ucs_linear_func_make(0, 0);

    if (ucs_test_all_flags(params->flags, UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                                          UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY)) {
        ucs_linear_func_add_inplace(&range->perf, uct_time);
    } else {
        piped_send_cost = ucs_linear_func_make(0, ucs_max(pack_time.m, uct_time.m));
        piped_size      = ucs_linear_func_make(-1.0 * frag_size, 1);

        /* Copy first fragment */
        if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
            ucs_linear_func_add_value_at(&range->perf, pack_time, frag_size);
        }

        /* Reach the point where we can start sending the last fragment */
        ucs_linear_func_add_inplace(&range->perf,
                                    ucs_linear_func_compose(piped_send_cost,
                                                            piped_size));

        /* Send last fragment */
        ucs_linear_func_add_value_at(&range->perf, uct_time, frag_size);
    }

    /* Receive last fragment */
    ucs_linear_func_add_value_at(&range->perf, recv_time, frag_size);
}

void ucp_proto_common_calc_perf(const ucp_proto_common_init_params_t *params,
                                const ucp_proto_common_perf_params_t *perf_params)
{
    ucp_context_h context  = params->super.worker->context;
    ucp_proto_caps_t *caps = params->super.caps;
    double bandwidth, overhead, latency;
    const uct_iface_attr_t *iface_attr;
    ucs_linear_func_t pack_time;
    ucs_linear_func_t uct_time;
    ucp_lane_index_t lane;
    uint32_t op_attr_mask;
    size_t frag_size;

    /* Remote access implies zero copy on receiver */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assert(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY);
    }

   /* TODO
     * - consider remote/local system device
     * - consider memory type for pack/unpack
     */

    bandwidth = 0;
    overhead  = 0;
    latency   = params->latency;

    /* Collect latency, overhead, bandwidth from all lanes */
    ucs_for_each_bit(lane, perf_params->lane_map) {
        iface_attr = ucp_proto_common_get_iface_attr(&params->super, lane);
        overhead  += iface_attr->overhead;
        latency    = ucs_max(ucp_tl_iface_latency(context, &iface_attr->latency),
                             latency);
        bandwidth += ucp_proto_common_iface_bandwidth(params, iface_attr);
    }

    /* Take fragment size from first lane */
    iface_attr       = ucp_proto_common_get_iface_attr(&params->super,
                                                       perf_params->lane0);
    frag_size        = ucp_proto_get_iface_attr_field(iface_attr,
                                                       params->fragsz_offset) -
                       params->hdr_size;

    caps->cfg_thresh   = params->cfg_thresh;
    caps->cfg_priority = params->cfg_priority;
    caps->min_length   = 0;
    caps->num_ranges   = 0;

    op_attr_mask = ucp_proto_select_op_attr_from_flags(
                            params->super.select_param->op_flags);
    uct_time     = ucs_linear_func_make(latency, 1.0 / bandwidth);
    pack_time    = ucs_linear_func_make(0, 1.0 / context->config.ext.bcopy_bw);

    if (op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) {
        /* Calculate time to complete the send operation locally */
        ucp_proto_common_calc_completion(params, perf_params, pack_time,
                                         uct_time, frag_size, latency);
    } else {
        /* Calculate the time it takes for the message to be received on the
         * remote side */
        ucp_proto_common_calc_latency(params, perf_params, pack_time, uct_time,
                                      frag_size, overhead);
    }

    ucp_proto_common_add_overheads(params, overhead, perf_params->reg_md_map);
}

void ucp_proto_request_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.state.uct_comp);

    /* request should NOT be on pending queue because when we decrement the last
     * refcount the request is not on the pending queue any more
     */
    ucp_proto_request_zcopy_complete(req, req->send.state.uct_comp.status);
}

void ucp_proto_request_select_error(ucp_request_t *req,
                                    ucp_proto_select_t *proto_select,
                                    ucp_worker_cfg_index_t rkey_cfg_index,
                                    const ucp_proto_select_param_t *sel_param,
                                    size_t msg_length)
{
    ucp_ep_h ep = req->send.ep;
    ucs_string_buffer_t strb;

    ucp_proto_select_param_str(sel_param, &strb);
    ucp_proto_select_dump(ep->worker, ep->cfg_index, rkey_cfg_index,
                          proto_select, stdout);
    ucs_fatal("req %p on ep %p to %s: could not find a protocol for %s "
              "length %zu",
              req, ep, ucp_ep_peer_name(ep), ucs_string_buffer_cstr(&strb),
              msg_length);
    ucs_string_buffer_cleanup(&strb);
}
