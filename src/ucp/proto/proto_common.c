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
        ucs_string_buffer_appendf(strb, ",mh%d", lpriv->memh_index);
    }
    if (lpriv->rkey_index != UCP_NULL_RESOURCE) {
        ucs_string_buffer_appendf(strb, ",rk%d", lpriv->rkey_index);
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

size_t ucp_proto_common_get_iface_attr_field(const uct_iface_attr_t *iface_attr,
                                             ptrdiff_t field_offset,
                                             size_t dfl_value)
{
    if (field_offset == UCP_PROTO_COMMON_OFFSET_INVALID) {
        return dfl_value;
    }

    return *(const size_t*)UCS_PTR_BYTE_OFFSET(iface_attr, field_offset);
}

size_t
ucp_proto_common_get_max_frag(const ucp_proto_common_init_params_t *params,
                              const uct_iface_attr_t *iface_attr)
{
    return ucp_proto_common_get_iface_attr_field(iface_attr,
                                                 params->max_frag_offs,
                                                 SIZE_MAX);
}

double
ucp_proto_common_iface_bandwidth(const ucp_proto_common_init_params_t *params,
                                 const uct_iface_attr_t *iface_attr)
{
    return ucp_tl_iface_bandwidth(params->super.worker->context,
                                  &iface_attr->bandwidth);
}

static ucp_lane_index_t
ucp_proto_common_find_lanes_internal(const ucp_proto_init_params_t *params,
                                     unsigned flags, ucp_lane_type_t lane_type,
                                     uint64_t tl_cap_flags,
                                     ucp_lane_index_t max_lanes,
                                     ucp_lane_map_t exclude_map,
                                     ucp_lane_index_t *lanes)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    ucp_context_h context                        = params->worker->context;
    const ucp_ep_config_key_t *ep_config_key     = params->ep_config_key;
    const ucp_rkey_config_key_t *rkey_config_key = params->rkey_config_key;
    const ucp_proto_select_param_t *select_param = params->select_param;
    const uct_iface_attr_t *iface_attr;
    ucp_lane_index_t lane, num_lanes;
    const uct_md_attr_t *md_attr;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t md_index;
    ucp_lane_map_t lane_map;
    char lane_desc[64];

    num_lanes = 0;

    ucp_proto_select_param_str(select_param, &sel_param_strb);
    if (rkey_config_key != NULL) {
        ucs_string_buffer_appendf(&sel_param_strb, "->");
        ucp_rkey_config_dump_brief(rkey_config_key, &sel_param_strb);
    }
    ucs_trace("selecting up to %d/%d lanes for %s %s", max_lanes,
              ep_config_key->num_lanes, params->proto_name,
              ucs_string_buffer_cstr(&sel_param_strb));
    ucs_log_indent(1);

    if (flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        if ((select_param->dt_class == UCP_DATATYPE_GENERIC) ||
            (select_param->dt_class == UCP_DATATYPE_IOV)) {
            /* Generic/IOV datatype cannot be used with zero-copy send */
            /* TODO support IOV registration */
            ucs_trace("datatype %s cannot be used with zcopy",
                      ucp_datatype_class_names[select_param->dt_class]);
            goto out;
        }
    } else if (!(flags & UCP_PROTO_COMMON_INIT_FLAG_MEM_TYPE) &&
               (select_param->dt_class != UCP_DATATYPE_GENERIC) &&
               !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(select_param->mem_type)) {
        /* If zero-copy is off, the memory must be host-accessible for
         * non-generic type (for generic type there is no buffer to access) */
        ucs_trace("memory type %s with datatype %s is not supported",
                  ucs_memory_type_names[select_param->mem_type],
                  ucp_datatype_class_names[select_param->dt_class]);
        goto out;
    }

    lane_map      = UCS_MASK(ep_config_key->num_lanes) & ~exclude_map;
    ucs_for_each_bit(lane, lane_map) {
        if (num_lanes >= max_lanes) {
            break;
        }

        ucs_assert(lane < UCP_MAX_LANES);
        rsc_index = ep_config_key->lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        snprintf(lane_desc, sizeof(lane_desc),
                 "lane[%d] " UCT_TL_RESOURCE_DESC_FMT, lane,
                 UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc));

        /* Check if lane type matches */
        ucs_assert(lane < UCP_MAX_LANES);
        if (!(ep_config_key->lanes[lane].lane_types & UCS_BIT(lane_type))) {
            ucs_trace("%s: no %s in name types", lane_desc,
                      ucp_lane_type_info[lane_type].short_name);
            continue;
        }

        /* Check iface capabilities */
        iface_attr = ucp_proto_common_get_iface_attr(params, lane);
        if (!ucs_test_all_flags(iface_attr->cap.flags, tl_cap_flags)) {
            ucs_trace("%s: no cap 0x%" PRIx64, lane_desc, tl_cap_flags);
            continue;
        }

        md_index = context->tl_rscs[rsc_index].md_index;
        md_attr  = &context->tl_mds[md_index].attr;

        /* Check memory registration capabilities for zero-copy case */
        if (flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
            if (md_attr->cap.flags & UCT_MD_FLAG_NEED_MEMH) {
                /* Memory domain must support registration on the relevant
                 * memory type */
                if (!(md_attr->cap.flags & UCT_MD_FLAG_REG) ||
                    !(md_attr->cap.reg_mem_types & UCS_BIT(select_param->mem_type))) {
                    ucs_trace("%s: no reg of mem type %s", lane_desc,
                              ucs_memory_type_names[select_param->mem_type]);
                    continue;
                }
            } else if (!(md_attr->cap.access_mem_types &
                         UCS_BIT(select_param->mem_type))) {
                /*
                 * Memory domain which does not require a registration for zero
                 * copy operation must be able to access the relevant memory type
                 */
                ucs_trace("%s: no access to mem type %s", lane_desc,
                          ucs_memory_type_names[select_param->mem_type]);
                continue;
            }
        }

        /* Check remote access capabilities */
        if (flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
            if (rkey_config_key == NULL) {
                ucs_trace("protocol requires remote access but remote key is "
                          "not present");
                goto out;
            }

            if (md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) {
                if (!(rkey_config_key->md_map &
                    UCS_BIT(ep_config_key->lanes[lane].dst_md_index))) {
                    ucs_trace("%s: no support of dst md map 0x%" PRIx64,
                              lane_desc, rkey_config_key->md_map);
                    continue;
                }
            } else if (!(md_attr->cap.access_mem_types &
                         UCS_BIT(rkey_config_key->mem_type))) {
                ucs_trace("%s: no access to remote mem type %s", lane_desc,
                          ucs_memory_type_names[rkey_config_key->mem_type]);
                continue;
            }
        }

        lanes[num_lanes++] = lane;
    }

out:
    ucs_trace("selected %d lanes", num_lanes);
    ucs_log_indent(-1);
    return num_lanes;
}

ucp_md_map_t
ucp_proto_common_reg_md_map(const ucp_proto_common_init_params_t *params,
                            ucp_lane_map_t lane_map)
{
    ucp_context_h context                        = params->super.worker->context;
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;
    ucp_md_map_t reg_md_map;
    ucp_lane_index_t lane;

    /* Register memory only for zero-copy send operations */
    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        return 0;
    }

    reg_md_map = 0;
    ucs_for_each_bit(lane, lane_map) {
        md_index = ucp_proto_common_get_md_index(&params->super, lane);
        md_attr  = &context->tl_mds[md_index].attr;

        /* Register if the memory domain support registration for the relevant
           memory type, and needs a local memory handle for zero-copy
           communication */
        if (ucs_test_all_flags(md_attr->cap.flags,
                               UCT_MD_FLAG_NEED_MEMH | UCT_MD_FLAG_REG) &&
            (md_attr->cap.reg_mem_types & UCS_BIT(select_param->mem_type))) {
            reg_md_map |= UCS_BIT(md_index);
        }
    }

    return reg_md_map;
}

ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_common_init_params_t *params,
                            ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
                            ucp_lane_index_t max_lanes,
                            ucp_lane_map_t exclude_map, ucp_lane_index_t *lanes)
{
    ucp_lane_index_t lane_index, lane, num_lanes, num_valid_lanes;
    const uct_iface_attr_t *iface_attr;
    size_t frag_size;

    num_lanes = ucp_proto_common_find_lanes_internal(&params->super,
                                                     params->flags, lane_type,
                                                     tl_cap_flags, max_lanes,
                                                     exclude_map, lanes);

    num_valid_lanes = 0;
    for (lane_index = 0; lane_index < num_lanes; ++lane_index) {
        lane       = lanes[lane_index];
        iface_attr = ucp_proto_common_get_iface_attr(&params->super, lane);
        frag_size  = ucp_proto_common_get_max_frag(params, iface_attr);
        /* Max fragment size should be larger than header size */
        if (frag_size <= params->hdr_size) {
            ucs_trace("lane[%d]: max fragment is too small %zu, need > %zu",
                      lane, frag_size, params->hdr_size);
            continue;
        }

        lanes[num_valid_lanes++] = lane;
    }

    if (num_valid_lanes != num_lanes) {
        ucs_assert(num_valid_lanes < num_lanes);
        ucs_trace("selected %d/%d valid lanes", num_valid_lanes, num_lanes);
    }

    return num_valid_lanes;
}

ucp_lane_index_t
ucp_proto_common_find_am_bcopy_lane(const ucp_proto_init_params_t *params)
{
    ucp_lane_index_t lane = UCP_NULL_LANE;
    ucp_lane_index_t num_lanes;

    num_lanes = ucp_proto_common_find_lanes_internal(
            params, UCP_PROTO_COMMON_INIT_FLAG_MEM_TYPE, UCP_LANE_TYPE_AM,
            UCT_IFACE_FLAG_AM_BCOPY, 1, 0, &lane);
    if (num_lanes == 0) {
        ucs_debug("no active message lane for %s", params->proto_name);
        return UCP_NULL_LANE;
    }

    ucs_assert(num_lanes == 1);

    return lane;
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

static ucs_linear_func_t
ucp_proto_common_get_reg_cost(const ucp_proto_common_init_params_t *params,
                              ucp_md_map_t reg_md_map)
{
    ucp_context_h context      = params->super.worker->context;
    ucs_linear_func_t reg_cost = ucs_linear_func_make(0, 0);
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        /* Go over all memory domains */
        ucs_for_each_bit(md_index, reg_md_map) {
            md_attr = &context->tl_mds[md_index].attr;
            ucs_linear_func_add_inplace(&reg_cost, md_attr->reg_cost);
        }
    }

    return reg_cost;
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

    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG) {
        range->max_length = frag_size;
    } else {
        range->max_length = SIZE_MAX;
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
                              size_t frag_size, double recv_overhead)
{
    ucs_linear_func_t piped_size, piped_send_cost, recv_time;
    ucp_proto_perf_range_t *range;

    recv_time         = ucp_proto_common_recv_time(params, recv_overhead,
                                                   pack_time);

    /* Performance for 0...frag_size */
    range             = &params->super.caps->ranges[params->super.caps->num_ranges++];
    range->max_length = frag_size;
    range->perf       = ucs_linear_func_add(uct_time, recv_time);
    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        ucs_linear_func_add_inplace(&range->perf, pack_time);
    }

    /* If the 1st range already covers up to SIZE_MAX, or the protocol should be
     * limited by single fragment - no more ranges are created
     */
    if ((range->max_length == SIZE_MAX) ||
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG)) {
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
    double overhead, latency, tl_latency;
    const uct_iface_attr_t *iface_attr;
    size_t frag_size, tl_min_length;
    ucs_linear_func_t extra_time;
    ucs_linear_func_t pack_time;
    ucs_linear_func_t uct_time;
    ucp_lane_index_t lane;
    uint32_t op_attr_mask;

    /* Remote access implies zero copy on receiver */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assert(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY);
    }

   /* TODO
     * - consider remote/local system device
     * - consider memory type for pack/unpack
     */
    caps->cfg_thresh   = params->cfg_thresh;
    caps->cfg_priority = params->cfg_priority;
    caps->num_ranges   = 0;
    caps->min_length   = 0;

    /* Collect latency and overhead from all lanes */
    overhead = 0;
    latency  = params->latency;
    ucs_for_each_bit(lane, perf_params->lane_map) {
        iface_attr    = ucp_proto_common_get_iface_attr(&params->super, lane);
        tl_latency    = ucp_tl_iface_latency(context, &iface_attr->latency);
        tl_min_length = ucp_proto_common_get_iface_attr_field(
                iface_attr, params->min_frag_offs, 0);

        overhead        += iface_attr->overhead;
        latency          = ucs_max(tl_latency, latency);
        caps->min_length = ucs_max(caps->min_length, tl_min_length);
    }

    /* Take fragment size from first lane */
    frag_size = perf_params->frag_size;
    if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE)) {
        /* if the data returns as a response, no need to subtract header size */
        frag_size -= params->hdr_size;
    }

    op_attr_mask  = ucp_proto_select_op_attr_from_flags(
                            params->super.select_param->op_flags);
    uct_time      = ucs_linear_func_make(latency, 1.0 / perf_params->bandwidth);
    pack_time     = ucs_linear_func_make(0, 1.0 / context->config.ext.bcopy_bw);
    extra_time    = ucp_proto_common_get_reg_cost(params, perf_params->reg_md_map);
    extra_time.c += overhead + params->overhead;

    if ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE)) {
        /* Calculate time to complete the send operation locally */
        ucp_proto_common_calc_completion(params, perf_params, pack_time,
                                         uct_time, frag_size, latency);
    } else {
        /* Calculate the time for message data transfer */
        ucp_proto_common_calc_latency(params, perf_params, pack_time, uct_time,
                                      frag_size, overhead);

        /* If we wait for response, add latency of sending the request */
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE) {
            extra_time.c += latency;
        }
    }

    ucp_proto_common_add_perf(&params->super, extra_time);
}

void ucp_proto_request_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.state.uct_comp);

    /* request should NOT be on pending queue because when we decrement the last
     * refcount the request is not on the pending queue any more
     */
    ucp_proto_request_zcopy_cleanup(req);
    ucp_request_complete_send(req, req->send.state.uct_comp.status);
}

void ucp_proto_request_select_error(ucp_request_t *req,
                                    ucp_proto_select_t *proto_select,
                                    ucp_worker_cfg_index_t rkey_cfg_index,
                                    const ucp_proto_select_param_t *sel_param,
                                    size_t msg_length)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    UCS_STRING_BUFFER_ONSTACK(proto_select_strb, UCP_PROTO_CONFIG_STR_MAX);
    ucp_ep_h ep = req->send.ep;

    ucp_proto_select_param_str(sel_param, &sel_param_strb);
    ucp_proto_select_dump(ep->worker, ep->cfg_index, rkey_cfg_index,
                          proto_select, &proto_select_strb);
    ucs_fatal("req %p on ep %p to %s: could not find a protocol for %s "
              "length %zu\navailable protocols:\n%s\n",
              req, ep, ucp_ep_peer_name(ep),
              ucs_string_buffer_cstr(&sel_param_strb), msg_length,
              ucs_string_buffer_cstr(&proto_select_strb));
}

void ucp_proto_request_abort(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(UCS_STATUS_IS_ERR(status));
    /*
     * TODO add a method to ucp_proto_t to abort a request (which is currently
     * not scheduled to a pending queue). The method should wait for UCT
     * completions and release associated resources, such as memory handles,
     * remote keys, request ID, etc.
     */
    ucs_fatal("abort request %p proto %s status %s: unimplemented", req,
              req->send.proto_config->proto->name, ucs_status_string(status));
}
