/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_common.inl"
#include <uct/api/v2/uct_v2.h>


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
    const uct_iface_attr_t *iface_attr;
    size_t uct_max_iov;

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

    /* Get max IOV from UCT capabilities */
    iface_attr  = ucp_proto_common_get_iface_attr(&params->super, lane);
    uct_max_iov = ucp_proto_common_get_iface_attr_field(iface_attr,
                                                        params->max_iov_offs,
                                                        SIZE_MAX);

    /* Final max_iov is limited both by UCP and UCT, so it can be uint8_t */
    UCS_STATIC_ASSERT(UCP_MAX_IOV <= UINT8_MAX);
    lane_priv->max_iov = ucs_min(uct_max_iov, UCP_MAX_IOV);
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

static void
ucp_proto_common_get_lane_distance(const ucp_proto_init_params_t *params,
                                   ucp_lane_index_t lane,
                                   ucs_sys_device_t sys_dev,
                                   ucs_sys_dev_distance_t *distance)
{
    ucp_context_h context       = params->worker->context;
    ucp_rsc_index_t rsc_index   = ucp_proto_common_get_rsc_index(params, lane);
    ucs_sys_device_t tl_sys_dev = context->tl_rscs[rsc_index].tl_rsc.sys_device;
    ucs_status_t status;

    status = ucs_topo_get_distance(sys_dev, tl_sys_dev, distance);
    ucs_assertv_always(status == UCS_OK, "sys_dev=%d tl_sys_dev=%d", sys_dev,
                       tl_sys_dev);
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
                                                 params->max_length);
}

static void ucp_proto_common_update_lane_perf_by_distance(
        ucp_proto_common_tl_perf_t *perf,
        const ucs_sys_dev_distance_t *distance)
{
    perf->bandwidth    = ucs_min(perf->bandwidth, distance->bandwidth);
    perf->sys_latency += distance->latency;
}

void ucp_proto_common_get_lane_perf(const ucp_proto_common_init_params_t *params,
                                    ucp_lane_index_t lane,
                                    ucp_proto_common_tl_perf_t *perf)
{
    const uct_iface_attr_t *iface_attr =
            ucp_proto_common_get_iface_attr(&params->super, lane);
    ucp_worker_h worker   = params->super.worker;
    ucp_context_h context = worker->context;
    const ucp_rkey_config_t *rkey_config;
    ucs_sys_dev_distance_t distance;

    perf->overhead  = iface_attr->overhead + params->overhead;
    perf->bandwidth = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);
    perf->latency   = ucp_tl_iface_latency(context, &iface_attr->latency) +
                      params->latency;

    perf->sys_latency = 0;
    perf->min_frag    = ucp_proto_common_get_iface_attr_field(
            iface_attr, params->min_frag_offs, 0);
    perf->max_frag    = ucp_proto_common_get_iface_attr_field(
            iface_attr, params->max_frag_offs, SIZE_MAX);

    /* For zero copy send, consider local system topology distance */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        ucp_proto_common_get_lane_distance(&params->super, lane,
                                           params->super.select_param->sys_dev,
                                           &distance);
        ucp_proto_common_update_lane_perf_by_distance(perf, &distance);
    }

    /* For remote memory access, consider remote system topology distance */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assertv(params->super.rkey_cfg_index < worker->rkey_config_count,
                    "rkey_cfg_index=%d rkey_config_count=%d",
                    params->super.rkey_cfg_index, worker->rkey_config_count);
        rkey_config = &worker->rkey_config[params->super.rkey_cfg_index];
        distance    = rkey_config->lanes_distance[lane];
        ucp_proto_common_update_lane_perf_by_distance(perf, &distance);
    }

    ucs_assert(perf->bandwidth > 1.0);
    ucs_assert(perf->overhead >= 0);
    ucs_assert(perf->max_frag > 0);
    ucs_assert(perf->sys_latency >= 0);
}

static ucp_lane_index_t ucp_proto_common_find_lanes_internal(
        const ucp_proto_init_params_t *params, uct_ep_operation_t memtype_op,
        unsigned flags, ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
        ucp_lane_index_t max_lanes, ucp_lane_map_t exclude_map,
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

    if (flags & UCP_PROTO_COMMON_INIT_FLAG_HDR_ONLY) {
        /* Skip send payload check */
    } else if (flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        if ((select_param->dt_class == UCP_DATATYPE_GENERIC)) {
            /* Generic/IOV datatype cannot be used with zero-copy send */
            ucs_trace("datatype %s cannot be used with zcopy",
                      ucp_datatype_class_names[select_param->dt_class]);
            goto out;
        }
    } else if (!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(select_param->mem_type) &&
               (memtype_op == UCT_EP_OP_LAST)) {
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
                                                     params->memtype_op,
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
ucp_proto_common_find_am_bcopy_hdr_lane(const ucp_proto_init_params_t *params)
{
    ucp_lane_index_t lane = UCP_NULL_LANE;
    ucp_lane_index_t num_lanes;

    num_lanes = ucp_proto_common_find_lanes_internal(
            params, UCT_EP_OP_LAST, UCP_PROTO_COMMON_INIT_FLAG_HDR_ONLY,
            UCP_LANE_TYPE_AM, UCT_IFACE_FLAG_AM_BCOPY, 1, 0, &lane);
    if (num_lanes == 0) {
        ucs_debug("no active message lane for %s", params->proto_name);
        return UCP_NULL_LANE;
    }

    ucs_assert(num_lanes == 1);

    return lane;
}

static ucs_linear_func_t
ucp_proto_common_memreg_time(const ucp_proto_common_init_params_t *params,
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

static ucs_status_t
ucp_proto_common_buffer_copy_time(ucp_worker_h worker, const char *title,
                                  ucs_memory_type_t local_mem_type,
                                  ucs_memory_type_t remote_mem_type,
                                  uct_ep_operation_t memtype_op,
                                  ucs_linear_func_t *copy_time)
{
    ucp_context_h context = worker->context;
    ucp_worker_iface_t *memtype_wiface;
    const ucp_ep_config_t *ep_config;
    uct_perf_attr_t perf_attr;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (UCP_MEM_IS_HOST(local_mem_type) && UCP_MEM_IS_HOST(remote_mem_type)) {
        *copy_time = ucs_linear_func_make(0,
                                          1.0 / context->config.ext.bcopy_bw);
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
    perf_attr.field_mask         = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH;
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

    rsc_index      = ep_config->key.lanes[lane].rsc_index;
    memtype_wiface = ucp_worker_iface(worker, rsc_index);

    status = uct_iface_estimate_perf(memtype_wiface->iface, &perf_attr);
    if (status != UCS_OK) {
        ucs_error("failed to get memtype wiface %p performance: %s",
                  memtype_wiface, ucs_status_string(status));
        return status;
    }

    copy_time->c = ucp_tl_iface_latency(context,
                                        &memtype_wiface->attr.latency) +
                   perf_attr.overhead;
    copy_time->m = 1.0 / ucp_tl_iface_bandwidth(context, &perf_attr.bandwidth);

    return UCS_OK;
}

/*
 * Calculate the performance pipelining an operation with performance 'perf1'
 * with an operation with performance 'perf2' by using fragments with size
 * 'frag_size' bytes.
*/
static ucs_linear_func_t ucp_proto_common_ppln_perf(ucs_linear_func_t perf1,
                                                    ucs_linear_func_t perf2,
                                                    double frag_size)
{
    double adjusted_frag_size = ucs_max(frag_size, 1.0);
    double max_m              = ucs_max(perf1.m, perf2.m);
    ucs_linear_func_t result;

    /* Pipeline overhead is maximal of both protocols' overheads */
    result.c = ucs_max(perf1.c, perf2.c);

    /*
     * Pipeline cost per byte is maximal of both protocols' cost per byte.
     * If the overhead per fragment is higher than the cost per byte, we take
     * the overhead per fragment divided by the fragment size.
     *
     * NOTE: We don't add the fragment overhead to the cost per byte, because we
     * assume multiple fragments are sent, so the overhead per fragment can be
     * hidden by the cost per byte of the same operation.
     */
    result.m = ucs_max(max_m, result.c / adjusted_frag_size);

    ucs_assert((result.m >= perf1.m) && (result.m >= perf2.m) &&
               (result.c >= perf1.c) && (result.c >= perf2.c));
    return result;
}

static ucs_linear_func_t
ucp_proto_common_ppln3_perf(ucs_linear_func_t perf1, ucs_linear_func_t perf2,
                            ucs_linear_func_t perf3, double frag_size)
{
    return ucp_proto_common_ppln_perf(ucp_proto_common_ppln_perf(perf1, perf2,
                                                                 frag_size),
                                      perf3, frag_size);
}

ucs_status_t
ucp_proto_common_init_caps(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *perf,
                           ucp_md_map_t reg_md_map)
{
    ucp_proto_caps_t *caps                       = params->super.caps;
    const ucp_proto_select_param_t *select_param = params->super.select_param;
    ucs_linear_func_t send_ovrh, xfer, recv_ovrh;
    ucp_proto_perf_range_t *range0, *range1;
    ucs_memory_type_t recv_mem_type;
    ucs_linear_func_t perf_multi;
    uint32_t op_attr_mask;
    ucs_status_t status;
    size_t frag_size;
    double frag_ovrh;

    /* Remote access implies zero copy on receiver */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assert(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY);
    }

    op_attr_mask = ucp_proto_select_op_attr_from_flags(select_param->op_flags);

    /* Calculate sender overhead */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) {
        send_ovrh = ucp_proto_common_memreg_time(params, reg_md_map);
    } else {
        ucs_assert(reg_md_map == 0);
        status = ucp_proto_common_buffer_copy_time(
                params->super.worker, "send-copy", UCS_MEMORY_TYPE_HOST,
                select_param->mem_type, params->memtype_op, &send_ovrh);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Add constant CPU overhead */
    send_ovrh.c += perf->overhead;

    /*
     * Add the latency of response/ACK back from the receiver.
     * It's counted as overhead because it does not keep the transport busy.
     */
    if (/* Protocol is waiting for response */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE) ||
        /* Send time is representing request completion, which in case of zcopy
           waits for ACK from remote side. */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
         (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY))) {
        send_ovrh.c += perf->latency;
    }

    /* Calculate transport time */
    if ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY)) {
        /* If we care only about time to start sending the message, ignore
           the transport time */
        xfer = ucs_linear_func_make(0, 0);
    } else {
        xfer = ucs_linear_func_make(perf->latency + perf->sys_latency,
                                    1.0 / perf->bandwidth);
    }

    /* Calculate receiver overhead */
    if (/* Don't care about receiver time for one-sided remote access */
        (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) ||
        /* Count only send completion time without waiting for a response */
        ((op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) &&
         !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE))) {
        recv_ovrh = ucs_linear_func_make(0, 0);
    } else {
        if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY) {
            /* Receiver has to register its buffer */
            recv_ovrh = ucp_proto_common_memreg_time(params, reg_md_map);
        } else {
            if (params->super.rkey_config_key == NULL) {
                /* Assume same memory type as sender */
                recv_mem_type = select_param->mem_type;
            } else {
                recv_mem_type = params->super.rkey_config_key->mem_type;
            }

            /* Receiver has to copy data */
            recv_ovrh = ucs_linear_func_make(0, 0); /* silence cppcheck */
            ucp_proto_common_buffer_copy_time(params->super.worker, "recv-copy",
                                              UCS_MEMORY_TYPE_HOST,
                                              recv_mem_type,
                                              UCT_EP_OP_PUT_SHORT, &recv_ovrh);
        }

        /* Receiver has to process the incoming message */
        if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS)) {
            /* latency measure: add remote-side processing time */
            recv_ovrh.c += perf->overhead;
        }
    }

    /* Initialize capabilities */
    caps->cfg_thresh   = params->cfg_thresh;
    caps->cfg_priority = params->cfg_priority;
    caps->min_length   = ucs_max(params->min_length, perf->min_frag);

    ucs_assert(perf->max_frag >= params->hdr_size);
    frag_size = ucs_min(params->max_length, perf->max_frag - params->hdr_size);

    /* First range represents sending the first fragment */
    range0             = &caps->ranges[0];
    range0->max_length = frag_size;
    range0->perf       = ucs_linear_func_add3(send_ovrh, xfer, recv_ovrh);

    /* Second range represents sending rest of the fragments, if applicable */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG) {
        /* If the 1st range already covers up to max_length, or the protocol is
         * limited to sending only one fragment - only one range is added.
         */
        caps->num_ranges = 1;
    } else {
        caps->num_ranges   = 2;
        range1             = &caps->ranges[1];
        range1->max_length = params->max_length;

        perf_multi = ucp_proto_common_ppln3_perf(send_ovrh, xfer, recv_ovrh,
                                                 frag_size);

        /* Overhead of sending one fragment before starting the pipeline */
        frag_ovrh = ucs_linear_func_apply(range0->perf, frag_size) -
                    ucs_linear_func_apply(perf_multi, frag_size);

        /* Apply the pipelining effect when sending multiple fragments */
        range1->perf = ucs_linear_func_add(perf_multi,
                                           ucs_linear_func_make(frag_ovrh, 0));
    }

    return UCS_OK;
}

void ucp_proto_request_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.state.uct_comp);

    /* request should NOT be on pending queue because when we decrement the last
     * refcount the request is not on the pending queue any more
     */
    ucp_proto_request_zcopy_cleanup(req, UCP_DT_MASK_ALL);
    ucp_request_complete_send(req, req->send.state.uct_comp.status);
}

void ucp_proto_trace_selected(ucp_request_t *req, size_t msg_length)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    UCS_STRING_BUFFER_ONSTACK(proto_config_strb, UCP_PROTO_CONFIG_STR_MAX);
    const ucp_proto_config_t *proto_config = req->send.proto_config;

    ucp_proto_select_param_str(&proto_config->select_param, &sel_param_strb);
    proto_config->proto->config_str(msg_length, msg_length, proto_config->priv,
                                    &proto_config_strb);
    ucp_trace_req(req, "%s length %zu using %s{%s}",
                  ucs_string_buffer_cstr(&sel_param_strb), msg_length,
                  proto_config->proto->name,
                  ucs_string_buffer_cstr(&proto_config_strb));
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
