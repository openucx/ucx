/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_debug.h"
#include "proto_common.inl"

#include <uct/api/v2/uct_v2.h>


ucp_rsc_index_t
ucp_proto_common_get_rsc_index(const ucp_proto_init_params_t *params,
                               ucp_lane_index_t lane)
{
    ucs_assert(lane < UCP_MAX_LANES);
    return params->ep_config_key->lanes[lane].rsc_index;
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
        lane_priv->md_index = md_index;
    } else {
        lane_priv->md_index = UCP_NULL_RESOURCE;
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

void ucp_proto_common_lane_priv_str(const ucp_proto_query_params_t *params,
                                    const ucp_proto_common_lane_priv_t *lpriv,
                                    int show_rsc, int show_path,
                                    ucs_string_buffer_t *strb)
{
    ucp_context_h context = params->worker->context;
    const ucp_ep_config_key_lane_t *ep_lane_cfg;
    const uct_iface_attr_t *iface_attr;
    const ucp_tl_resource_desc_t *rsc;

    ucs_assert(lpriv->lane < UCP_MAX_LANES);
    ep_lane_cfg = &params->ep_config_key->lanes[lpriv->lane];
    if (show_rsc) {
        rsc = &context->tl_rscs[ep_lane_cfg->rsc_index];
        ucs_string_buffer_appendf(strb, UCT_TL_RESOURCE_DESC_FMT,
                                  UCT_TL_RESOURCE_DESC_ARG(&rsc->tl_rsc));
    }

    iface_attr = ucp_worker_iface_get_attr(params->worker,
                                           ep_lane_cfg->rsc_index);
    if (show_path && (iface_attr->dev_num_paths > 1)) {
        if (show_rsc) {
            ucs_string_buffer_appendf(strb, "/");
        }
        ucs_string_buffer_appendf(strb, "path%d", ep_lane_cfg->path_index);
    }
}

ucp_md_index_t
ucp_proto_common_get_md_index(const ucp_proto_init_params_t *params,
                              ucp_lane_index_t lane)
{
    ucp_rsc_index_t rsc_index = ucp_proto_common_get_rsc_index(params, lane);
    return params->worker->context->tl_rscs[rsc_index].md_index;
}

ucs_sys_device_t
ucp_proto_common_get_sys_dev(const ucp_proto_init_params_t *params,
                             ucp_lane_index_t lane)
{
    ucp_rsc_index_t rsc_index = ucp_proto_common_get_rsc_index(params, lane);
    return params->worker->context->tl_rscs[rsc_index].tl_rsc.sys_device;
}

void ucp_proto_common_get_lane_distance(const ucp_proto_init_params_t *params,
                                        ucp_lane_index_t lane,
                                        ucs_sys_device_t sys_dev,
                                        ucs_sys_dev_distance_t *distance)
{
    ucp_context_h context     = params->worker->context;
    ucp_rsc_index_t rsc_index = ucp_proto_common_get_rsc_index(params, lane);
    ucs_sys_device_t tl_sys_dev;
    ucs_status_t status;

    if (rsc_index == UCP_NULL_RESOURCE) {
        *distance = ucs_topo_default_distance;
        return;
    }

    tl_sys_dev = context->tl_rscs[rsc_index].tl_rsc.sys_device;
    status     = ucs_topo_get_distance(sys_dev, tl_sys_dev, distance);
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

static void
ucp_proto_common_get_frag_size(const ucp_proto_common_init_params_t *params,
                               const uct_iface_attr_t *iface_attr,
                               size_t *min_frag_p, size_t *max_frag_p)
{
    *min_frag_p = ucp_proto_common_get_iface_attr_field(iface_attr,
                                                        params->min_frag_offs,
                                                        0);
    *max_frag_p = ucp_proto_common_get_iface_attr_field(iface_attr,
                                                        params->max_frag_offs,
                                                        SIZE_MAX);
}

static void ucp_proto_common_update_lane_perf_by_distance(
        ucp_proto_common_tl_perf_t *perf,
        const ucs_sys_dev_distance_t *distance)
{
    perf->bandwidth    = ucs_min(perf->bandwidth, distance->bandwidth);
    perf->sys_latency += distance->latency;
}

ucs_status_t
ucp_proto_common_lane_perf_attr(const ucp_proto_init_params_t *params,
                                ucp_lane_index_t lane, uct_ep_operation_t op,
                                uint64_t uct_field_mask,
                                uct_perf_attr_t* perf_attr)
{
    ucp_worker_h worker        = params->worker;
    ucp_rsc_index_t rsc_index  = ucp_proto_common_get_rsc_index(params, lane);
    ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);
    ucs_status_t status;

    /* Use the v2 API to query overhead and BW */
    perf_attr->field_mask = UCT_PERF_ATTR_FIELD_OPERATION | uct_field_mask;
    perf_attr->operation  = op;

    status = uct_iface_estimate_perf(wiface->iface, perf_attr);
    if (status != UCS_OK) {
        ucs_error("failed to get iface %p performance: %s", wiface->iface,
                  ucs_status_string(status));
    }

    return status;
}

ucs_status_t
ucp_proto_common_get_lane_perf(const ucp_proto_common_init_params_t *params,
                               ucp_lane_index_t lane,
                               ucp_proto_common_tl_perf_t *perf)
{
    ucp_worker_h worker        = params->super.worker;
    ucp_context_h context      = worker->context;
    ucp_rsc_index_t rsc_index  = ucp_proto_common_get_rsc_index(&params->super,
                                                                lane);
    ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);
    const ucp_rkey_config_t *rkey_config;
    ucs_sys_dev_distance_t distance;
    size_t tl_min_frag, tl_max_frag;
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    ucp_proto_common_get_frag_size(params, &wiface->attr, &tl_min_frag,
                                   &tl_max_frag);

    status = ucp_proto_common_lane_perf_attr(&params->super, lane,
            params->send_op, UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
            UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
            UCT_PERF_ATTR_FIELD_RECV_OVERHEAD | UCT_PERF_ATTR_FIELD_BANDWIDTH |
            UCT_PERF_ATTR_FIELD_LATENCY, &perf_attr);
    if (status != UCS_OK) {
        return status;
    }

    perf->send_pre_overhead  = perf_attr.send_pre_overhead + params->overhead;
    perf->send_post_overhead = perf_attr.send_post_overhead;
    perf->recv_overhead      = perf_attr.recv_overhead + params->overhead;
    perf->bandwidth          = ucp_tl_iface_bandwidth(context,
                                                      &perf_attr.bandwidth);
    perf->latency            = ucp_tl_iface_latency(context,
                                                    &perf_attr.latency) +
                               params->latency;
    perf->sys_latency        = 0;
    perf->min_length         = ucs_max(params->min_length, tl_min_frag);
    perf->max_frag           = tl_max_frag;

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
    ucs_assert(perf->send_pre_overhead >= 0);
    ucs_assert(perf->send_post_overhead >= 0);
    ucs_assert(perf->recv_overhead >= 0);
    ucs_assertv(perf->max_frag >= params->hdr_size, "max_frag=%zu hdr_size=%zu",
                perf->max_frag, params->hdr_size);
    ucs_assert(perf->sys_latency >= 0);

    return UCS_OK;
}

static ucp_lane_index_t ucp_proto_common_find_lanes_internal(
        const ucp_proto_init_params_t *params, uct_ep_operation_t memtype_op,
        unsigned flags, ptrdiff_t max_iov_offs, size_t min_iov,
        ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
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
    size_t max_iov;

    if (max_lanes == 0) {
        return 0;
    }

    ucp_proto_select_info_str(params->worker, params->rkey_cfg_index,
                              params->select_param, ucp_operation_names,
                              &sel_param_strb);

    num_lanes = 0;
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

            if (((md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) ||
                 (flags & UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR)) &&
                !(rkey_config_key->md_map &
                  UCS_BIT(ep_config_key->lanes[lane].dst_md_index))) {
                /* If remote key required remote memory domain should be
                 * available */
                ucs_trace("%s: no support of dst md map 0x%" PRIx64,
                          lane_desc, rkey_config_key->md_map);
                continue;
            }

            if (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) &&
                !(md_attr->cap.access_mem_types &
                  UCS_BIT(rkey_config_key->mem_type))) {
                /* Remote memory domain without remote key must be able to
                 * access relevant memory type */
                ucs_trace("%s: no access to remote mem type %s", lane_desc,
                          ucs_memory_type_names[rkey_config_key->mem_type]);
                continue;
            }
        }

        max_iov = ucp_proto_common_get_iface_attr_field(iface_attr,
                                                        max_iov_offs, SIZE_MAX);
        if (max_iov < min_iov) {
            continue;
        }

        ucs_trace("%s: added as lane %d", lane_desc, lane);
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
    size_t tl_min_frag, tl_max_frag;

    num_lanes = ucp_proto_common_find_lanes_internal(
            &params->super, params->memtype_op, params->flags,
            params->max_iov_offs, params->min_iov, lane_type, tl_cap_flags,
            max_lanes, exclude_map, lanes);

    num_valid_lanes = 0;
    for (lane_index = 0; lane_index < num_lanes; ++lane_index) {
        lane       = lanes[lane_index];
        iface_attr = ucp_proto_common_get_iface_attr(&params->super, lane);

        ucp_proto_common_get_frag_size(params, iface_attr, &tl_min_frag,
                                       &tl_max_frag);

        /* Minimal fragment size must be 0, unless 'MIN_FRAG' flag is set */
        if (!(params->flags & UCP_PROTO_COMMON_INIT_FLAG_MIN_FRAG) &&
            (tl_min_frag > 0)) {
            ucs_trace("lane[%d]: minimal fragment %zu is not 0", lane,
                      tl_min_frag);
            continue;
        }

        /* Maximal fragment size should be larger than header size */
        if (tl_max_frag <= params->hdr_size) {
            ucs_trace("lane[%d]: max fragment is too small %zu, need > %zu",
                      lane, tl_max_frag, params->hdr_size);
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
            UCP_PROTO_COMMON_OFFSET_INVALID, 1, UCP_LANE_TYPE_AM,
            UCT_IFACE_FLAG_AM_BCOPY, 1, 0, &lane);
    if (num_lanes == 0) {
        ucs_debug("no active message lane for %s", params->proto_name);
        return UCP_NULL_LANE;
    }

    ucs_assert(num_lanes == 1);

    return lane;
}

ucs_linear_func_t
ucp_proto_common_memreg_time(const ucp_proto_common_init_params_t *params,
                             ucp_md_map_t reg_md_map)
{
    ucp_context_h context      = params->super.worker->context;
    ucs_linear_func_t reg_cost = ucs_linear_func_make(0, 0);
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;

    /* Go over all memory domains */
    ucs_for_each_bit(md_index, reg_md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        ucs_linear_func_add_inplace(&reg_cost, md_attr->reg_cost);
        ucs_trace("md %s" UCP_PROTO_PERF_FUNC_FMT(reg_cost),
                  context->tl_mds[md_index].rsc.md_name,
                  UCP_PROTO_PERF_FUNC_ARG(&md_attr->reg_cost));
    }

    return reg_cost;
}

ucs_status_t
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
                                   UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_RECV_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH |
                                   UCT_PERF_ATTR_FIELD_LATENCY;
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

    /* all allowed copy operations are one-sided */
    ucs_assert(perf_attr.recv_overhead < 1e-15);
    copy_time->c = ucp_tl_iface_latency(context, &perf_attr.latency) +
                   perf_attr.send_pre_overhead + perf_attr.send_post_overhead +
                   perf_attr.recv_overhead;
    copy_time->m = 1.0 / ucp_tl_iface_bandwidth(context, &perf_attr.bandwidth);

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
    UCS_STRING_BUFFER_ONSTACK(strb, UCP_PROTO_CONFIG_STR_MAX);

    ucp_proto_config_info_str(req->send.ep->worker, req->send.proto_config,
                              msg_length, &strb);
    ucp_trace_req(req, "%s", ucs_string_buffer_cstr(&strb));
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

    ucp_proto_select_param_str(sel_param, ucp_operation_names, &sel_param_strb);
    ucp_proto_select_info(ep->worker, ep->cfg_index, rkey_cfg_index,
                          proto_select, &proto_select_strb);
    ucs_fatal("req %p on ep %p to %s: could not find a protocol for %s "
              "length %zu\navailable protocols:\n%s\n",
              req, ep, ucp_ep_peer_name(ep),
              ucs_string_buffer_cstr(&sel_param_strb), msg_length,
              ucs_string_buffer_cstr(&proto_select_strb));
}

void ucp_proto_common_zcopy_adjust_min_frag_always(ucp_request_t *req,
                                                   size_t min_frag_diff,
                                                   uct_iov_t *iov,
                                                   size_t iovcnt,
                                                   size_t *offset_p)
{
    if (ucs_likely(*offset_p > 0)) {
        /* Move backward: the first IOV element would send additional
           overlapping data before its start, to reach min_frag length */
        ucs_assert(*offset_p >= min_frag_diff);
        *offset_p -= min_frag_diff;

        ucs_assert(iov[0].count == 1);
        iov[0].buffer  = UCS_PTR_BYTE_OFFSET(iov[0].buffer, -min_frag_diff);
        iov[0].length += min_frag_diff;
    } else {
        /* Move forward: the last IOV element would send additional overlapping
           data after its end, to reach min_frag length */
        ucs_assert(iov[iovcnt - 1].count == 1);
        iov[iovcnt - 1].length += min_frag_diff;
    }
}

void ucp_proto_request_abort(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(UCS_STATUS_IS_ERR(status));
    ucs_debug("abort request %p proto %s status %s", req,
              req->send.proto_config->proto->name, ucs_status_string(status));

    req->send.proto_config->proto->abort(req, status);
}

void ucp_proto_request_bcopy_abort(ucp_request_t *request, ucs_status_t status)
{
    ucp_datatype_iter_cleanup(&request->send.state.dt_iter, UCP_DT_MASK_ALL);
    ucp_request_complete_send(request, status);
}

int ucp_proto_is_short_supported(const ucp_proto_select_param_t *select_param)
{
    /* Short protocol requires contig/host */
    return (select_param->dt_class == UCP_DATATYPE_CONTIG) &&
           UCP_MEM_IS_HOST(select_param->mem_type);
}
