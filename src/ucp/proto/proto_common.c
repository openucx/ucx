/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_debug.h"
#include "proto_common.inl"

#include <ucp/am/ucp_am.inl>
#include <ucp/wireup/wireup.h>
#include <uct/api/v2/uct_v2.h>


ucp_proto_common_init_params_t
ucp_proto_common_init_params(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_common_init_params_t params = {
        .super         = *init_params,
        .latency       = 0,
        .overhead      = 0,
        .cfg_thresh    = UCS_MEMUNITS_AUTO,
        .cfg_priority  = 0,
        .min_length    = 0,
        .max_length    = SIZE_MAX,
        .min_iov       = 0,
        .min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .hdr_size      = 0,
        .send_op       = UCT_EP_OP_LAST,
        .memtype_op    = UCT_EP_OP_LAST,
        .flags         = 0,
        .exclude_map   = 0
    };
    return params;
}

int ucp_proto_common_init_check_err_handling(
        const ucp_proto_common_init_params_t *init_params)
{
    return (init_params->flags & UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING) ||
           (init_params->super.ep_config_key->err_mode ==
            UCP_ERR_HANDLING_MODE_NONE);
}

ucp_rsc_index_t
ucp_proto_common_get_rsc_index(const ucp_proto_init_params_t *params,
                               ucp_lane_index_t lane)
{
    ucs_assert(lane < UCP_MAX_LANES);
    return params->ep_config_key->lanes[lane].rsc_index;
}

static size_t
ucp_proto_common_get_seg_size(const ucp_proto_common_init_params_t *params,
                              ucp_lane_index_t lane)
{
    ucs_assert(lane < UCP_MAX_LANES);
    return params->super.ep_config_key->lanes[lane].seg_size;
}

ucp_memory_info_t ucp_proto_common_select_param_mem_info(
                                   const ucp_proto_select_param_t *select_param)
{
    ucp_memory_info_t mem_info = {
        .type = select_param->mem_type,
        .sys_dev = select_param->sys_dev
    };

    return mem_info;
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

/* Pack/unpack local distance to make it equal to the remote one */
static void
ucp_proto_common_fp8_pack_unpack_distance(ucs_sys_dev_distance_t *distance)
{
    distance->latency   = ucp_wireup_fp8_pack_unpack_latency(distance->latency);
    distance->bandwidth = UCS_FP8_PACK_UNPACK(BANDWIDTH, distance->bandwidth);
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

    ucp_proto_common_fp8_pack_unpack_distance(distance);
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
                               ucp_lane_index_t lane, size_t *min_frag_p,
                               size_t *max_frag_p)
{
    ucp_context_h context = params->super.worker->context;
    *min_frag_p = ucp_proto_common_get_iface_attr_field(iface_attr,
                                                        params->min_frag_offs,
                                                        0);
    *max_frag_p = ucp_proto_common_get_iface_attr_field(iface_attr,
                                                        params->max_frag_offs,
                                                        SIZE_MAX);

    /* Adjust maximum fragment size taking into account segment size to prevent
       sending more than the remote side supports. */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE) {
        *max_frag_p = ucs_min(ucp_proto_common_get_seg_size(params, lane),
                              *max_frag_p);
    }

    /* Force upper bound on fragment size according to user configuration. */
    if (ucs_test_all_flags(params->flags,
                           UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                           UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) &&
        (context->config.ext.rma_zcopy_max_seg_size != UCS_MEMUNITS_AUTO)) {
        *max_frag_p = ucs_min(*max_frag_p,
                              context->config.ext.rma_zcopy_max_seg_size);
    }
}

/* Update 'perf' with the distance */
static void ucp_proto_common_update_lane_perf_by_distance(
        ucp_proto_common_tl_perf_t *perf, ucp_proto_perf_node_t *perf_node,
        const ucs_sys_dev_distance_t *distance, const char *perf_name,
        const char *perf_fmt, ...)
{
    ucp_proto_perf_node_t *sys_perf_node;
    ucs_linear_func_t distance_func;
    char perf_node_desc[128];
    va_list ap;

    distance_func.c = distance->latency;
    distance_func.m = 1.0 / distance->bandwidth;

    if (ucs_linear_func_is_zero(distance_func, UCP_PROTO_PERF_EPSILON)) {
        return;
    }

    perf->bandwidth    = ucs_min(perf->bandwidth, distance->bandwidth);
    perf->sys_latency += distance->latency;

    va_start(ap, perf_fmt);
    ucs_vsnprintf_safe(perf_node_desc, sizeof(perf_node_desc), perf_fmt, ap);
    va_end(ap);

    sys_perf_node = ucp_proto_perf_node_new_data(perf_name, "%s",
                                                 perf_node_desc);
    ucp_proto_perf_node_add_data(sys_perf_node, "", distance_func);
    ucp_proto_perf_node_own_child(perf_node, &sys_perf_node);
}

void ucp_proto_common_lane_perf_node(ucp_context_h context,
                                     ucp_rsc_index_t rsc_index,
                                     const uct_perf_attr_t *perf_attr,
                                     ucp_proto_perf_node_t **perf_node_p)
{
    const uct_tl_resource_desc_t *tl_rsc = &context->tl_rscs[rsc_index].tl_rsc;
    ucp_proto_perf_node_t *perf_node;

    if (perf_attr->operation == UCT_EP_OP_LAST) {
        *perf_node_p = NULL;
        return;
    }

    perf_node = ucp_proto_perf_node_new_data(
            uct_ep_operation_names[perf_attr->operation],
            UCT_TL_RESOURCE_DESC_FMT, UCT_TL_RESOURCE_DESC_ARG(tl_rsc));

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {
        ucp_proto_perf_node_add_bandwidth(perf_node, "bw/proc",
                                          perf_attr->bandwidth.dedicated);
        ucp_proto_perf_node_add_bandwidth(perf_node, "bw/node",
                                          perf_attr->bandwidth.shared);
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_LATENCY) {
        ucp_proto_perf_node_add_scalar(perf_node, "lat", perf_attr->latency.c);
        ucp_proto_perf_node_add_scalar(perf_node, "lat/ep",
                                       perf_attr->latency.m);
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD) {
        ucp_proto_perf_node_add_scalar(perf_node, "send-pre",
                                       perf_attr->send_pre_overhead);
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD) {
        ucp_proto_perf_node_add_scalar(perf_node, "send-post",
                                       perf_attr->send_post_overhead);
    }

    *perf_node_p = perf_node;
}

static void ucp_proto_common_tl_perf_reset(ucp_proto_common_tl_perf_t *tl_perf)
{
    tl_perf->send_pre_overhead  = 0;
    tl_perf->send_post_overhead = 0;
    tl_perf->recv_overhead      = 0;
    tl_perf->bandwidth          = 0;
    tl_perf->latency            = 0;
    tl_perf->sys_latency        = 0;
    tl_perf->min_length         = 0;
    tl_perf->max_frag           = SIZE_MAX;
}

ucs_status_t
ucp_proto_common_get_lane_perf(const ucp_proto_common_init_params_t *params,
                               ucp_lane_index_t lane,
                               ucp_proto_common_tl_perf_t *tl_perf,
                               ucp_proto_perf_node_t **perf_node_p)
{
    ucp_worker_h worker        = params->super.worker;
    ucp_context_h context      = worker->context;
    ucp_rsc_index_t rsc_index  = ucp_proto_common_get_rsc_index(&params->super,
                                                                lane);
    ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);
    ucp_proto_perf_node_t *perf_node, *lane_perf_node;
    const ucp_rkey_config_t *rkey_config;
    ucs_sys_dev_distance_t distance;
    size_t tl_min_frag, tl_max_frag;
    uct_perf_attr_t perf_attr;
    ucs_sys_device_t sys_dev;
    ucs_status_t status;
    char bdf_name[32];

    if (lane == UCP_NULL_LANE) {
        ucp_proto_common_tl_perf_reset(tl_perf);
        *perf_node_p = NULL;
        return UCS_OK;
    }

    ucp_proto_common_get_frag_size(params, &wiface->attr, lane, &tl_min_frag,
                                   &tl_max_frag);

    perf_node = ucp_proto_perf_node_new_data("lane", "%u ppn %u eps",
                                             context->config.est_num_ppn,
                                             context->config.est_num_eps);

    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                           UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_RECV_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_BANDWIDTH |
                           UCT_PERF_ATTR_FIELD_LATENCY;
    perf_attr.operation  = params->send_op;

    status = ucp_worker_iface_estimate_perf(wiface, &perf_attr);
    if (status != UCS_OK) {
        goto err_deref_perf_node;
    }

    tl_perf->send_pre_overhead  = perf_attr.send_pre_overhead + params->overhead;
    tl_perf->send_post_overhead = perf_attr.send_post_overhead;
    tl_perf->recv_overhead      = perf_attr.recv_overhead + params->overhead;
    tl_perf->bandwidth          = ucp_tl_iface_bandwidth(context,
                                                         &perf_attr.bandwidth);
    tl_perf->latency            = ucp_tl_iface_latency(context,
                                                       &perf_attr.latency) +
                                  params->latency;
    tl_perf->sys_latency        = 0;
    tl_perf->min_length         = ucs_max(params->min_length, tl_min_frag);
    tl_perf->max_frag           = tl_max_frag;

    ucp_proto_common_lane_perf_node(context, rsc_index, &perf_attr,
                                    &lane_perf_node);
    ucp_proto_perf_node_own_child(perf_node, &lane_perf_node);

    /* If reg_mem_info type is not unknown we assume the protocol is going to
     * send that mem type in a zero copy fashion. So, need to consider the
     * system device distance. */
    if (params->reg_mem_info.type != UCS_MEMORY_TYPE_UNKNOWN) {
        sys_dev = params->reg_mem_info.sys_dev;

        ucs_assertv((sys_dev == params->super.select_param->sys_dev) ||
                    !(params->flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY),
                    "flags=0x%x sys_dev=%u select_param->sys_dev=%u",
                    params->flags, sys_dev,
                    params->super.select_param->sys_dev);

        ucp_proto_common_get_lane_distance(&params->super, lane, sys_dev,
                                           &distance);
        ucp_proto_common_update_lane_perf_by_distance(
                tl_perf, perf_node, &distance, "local system", "%s %s",
                ucs_topo_sys_device_get_name(sys_dev),
                ucs_topo_sys_device_bdf_name(sys_dev, bdf_name,
                                             sizeof(bdf_name)));
    }

    /* For remote memory access, consider remote system topology distance */
    if (params->flags & UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS) {
        ucs_assertv(params->super.rkey_cfg_index < worker->rkey_config_count,
                    "rkey_cfg_index=%d rkey_config_count=%d",
                    params->super.rkey_cfg_index, worker->rkey_config_count);
        rkey_config = &worker->rkey_config[params->super.rkey_cfg_index];
        distance    = rkey_config->lanes_distance[lane];
        ucp_proto_common_update_lane_perf_by_distance(
                tl_perf, perf_node, &distance, "remote system", "sys-dev %d %s",
                rkey_config->key.sys_dev,
                ucs_memory_type_names[rkey_config->key.mem_type]);
    }

    ucs_assert(tl_perf->bandwidth > 1.0);
    ucs_assert(tl_perf->send_pre_overhead >= 0);
    ucs_assert(tl_perf->send_post_overhead >= 0);
    ucs_assert(tl_perf->recv_overhead >= 0);
    ucs_assertv(tl_perf->max_frag >= params->hdr_size,
                "max_frag=%zu hdr_size=%zu", tl_perf->max_frag,
                params->hdr_size);
    ucs_assert(tl_perf->sys_latency >= 0);

    ucp_proto_perf_node_add_bandwidth(perf_node, "bw", tl_perf->bandwidth);
    ucp_proto_perf_node_add_scalar(perf_node, "lat", tl_perf->latency);
    ucp_proto_perf_node_add_scalar(perf_node, "sys-lat", tl_perf->sys_latency);
    ucp_proto_perf_node_add_scalar(perf_node, "send-pre",
                                   tl_perf->send_pre_overhead);
    ucp_proto_perf_node_add_scalar(perf_node, "send-post",
                                   tl_perf->send_post_overhead);
    ucp_proto_perf_node_add_scalar(perf_node, "recv", tl_perf->recv_overhead);

    *perf_node_p = perf_node;
    return UCS_OK;

err_deref_perf_node:
    ucp_proto_perf_node_deref(&perf_node);
    return status;
}

ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_init_params_t *params,
                            uct_ep_operation_t memtype_op, unsigned flags,
                            ptrdiff_t max_iov_offs, size_t min_iov,
                            ucp_lane_type_t lane_type,
                            ucs_memory_type_t reg_mem_type,
                            uint64_t tl_cap_flags, ucp_lane_index_t max_lanes,
                            ucp_lane_map_t exclude_map, ucp_lane_index_t *lanes)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    ucp_context_h context                        = params->worker->context;
    const ucp_ep_config_key_t *ep_config_key     = params->ep_config_key;
    const ucp_rkey_config_key_t *rkey_config_key = params->rkey_config_key;
    const ucp_proto_select_param_t *select_param = params->select_param;
    const uct_iface_attr_t *iface_attr;
    ucp_lane_index_t lane, num_lanes;
    const uct_md_attr_v2_t *md_attr;
    const uct_component_attr_t *cmpt_attr;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t md_index;
    ucp_lane_map_t lane_map;
    char lane_desc[64];
    size_t max_iov;

    if (max_lanes == 0) {
        return 0;
    }

    ucp_proto_select_info_str(params->worker, params->rkey_cfg_index,
                              select_param, ucp_operation_names,
                              &sel_param_strb);

    num_lanes = 0;
    ucs_trace("selecting up to %d/%d lanes for %s %s", max_lanes,
              ep_config_key->num_lanes,
              ucp_proto_id_field(params->proto_id, name),
              ucs_string_buffer_cstr(&sel_param_strb));
    ucs_log_indent(1);

    if ((flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY) &&
        (select_param->dt_class == UCP_DATATYPE_GENERIC)) {
        /* Generic/IOV datatype cannot be used with zero-copy send */
        ucs_trace("datatype %s cannot be used with zcopy",
                  ucp_datatype_class_names[select_param->dt_class]);
        goto out;
    }

    lane_map = UCS_MASK(ep_config_key->num_lanes) & ~exclude_map;
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
        if ((lane_type != UCP_LANE_TYPE_LAST) &&
            !(ep_config_key->lanes[lane].lane_types & UCS_BIT(lane_type))) {
            ucs_trace("%s: no %s in lane types", lane_desc,
                      ucp_lane_type_info[lane_type].short_name);
            continue;
        }

        /* Check iface capabilities */
        iface_attr = ucp_proto_common_get_iface_attr(params, lane);
        if (!ucs_test_all_flags(iface_attr->cap.flags, tl_cap_flags)) {
            ucs_trace("%s: no cap 0x%" PRIx64, lane_desc, tl_cap_flags);
            continue;
        }

        md_index  = context->tl_rscs[rsc_index].md_index;
        md_attr   = &context->tl_mds[md_index].attr;
        cmpt_attr = ucp_cmpt_attr_by_md_index(context, md_index);

        if ((flags & UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR) &&
            !(cmpt_attr->flags & UCT_COMPONENT_FLAG_RKEY_PTR)) {
            ucs_trace("protocol requires rkey ptr but it is not "
                      "supported by the component");
            continue;
        }

        /* Check memory registration capabilities for zero-copy case */
        if (reg_mem_type != UCS_MEMORY_TYPE_UNKNOWN) {
            ucs_assertv((reg_mem_type == select_param->mem_type) ||
                        !(flags & UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY),
                        "flags=0x%x reg_mem_type=%s select_param->mem_type=%s",
                        flags, ucs_memory_type_names[reg_mem_type],
                        ucs_memory_type_names[select_param->mem_type]);

            if (md_attr->flags & UCT_MD_FLAG_NEED_MEMH) {
                /* Memory domain must support registration on the relevant
                 * memory type */
                if (!(context->reg_md_map[reg_mem_type] & UCS_BIT(md_index))) {
                    ucs_trace("%s: md %s cannot register %s memory", lane_desc,
                              context->tl_mds[md_index].rsc.md_name,
                              ucs_memory_type_names[reg_mem_type]);
                    continue;
                }
            } else if (!(md_attr->access_mem_types & UCS_BIT(reg_mem_type))) {
                /*
                 * Memory domain which does not require a registration for zero
                 * copy operation must be able to access the relevant memory type
                 */
                ucs_trace("%s: no access to mem type %s", lane_desc,
                          ucs_memory_type_names[reg_mem_type]);
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

            if (((md_attr->flags & UCT_MD_FLAG_NEED_RKEY) ||
                 (flags & UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR)) &&
                !(rkey_config_key->md_map &
                  UCS_BIT(ep_config_key->lanes[lane].dst_md_index))) {
                /* If remote key required remote memory domain should be
                 * available */
                ucs_trace("%s: no support of dst md map 0x%" PRIx64,
                          lane_desc, rkey_config_key->md_map);
                continue;
            }

            if (!(md_attr->flags & UCT_MD_FLAG_NEED_RKEY) &&
                !(md_attr->access_mem_types &
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
    const uct_md_attr_v2_t *md_attr;
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
        if ((md_attr->flags & UCT_MD_FLAG_NEED_MEMH) &&
            (context->reg_md_map[select_param->mem_type] & UCS_BIT(md_index))) {
            reg_md_map |= UCS_BIT(md_index);
        }
    }

    return reg_md_map;
}

ucp_lane_index_t ucp_proto_common_find_lanes_with_min_frag(
        const ucp_proto_common_init_params_t *params, ucp_lane_type_t lane_type,
        uint64_t tl_cap_flags, ucp_lane_index_t max_lanes,
        ucp_lane_map_t exclude_map, ucp_lane_index_t *lanes)
{
    ucp_lane_index_t lane_index, lane, num_lanes, num_valid_lanes;
    const uct_iface_attr_t *iface_attr;
    size_t tl_min_frag, tl_max_frag;

    num_lanes = ucp_proto_common_find_lanes(
                   &params->super, params->memtype_op, params->flags,
                   params->max_iov_offs, params->min_iov, lane_type,
                   params->reg_mem_info.type, tl_cap_flags, max_lanes,
                   exclude_map, lanes);

    num_valid_lanes = 0;
    for (lane_index = 0; lane_index < num_lanes; ++lane_index) {
        lane       = lanes[lane_index];
        iface_attr = ucp_proto_common_get_iface_attr(&params->super, lane);

        ucp_proto_common_get_frag_size(params, iface_attr, lane, &tl_min_frag,
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

void ucp_proto_request_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    /* request should NOT be on pending queue because when we decrement the last
     * refcount the request is not on the pending queue any more
     */
    ucp_proto_request_zcopy_complete(req, req->send.state.uct_comp.status);
}

int ucp_proto_is_short_supported(const ucp_proto_select_param_t *select_param)
{
    return (select_param->dt_class == UCP_DATATYPE_CONTIG);
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
                          proto_select, 1, &proto_select_strb);
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

ucs_status_t
ucp_proto_request_init(ucp_request_t *req,
                       const ucp_proto_select_param_t *select_param)
{
    ucp_ep_h ep         = req->send.ep;
    ucp_worker_h worker = ep->worker;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_t *proto_select;
    size_t msg_length;

    proto_select = ucp_proto_select_get(worker, ep->cfg_index,
                                        req->send.proto_config->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return UCS_OK;
    }

    msg_length = req->send.state.dt_iter.length;
    if (ucp_proto_config_is_am(req->send.proto_config)) {
        msg_length += req->send.msg_proto.am.header.length;
    }

    /* Select from protocol hash according to saved request parameters */
    return ucp_proto_request_lookup_proto(worker, ep, req, proto_select,
                                          rkey_cfg_index, select_param,
                                          msg_length);
}

void ucp_proto_request_restart(ucp_request_t *req)
{
    const ucp_proto_config_t *proto_config = req->send.proto_config;
    ucp_proto_select_param_t select_param  = proto_config->select_param;
    ucs_status_t status;

    ucp_trace_req(req, "proto %s at stage %d restarting",
                  proto_config->proto->name, req->send.proto_stage);

    status = proto_config->proto->reset(req);
    if (status != UCS_OK) {
        ucs_assert_always(status == UCS_ERR_CANCELED);
        return;
    }

    /* Select a protocol with resume request support */
    if (!ucp_datatype_iter_is_begin(&req->send.state.dt_iter)) {
        select_param.op_id_flags |= UCP_PROTO_SELECT_OP_FLAG_RESUME;
    }

    status = ucp_proto_request_init(req, &select_param);
    if (status == UCS_OK) {
        ucp_request_send(req);
    } else {
        ucp_proto_request_abort(req, status);
    }
}

void ucp_proto_request_abort(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(UCS_STATUS_IS_ERR(status));
    ucs_debug("abort request %p proto %s status %s", req,
              req->send.proto_config->proto->name, ucs_status_string(status));

    req->send.proto_config->proto->abort(req, status);
}

void ucp_proto_request_bcopy_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0, UCP_DT_MASK_ALL);
    ucp_request_complete_send(req, status);
}

void ucp_proto_request_bcopy_id_abort(ucp_request_t *req,
                                      ucs_status_t status)
{
    ucp_send_request_id_release(req);
    ucp_proto_request_bcopy_abort(req, status);
}

ucs_status_t ucp_proto_request_bcopy_reset(ucp_request_t *req)
{
    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    return UCS_OK;
}

ucs_status_t ucp_proto_request_bcopy_id_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucp_send_request_id_release(req);
        return ucp_proto_request_bcopy_reset(req);
    }

    return UCS_OK;
}

void ucp_proto_request_zcopy_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
}

ucs_status_t ucp_proto_request_zcopy_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucp_proto_request_zcopy_clean(req, UCP_DT_MASK_ALL);
    }

    return UCS_OK;
}

ucs_status_t ucp_proto_request_zcopy_id_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucp_send_request_id_release(req);
        ucp_proto_request_zcopy_clean(req, UCP_DT_MASK_ALL);
    }

    return UCS_OK;
}

static void ucp_proto_stub_fatal_not_implemented(const char *func_name,
                                                 ucp_request_t *req)
{
    ucs_fatal("'%s' is not implemented for protocol %s (req: %p)", func_name,
              req->send.proto_config->proto->name, req);
}

void ucp_proto_abort_fatal_not_implemented(ucp_request_t *req,
                                           ucs_status_t status)
{
    ucp_proto_stub_fatal_not_implemented("abort", req);
}

void ucp_proto_reset_fatal_not_implemented(ucp_request_t *req)
{
    ucp_proto_stub_fatal_not_implemented("reset", req);
}

void ucp_proto_fatal_invalid_stage(ucp_request_t *req, const char *func_name)
{
    ucs_fatal("req %p: proto %s is in invalid stage %d on %s", req,
              req->send.proto_config->proto->name, req->send.proto_stage,
              func_name);
}
