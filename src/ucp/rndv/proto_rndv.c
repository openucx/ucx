/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"
#include "rndv_mtype.inl"

#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_common.inl>


static void
ucp_proto_rndv_ctrl_get_md_map(const ucp_proto_rndv_ctrl_init_params_t *params,
                               ucp_proto_rndv_ctrl_priv_t *rpriv)
{
    ucp_context_h context                    = params->super.super.worker->context;
    const ucp_ep_config_key_t *ep_config_key = params->super.super.ep_config_key;
    ucs_sys_dev_distance_t *sys_distance     = rpriv->sys_dev_distance;
    ucp_rsc_index_t mem_sys_dev, ep_sys_dev;
    const uct_iface_attr_t *iface_attr;
    const uct_md_attr_v2_t *md_attr;
    const uct_component_attr_t *cmpt_attr;
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;
    ucs_status_t status;

    /* md_map is all lanes which support get_zcopy on the given mem_type and
     * require remote key
     */
    rpriv->md_map      = 0;
    rpriv->sys_dev_map = 0;

    if (params->super.super.select_param->dt_class != UCP_DATATYPE_CONTIG) {
        return;
    }

    for (lane = 0; lane < ep_config_key->num_lanes; ++lane) {
        if (ep_config_key->lanes[lane].rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        iface_attr = ucp_proto_common_get_iface_attr(&params->super.super,
                                                     lane);
        ep_sys_dev = ucp_proto_common_get_sys_dev(&params->super.super, lane);
        md_index   = ucp_proto_common_get_md_index(&params->super.super, lane);

        ucs_assertv(md_index < UCP_MAX_MDS, "md_index=%u", md_index);

        cmpt_attr  = ucp_cmpt_attr_by_md_index(context, md_index);
        md_attr    = &context->tl_mds[md_index].attr;

        /* Check the lane supports get_zcopy or rkey_ptr */
        if (!(cmpt_attr->flags & UCT_COMPONENT_FLAG_RKEY_PTR) &&
            !(iface_attr->cap.flags &
              (UCT_IFACE_FLAG_GET_ZCOPY | UCT_IFACE_FLAG_PUT_ZCOPY))) {
            continue;
        }

        if (!(md_attr->flags & UCT_MD_FLAG_NEED_RKEY)) {
            continue;
        }

        /* Check that memory domain is requested by the protocol or
         * it is capable of registering the memory type
         */
        if (!(params->md_map & UCS_BIT(md_index)) &&
            !(context->reg_md_map[params->mem_info.type] & UCS_BIT(md_index))) {
            continue;
        }

        ucs_trace_req("lane[%d]: selected md %s index %u", lane,
                      context->tl_mds[md_index].rsc.md_name, md_index);
        rpriv->md_map |= UCS_BIT(md_index);

        if (ep_sys_dev >= UCP_MAX_SYS_DEVICES) {
            continue;
        }

        rpriv->sys_dev_map |= UCS_BIT(ep_sys_dev);
    }

    mem_sys_dev = params->super.super.select_param->sys_dev;
    ucs_for_each_bit(ep_sys_dev, rpriv->sys_dev_map) {
        status = ucs_topo_get_distance(mem_sys_dev, ep_sys_dev, sys_distance);
        ucs_assertv_always(status == UCS_OK, "mem_info->sys_dev=%d sys_dev=%d",
                           mem_sys_dev, ep_sys_dev);
        ++sys_distance;
    }
}

static ucp_md_map_t
ucp_proto_rndv_md_map_to_remote(const ucp_proto_rndv_ctrl_init_params_t *params,
                                ucp_md_map_t md_map)
{
    ucp_worker_h worker   = params->super.super.worker;
    ucp_context_h context = worker->context;
    const ucp_ep_config_key_lane_t *lane_cfg;
    const ucp_ep_config_t *ep_config;
    uint64_t remote_md_map;

    ep_config     = &ucs_array_elem(&worker->ep_config,
                                    params->super.super.ep_cfg_index);
    remote_md_map = 0;

    ucs_carray_for_each(lane_cfg, ep_config->key.lanes,
                        ep_config->key.num_lanes) {
        if (lane_cfg->rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        if (md_map & UCS_BIT(context->tl_rscs[lane_cfg->rsc_index].md_index)) {
            remote_md_map |= UCS_BIT(lane_cfg->dst_md_index);
        }
    }

    return remote_md_map;
}

/*
 * Init protocols that can be used by the remote peer
 * (assuming peer has the same configuration)
 */
static ucs_status_t ucp_proto_rndv_ctrl_select_remote_proto(
        const ucp_proto_rndv_ctrl_init_params_t *params,
        const ucp_proto_select_param_t *remote_select_param,
        ucp_md_map_t md_map, ucp_proto_select_elem_t **remote_proto)
{
    ucp_worker_h worker                 = params->super.super.worker;
    ucp_worker_cfg_index_t ep_cfg_index = params->super.super.ep_cfg_index;
    const ucp_ep_config_t *ep_config    = &ucs_array_elem(&worker->ep_config,
                                                          ep_cfg_index);
    ucs_sys_dev_distance_t lanes_distance[UCP_MAX_LANES];
    ucp_rkey_config_key_t rkey_config_key;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_rkey_config_t *rkey_config;
    ucs_status_t status;
    ucp_lane_index_t lane;

    /* Construct remote key for remote protocol lookup according to the local
     * buffer properties (since remote side is expected to access the local
     * buffer)
     */
    rkey_config_key.md_map       = ucp_proto_rndv_md_map_to_remote(params,
                                                                   md_map);
    rkey_config_key.ep_cfg_index = ep_cfg_index;
    rkey_config_key.sys_dev      = params->mem_info.sys_dev;
    rkey_config_key.mem_type     = params->mem_info.type;

    rkey_config_key.unreachable_md_map = 0;

    for (lane = 0; lane < ep_config->key.num_lanes; ++lane) {
        ucp_proto_common_get_lane_distance(&params->super.super, lane,
                                           params->mem_info.sys_dev,
                                           &lanes_distance[lane]);
    }

    status = ucp_worker_rkey_config_get(worker, &rkey_config_key,
                                        lanes_distance, &rkey_cfg_index);
    if (status != UCS_OK) {
        return status;
    }

    ucs_trace("rndv select remote protocols rkey_config->md_map=0x%" PRIx64,
              rkey_config_key.md_map);

    rkey_config = &worker->rkey_config[rkey_cfg_index];
    *remote_proto = ucp_proto_select_lookup_slow(worker,
                                                 &rkey_config->proto_select, 1,
                                                 ep_cfg_index, rkey_cfg_index,
                                                 remote_select_param);
    if (*remote_proto == NULL) {
        ucs_debug("%s: did not find protocol for %s",
                  ucp_proto_id_field(params->super.super.proto_id, name),
                  ucp_operation_names[params->remote_op_id]);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_ctrl_perf(const ucp_proto_init_params_t *params,
                         ucp_lane_index_t lane, double *send_time,
                         double *receive_time)
{
    ucp_context_t *context = params->worker->context;
    ucp_worker_iface_t *wiface;
    uct_perf_attr_t perf_attr;
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;

    if (lane == UCP_NULL_LANE) {
        *send_time = *receive_time = 0;
        return UCS_OK;
    }

    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                           UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_RECV_OVERHEAD |
                           UCT_PERF_ATTR_FIELD_LATENCY;
    perf_attr.operation  = UCT_EP_OP_AM_BCOPY;

    rsc_index = params->ep_config_key->lanes[lane].rsc_index;
    wiface    = ucp_worker_iface(params->worker, rsc_index);
    status    = ucp_worker_iface_estimate_perf(wiface, &perf_attr);
    if (status != UCS_OK) {
        return status;
    }

    *send_time    = perf_attr.send_pre_overhead + perf_attr.send_post_overhead;
    *receive_time = perf_attr.recv_overhead +
                    ucp_tl_iface_latency(context, &perf_attr.latency);

    return UCS_OK;
}

static ucp_proto_select_param_t ucp_proto_rndv_remote_select_param_init(
        const ucp_proto_rndv_ctrl_init_params_t *params)
{
    const ucp_proto_init_params_t *init_params   = &params->super.super;
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    ucp_proto_select_param_t remote_select_param;
    ucp_memory_info_t mem_info;
    uint32_t op_attr_mask;

    op_attr_mask = ucp_proto_select_op_attr_unpack(select_param->op_attr) &
                   UCP_OP_ATTR_FLAG_MULTI_SEND;
    /* Construct select parameter for the remote protocol */
    if (init_params->rkey_config_key == NULL) {
        /* Remote buffer is unknown, assume same params as local */
        mem_info.type    = select_param->mem_type;
        mem_info.sys_dev = select_param->sys_dev;
        ucp_proto_select_param_init(&remote_select_param, params->remote_op_id,
                                    op_attr_mask, 0, select_param->dt_class,
                                    &mem_info, select_param->sg_count);
    } else {
        /* If we know the remote buffer parameters, these are actually the local
         * parameters for the remote protocol
         */
        mem_info.sys_dev = init_params->rkey_config_key->sys_dev;
        mem_info.type    = init_params->rkey_config_key->mem_type;
        ucp_proto_select_param_init(&remote_select_param, params->remote_op_id,
                                    op_attr_mask, 0, UCP_DATATYPE_CONTIG,
                                    &mem_info, 1);
    }

    return remote_select_param;
}

static void
ucp_proto_rndv_ctrl_init_priv(const ucp_proto_rndv_ctrl_init_params_t *params,
                              ucp_proto_rndv_ctrl_priv_t *rpriv,
                              ucp_lane_index_t lane)
{
    const ucp_proto_init_params_t *init_params = &params->super.super;

    /* Initialize estimated memory registration map */
    ucp_proto_rndv_ctrl_get_md_map(params, rpriv);

    /* Use only memory domains for which the unpacking of the remote key was
     * successful */
    if (init_params->rkey_config_key != NULL) {
        rpriv->md_map &= ~init_params->rkey_config_key->unreachable_md_map;
    }

    rpriv->lane             = lane;
    rpriv->packed_rkey_size = ucp_rkey_packed_size(
            init_params->worker->context, rpriv->md_map,
            init_params->select_param->sys_dev, rpriv->sys_dev_map);
}

static ucs_status_t
ucp_proto_rndv_ctrl_range_init(const ucp_proto_rndv_ctrl_init_params_t *params,
                               const ucp_proto_rndv_ctrl_priv_t *priv,
                               ucp_proto_perf_range_t *ctrl_perf)
{
    double send_time, receive_time, ctrl_latency;
    ucp_proto_perf_node_t *memreg_perf_node;
    ucs_linear_func_t memreg_time;
    ucs_status_t status;

    ctrl_perf->node = ucp_proto_perf_node_new_data(params->ctrl_msg_name, "");
    ucp_proto_perf_node_add_child(ctrl_perf->node, params->unpack_perf_node);

    /* Set send_overheads to the time to send and receive RTS message */
    status = ucp_proto_rndv_ctrl_perf(&params->super.super, priv->lane,
                                      &send_time, &receive_time);
    if (status != UCS_OK) {
        return status;
    }

    ucp_proto_init_memreg_time(&params->super, priv->md_map, &memreg_time,
                               &memreg_perf_node);
    ucp_proto_perf_node_own_child(ctrl_perf->node, &memreg_perf_node);

    ctrl_latency = send_time + receive_time + params->super.overhead * 2;
    ucs_trace("rndv" UCP_PROTO_TIME_FMT(ctrl_latency),
              UCP_PROTO_TIME_ARG(ctrl_latency));
    ctrl_perf->perf[UCP_PROTO_PERF_TYPE_SINGLE] =
    ctrl_perf->perf[UCP_PROTO_PERF_TYPE_MULTI]  =
    ctrl_perf->perf[UCP_PROTO_PERF_TYPE_CPU]    = ucs_linear_func_add3(
            memreg_time, ucs_linear_func_make(ctrl_latency, 0.0),
            params->unpack_time);
    ucp_proto_perf_range_add_data(ctrl_perf);

    return UCS_OK;
}

/* Copy performance ranges from the remote protocol add CTRL overheads */
static ucs_status_t
ucp_proto_rndv_ctrl_init_parallel_stages(ucp_proto_perf_range_t *ctrl_ranges,
                                         unsigned num_ctrl_msgs,
                                         ucp_proto_caps_t *proto_caps,
                                         double bias,
                                         ucp_proto_caps_t *output_caps,
                                         size_t min_length, size_t max_length)
{
    const ucp_proto_perf_range_t *parallel_stages[3];
    UCS_STRING_BUFFER_ONSTACK(ctrl_msgs_buf, 256);
    UCS_STRING_BUFFER_ONSTACK(agg_buf, 256);
    ucp_proto_perf_range_t *range;
    const char *ctrl_name;
    size_t range_max_length;
    ucs_status_t status;
    size_t i;

    ucs_assertv(num_ctrl_msgs <= 2, "num_ctrl_msgs=%u", num_ctrl_msgs);
    for (i = 0; i < num_ctrl_msgs; ++i) {
        ctrl_name          = ucp_proto_perf_node_name(ctrl_ranges[i].node);
        parallel_stages[i] = &ctrl_ranges[i];
        ucs_string_buffer_appendf(
                &ctrl_msgs_buf, "%s"UCP_PROTO_PERF_NODE_NEW_LINE, ctrl_name);

        ucs_trace("aggregate proto caps with %s" UCP_PROTO_PERF_FUNC_TYPES_FMT,
                  ctrl_name, UCP_PROTO_PERF_FUNC_TYPES_ARG(ctrl_ranges[i].perf));
    }

    ucs_carray_for_each(range, proto_caps->ranges, proto_caps->num_ranges) {
        range_max_length = ucs_min(range->max_length, max_length);
        if (range_max_length < min_length) {
            continue;
        }

        ucs_trace("%s range max %zu" UCP_PROTO_PERF_FUNC_TYPES_FMT,
                  ucp_proto_perf_node_name(range->node), range->max_length,
                  UCP_PROTO_PERF_FUNC_TYPES_ARG(range->perf));

        ucs_string_buffer_reset(&agg_buf);
        ucs_string_buffer_appendf(&agg_buf, "%s%s",
                                  ucs_string_buffer_cstr(&ctrl_msgs_buf),
                                  ucp_proto_perf_node_name(range->node));

        parallel_stages[num_ctrl_msgs] = range;
        status                         = ucp_proto_init_parallel_stages(
                ucs_string_buffer_cstr(&agg_buf), min_length, range_max_length,
                bias, parallel_stages, num_ctrl_msgs + 1, output_caps);
        if (status != UCS_OK) {
            return status;
        }

        if (range_max_length == max_length) {
            break;
        }
        min_length = range_max_length + 1;
    };

    return UCS_OK;
}

void
ucp_proto_rndv_set_variant_config(const ucp_proto_init_params_t *init_params,
                                  const ucp_proto_init_elem_t *proto,
                                  const ucp_proto_select_param_t *select_param,
                                  void *priv, ucp_proto_config_t *proto_config)
{
    proto_config->proto          = ucp_protocols[proto->proto_id];
    proto_config->priv           = priv;
    proto_config->ep_cfg_index   = init_params->ep_cfg_index;
    proto_config->rkey_cfg_index = init_params->rkey_cfg_index;
    proto_config->select_param   = *select_param;
}

void ucp_proto_rndv_ctrl_probe(const ucp_proto_rndv_ctrl_init_params_t *params,
                               void *priv, size_t priv_size)
{
    const ucp_proto_init_params_t *init_params = &params->super.super;
    ucp_proto_rndv_ctrl_priv_t *rpriv          = priv;
    const ucp_proto_select_init_protocols_t *remote_proto_init;
    ucp_proto_select_param_t remote_select_param;
    ucp_proto_select_elem_t *remote_proto_select;
    ucp_proto_init_elem_t *remote_proto;
    ucp_proto_caps_t result_caps, *caps;
    ucp_proto_perf_range_t ctrl_perf;
    ucs_status_t status;
    size_t min_length, max_length, cfg_thresh, cfg_priority;
    void *remote_priv;
    const char *proto_name;

    ucs_assert(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE);
    ucs_assert(!(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG));

    if (!ucp_proto_common_init_check_err_handling(&params->super)) {
        return;
    }

    /* Initialize 'rpriv' structure */
    ucp_proto_rndv_ctrl_init_priv(params, rpriv, params->lane);

    /* Init remote proto */
    remote_select_param = ucp_proto_rndv_remote_select_param_init(params);
    status              = ucp_proto_rndv_ctrl_select_remote_proto(
            params, &remote_select_param, rpriv->md_map, &remote_proto_select);
    if (status != UCS_OK) {
        return;
    }
    remote_proto_init = &remote_proto_select->proto_init;

    /* Add variants for each remote proto */
    ucs_trace("add variants for %s: min_length=%zu max_length=%zu",
              ucp_proto_id_field(init_params->proto_id, name),
              params->super.min_length, params->super.max_length);
    ucs_log_indent(+1);
    ucs_array_for_each(remote_proto, &remote_proto_init->protocols) {
        if (ucp_proto_id_field(remote_proto->proto_id, flags) &
            UCP_PROTO_FLAG_INVALID) {
            continue;
        }

        ucs_assert(remote_proto->caps.num_ranges > 0);

        proto_name = ucp_proto_id_field(remote_proto->proto_id, name);
        caps       = &remote_proto->caps;
        max_length = caps->ranges[caps->num_ranges - 1].max_length;
        min_length = caps->min_length;
        ucs_trace("%s proto variant: min_length=%zu max_length=%zu "
                  "cfg_thresh=%zu cfg_priority=%u", proto_name, 
                  min_length, max_length, remote_proto->cfg_thresh,
                  remote_proto->cfg_priority);

        max_length = ucs_min(max_length, params->super.max_length);
        min_length = ucs_max(min_length, params->super.min_length);
        if (max_length < min_length) {
            ucs_trace("no available ranges for %s", proto_name);
            continue;
        }

        remote_priv = &ucs_array_elem(&remote_proto_init->priv_buf,
                                      remote_proto->priv_offset);
        ucp_proto_rndv_set_variant_config(init_params, remote_proto,
                                          &remote_select_param, remote_priv,
                                          &rpriv->remote_proto_config);

        /* Create separate CTRL message perf node for each variant */
        status = ucp_proto_rndv_ctrl_range_init(params, rpriv, &ctrl_perf);
        if (status != UCS_OK) {
            ucs_trace("failed to init perf data for %s", proto_name);
            continue;
        }

        ucp_proto_common_init_base_caps(&params->super, &result_caps,
                                        min_length);

        status = ucp_proto_rndv_ctrl_init_parallel_stages(&ctrl_perf, 1, caps,
                                                          params->perf_bias,
                                                          &result_caps,
                                                          min_length,
                                                          max_length);
        ucp_proto_perf_node_deref(&ctrl_perf.node);
        if (status != UCS_OK) {
            ucs_trace("failed to calculate parallel stages for %s", proto_name);
            continue;
        }

        cfg_thresh   = params->super.cfg_thresh;
        cfg_priority = params->super.cfg_priority;
        if ((remote_proto->cfg_thresh != UCS_MEMUNITS_INF) &&
            (remote_proto->cfg_thresh != UCS_MEMUNITS_AUTO)) {
            /* Consider remote priority and threshold only if RNDV_SCHEME or
             * BCOPY/ZCOPY thresh are set to force these settings */
            cfg_priority = remote_proto->cfg_priority;
            cfg_thresh   = (cfg_thresh == UCS_MEMUNITS_AUTO) ?
                           remote_proto->cfg_thresh :
                           ucs_max(cfg_thresh, remote_proto->cfg_thresh);
        }

        /* Remote variants priorities are used to respect RNDV_SCHEME setting
         * so they should contain value greater than RTS `cfg_thresh`.
         * Equality is allowed for RTR remote variants.
         */
        ucs_assertv(params->super.cfg_priority <= remote_proto->cfg_priority,
                    "remote_proto=%s params->super.cfg_priority=%u "
                    "remote_proto->cfg_priority=%u", proto_name,
                    params->super.cfg_priority, remote_proto->cfg_priority);

        status = ucp_proto_rndv_predict_prev_stages(init_params, &result_caps);
        if (status != UCS_OK) {
            continue;
        }

        ucp_proto_select_add_proto(init_params, cfg_thresh, cfg_priority,
                                   &result_caps, rpriv, priv_size);
    }
    ucs_log_indent(-1);
}

size_t ucp_proto_rndv_thresh(const ucp_proto_init_params_t *init_params)
{
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    const ucp_context_config_t *cfg = &init_params->worker->context->config.ext;
    size_t rndv_thresh;

    /* Explicit configuration for rendezvous threshold (either fine-grained or
     * coarse-grained) has precedence over rndv_send_nbr_thresh
     */
    if (ucp_ep_config_is_inter_node(init_params->ep_config_key)) {
        rndv_thresh = cfg->rndv_inter_thresh;
    } else {
        rndv_thresh = cfg->rndv_intra_thresh;
    }

    if ((rndv_thresh == UCS_MEMUNITS_AUTO) &&
        (ucp_proto_select_op_attr_unpack(select_param->op_attr) &
         UCP_OP_ATTR_FLAG_FAST_CMPL) &&
        ucs_likely(UCP_MEM_IS_HOST(select_param->mem_type))) {
        rndv_thresh = cfg->rndv_send_nbr_thresh;
    }

    return rndv_thresh;
}

ucp_lane_index_t
ucp_proto_rndv_find_ctrl_lane(const ucp_proto_init_params_t *params)
{
    ucp_lane_index_t lane, num_lanes;

    num_lanes = ucp_proto_common_find_lanes(params, UCT_EP_OP_LAST,
                                            UCP_PROTO_COMMON_INIT_FLAG_HDR_ONLY,
                                            UCP_PROTO_COMMON_OFFSET_INVALID, 1,
                                            UCP_LANE_TYPE_AM,
                                            UCT_IFACE_FLAG_AM_BCOPY, 1, 0,
                                            &lane);
    if (num_lanes == 0) {
        ucs_debug("no active message lane for %s",
                  ucp_proto_id_field(params->proto_id, name));
        return UCP_NULL_LANE;
    }

    ucs_assertv(num_lanes == 1, "proto=%s num_lanes=%u",
                ucp_proto_id_field(params->proto_id, name), num_lanes);
    return lane;
}

static void
ucp_proto_rndv_rtr_common_params_init(
        const ucp_proto_init_params_t *init_params,
        ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t tmp_params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rtr,
        .super.cfg_priority  = 80,
        .super.min_length    = 1,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_rndv_rtr_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .remote_op_id        = UCP_OP_ID_RNDV_SEND,
        .lane                = ucp_proto_rndv_find_ctrl_lane(init_params),
        .perf_bias           = 0.0,
        .md_map              = 0,
        .ctrl_msg_name       = UCP_PROTO_RNDV_RTR_NAME
    };

    *params = tmp_params;
}

void
ucp_proto_rndv_rtr_params_init(const ucp_proto_init_params_t *init_params,
                               ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context = init_params->worker->context;

    ucp_proto_rndv_rtr_common_params_init(init_params, params);

    params->super.cfg_thresh = ucp_proto_rndv_cfg_thresh(context,
                               UCS_BIT(UCP_RNDV_MODE_PUT_ZCOPY) |
                               UCS_BIT(UCP_RNDV_MODE_AM));
    params->super.max_length = SIZE_MAX;
    params->unpack_time      = UCS_LINEAR_FUNC_ZERO;
    params->unpack_perf_node = NULL;
    params->mem_info.type    = init_params->select_param->mem_type;
    params->mem_info.sys_dev = init_params->select_param->sys_dev;
}

ucs_status_t
ucp_proto_rndv_rtr_mtype_params_init(const ucp_proto_init_params_t *init_params,
                                     ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context = init_params->worker->context;
    ucp_md_map_t dummy_md_map;
    ucp_md_index_t md_index;
    ucs_status_t status;

    ucp_proto_rndv_rtr_common_params_init(init_params, params);

    params->super.cfg_thresh = ucp_proto_rndv_cfg_thresh(context,
                               UCS_BIT(UCP_RNDV_MODE_PUT_PIPELINE));
    params->mem_info.type    = context->config.ext.rndv_frag_mem_type;
    params->mem_info.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;

    status = ucp_proto_rndv_mtype_init(init_params, &dummy_md_map,
                                       &params->super.max_length);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_mm_get_alloc_md_index(context, &md_index,
                                       params->mem_info.type);
    if ((status == UCS_OK) && (md_index != UCP_NULL_RESOURCE)) {
        params->md_map |= UCS_BIT(md_index);
    }

    return ucp_proto_init_buffer_copy_time(
            init_params->worker, "rtr/mtype unpack", params->mem_info.type,
            init_params->select_param->mem_type, UCT_EP_OP_PUT_ZCOPY,
            &params->unpack_time, &params->unpack_perf_node);
}

static void
ucp_proto_rndv_rts_params_init(const ucp_proto_init_params_t *init_params,
                                    ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t tmp_params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rts,
        .super.cfg_thresh    = ucp_proto_rndv_thresh(init_params),
        .super.cfg_priority  = 60,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .remote_op_id        = UCP_OP_ID_RNDV_RECV,
        .lane                = ucp_proto_rndv_find_ctrl_lane(init_params),
        .unpack_time         = UCS_LINEAR_FUNC_ZERO,
        .perf_bias           = context->config.ext.rndv_perf_diff / 100.0,
        .mem_info.type       = init_params->select_param->mem_type,
        .mem_info.sys_dev    = init_params->select_param->sys_dev,
        .ctrl_msg_name       = UCP_PROTO_RNDV_RTS_NAME,
        .md_map              = 0
    };

    *params = tmp_params;
}

void ucp_proto_rndv_rts_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_rndv_ctrl_init_params_t params;
    ucp_proto_rndv_ctrl_priv_t rpriv;

    ucp_proto_rndv_rts_params_init(init_params, &params);

    ucp_proto_rndv_ctrl_probe(&params, &rpriv, sizeof(rpriv));
}

void ucp_proto_rndv_rts_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t remote_attr;

    ucp_proto_config_query(params->worker, &rpriv->remote_proto_config,
                           params->msg_length, &remote_attr);

    attr->is_estimation  = 1;
    attr->max_msg_length = remote_attr.max_msg_length;
    attr->lane_map       = UCS_BIT(rpriv->lane);

    ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "rendezvous %s",
                      remote_attr.desc);
    ucs_strncpy_safe(attr->config, remote_attr.config, sizeof(attr->config));
}

void ucp_proto_rndv_rts_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_am_release_user_header(req);

    if (ucp_request_memh_invalidate(req, status)) {
        ucp_proto_rndv_rts_reset(req);
        return;
    }

    ucp_proto_rndv_rts_reset(req);
    ucp_request_complete_send(req, status);
}

ucs_status_t ucp_proto_rndv_rts_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucs_assert(req->send.state.completed_size == 0);
    }

    return ucp_proto_request_zcopy_id_reset(req);
}

static ucs_status_t
ucp_proto_rndv_ack_perf(const ucp_proto_init_params_t *init_params,
                        ucp_lane_index_t lane, ucs_linear_func_t *ack_perf,
                        ucs_linear_func_t overhead)
{
    double send_time, receive_time;
    ucs_linear_func_t ack_func;
    ucs_status_t status;

    status = ucp_proto_rndv_ctrl_perf(init_params, lane, &send_time,
                                      &receive_time);
    if (status != UCS_OK) {
        return status;
    }
    ack_func = ucs_linear_func_make(send_time + receive_time, 0);

    ack_perf[UCP_PROTO_PERF_TYPE_SINGLE] =
            ucs_linear_func_add(ack_func, overhead);
    ack_perf[UCP_PROTO_PERF_TYPE_MULTI] =
    ack_perf[UCP_PROTO_PERF_TYPE_CPU] =
            ucs_linear_func_add(ucs_linear_func_make(send_time, 0), overhead);

    return UCS_OK;
}

ucs_status_t ucp_proto_rndv_ack_init(const ucp_proto_init_params_t *init_params,
                                     const char *name,
                                     const ucp_proto_caps_t *input_caps,
                                     ucs_linear_func_t overhead,
                                     ucp_proto_rndv_ack_priv_t *apriv,
                                     ucp_proto_caps_t *caps)
{
    const ucp_proto_perf_range_t *parallel_stages[2];
    ucp_proto_perf_range_t ack_range;
    ucs_status_t status;
    size_t min_length;
    unsigned i;

    if (ucp_proto_rndv_init_params_is_ppln_frag(init_params)) {
        /* Not sending ACK */
        apriv->lane = UCP_NULL_LANE;
    } else {
        apriv->lane = ucp_proto_rndv_find_ctrl_lane(init_params);
        if (apriv->lane == UCP_NULL_LANE) {
            return UCS_ERR_NO_ELEM;
        }
    }

    status = ucp_proto_rndv_ack_perf(init_params, apriv->lane, ack_range.perf,
                                     overhead);
    if (status != UCS_OK) {
        return status;
    }

    ack_range.node = ucp_proto_perf_node_new_data(name, "");
    ucp_proto_perf_node_add_data(ack_range.node, "ovrh", overhead);
    ucp_proto_perf_range_add_data(&ack_range);

    /* Copy basic capabilities from bulk protocol */
    caps->min_length = input_caps->min_length;
    caps->num_ranges = 0;

    min_length = input_caps->min_length;

    /* Create ranges by adding latency and overhead to bulk protocol ranges */
    for (i = 0; i < input_caps->num_ranges; ++i) {
        ack_range.max_length = input_caps->ranges[i].max_length;

        parallel_stages[0] = &input_caps->ranges[i];
        parallel_stages[1] = &ack_range;

        status = ucp_proto_init_parallel_stages(
                ucp_proto_id_field(init_params->proto_id, name), min_length,
                ack_range.max_length, 0, parallel_stages, 2, caps);
        if (status != UCS_OK) {
            break;
        }

        min_length = ack_range.max_length - 1;
    }

    ucp_proto_perf_node_deref(&ack_range.node);

    return status;
}

ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *params,
                         const char *name, const char *ack_name,
                         ucp_proto_rndv_bulk_priv_t *rpriv,
                         ucp_proto_caps_t *caps)
{
    const ucp_proto_init_params_t *init_params = &params->super.super;
    ucp_context_t *context                     = init_params->worker->context;
    ucp_proto_multi_priv_t *mpriv              = &rpriv->mpriv;
    ucp_proto_caps_t multi_caps;
    ucs_status_t status;

    status = ucp_proto_multi_init(params, &multi_caps, mpriv);
    if (status != UCS_OK) {
        return status;
    }

    /* Adjust align split threshold by user configuration */
    mpriv->align_thresh = ucs_max(context->config.ext.rndv_align_thresh,
                                  mpriv->align_thresh + mpriv->min_frag);

    /* Add ack latency */
    status = ucp_proto_rndv_ack_init(init_params, ack_name, &multi_caps,
                                     ucs_linear_func_make(150e-9, 0),
                                     &rpriv->super, caps);
    ucp_proto_select_caps_cleanup(&multi_caps);
    if (status != UCS_OK) {
        return status;
    }

    return ucp_proto_rndv_predict_prev_stages(init_params, caps);
}

size_t ucp_proto_rndv_common_pack_ack(void *dest, void *arg)
{
    ucp_request_t *req = arg;

    return ucp_proto_rndv_pack_ack(req, dest, req->send.state.dt_iter.length);
}

ucs_status_t ucp_proto_rndv_ats_complete(ucp_request_t *req)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, UCP_DT_MASK_ALL);
    return ucp_proto_rndv_recv_complete(req);
}

ucs_status_t ucp_proto_rndv_ats_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_rndv_ack_progress(req, req->send.proto_config->priv,
                                       UCP_AM_ID_RNDV_ATS,
                                       ucp_proto_rndv_common_pack_ack,
                                       ucp_proto_rndv_ats_complete);
}

void ucp_proto_rndv_bulk_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_bulk_priv_t *rpriv     = params->priv;
    ucp_proto_query_params_t multi_query_params = {
        .proto         = params->proto,
        .priv          = &rpriv->mpriv,
        .worker        = params->worker,
        .select_param  = params->select_param,
        .ep_config_key = params->ep_config_key,
        .msg_length    = params->msg_length
    };

    attr->max_msg_length = SIZE_MAX;
    attr->is_estimation  = 0;
    ucp_proto_multi_query_config(&multi_query_params, attr);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_rndv_send_reply,
                 (worker, req, op_id, op_attr_mask, length, rkey_buffer,
                  rkey_length, sg_count),
                 ucp_worker_h worker, ucp_request_t *req,
                 ucp_operation_id_t op_id, uint32_t op_attr_mask, size_t length,
                 const void *rkey_buffer, size_t rkey_length, uint8_t sg_count)
{
    ucp_ep_h ep                = req->send.ep;
    ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t sel_param;
    ucp_proto_select_t *proto_select;
    ucs_status_t status;
    ucp_rkey_h rkey;

    ucs_assert((op_id >= UCP_OP_ID_RNDV_FIRST) &&
               (op_id < UCP_OP_ID_RNDV_LAST));

    if (rkey_length > 0) {
        ucs_assert(rkey_buffer != NULL);
        /* Do not unpack rkeys from MDs with rkey_ptr capability, except
         * rkey_ptr_lane. Examples are: sysv and posix. Such keys, if packed,
         * are unpacked only once and cached in the peer_mem hash on ep. It is
         * done by the specific protocols (if selected) which use them.
         */
        status = ucp_ep_rkey_unpack_internal(
                  ep, rkey_buffer, rkey_length, ep_config->key.reachable_md_map,
                  ep_config->rndv.proto_rndv_rkey_skip_mds, &rkey);
        if (status != UCS_OK) {
            goto err;
        }

        proto_select   = &ucp_rkey_config(worker, rkey)->proto_select;
        rkey_cfg_index = rkey->cfg_index;
    } else {
        /* No remote key, use endpoint protocols */
        proto_select   = &ep_config->proto_select;
        rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        rkey           = NULL;
    }

    ucp_proto_select_param_init(&sel_param, op_id, op_attr_mask,
                                UCP_PROTO_SELECT_OP_FLAG_PREV_STAGES,
                                req->send.state.dt_iter.dt_class,
                                &req->send.state.dt_iter.mem_info, sg_count);

    status = UCS_PROFILE_CALL(ucp_proto_request_lookup_proto, worker, ep, req,
                              proto_select, rkey_cfg_index, &sel_param, length);
    if (status != UCS_OK) {
        goto err_destroy_rkey;
    }

    req->send.rndv.rkey        = rkey;
    /* Caching rkey_buffer pointer for later unpacking of shm keys in
     * rkey_ptr mtype ppln protocol. */
    req->send.rndv.rkey_buffer = rkey_buffer;

    ucp_trace_req(req,
                  "%s rva 0x%" PRIx64 " length %zd rreq_id 0x%" PRIx64
                  " with protocol %s",
                  ucp_operation_names[ucp_proto_select_op_id(&sel_param)],
                  req->send.rndv.remote_address, length,
                  req->send.rndv.remote_req_id,
                  req->send.proto_config->proto->name);
    return UCS_OK;

err_destroy_rkey:
    if (rkey != NULL) {
        ucp_rkey_destroy(rkey);
    }
err:
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_check_rkey_length(uint64_t address, size_t rkey_length,
                                 const char *title)
{
    ucs_assertv((ssize_t)rkey_length >= 0, "%s rkey_length=%zd", title,
                (ssize_t)rkey_length);
    ucs_assertv((address != 0) == (rkey_length > 0),
                "%s rts->address=0x%" PRIx64 " rkey_length=%zu", title, address,
                rkey_length);
}

UCS_PROFILE_FUNC_VOID(ucp_proto_rndv_receive_start,
                      (worker, recv_req, rts, rkey_buffer, rkey_length),
                      ucp_worker_h worker, ucp_request_t *recv_req,
                      const ucp_rndv_rts_hdr_t *rts, const void *rkey_buffer,
                      size_t rkey_length)
{
    ucp_operation_id_t op_id;
    ucs_status_t status;
    ucp_request_t *req;
    uint8_t sg_count;
    ucp_ep_h ep;

    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, rts->sreq.ep_id, {
        ucp_proto_rndv_recv_req_complete(recv_req, UCS_ERR_CANCELED);
        return;
    }, "RTS on non-existing endpoint");

    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_error("failed to allocate rendezvous reply");
        return;
    }

    /* Initialize send request */
    ucp_proto_request_send_init(req, ep, 0);
    req->send.rndv.remote_req_id  = rts->sreq.req_id;
    req->send.rndv.remote_address = rts->address;
    req->send.rndv.offset         = 0;
    ucp_request_set_super(req, recv_req);

    if (ucs_likely(rts->size <= recv_req->recv.dt_iter.length)) {
        ucp_proto_rndv_check_rkey_length(rts->address, rkey_length, "rts");
        op_id            = UCP_OP_ID_RNDV_RECV;
        recv_req->status = UCS_OK;
        UCS_PROFILE_CALL_VOID(ucp_datatype_iter_move, &req->send.state.dt_iter,
                              &recv_req->recv.dt_iter, rts->size, &sg_count);
    } else {
        /* Short receive: complete with error, and send reply to sender */
        rkey_length      = 0; /* Override rkey length to disable data fetch */
        op_id            = UCP_OP_ID_RNDV_RECV_DROP;
        recv_req->status = UCS_ERR_MESSAGE_TRUNCATED;
        ucp_datatype_iter_cleanup(&recv_req->recv.dt_iter, 1, UCP_DT_MASK_ALL);
        ucp_datatype_iter_init_null(&req->send.state.dt_iter, rts->size,
                                    &sg_count);
    }

    status = ucp_proto_rndv_send_reply(worker, req, op_id,
                                       recv_req->recv.op_attr, rts->size,
                                       rkey_buffer, rkey_length, sg_count);
    if (status != UCS_OK) {
        ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, UCP_DT_MASK_ALL);
        ucs_mpool_put(req);
        return;
    }

#if ENABLE_DEBUG_DATA
    recv_req->recv.proto_rndv_config  = req->send.proto_config;
    recv_req->recv.proto_rndv_request = req;
#endif

    UCS_PROFILE_CALL_VOID(ucp_request_send, req);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_rndv_send_start,
                 (worker, req, op_attr_mask, rtr, header_length, sg_count),
                 ucp_worker_h worker, ucp_request_t *req, uint32_t op_attr_mask,
                 const ucp_rndv_rtr_hdr_t *rtr, size_t header_length,
                 uint8_t sg_count)
{
    ucs_status_t status;
    size_t rkey_length;

    ucs_assert(header_length >= sizeof(*rtr));
    rkey_length = header_length - sizeof(*rtr);

    ucp_proto_rndv_check_rkey_length(rtr->address, rkey_length, "rtr");
    req->send.rndv.remote_address = rtr->address;
    req->send.rndv.remote_req_id  = rtr->rreq_id;
    req->send.rndv.offset         = rtr->offset;

    ucs_assert(rtr->size == req->send.state.dt_iter.length);
    status = ucp_proto_rndv_send_reply(worker, req, UCP_OP_ID_RNDV_SEND,
                                       op_attr_mask, rtr->size, rtr + 1,
                                       rkey_length, sg_count);
    if (status != UCS_OK) {
        return status;
    }

    UCS_PROFILE_CALL_VOID(ucp_request_send, req);
    return UCS_OK;
}

static void ucp_proto_rndv_send_complete_one(void *request, ucs_status_t status,
                                             void *user_data)
{
    ucp_request_t *freq = (ucp_request_t*)request - 1;
    ucp_request_t *req;

    req = ucp_request_user_data_get_super(request, user_data);

    if (!ucp_proto_rndv_frag_complete(req, freq, "rndv_send")) {
        return;
    }

    ucp_send_request_id_release(req);
    ucp_proto_request_zcopy_complete(req, status);
}

ucs_status_t
ucp_proto_rndv_handle_rtr(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker           = arg;
    const ucp_rndv_rtr_hdr_t *rtr = data;
    const ucp_proto_select_param_t *select_param;
    ucp_request_t *req, *freq;
    uint32_t op_attr_mask;
    ucs_status_t status;
    uint8_t sg_count;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rtr->sreq_id, 0, return UCS_OK,
                               "RTR %p", rtr);

    ucp_trace_req(req, "recv RTR offset %zu length %zu/%zu req %p", rtr->offset,
                  rtr->size, req->send.state.dt_iter.length, req);

    if (req->flags & UCP_REQUEST_FLAG_OFFLOADED) {
        ucp_tag_offload_cancel_rndv(req);
        ucs_assert(!ucp_ep_use_indirect_id(req->send.ep));
    }

    /* RTR covers the whole send request - use the send request directly */
    ucs_assert(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED);

    select_param = &req->send.proto_config->select_param;
    op_attr_mask = ucp_proto_select_op_attr_unpack(select_param->op_attr);

    if (rtr->size == req->send.state.dt_iter.length) {
        /* RTR covers the whole send request - use the send request directly */
        ucs_assert(rtr->offset == 0);

        ucp_send_request_id_release(req);
        ucp_datatype_iter_rewind(&req->send.state.dt_iter, UCP_DT_MASK_ALL);
        req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;

        sg_count = select_param->sg_count;
        status   = ucp_proto_rndv_send_start(worker, req, op_attr_mask, rtr,
                                             length, sg_count);
        if (status != UCS_OK) {
            goto err_request_fail;
        }
    } else {
        ucs_assertv(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG,
                    "fragmented rendezvous is not supported with datatype %s",
                    ucp_datatype_class_names[req->send.state.dt_iter.dt_class]);

        /* Partial RTR, its "offset" and "size" fields specify part to send */
        status = ucp_proto_rndv_frag_request_alloc(worker, req, &freq);
        if (status != UCS_OK) {
            goto err_request_fail;
        }

        /* When this fragment is completed, count total size and complete the
           super request if needed */
        ucp_request_set_callback(freq, send.cb,
                                 ucp_proto_rndv_send_complete_one);

        ucp_datatype_iter_slice(&req->send.state.dt_iter, rtr->offset,
                                rtr->size, &freq->send.state.dt_iter);

        /* Send rendezvous fragment, when it's completed update 'remaining'
         * and complete 'req' when it reaches zero
         * TODO can rndv/ppln be selected here (and not just single frag)?
         */
        status = ucp_proto_rndv_send_start(worker, freq,
                                           op_attr_mask |
                                           UCP_OP_ATTR_FLAG_MULTI_SEND,
                                           rtr, length, 1);
        if (status != UCS_OK) {
            goto err_put_freq;
        }
    }

    return UCS_OK;

err_put_freq:
    ucp_request_put(freq);
err_request_fail:
    ucp_proto_request_abort(req, status);
    return UCS_OK;
}

void ucp_proto_rndv_bulk_request_init_lane_idx(
        ucp_request_t *req, const ucp_proto_rndv_bulk_priv_t *rpriv)
{
    size_t total_length = ucp_proto_rndv_request_total_length(req);
    size_t max_frag_sum = rpriv->mpriv.max_frag_sum;
    const ucp_proto_multi_lane_priv_t *lpriv;
    size_t end_offset, rel_offset;
    ucp_lane_index_t lane_idx;

    lane_idx = 0;
    if (ucs_likely(total_length < max_frag_sum)) {
        /* Size is smaller than frag sum - scale the total length by the weight
           of each lane */
        do {
            lpriv      = &rpriv->mpriv.lanes[lane_idx++];
            end_offset = ucp_proto_multi_scaled_length(lpriv->weight_sum,
                                                       total_length);
        } while (req->send.rndv.offset >= end_offset);
    } else {
        /* Find the lane which needs to send the current fragment */
        rel_offset = req->send.rndv.offset % rpriv->mpriv.max_frag_sum;
        do {
            lpriv = &rpriv->mpriv.lanes[lane_idx++];
        } while (rel_offset >= lpriv->max_frag_sum);
    }

    req->send.multi_lane_idx = lane_idx - 1;
}

static int
ucp_proto_rndv_init_params_incl_prev_stages(
        const ucp_proto_init_params_t *params)
{
    if (!ucp_proto_init_check_op(params, UCP_PROTO_RNDV_OP_ID_MASK)) {
        return 0;
    }

    return ucp_proto_select_op_flags(params->select_param) &
            UCP_PROTO_SELECT_OP_FLAG_PREV_STAGES;
}

static int
ucp_proto_rndv_predict_rtr_mtype(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context          = init_params->worker->context;
    ucs_memory_type_t frag_mem_type = context->config.ext.rndv_frag_mem_type;
    ucs_memory_type_t rkey_mem_type;

    if (init_params->rkey_config_key == NULL) {
        return 0;
    }
    rkey_mem_type = init_params->rkey_config_key->mem_type;

    if ((rkey_mem_type == init_params->select_param->mem_type) ||
        (rkey_mem_type != frag_mem_type)) {
        return 0;
    }

    /* Report `true` only if `rkey_mem_type` equal to `frag_mem_type` and
     * different from `select_params->mem_type` */
    return 1;
}

ucs_status_t
ucp_proto_rndv_predict_prev_stages(const ucp_proto_init_params_t *init_params,
                                   ucp_proto_caps_t *caps)
{
    size_t num_stages = 0;
    ucp_proto_perf_range_t ctrl_stages[2];
    ucp_proto_rndv_ctrl_init_params_t rts_params, rtr_params;
    ucp_md_map_t remote_md_map, local_md_map;
    ucp_proto_rndv_ctrl_priv_t rpriv;
    ucp_proto_caps_t result_caps;
    ucs_status_t status;

    if (!ucp_proto_rndv_init_params_incl_prev_stages(init_params)) {
        /* Do not include previous stages */
        return UCS_OK;
    }

    ucp_proto_rndv_rts_params_init(init_params, &rts_params);
    if (rts_params.lane == UCP_NULL_LANE) {
        return UCS_ERR_NO_ELEM;
    }

    /* Init RTS priv */
    ucp_proto_rndv_ctrl_init_priv(&rts_params, &rpriv, rts_params.lane);
    local_md_map  = rpriv.md_map;
    remote_md_map = ucp_proto_rndv_md_map_to_remote(&rts_params, rpriv.md_map);

    /* Turn RTS md_map to remote in case if RTS was sent by peer */
    if (ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_RNDV_RECV))) {
        rpriv.md_map = remote_md_map;
    }
    status = ucp_proto_rndv_ctrl_range_init(&rts_params, &rpriv,
                                            &ctrl_stages[num_stages++]);
    if (status != UCS_OK) {
        return status;
    }

    /* Predict RTR */
    if (ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_RNDV_SEND))) {
        status     = UCS_ERR_UNSUPPORTED;
        if (ucp_proto_rndv_predict_rtr_mtype(init_params)) {
            status = ucp_proto_rndv_rtr_mtype_params_init(init_params,
                                                          &rtr_params);
        }
        if (status != UCS_OK) {
            ucp_proto_rndv_rtr_params_init(init_params, &rtr_params);
        }

        rpriv.md_map = local_md_map;
        status       = ucp_proto_rndv_ctrl_range_init(
                &rtr_params, &rpriv, &ctrl_stages[num_stages++]);
        if (status != UCS_OK) {
            num_stages--;
            goto out;
        }
        ucp_proto_perf_node_deref(&rtr_params.unpack_perf_node);
    }

    result_caps            = *caps;
    result_caps.num_ranges = 0;
    status                 = ucp_proto_rndv_ctrl_init_parallel_stages(
            ctrl_stages, num_stages, caps, 0, &result_caps, 0, SIZE_MAX);
    if (status != UCS_OK) {
        goto out;
    }

    ucp_proto_select_caps_cleanup(caps);
    *caps = result_caps;

out:
    do {
        ucp_proto_perf_node_deref(&ctrl_stages[--num_stages].node);
    } while (num_stages > 0);

    return status;
}
