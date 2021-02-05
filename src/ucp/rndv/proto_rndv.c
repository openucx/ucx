/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.h"

#include <ucp/proto/proto_common.inl>


static ucp_md_map_t
ucp_proto_rndv_ctrl_reg_md_map(const ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_worker_h worker                      = params->super.super.worker;
    const ucp_ep_config_key_t *ep_config_key = params->super.super.ep_config_key;
    const uct_iface_attr_t *iface_attr;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;
    ucp_md_map_t reg_md_map;
    ucp_lane_index_t lane;

    /* md_map is all lanes which support get_zcopy on the given mem_type and
     * require remote key
     */
    reg_md_map = 0;
    for (lane = 0; lane < ep_config_key->num_lanes; ++lane) {
        if (ep_config_key->lanes[lane].rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        /* Check the lane supports get_zcopy */
        iface_attr = ucp_proto_common_get_iface_attr(&params->super.super,
                                                     lane);
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY)) {
            continue;
        }

        /* Check the memory domain requires remote key, and capable of
         * registering the memory type
         */
        md_index = ucp_proto_common_get_md_index(&params->super.super, lane);
        md_attr  = &worker->context->tl_mds[md_index].attr;
        if (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) ||
            !(md_attr->cap.reg_mem_types & UCS_BIT(params->mem_info.type))) {
            continue;
        }

        reg_md_map |= UCS_BIT(md_index);
    }

    return reg_md_map;
}

/*
 * Select (guess) the protocol that would be used by the remote peer.
 * We report the rendezvous protocol performance according to the protocol we
 * think the remote peer would select.
 */
static ucs_status_t ucp_proto_rndv_ctrl_select_remote_proto(
        const ucp_proto_rndv_ctrl_init_params_t *params,
        const ucp_proto_select_param_t *remote_select_param,
        ucp_proto_rndv_ctrl_priv_t *rpriv)
{
    ucp_worker_h worker                 = params->super.super.worker;
    ucp_worker_cfg_index_t ep_cfg_index = params->super.super.ep_cfg_index;
    ucp_rkey_config_key_t rkey_config_key;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_elem_t *select_elem;
    ucp_rkey_config_t *rkey_config;
    ucs_status_t status;

    /* Construct remote key for remote protocol lookup according to the local
     * buffer properties (since remote side is expected to access the local
     * buffer)
     */
    rkey_config_key.md_map       = rpriv->md_map;
    rkey_config_key.ep_cfg_index = ep_cfg_index;
    rkey_config_key.mem_type     = params->mem_info.type;

    status = ucp_worker_get_rkey_config(worker, &rkey_config_key,
                                        &rkey_cfg_index);
    if (status != UCS_OK) {
        return status;
    }

    rkey_config = &worker->rkey_config[rkey_cfg_index];
    select_elem = ucp_proto_select_lookup_slow(worker,
                                               &rkey_config->proto_select,
                                               ep_cfg_index, rkey_cfg_index,
                                               remote_select_param);
    if (select_elem == NULL) {
        ucs_debug("%s: did not find protocol for %s",
                  params->super.super.proto_name,
                  ucp_operation_names[params->remote_op_id]);
        return UCS_ERR_UNSUPPORTED;
    }

    rpriv->remote_proto = *select_elem;
    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_ctrl_init(const ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context             = params->super.super.worker->context;
    ucp_proto_rndv_ctrl_priv_t *rpriv = params->super.super.priv;
    const ucp_proto_perf_range_t *remote_perf_range;
    ucp_proto_select_param_t remote_select_param;
    ucp_proto_perf_range_t *perf_range;
    const uct_iface_attr_t *iface_attr;
    ucs_linear_func_t send_overheads;
    ucs_memory_info_t mem_info;
    ucp_md_index_t md_index;
    ucp_proto_caps_t *caps;
    ucp_md_map_t md_map;
    ucs_status_t status;
    double rts_latency;

    ucs_assert(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE);
    ucs_assert(!(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG));

    /* Find lane to send the initial message */
    rpriv->lane = ucp_proto_common_find_am_bcopy_lane(&params->super.super);
    if (rpriv->lane == UCP_NULL_LANE) {
        return UCS_ERR_NO_ELEM;
    }

    /* Construct select parameter for the remote protocol */
    if (params->super.super.rkey_config_key == NULL) {
        /* Remote buffer is unknown, assume same params as local */
        remote_select_param          = *params->super.super.select_param;
        remote_select_param.op_id    = params->remote_op_id;
        remote_select_param.op_flags = 0;
    } else {
        /* If we know the remote buffer parameters, these are actually the local
         * parameters for the remote protocol
         */
        mem_info.type    = params->super.super.rkey_config_key->mem_type;
        mem_info.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        ucp_proto_select_param_init(&remote_select_param, params->remote_op_id,
                                    0, UCP_DATATYPE_CONTIG, &mem_info, 1);
    }

    /* Initialize estimated memory registration map */
    md_map = ucp_proto_rndv_ctrl_reg_md_map(params);
    if (md_map == 0) {
        ucs_trace("registration map is 0, memory type %s",
                  ucs_memory_type_names[params->mem_info.type]);
        return UCS_ERR_UNSUPPORTED;
    }

    rpriv->md_map           = md_map;
    rpriv->packed_rkey_size = ucp_rkey_packed_size(context, rpriv->md_map);

    /* Guess the protocol the remote side will select */
    status = ucp_proto_rndv_ctrl_select_remote_proto(params,
                                                     &remote_select_param,
                                                     rpriv);
    if (status != UCS_OK) {
        return status;
    }

    /* Set send_overheads to the time to send and receive RTS message */
    iface_attr     = ucp_proto_common_get_iface_attr(&params->super.super,
                                                     rpriv->lane);
    rts_latency    = (iface_attr->overhead * 2) +
                     ucp_tl_iface_latency(context, &iface_attr->latency);
    send_overheads = ucs_linear_func_make(rts_latency, 0.0);

    /* Add registration cost to send_overheads */
    ucs_for_each_bit(md_index, rpriv->md_map) {
        ucs_linear_func_add_inplace(&send_overheads,
                                    context->tl_mds[md_index].attr.reg_cost);
    }

    /* Set rendezvous protocol properties */
    *params->super.super.priv_size         = sizeof(ucp_proto_rndv_ctrl_priv_t);
    params->super.super.caps->cfg_thresh   = params->super.cfg_thresh;
    params->super.super.caps->cfg_priority = params->super.cfg_priority;
    params->super.super.caps->min_length   = 0;
    params->super.super.caps->num_ranges   = 0;

    /* Copy performance ranges from the remote protocol, and add overheads */
    remote_perf_range = rpriv->remote_proto.perf_ranges;
    caps              = params->super.super.caps;
    do {
        perf_range             = &caps->ranges[caps->num_ranges];
        perf_range->max_length = remote_perf_range->max_length;

        /* Add send overheads and apply perf_bias */
        perf_range->perf = ucs_linear_func_compose(
                ucs_linear_func_make(0, 1.0 - params->perf_bias),
                ucs_linear_func_add(remote_perf_range->perf, send_overheads));

        ++caps->num_ranges;
    } while ((remote_perf_range++)->max_length != SIZE_MAX);

    return UCS_OK;
}

void ucp_proto_rndv_ctrl_config_str(size_t min_length, size_t max_length,
                                    const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = priv;
    const ucp_proto_threshold_elem_t *thresh_elem;
    size_t range_start, range_end;
    const ucp_proto_t *proto;
    ucp_md_index_t md_index;
    char str[64];

    /* Print message lane and memory domains list */
    ucs_string_buffer_appendf(strb, "am-ln:%d mds:{", rpriv->lane);
    ucs_for_each_bit(md_index, rpriv->md_map) {
        ucs_string_buffer_appendf(strb, "%d,", md_index);
    }
    ucs_string_buffer_rtrim(strb, ",");
    ucs_string_buffer_appendf(strb, "} ");

    /* Print estimated remote protocols for each message size */
    thresh_elem = rpriv->remote_proto.thresholds;
    range_start = 0;
    do {
        range_end = thresh_elem->max_msg_length;

        /* Print only protocols within the range provided by {min,max}_length */
        if ((range_end >= min_length) && (range_start <= max_length)) {
            proto = thresh_elem->proto_config.proto;
            ucs_string_buffer_appendf(strb, "%s(", proto->name);
            proto->config_str(range_start, range_end,
                              thresh_elem->proto_config.priv, strb);
            ucs_string_buffer_appendf(strb, ")");

            if (range_end < max_length) {
                ucs_memunits_to_str(thresh_elem->max_msg_length, str,
                                    sizeof(str));
                ucs_string_buffer_appendf(strb, "<=%s<", str);
            }
        }

        ++thresh_elem;
        range_start = range_end + 1;
    } while (range_end < max_length);

    ucs_string_buffer_rtrim(strb, "<");
}

ucs_status_t ucp_proto_rndv_rts_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super        = *init_params,
        .super.latency      = 0,
        .super.overhead     = 40e-9,
        .super.cfg_thresh   = context->config.ext.rndv_thresh,
        .super.cfg_priority = 60,
        .super.flags        = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .remote_op_id       = UCP_OP_ID_RNDV_RECV,
        .perf_bias          = context->config.ext.rndv_perf_diff / 100.0,
        .mem_info.type      = init_params->select_param->mem_type,
        .mem_info.sys_dev   = init_params->select_param->sys_dev
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_TAG_SEND);

    ucp_proto_rndv_ctrl_init(&params);

    /* TODO enable when progress logic is added */
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *init_params)
{
    ucp_proto_rndv_bulk_priv_t *rpriv    = init_params->super.super.priv;
    ucp_proto_multi_init_params_t params = *init_params;
    ucs_status_t status;
    size_t mpriv_size;

    /* Change priv pointer, since proto_multi priv is not the first element in
     * ucp_proto_rndv_bulk_priv_t struct. Later on, we also update priv size.
     */
    params.super.super.priv      = &rpriv->mpriv;
    params.super.super.priv_size = &mpriv_size;

    status = ucp_proto_multi_init(&params);
    if (status != UCS_OK) {
        return status;
    }

    rpriv->am_lane = ucp_proto_common_find_am_bcopy_lane(&params.super.super);
    if (rpriv->am_lane == UCP_NULL_LANE) {
        return UCS_ERR_NO_ELEM;
    }

    /* Update private data size based of ucp_proto_multi_priv_t variable size */
    *init_params->super.super.priv_size =
            ucs_offsetof(ucp_proto_rndv_bulk_priv_t, mpriv) + mpriv_size;
    return UCS_OK;
}

void ucp_proto_rndv_bulk_config_str(size_t min_length, size_t max_length,
                                    const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_bulk_priv_t *rpriv = priv;

    ucp_proto_multi_config_str(min_length, max_length, &rpriv->mpriv, strb);
    ucs_string_buffer_appendf(strb, " am-ln:%d", rpriv->am_lane);
}

int ucp_proto_rndv_check_params(const ucp_proto_init_params_t *init_params,
                                ucp_operation_id_t op_id,
                                ucp_rndv_mode_t rndv_mode)
{
    ucp_context_h context = init_params->worker->context;

    return (init_params->select_param->op_id == op_id) &&
           (init_params->select_param->dt_class == UCP_DATATYPE_CONTIG) &&
           ((context->config.ext.rndv_mode == UCP_RNDV_MODE_AUTO) ||
            (context->config.ext.rndv_mode == rndv_mode));
}
