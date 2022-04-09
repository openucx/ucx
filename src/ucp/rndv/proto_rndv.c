/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"

#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_common.inl>


static void
ucp_proto_rndv_ctrl_get_md_map(const ucp_proto_rndv_ctrl_init_params_t *params,
                               ucp_md_map_t *md_map,
                               ucp_sys_dev_map_t *sys_dev_map,
                               ucs_sys_dev_distance_t *sys_distance)
{
    ucp_worker_h worker                      = params->super.super.worker;
    const ucp_ep_config_key_t *ep_config_key = params->super.super.ep_config_key;
    ucp_rsc_index_t mem_sys_dev, ep_sys_dev;
    const uct_iface_attr_t *iface_attr;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;
    ucs_status_t status;

    /* md_map is all lanes which support get_zcopy on the given mem_type and
     * require remote key
     */
    *md_map      = 0;
    *sys_dev_map = 0;

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
        md_attr    = &worker->context->tl_mds[md_index].attr;

        /* Check the lane supports get_zcopy or rkey_ptr */
        if (!(md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR) &&
            !(iface_attr->cap.flags &
              (UCT_IFACE_FLAG_GET_ZCOPY | UCT_IFACE_FLAG_PUT_ZCOPY))) {
            continue;
        }

        /* Check the memory domain requires remote key, and capable of
         * registering the memory type
         */
        if (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) ||
            !(md_attr->cap.reg_mem_types & UCS_BIT(params->mem_info.type))) {
            continue;
        }

        ucs_trace_req("lane[%d]: selected md %s\n", lane,
                      worker->context->tl_mds[md_index].rsc.md_name);
        *md_map |= UCS_BIT(md_index);

        if (ep_sys_dev >= UCP_MAX_SYS_DEVICES) {
            continue;
        }

        mem_sys_dev   = params->super.super.select_param->sys_dev;
        *sys_dev_map |= UCS_BIT(ep_sys_dev);

        status = ucs_topo_get_distance(mem_sys_dev, ep_sys_dev, sys_distance);
        ucs_assertv_always(status == UCS_OK, "mem_info->sys_dev=%d sys_dev=%d",
                           mem_sys_dev, ep_sys_dev);
        ++sys_distance;
    }
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
    const ucp_ep_config_t *ep_config    = &worker->ep_config[ep_cfg_index];
    ucs_sys_dev_distance_t lanes_distance[UCP_MAX_LANES];
    const ucp_proto_select_elem_t *select_elem;
    ucp_rkey_config_key_t rkey_config_key;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_rkey_config_t *rkey_config;
    ucs_status_t status;
    ucp_lane_index_t lane;

    /* Construct remote key for remote protocol lookup according to the local
     * buffer properties (since remote side is expected to access the local
     * buffer)
     */
    rkey_config_key.md_map       = rpriv->md_map;
    rkey_config_key.ep_cfg_index = ep_cfg_index;
    rkey_config_key.sys_dev      = params->mem_info.sys_dev;
    rkey_config_key.mem_type     = params->mem_info.type;
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
ucp_proto_rndv_ctrl_perf(const ucp_proto_init_params_t *params,
                         ucp_lane_index_t lane, double *send_time,
                         double *receive_time)
{
    ucp_context_t *context = params->worker->context;
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    if (lane == UCP_NULL_LANE) {
        *send_time = *receive_time = 0;
        return UCS_OK;
    }

    status = ucp_proto_common_lane_perf_attr(params, lane, UCT_EP_OP_AM_BCOPY,
            UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
            UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
            UCT_PERF_ATTR_FIELD_RECV_OVERHEAD | UCT_PERF_ATTR_FIELD_LATENCY,
            &perf_attr);
    if (status != UCS_OK) {
        return status;
    }

    *send_time    = perf_attr.send_pre_overhead + perf_attr.send_post_overhead;
    *receive_time = perf_attr.recv_overhead +
                    ucp_tl_iface_latency(context, &perf_attr.latency);
    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_ctrl_init(const ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context             = params->super.super.worker->context;
    ucp_proto_rndv_ctrl_priv_t *rpriv = params->super.super.priv;
    UCS_ARRAY_DEFINE_ONSTACK(list, ucp_proto_perf_envelope, 2);
    size_t min_length, max_length, range_max_length;
    const ucp_proto_select_param_t *select_param;
    const ucp_proto_select_range_t *remote_range;
    ucp_proto_select_param_t remote_select_param;
    ucp_proto_perf_range_t send_perf;
    ucs_linear_func_t rndv_bias;
    double send_time, receive_time;
    ucp_memory_info_t mem_info;
    ucs_status_t status;
    double ctrl_latency;

    ucs_assert(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE);
    ucs_assert(!(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG));

    select_param                   = params->super.super.select_param;
    *params->super.super.priv_size = sizeof(ucp_proto_rndv_ctrl_priv_t);

    /* Find lane to send the initial message */
    rpriv->lane = ucp_proto_common_find_am_bcopy_hdr_lane(&params->super.super);
    if (rpriv->lane == UCP_NULL_LANE) {
        return UCS_ERR_NO_ELEM;
    }

    /* Construct select parameter for the remote protocol */
    if (params->super.super.rkey_config_key == NULL) {
        /* Remote buffer is unknown, assume same params as local */
        remote_select_param          = *select_param;
        remote_select_param.op_id    = params->remote_op_id;
        remote_select_param.op_flags = UCP_PROTO_SELECT_OP_FLAG_INTERNAL;
    } else {
        /* If we know the remote buffer parameters, these are actually the local
         * parameters for the remote protocol
         */
        mem_info.sys_dev = params->super.super.rkey_config_key->sys_dev;
        mem_info.type    = params->super.super.rkey_config_key->mem_type;
        ucp_proto_select_param_init(&remote_select_param, params->remote_op_id,
                                    0, UCP_PROTO_SELECT_OP_FLAG_INTERNAL,
                                    UCP_DATATYPE_CONTIG, &mem_info, 1);
    }

    /* Initialize estimated memory registration map */
    ucp_proto_rndv_ctrl_get_md_map(params, &rpriv->md_map, &rpriv->sys_dev_map,
                                   rpriv->sys_dev_distance);
    rpriv->packed_rkey_size = ucp_rkey_packed_size(context, rpriv->md_map,
                                                   select_param->sys_dev,
                                                   rpriv->sys_dev_map);

    /* Guess the protocol the remote side will select */
    status = ucp_proto_rndv_ctrl_select_remote_proto(params,
                                                     &remote_select_param,
                                                     rpriv);
    if (status != UCS_OK) {
        return status;
    }

    if (!ucp_proto_select_get_valid_range(rpriv->remote_proto.thresholds,
                                          &min_length, &max_length)) {
        return UCS_ERR_UNSUPPORTED;
    }

    max_length = ucs_min(params->super.max_length, max_length);

    ucs_assert(params->super.send_op == UCT_EP_OP_AM_BCOPY);
    /* Set send_overheads to the time to send and receive RTS message */
    status = ucp_proto_rndv_ctrl_perf(&params->super.super, rpriv->lane,
                                      &send_time, &receive_time);
    if (status != UCS_OK) {
        return status;
    }

    ctrl_latency = send_time + receive_time + params->super.overhead * 2;
    ucs_trace("rndv" UCP_PROTO_TIME_FMT(ctrl_latency),
              UCP_PROTO_TIME_ARG(ctrl_latency));
    send_perf.perf[UCP_PROTO_PERF_TYPE_SINGLE] =
    send_perf.perf[UCP_PROTO_PERF_TYPE_MULTI]  = ucs_linear_func_add3(
            ucp_proto_common_memreg_time(&params->super, rpriv->md_map),
            ucs_linear_func_make(ctrl_latency, 0.0), params->unpack_time);

    /* Set rendezvous protocol properties */
    ucp_proto_common_init_base_caps(&params->super, min_length);

    /* Copy performance ranges from the remote protocol, and add overheads */
    remote_range = rpriv->remote_proto.perf_ranges;
    rndv_bias    = ucs_linear_func_make(0, 1.0 - params->perf_bias);

    ucp_proto_perf_envelope_append(&list, "ctrl", &send_perf, SIZE_MAX,
                                   rndv_bias);
    do {
        range_max_length = ucs_min(remote_range->super.max_length, max_length);
        if (range_max_length < params->super.super.caps->min_length) {
            continue;
        }

        ucp_proto_perf_envelope_append(&list, remote_range->super.name,
                                       &remote_range->super, SIZE_MAX,
                                       rndv_bias);
        status = ucp_proto_common_add_perf_ranges(&params->super, min_length,
                                                  range_max_length, &list);
        if (status != UCS_OK) {
            return status;
        }

        min_length = range_max_length - 1;
        ucs_array_set_length(&list, 1);
    } while ((remote_range++)->super.max_length < max_length);

    return UCS_OK;
}

ucs_status_t ucp_proto_rndv_rts_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 40e-9,
        .super.cfg_thresh    = context->config.ext.rndv_thresh,
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
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .remote_op_id        = UCP_OP_ID_RNDV_RECV,
        .unpack_time         = ucs_linear_func_make(0, 0),
        .perf_bias           = context->config.ext.rndv_perf_diff / 100.0,
        .mem_info.type       = init_params->select_param->mem_type,
        .mem_info.sys_dev    = init_params->select_param->sys_dev
    };

    return ucp_proto_rndv_ctrl_init(&params);
}

void ucp_proto_rndv_rts_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t remote_attr;

    ucp_proto_select_elem_query(params->worker, &rpriv->remote_proto,
                                params->msg_length, &remote_attr);

    attr->is_estimation  = 1;
    attr->max_msg_length = SIZE_MAX;

    ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "rendezvous %s",
                      remote_attr.desc);
    ucs_strncpy_safe(attr->config, remote_attr.config, sizeof(attr->config));
}

void ucp_proto_rndv_rts_abort(ucp_request_t *req, ucs_status_t status)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        ucp_send_request_id_release(req);
        ucp_datatype_iter_mem_dereg(req->send.ep->worker->context,
                                    &req->send.state.dt_iter, UCP_DT_MASK_ALL);
    }

    ucp_request_complete_send(req, status);
}

static ucs_status_t
ucp_proto_rndv_ack_perf(const ucp_proto_init_params_t *init_params,
                        ucp_lane_index_t lane, ucs_linear_func_t *ack_perf)
{
    double send_time, receive_time;
    ucs_status_t status;

    status = ucp_proto_rndv_ctrl_perf(init_params, lane, &send_time,
                                      &receive_time);
    if (status != UCS_OK) {
        return status;
    }

    ack_perf[UCP_PROTO_PERF_TYPE_SINGLE] =
            ucs_linear_func_make(send_time + receive_time, 0);
    ack_perf[UCP_PROTO_PERF_TYPE_MULTI] = ucs_linear_func_make(send_time, 0);

    return UCS_OK;
}

ucs_status_t ucp_proto_rndv_ack_init(const ucp_proto_init_params_t *init_params,
                                     ucp_proto_rndv_ack_priv_t *apriv)
{
    ucp_proto_caps_t *caps = init_params->caps;
    ucs_linear_func_t ack_perf[UCP_PROTO_PERF_TYPE_LAST];
    ucp_proto_perf_type_t perf_type;
    ucs_status_t status;
    unsigned i;

    if (ucp_proto_rndv_init_params_is_ppln_frag(init_params)) {
        /* Not sending ACK */
        apriv->lane = UCP_NULL_LANE;
    } else {
        apriv->lane = ucp_proto_common_find_am_bcopy_hdr_lane(init_params);
        if (apriv->lane == UCP_NULL_LANE) {
            return UCS_ERR_NO_ELEM;
        }
    }

    status = ucp_proto_rndv_ack_perf(init_params, apriv->lane, ack_perf);
    if (status != UCS_OK) {
        return status;
    }

    for (i = 0; i < caps->num_ranges; ++i) {
        for (perf_type = 0; perf_type < UCP_PROTO_PERF_TYPE_LAST; ++perf_type) {
            ucs_linear_func_add_inplace(&caps->ranges[i].perf[perf_type],
                                        ack_perf[perf_type]);
            ucs_trace("range[%d] %s" UCP_PROTO_PERF_FUNC_FMT(ack)
                      UCP_PROTO_PERF_FUNC_FMT(total),
                      i, ucp_proto_perf_types[perf_type],
                      UCP_PROTO_PERF_FUNC_ARG(&ack_perf[perf_type]),
                      UCP_PROTO_PERF_FUNC_ARG(&caps->ranges[i].perf[perf_type]));
        }
    }

    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *init_params,
                         ucp_proto_rndv_bulk_priv_t *rpriv, size_t *priv_size_p)
{
    ucs_status_t status;
    size_t mpriv_size;

    status = ucp_proto_multi_init(init_params, &rpriv->mpriv, &mpriv_size);
    if (status != UCS_OK) {
        return status;
    }

    /* Add ack latency */
    status = ucp_proto_rndv_ack_init(&init_params->super.super, &rpriv->super);
    if (status != UCS_OK) {
        return status;
    }

    /* Update private data size based of ucp_proto_multi_priv_t variable size */
    *priv_size_p = ucs_offsetof(ucp_proto_rndv_bulk_priv_t, mpriv) + mpriv_size;
    return UCS_OK;
}

static size_t ucp_proto_rndv_ats_pack_ack(void *dest, void *arg)
{
    ucp_request_t *req = arg;

    return ucp_proto_rndv_pack_ack(req, dest, req->send.state.dt_iter.length);
}

ucs_status_t ucp_proto_rndv_ats_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_rndv_ack_progress(req, req->send.proto_config->priv,
                                       UCP_AM_ID_RNDV_ATS,
                                       ucp_proto_rndv_ats_pack_ack,
                                       ucp_proto_rndv_recv_complete);
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

static ucs_status_t
ucp_proto_rndv_send_reply(ucp_worker_h worker, ucp_request_t *req,
                          ucp_operation_id_t op_id, uint32_t op_attr_mask,
                          size_t length, const void *rkey_buffer,
                          size_t rkey_length, uint8_t sg_count)
{
    ucp_ep_h ep = req->send.ep;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t sel_param;
    ucp_proto_select_t *proto_select;
    ucs_status_t status;
    ucp_rkey_h rkey;

    ucs_assert((op_id >= UCP_OP_ID_RNDV_FIRST) &&
               (op_id < UCP_OP_ID_RNDV_LAST));

    if (rkey_length > 0) {
        ucs_assert(rkey_buffer != NULL);
        status = ucp_ep_rkey_unpack_reachable(ep, rkey_buffer, rkey_length,
                                              &rkey);
        if (status != UCS_OK) {
            goto err;
        }

        proto_select   = &ucp_rkey_config(worker, rkey)->proto_select;
        rkey_cfg_index = rkey->cfg_index;
    } else {
        /* No remote key, use endpoint protocols */
        proto_select   = &ucp_ep_config(ep)->proto_select;
        rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        rkey           = NULL;
    }

    ucp_proto_select_param_init(&sel_param, op_id, op_attr_mask, 0,
                                req->send.state.dt_iter.dt_class,
                                &req->send.state.dt_iter.mem_info, sg_count);

    status = UCS_PROFILE_CALL(ucp_proto_request_lookup_proto, worker, ep, req,
                              proto_select, rkey_cfg_index, &sel_param, length);
    if (status != UCS_OK) {
        goto err_destroy_rkey;
    }

    req->send.rndv.rkey = rkey;

    ucp_trace_req(req,
                  "%s rva 0x%" PRIx64 " length %zd rreq_id 0x%" PRIx64 " with protocol %s",
                  ucp_operation_names[op_id], req->send.rndv.remote_address,
                  length, req->send.rndv.remote_req_id,
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

void ucp_proto_rndv_receive_start(ucp_worker_h worker, ucp_request_t *recv_req,
                                  const ucp_rndv_rts_hdr_t *rts,
                                  const void *rkey_buffer, size_t rkey_length)
{
    ucp_operation_id_t op_id;
    ucs_status_t status;
    ucp_request_t *req;
    uint8_t sg_count;
    ucp_ep_h ep;

    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, rts->sreq.ep_id, return,
                                  "RTS on non-existing endpoint");

    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_error("failed to allocate rendezvous reply");
        return;
    }

    /* Initialize send request */
    req->flags                    = 0;
    req->send.ep                  = ep;
    req->send.rndv.remote_req_id  = rts->sreq.req_id;
    req->send.rndv.remote_address = rts->address;
    req->send.rndv.offset         = 0;
    ucp_request_set_super(req, recv_req);

    if (ucs_likely(rts->size <= recv_req->recv.length)) {
        ucp_proto_rndv_check_rkey_length(rts->address, rkey_length, "rts");
        op_id            = UCP_OP_ID_RNDV_RECV;
        recv_req->status = UCS_OK;
        UCS_PROFILE_CALL_VOID(ucp_datatype_iter_init_from_dt_state,
                              worker->context, recv_req->recv.buffer, rts->size,
                              recv_req->recv.datatype, &recv_req->recv.state,
                              &req->send.state.dt_iter, &sg_count);
    } else {
        /* Short receive: complete with error, and send reply to sender */
        rkey_length      = 0; /* Override rkey length to disable data fetch */
        op_id            = UCP_OP_ID_RNDV_RECV_DROP;
        recv_req->status = UCS_ERR_MESSAGE_TRUNCATED;
        ucp_request_recv_generic_dt_finish(recv_req);
        ucp_datatype_iter_init_null(&req->send.state.dt_iter, rts->size,
                                    &sg_count);
    }

    status = ucp_proto_rndv_send_reply(worker, req, op_id, 0, rts->size,
                                       rkey_buffer, rkey_length, sg_count);
    if (status != UCS_OK) {
        ucp_datatype_iter_cleanup(&req->send.state.dt_iter, UCP_DT_MASK_ALL);
        ucs_mpool_put(req);
        return;
    }

#if ENABLE_DEBUG_DATA
    recv_req->recv.proto_rndv_config = req->send.proto_config;
#endif

    UCS_PROFILE_CALL_VOID(ucp_request_send, req);
}

static ucs_status_t
ucp_proto_rndv_send_start(ucp_worker_h worker, ucp_request_t *req,
                          uint32_t op_attr_mask, const ucp_rndv_rtr_hdr_t *rtr,
                          size_t header_length, uint8_t sg_count)
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

    if (!ucp_proto_rndv_frag_complete(req, freq, "rdnv_send")) {
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
    ucp_request_t *req, *freq;
    ucs_status_t status;
    uint8_t sg_count;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rtr->sreq_id, 0, return UCS_OK,
                               "RTR %p", rtr);

    ucp_trace_req(req, "RTR offset %zu length %zu/%zu req %p", rtr->offset,
                  rtr->size, req->send.state.dt_iter.length, req);

    /* RTR covers the whole send request - use the send request directly */
    ucs_assert(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED);

    if (rtr->size == req->send.state.dt_iter.length) {
        /* RTR covers the whole send request - use the send request directly */
        ucs_assert(rtr->offset == 0);

        ucp_datatype_iter_mem_dereg(worker->context, &req->send.state.dt_iter,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
        ucp_send_request_id_release(req);
        req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;

        sg_count = req->send.proto_config->select_param.sg_count;
        status   = ucp_proto_rndv_send_start(worker, req, 0, rtr, length,
                                             sg_count);
        if (status != UCS_OK) {
            goto err_request_fail;
        }
    } else {
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
                                rtr->size, &freq->send.state.dt_iter,
                                &sg_count);

        /* Send rendezvous fragment, when it's completed update 'remaining'
         * and complete 'req' when it reaches zero
         * TODO can rndv/ppln be selected here (and not just single frag)?
         */
        status = ucp_proto_rndv_send_start(worker, freq,
                                           UCP_OP_ATTR_FLAG_MULTI_SEND, rtr,
                                           length, sg_count);
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
