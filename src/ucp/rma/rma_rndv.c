/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"
#include "rma_rndv.h"

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.inl>
#include <ucp/rndv/proto_rndv.inl>


#define UCP_PROTO_RMA_RNDV_RTS_NAME "RMA_RTS"


static void
ucp_rma_rndv_dt_iter_init(ucp_datatype_iter_t *dt_iter, uint64_t address,
                          size_t length, ucs_memory_type_t mem_type,
                          ucs_sys_device_t sys_dev)
{
    dt_iter->dt_class           = UCP_DATATYPE_CONTIG;
    dt_iter->mem_info.type      = mem_type;
    dt_iter->mem_info.sys_dev   = sys_dev;
    dt_iter->length             = length;
    dt_iter->offset             = 0;
    dt_iter->type.contig.buffer = (void*)(uintptr_t)address;
    dt_iter->type.contig.memh   = NULL;
}

static int
ucp_proto_rma_rndv_probe_check(const ucp_proto_init_params_t *init_params,
                               ucp_operation_id_t op_id)
{
    const ucp_proto_select_param_t *sel_param = init_params->select_param;

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(op_id)) ||
        ucp_proto_rndv_init_params_is_ppln_frag(init_params) ||
        (sel_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (init_params->rkey_config_key == NULL)) {
        return 0;
    }

    return !UCP_MEM_IS_HOST(sel_param->mem_type) ||
           !UCP_MEM_IS_HOST(init_params->rkey_config_key->mem_type);
}


static size_t ucp_proto_put_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *req          = arg;
    ucp_rma_rndv_rts_hdr_t *rts = dest;
    ucp_rkey_config_t *rkey_config;

    rkey_config = ucp_rkey_config(req->send.ep->worker, req->send.rma.rkey);

    rts->super.hdr    = 0;
    rts->super.opcode = UCP_RNDV_RTS_RMA;
    rts->address      = req->send.rma.remote_addr;
    rts->sys_dev      = rkey_config->key.sys_dev;
    rts->mem_type     = req->send.rma.rkey->mem_type;

    return ucp_proto_rndv_rts_pack(req, &rts->super, sizeof(*rts));
}

static ucs_status_t ucp_proto_put_rndv_init(ucp_request_t *req)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    int was_initialized;
    ucs_status_t status;

    was_initialized = req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    status          = ucp_proto_rndv_rts_request_init(req);
    if ((status != UCS_OK) || was_initialized) {
        return status;
    }

    /* Nested RNDV data protocols are not RMA protocols, so the wrapper handles
     * RMA fence ordering before exposing the operation to the peer. */
    return ucp_ep_rma_handle_fence(req->send.ep, req, UCS_BIT(rpriv->lane));
}

static ucs_status_t ucp_proto_put_rndv_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    size_t max_rts_size;
    ucs_status_t status;
    ucp_ep_h ep;

    status = ucp_proto_put_rndv_init(req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    ep           = req->send.ep;
    rpriv        = req->send.proto_config->priv;
    max_rts_size = sizeof(ucp_rma_rndv_rts_hdr_t) + rpriv->packed_rkey_size;

    /* Both the RNDV request and the remote completion must complete to unblock
     * flush, and they may complete in any order. */
    req->flags |= UCP_REQUEST_FLAG_RNDV_FLUSH;
    ucp_worker_flush_ops_count_add(ep->worker, +2);
    status = ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RNDV_RTS,
                                            rpriv->lane,
                                            ucp_proto_put_rndv_rts_pack, req,
                                            max_rts_size, 0);
    if (status != UCS_OK) {
        if (status == UCS_ERR_NO_RESOURCE) {
            req->send.lane = rpriv->lane;
        }
        goto err_flush_count;
    }

    ucp_ep_rma_remote_request_sent(ep);
    return UCS_OK;

err_flush_count:
    req->flags &= ~UCP_REQUEST_FLAG_RNDV_FLUSH;
    ucp_worker_flush_ops_count_add(ep->worker, -2);
    if (status == UCS_ERR_NO_RESOURCE) {
        return status;
    }

    ucp_proto_request_abort(req, status);
    return UCS_OK;
}

static void
ucp_proto_put_rndv_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rts,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 5,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_rma_rndv_rts_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        /* For performance modeling, this control protocol is followed on the
         * peer by a regular RNDV receive flow over the final RMA address. */
        .remote_op_id        = UCP_OP_ID_RNDV_RECV,
        .lane                = ucp_proto_rndv_find_ctrl_lane(init_params),
        .unpack_perf         = NULL,
        .perf_bias           = 0,
        .ctrl_msg_name       = UCP_PROTO_RMA_RNDV_RTS_NAME,
        .md_map              = 0
    };
    ucp_proto_rndv_ctrl_priv_t rpriv = {0};

    if (!ucp_proto_rma_rndv_probe_check(init_params, UCP_OP_ID_PUT)) {
        return;
    }

    ucp_proto_rndv_ctrl_probe(&params, &rpriv, sizeof(rpriv));
}

static void
ucp_rma_rndv_send_ats_err(ucp_ep_h ep, ucs_ptr_map_key_t remote_req_id,
                          ucs_status_t status)
{
    ucp_request_t *req;

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        ucs_error("failed to allocate RMA RNDV error ATS");
        return;
    }

    ucp_proto_request_send_init(req, ep, 0);
    ucp_rndv_req_send_ack(req, 0, remote_req_id, status, UCP_AM_ID_RNDV_ATS,
                          "send_ats_err");
}

static void ucp_proto_rma_rndv_query(const ucp_proto_query_params_t *params,
                                     ucp_proto_query_attr_t *attr,
                                     const char *desc)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t remote_attr;

    ucp_proto_config_query(params->worker, &rpriv->remote_proto_config,
                           params->msg_length, &remote_attr);

    attr->is_estimation  = 1;
    attr->max_msg_length = remote_attr.max_msg_length;
    attr->lane_map       = UCS_BIT(rpriv->lane) | remote_attr.lane_map;

    ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "%s using %s", desc,
                      remote_attr.desc);
    ucs_snprintf_safe(attr->config, sizeof(attr->config), "%s",
                      remote_attr.config);
}

static void
ucp_proto_put_rndv_query(const ucp_proto_query_params_t *params,
                         ucp_proto_query_attr_t *attr)
{
    ucp_proto_rma_rndv_query(params, attr, UCP_PROTO_RNDV_DESC);
}

static void
ucp_proto_get_rndv_query(const ucp_proto_query_params_t *params,
                         ucp_proto_query_attr_t *attr)
{
    ucp_proto_rma_rndv_query(params, attr, UCP_PROTO_RNDV_DESC);
}

static void
ucp_proto_get_rndv_add_variant(
        const ucp_proto_init_params_t *init_params,
        const ucp_proto_select_param_t *select_param,
        ucp_worker_cfg_index_t rkey_cfg_index, ucp_lane_index_t lane,
        ucp_proto_init_elem_t *proto, const void *proto_priv)
{
    ucp_context_h context = init_params->worker->context;
    const ucp_proto_perf_t *perf_elems[1];
    ucp_proto_rndv_ctrl_priv_t rpriv = {0};
    ucp_proto_init_params_t variant_params;
    UCS_STRING_BUFFER_ONSTACK(perf_name, 128);
    ucp_proto_perf_t *perf;
    size_t cfg_thresh;
    ucs_status_t status;

    perf_elems[0] = proto->perf;
    ucs_string_buffer_appendf(&perf_name, "%s" UCP_PROTO_PERF_NODE_NEW_LINE
                              "%s", UCP_PROTO_RNDV_RTR_REQ_NAME,
                              ucp_proto_perf_name(proto->perf));
    status = ucp_proto_perf_aggregate(ucs_string_buffer_cstr(&perf_name),
                                      perf_elems, 1, &perf);
    if (status != UCS_OK) {
        return;
    } else if (ucp_proto_perf_is_empty(perf)) {
        ucp_proto_perf_destroy(perf);
        return;
    }

    rpriv.lane = lane;

    variant_params                = *init_params;
    variant_params.rkey_cfg_index = rkey_cfg_index;
    ucp_proto_rndv_set_variant_config(&variant_params, proto, select_param,
                                      proto_priv, &rpriv.remote_proto_config);

    cfg_thresh = context->config.ext.zcopy_thresh;
    if (proto->cfg_thresh != UCS_MEMUNITS_AUTO) {
        cfg_thresh = proto->cfg_thresh;
    }

    ucp_proto_select_add_proto(init_params, cfg_thresh, 6, perf, &rpriv,
                               sizeof(rpriv));
}

static void
ucp_proto_get_rndv_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_worker_h worker                     = init_params->worker;
    const ucp_proto_select_init_protocols_t *proto_init;
    ucp_proto_select_param_t rndv_sel_param;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_elem_t *select_elem;
    ucp_proto_select_t *proto_select;
    ucp_proto_init_elem_t *proto;
    ucp_memory_info_t mem_info;
    ucp_lane_index_t lane;
    const void *priv;

    if (!ucp_proto_rma_rndv_probe_check(init_params, UCP_OP_ID_GET)) {
        return;
    }

    lane = ucp_proto_rndv_find_ctrl_lane(init_params);
    if (lane == UCP_NULL_LANE) {
        return;
    }

    mem_info = ucp_proto_common_select_param_mem_info(
            init_params->select_param);
    ucp_proto_select_param_init(&rndv_sel_param, UCP_OP_ID_RNDV_RECV, 0,
                                0, UCP_DATATYPE_CONTIG, &mem_info, 1);

    proto_select = ucp_proto_select_get(worker, init_params->ep_cfg_index,
                                        init_params->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return;
    }

    select_elem = ucp_proto_select_lookup_slow(worker, proto_select, 1,
                                               init_params->ep_cfg_index,
                                               rkey_cfg_index,
                                               &rndv_sel_param);
    if (select_elem == NULL) {
        return;
    }

    proto_init = &select_elem->proto_init;
    ucs_array_for_each(proto, &proto_init->protocols) {
        if (ucp_proto_id_field(proto->proto_id, flags) &
            UCP_PROTO_FLAG_INVALID) {
            continue;
        }

        priv = &ucs_array_elem(&proto_init->priv_buf, proto->priv_offset);
        ucp_proto_get_rndv_add_variant(init_params, &rndv_sel_param,
                                       rkey_cfg_index, lane, proto, priv);
    }
}

static void
ucp_proto_get_rndv_abort(ucp_request_t *req, ucs_status_t status)
{
    if (req->id != UCS_PTR_MAP_KEY_INVALID) {
        ucp_send_request_id_release(req);
    }

    if (req->send.state.dt_iter.dt_class != UCP_DATATYPE_CLASS_MASK) {
        ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                                  UCP_DT_MASK_ALL);
    }

    ucp_request_complete_send(req, status);
}

static ucs_status_t ucp_proto_get_rndv_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        if (req->id != UCS_PTR_MAP_KEY_INVALID) {
            ucp_send_request_id_release(req);
        }

        if (req->send.state.dt_iter.dt_class != UCP_DATATYPE_CLASS_MASK) {
            ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                                      UCP_DT_MASK_ALL);
        }
    }

    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    return UCS_OK;
}

ucp_request_t *ucp_rma_rndv_flush_open(ucp_request_t *rndv_req)
{
    ucp_request_t *recv_req = rndv_req;
    ucp_ep_h ep            = rndv_req->send.ep;

    if (!(rndv_req->flags & UCP_REQUEST_FLAG_RNDV_GET_REQ)) {
        return NULL;
    }

    while (!(recv_req->flags & UCP_REQUEST_FLAG_RNDV_RECV_INTERNAL)) {
        ucs_assert(recv_req->flags & UCP_REQUEST_FLAG_RNDV_GET_REQ);
        recv_req = ucp_request_get_super(recv_req);
    }

    if (recv_req->flags & UCP_REQUEST_FLAG_RNDV_START_FLUSH) {
        /* Account the RMA GET/RNDV nested transfer as an RMA remote operation,
         * so RMA flush waits for the internal receive request to complete.
         * Claim before issuing it, since SELF can complete inline. */
        recv_req->flags &= ~UCP_REQUEST_FLAG_RNDV_START_FLUSH;
        ucp_worker_flush_ops_count_add(ep->worker, +1);
        return recv_req;
    }

    return NULL;
}

void ucp_rma_rndv_flush_close(ucp_request_t *recv_req, ucp_ep_h ep,
                              ucs_status_t status)
{
    if (recv_req != NULL) {
        if (!UCS_STATUS_IS_ERR(status)) {
            /* recv_req may complete inline, so only touch ep on success. */
            ucp_ep_rma_remote_request_sent(ep);
        } else {
            ucp_worker_flush_ops_count_add(ep->worker, -1);
            recv_req->flags |= UCP_REQUEST_FLAG_RNDV_START_FLUSH;
        }
    }
}

static void ucp_rma_rndv_get_recv_complete(ucp_request_t *recv_req)
{
    ucp_request_t *get_req = ucp_request_get_super(recv_req);
    ucp_ep_h ep            = get_req->send.ep;
    int start_flush;

    start_flush = recv_req->flags & UCP_REQUEST_FLAG_RNDV_START_FLUSH;
    ucp_request_complete_send(get_req, recv_req->status);
    if (!start_flush) {
        ucp_ep_rma_remote_request_completed(ep);
    }
    ucp_request_put(recv_req);
}

static ucs_status_t ucp_proto_get_rndv_init(ucp_request_t *get_req,
                                            ucp_request_t **rndv_req_p)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = get_req->send.proto_config->priv;
    ucp_worker_h worker                     = get_req->send.ep->worker;
    ucp_request_t *recv_req;
    ucp_request_t *rndv_req;
    uint8_t UCS_V_UNUSED sg_count;
    ucs_status_t status;
    uint64_t address;
    size_t length;

    status = ucp_ep_rma_handle_fence(get_req->send.ep, get_req,
                                     UCS_BIT(rpriv->lane));
    if (status != UCS_OK) {
        return status;
    }

    address              = get_req->send.rma.remote_addr;
    length               = get_req->send.state.dt_iter.length;
    get_req->send.buffer =
            get_req->send.state.dt_iter.type.contig.buffer;
    get_req->send.length = length;

    recv_req = ucp_request_get(worker);
    if (recv_req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    rndv_req = ucp_request_get(worker);
    if (rndv_req == NULL) {
        ucp_request_put(recv_req);
        return UCS_ERR_NO_MEMORY;
    }

    get_req->flags                 |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    recv_req->flags                 = UCP_REQUEST_FLAG_RNDV_RECV_INTERNAL |
                                      UCP_REQUEST_FLAG_RNDV_START_FLUSH;
    recv_req->recv.worker           = worker;
    recv_req->recv.op_attr          = 0;
    recv_req->recv.remote_req_id    = UCS_PTR_MAP_KEY_INVALID;
    recv_req->recv.rndv.complete_cb = ucp_rma_rndv_get_recv_complete;
    recv_req->status                = UCS_OK;
    ucp_request_set_super(recv_req, get_req);

    UCS_PROFILE_CALL_VOID(ucp_datatype_iter_move, &recv_req->recv.dt_iter,
                          &get_req->send.state.dt_iter, length, &sg_count);

    ucp_proto_request_send_init(rndv_req, get_req->send.ep,
                                UCP_REQUEST_FLAG_RNDV_GET_REQ);
    ucp_request_set_super(rndv_req, recv_req);
    rndv_req->send.rndv.remote_req_id      = UCS_PTR_MAP_KEY_INVALID;
    rndv_req->send.rndv.remote_address     = address;
    rndv_req->send.rndv.rkey               = get_req->send.rma.rkey;
    rndv_req->send.rndv.offset             = 0;

    UCS_PROFILE_CALL_VOID(ucp_datatype_iter_move,
                          &rndv_req->send.state.dt_iter,
                          &recv_req->recv.dt_iter, length, &sg_count);
    ucp_proto_request_set_proto(rndv_req, &rpriv->remote_proto_config, length);

    *rndv_req_p = rndv_req;
    return UCS_OK;
}

static ucs_status_t ucp_proto_get_rndv_progress(uct_pending_req_t *self)
{
    ucp_request_t *get_req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    ucp_request_t *rndv_req;
    ucs_status_t status;

    if (get_req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        return UCS_OK;
    }

    rpriv              = get_req->send.proto_config->priv;
    get_req->send.lane = rpriv->lane;
    status             = ucp_ep_resolve_remote_id(get_req->send.ep,
                                                  rpriv->lane);
    if (status == UCS_ERR_NO_RESOURCE) {
        return status;
    } else if (status != UCS_OK) {
        ucp_proto_request_abort(get_req, status);
        return UCS_OK;
    }

    status = ucp_proto_get_rndv_init(get_req, &rndv_req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(get_req, status);
        return UCS_OK;
    }

    ucp_request_send(rndv_req);
    return UCS_OK;
}

static void ucp_rma_rndv_put_recv_complete(ucp_request_t *recv_req)
{
    ucp_worker_h worker = recv_req->recv.worker;
    ucp_ep_h ep;

    UCP_WORKER_GET_EP_BY_ID(&ep, worker, recv_req->recv.rndv.ep_id, {
        ucp_request_put(recv_req);
        return;
    }, "RMA RNDV PUT completion");

    ucp_rma_sw_send_cmpl(ep);
    if (recv_req->status != UCS_OK) {
        ucp_rma_rndv_send_ats_err(ep, recv_req->recv.remote_req_id,
                                  recv_req->status);
    }

    ucp_request_put(recv_req);
}

ucs_status_t ucp_rma_rndv_process_rts(ucp_worker_h worker,
                                      const ucp_rma_rndv_rts_hdr_t *rts,
                                      size_t length)
{
    const void *rkey_buffer;
    ucp_request_t *recv_req;
    ucp_ep_h ep;

    if (length < sizeof(*rts)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    recv_req = ucp_request_get(worker);
    if (recv_req == NULL) {
        ucs_error("failed to allocate RMA RNDV PUT receive request");
        UCP_WORKER_GET_EP_BY_ID(&ep, worker, rts->super.sreq.ep_id,
                                return UCS_OK, "RMA RNDV PUT error");
        ucp_rma_sw_send_cmpl(ep);
        ucp_rma_rndv_send_ats_err(ep, rts->super.sreq.req_id,
                                  UCS_ERR_NO_MEMORY);
        return UCS_OK;
    }

    recv_req->flags              = UCP_REQUEST_FLAG_RNDV_RECV_INTERNAL;
    recv_req->recv.worker        = worker;
    recv_req->recv.op_attr       = 0;
    recv_req->recv.remote_req_id = rts->super.sreq.req_id;
    ucp_rma_rndv_dt_iter_init(&recv_req->recv.dt_iter, rts->address,
                              rts->super.size, rts->mem_type, rts->sys_dev);
    recv_req->recv.rndv.ep_id                 = rts->super.sreq.ep_id;
    recv_req->recv.rndv.complete_cb           = ucp_rma_rndv_put_recv_complete;

    rkey_buffer = UCS_PTR_BYTE_OFFSET(rts, sizeof(*rts));
    ucp_proto_rndv_receive_start(worker, recv_req, &rts->super, rkey_buffer,
                                 length - sizeof(*rts));
    return UCS_OK;
}

ucp_proto_t ucp_put_rndv_proto = {
    .name     = "put/rndv",
    .desc     = UCP_PROTO_RNDV_DESC,
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_CONTIG),
    .probe    = ucp_proto_put_rndv_probe,
    .query    = ucp_proto_put_rndv_query,
    .progress = {ucp_proto_put_rndv_progress},
    .abort    = ucp_proto_rndv_rts_abort,
    .reset    = ucp_proto_rndv_rts_reset
};

ucp_proto_t ucp_get_rndv_proto = {
    .name     = "get/rndv",
    .desc     = UCP_PROTO_RNDV_DESC,
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_CONTIG),
    .probe    = ucp_proto_get_rndv_probe,
    .query    = ucp_proto_get_rndv_query,
    .progress = {ucp_proto_get_rndv_progress},
    .abort    = ucp_proto_get_rndv_abort,
    .reset    = ucp_proto_get_rndv_reset
};
