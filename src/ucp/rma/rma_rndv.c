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

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.inl>
#include <ucp/rndv/proto_rndv.inl>


#define UCP_PROTO_RMA_RNDV_PUT_RTS_NAME "RMA PUT RTS"
#define UCP_PROTO_RMA_RNDV_PUT_DESC     "RMA PUT rendezvous"


enum {
    UCP_RMA_RNDV_AM_PUT_RTS
};


typedef struct {
    ucp_rndv_rts_hdr_t super;
    uint64_t           address;
    ucs_sys_device_t   sys_dev;
    ucs_memory_type_t  mem_type;
} UCS_S_PACKED ucp_rma_rndv_put_rts_hdr_t;



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

static size_t ucp_proto_put_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *req              = arg;
    ucp_rma_rndv_put_rts_hdr_t *rts = dest;
    ucp_rkey_config_t *rkey_config;

    rkey_config = ucp_rkey_config(req->send.ep->worker, req->send.rma.rkey);

    rts->super.hdr    = UCP_RMA_RNDV_AM_PUT_RTS;
    rts->super.opcode = UCP_RNDV_RTS_TAG_OK;
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
    max_rts_size = sizeof(ucp_rma_rndv_put_rts_hdr_t) +
                   rpriv->packed_rkey_size;

    ucp_worker_flush_ops_count_add(ep->worker, +1);
    status = ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RMA_RNDV,
                                            rpriv->lane,
                                            ucp_proto_put_rndv_rts_pack, req,
                                            max_rts_size, 0);
    if (status == UCS_ERR_NO_RESOURCE) {
        ucp_worker_flush_ops_count_add(ep->worker, -1);
        req->send.lane = rpriv->lane;
        return status;
    } else if (status != UCS_OK) {
        ucp_worker_flush_ops_count_add(ep->worker, -1);
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    ucp_ep_rma_remote_request_sent(ep);
    return UCS_OK;
}

static void
ucp_proto_put_rndv_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    const ucp_proto_select_param_t *sel_param = init_params->select_param;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rts,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 5,
        .super.min_length    = 1,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_rma_rndv_put_rts_hdr_t),
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
        .ctrl_msg_name       = UCP_PROTO_RMA_RNDV_PUT_RTS_NAME,
        .md_map              = 0
    };
    ucp_proto_rndv_ctrl_priv_t rpriv;

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT)) ||
        (sel_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (init_params->rkey_config_key == NULL)) {
        return;
    }

    if (UCP_MEM_IS_HOST(sel_param->mem_type) &&
        UCP_MEM_IS_HOST(init_params->rkey_config_key->mem_type)) {
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
    attr->lane_map       = UCS_BIT(rpriv->lane);

    ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "%s using %s", desc,
                      remote_attr.desc);
    ucs_snprintf_safe(attr->config, sizeof(attr->config), "ctrl lane %u, %s",
                      rpriv->lane, remote_attr.config);
}

static void ucp_proto_put_rndv_query(const ucp_proto_query_params_t *params,
                                     ucp_proto_query_attr_t *attr)
{
    ucp_proto_rma_rndv_query(params, attr, UCP_PROTO_RMA_RNDV_PUT_DESC);
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

static ucs_status_t
ucp_rma_rndv_handle_put_rts(ucp_worker_h worker, void *data, size_t length)
{
    const ucp_rma_rndv_put_rts_hdr_t *rts = data;
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

UCS_PROFILE_FUNC(ucs_status_t, ucp_rma_rndv_handler,
                 (arg, data, length, am_flags), void *arg, void *data,
                 size_t length, unsigned am_flags)
{
    const ucp_rndv_rts_hdr_t *hdr = data;
    ucp_worker_h worker           = arg;

    if (length < sizeof(*hdr)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (hdr->hdr) {
    case UCP_RMA_RNDV_AM_PUT_RTS:
        return ucp_rma_rndv_handle_put_rts(worker, data, length);
    default:
        ucs_debug("unexpected RMA RNDV AM sub-id %" PRIu64, hdr->hdr);
        return UCS_ERR_UNSUPPORTED;
    }
}

static void
ucp_rma_rndv_dump_packet(ucp_worker_h worker, uct_am_trace_type_t type,
                         uint8_t id, const void *data, size_t length,
                         char *buffer, size_t max)
{
    const ucp_rma_rndv_put_rts_hdr_t *put_rts = data;
    const ucp_rndv_rts_hdr_t *hdr             = data;

    if (length < sizeof(*hdr)) {
        return;
    }

    switch (hdr->hdr) {
    case UCP_RMA_RNDV_AM_PUT_RTS:
        if (length < sizeof(*put_rts)) {
            return;
        }

        snprintf(buffer, max, "RMA_PUT_RTS [src 0x%" PRIx64
                 " dst 0x%" PRIx64 " len %zu req_id 0x%" PRIx64
                 " ep_id 0x%" PRIx64 " %s]", put_rts->super.address,
                 put_rts->address, put_rts->super.size,
                 put_rts->super.sreq.req_id, put_rts->super.sreq.ep_id,
                 ucs_memory_type_names[put_rts->mem_type]);
        break;
    default:
        snprintf(buffer, max, "RMA_RNDV [sub-id %" PRIu64 "]", hdr->hdr);
        break;
    }
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_RMA, UCP_AM_ID_RMA_RNDV,
                         ucp_rma_rndv_handler, ucp_rma_rndv_dump_packet, 0);

ucp_proto_t ucp_put_rndv_proto = {
    .name     = "put/rndv",
    .desc     = UCP_PROTO_RMA_RNDV_PUT_DESC,
    .flags    = 0,
    .probe    = ucp_proto_put_rndv_probe,
    .query    = ucp_proto_put_rndv_query,
    .progress = {ucp_proto_put_rndv_progress},
    .abort    = ucp_proto_rndv_rts_abort,
    .reset    = ucp_proto_rndv_rts_reset
};
