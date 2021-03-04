/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_RNDV_INL_
#define UCP_PROTO_RNDV_INL_

#include "proto_rndv.h"

#include <ucp/core/ucp_rkey.inl>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_single.inl>
#include <ucp/proto/proto_multi.inl>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_rts_request_init(ucp_request_t *req)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    ucp_ep_h ep                             = req->send.ep;
    ucs_status_t status;

    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        return UCS_OK;
    }

    status = ucp_ep_resolve_remote_id(req->send.ep, rpriv->lane);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_datatype_iter_mem_reg(ep->worker->context,
                                       &req->send.state.dt_iter, rpriv->md_map,
                                       UCT_MD_MEM_ACCESS_RMA |
                                       UCT_MD_MEM_FLAG_HIDE_ERRORS);
    if (status != UCS_OK) {
        return status;
    }

    req->send.msg_proto.sreq_id = ucp_send_request_get_id(req);
    req->flags                 |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_ats_handler(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker        = arg;
    const ucp_reply_hdr_t *ats = data;
    ucp_request_t *req;

    UCP_WORKER_EXTRACT_REQUEST_BY_ID(&req, worker, ats->req_id, return UCS_OK,
                                     "ATS %p", ats);
    ucp_proto_request_zcopy_complete(req, ats->status);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE size_t ucp_proto_rndv_rts_pack(
        ucp_request_t *req, ucp_rndv_rts_hdr_t *rts, size_t hdr_len)
{
    void *rkey_buffer = UCS_PTR_BYTE_OFFSET(rts, hdr_len);
    size_t rkey_size;

    rts->sreq.req_id = req->send.msg_proto.sreq_id;
    rts->sreq.ep_id  = ucp_send_request_get_ep_remote_id(req);
    rts->size        = req->send.state.dt_iter.length;

    if (req->send.state.dt_iter.type.contig.reg.md_map == 0) {
        rts->address = 0;
        rkey_size    = 0;
    } else {
        rts->address = (uintptr_t)req->send.state.dt_iter.type.contig.buffer;
        rkey_size    = ucp_proto_request_pack_rkey(req, rkey_buffer);
    }

    return hdr_len + rkey_size;
}

static ucs_status_t UCS_F_ALWAYS_INLINE
ucp_proto_rndv_ack_progress(ucp_request_t *req, ucp_am_id_t am_id,
                            ucp_proto_complete_cb_t complete_func)
{
    const ucp_proto_rndv_ack_priv_t *apriv = req->send.proto_config->priv;

    ucs_assert(ucp_datatype_iter_is_end(&req->send.state.dt_iter));

    return ucp_proto_am_bcopy_single_progress(req, am_id, apriv->lane,
                                              ucp_proto_rndv_pack_ack, req,
                                              sizeof(ucp_reply_hdr_t),
                                              complete_func);
}

#endif
