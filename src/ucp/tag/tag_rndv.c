/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tag_rndv.h"
#include "tag_match.inl"

#include <ucp/proto/proto_single.inl>
#include <ucp/rndv/proto_rndv.inl>
#include <ucp/rndv/rndv.inl>


void ucp_tag_rndv_matched(ucp_worker_h worker, ucp_request_t *rreq,
                          const ucp_rndv_rts_hdr_t *rts_hdr, size_t hdr_length)
{
    /* rreq is the receive request on the receiver's side */
    ucs_assert(ucp_rndv_rts_is_tag(rts_hdr));
    rreq->recv.tag.info.sender_tag = ucp_tag_hdr_from_rts(rts_hdr)->tag;
    rreq->recv.tag.info.length     = rts_hdr->size;

    ucp_rndv_receive_start(worker, rreq, rts_hdr, rts_hdr + 1,
                           hdr_length - sizeof(*rts_hdr));
}

ucs_status_t ucp_tag_rndv_process_rts(ucp_worker_h worker,
                                      ucp_rndv_rts_hdr_t *rts_hdr,
                                      size_t length, unsigned tl_flags)
{
    ucp_recv_desc_t *rdesc;
    ucp_request_t *rreq;
    ucs_status_t status;

    ucs_assert(ucp_rndv_rts_is_tag(rts_hdr));

    rreq = ucp_tag_exp_search(&worker->tm, ucp_tag_hdr_from_rts(rts_hdr)->tag);
    if (rreq != NULL) {
        /* Cancel req in transport if it was offloaded, because it arrived
           as unexpected */
        ucp_tag_offload_try_cancel(worker, rreq, UCP_TAG_OFFLOAD_CANCEL_FORCE);
        ucp_tag_rndv_matched(worker, rreq, rts_hdr, length);

        UCP_WORKER_STAT_RNDV(worker, EXP, 1);
        return UCS_OK;
    }

    ucs_assert(length >= sizeof(*rts_hdr));

    status = ucp_recv_desc_init(worker, rts_hdr, length, 0, tl_flags,
                                sizeof(*rts_hdr), UCP_RECV_DESC_FLAG_RNDV, 0, 1,
                                "tag_rndv_process_rts", &rdesc);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucs_assert(ucp_rdesc_get_tag(rdesc) ==
                   ucp_tag_hdr_from_rts(rts_hdr)->tag);
        ucp_tag_unexp_recv(&worker->tm, rdesc,
                           ucp_tag_hdr_from_rts(rts_hdr)->tag);
    }

    return status;
}

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq         = arg;
    ucp_rndv_rts_hdr_t *rts_hdr = dest;

    ucp_tag_hdr_from_rts(rts_hdr)->tag = sreq->send.msg_proto.tag;

    return ucp_rndv_rts_pack(sreq, rts_hdr, UCP_RNDV_RTS_TAG_OK);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_tag_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_rndv_send_rts(sreq, ucp_tag_rndv_rts_pack,
                             sizeof(ucp_rndv_rts_hdr_t));
}

ucs_status_t
ucp_tag_send_start_rndv(ucp_request_t *sreq, const ucp_request_param_t *param)
{
    ucp_ep_h ep = sreq->send.ep;
    ucs_status_t status;

    ucp_trace_req(sreq, "start_rndv to %s buffer %p length %zu mem_type:%s",
                  ucp_ep_peer_name(ep), sreq->send.buffer,
                  sreq->send.length, ucs_memory_type_names[sreq->send.mem_type]);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    status = ucp_ep_resolve_remote_id(ep, sreq->send.lane);
    if (status != UCS_OK) {
        return status;
    }

    ucp_send_request_id_alloc(sreq);

    if (ucp_ep_config_key_has_tag_lane(&ucp_ep_config(ep)->key)) {
        status = ucp_tag_offload_start_rndv(sreq, param);
    } else {
        ucs_assert(sreq->send.lane == ucp_ep_get_am_lane(ep));
        sreq->send.uct.func = ucp_proto_progress_tag_rndv_rts;
        status              = ucp_rndv_reg_send_buffer(sreq, param);
    }

    return status;
}

static size_t ucp_tag_rndv_proto_rts_pack(void *dest, void *arg)
{
    ucp_rndv_rts_hdr_t *tag_rts = dest;
    ucp_request_t *req          = arg;

    tag_rts->opcode                    = UCP_RNDV_RTS_TAG_OK;
    ucp_tag_hdr_from_rts(tag_rts)->tag = req->send.msg_proto.tag;

    return ucp_proto_rndv_rts_pack(req, tag_rts, sizeof(*tag_rts));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_rndv_rts_progress, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    size_t max_rts_size;
    ucs_status_t status;

    rpriv        = req->send.proto_config->priv;
    max_rts_size = sizeof(ucp_rndv_rts_hdr_t) + rpriv->packed_rkey_size;

    status = UCS_PROFILE_CALL(ucp_proto_rndv_rts_request_init, req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    return UCS_PROFILE_CALL(ucp_proto_am_bcopy_single_progress, req,
                            UCP_AM_ID_RNDV_RTS, rpriv->lane,
                            ucp_tag_rndv_proto_rts_pack, req, max_rts_size,
                            NULL);
}

ucs_status_t ucp_tag_rndv_rts_init(const ucp_proto_init_params_t *init_params)
{
    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_TAG_SEND);
    if (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_rts_init(init_params);
}

ucp_proto_t ucp_tag_rndv_proto = {
    .name     = "tag/rndv",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_tag_rndv_rts_init,
    .query    = ucp_proto_rndv_rts_query,
    .progress = {ucp_tag_rndv_rts_progress},
    .abort    = ucp_proto_rndv_rts_abort
};
