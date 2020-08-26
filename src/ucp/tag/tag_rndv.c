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


void ucp_tag_rndv_matched(ucp_worker_h worker, ucp_request_t *rreq,
                          const ucp_rndv_rts_hdr_t *rts_hdr)
{
    ucs_assert(rts_hdr->flags & UCP_RNDV_RTS_FLAG_TAG);

    /* rreq is the receive request on the receiver's side */
    rreq->recv.tag.info.sender_tag = rts_hdr->tag.tag;
    rreq->recv.tag.info.length     = rts_hdr->size;

    ucp_rndv_receive(worker, rreq, rts_hdr);
}

ucs_status_t ucp_tag_rndv_process_rts(ucp_worker_h worker,
                                      ucp_rndv_rts_hdr_t *rts_hdr,
                                      size_t length, unsigned tl_flags)
{
    ucp_recv_desc_t *rdesc;
    ucp_request_t *rreq;
    ucs_status_t status;

    ucs_assert(rts_hdr->flags & UCP_RNDV_RTS_FLAG_TAG);

    rreq = ucp_tag_exp_search(&worker->tm, rts_hdr->tag.tag);
    if (rreq != NULL) {
        ucp_tag_rndv_matched(worker, rreq, rts_hdr);

        /* Cancel req in transport if it was offloaded, because it arrived
           as unexpected */
        ucp_tag_offload_try_cancel(worker, rreq, UCP_TAG_OFFLOAD_CANCEL_FORCE);

        UCP_WORKER_STAT_RNDV(worker, EXP, 1);
        return UCS_OK;
    }

    ucs_assert(length >= sizeof(*rts_hdr));

    status = ucp_recv_desc_init(worker, rts_hdr, length, 0, tl_flags,
                                sizeof(*rts_hdr), UCP_RECV_DESC_FLAG_RNDV,
                                0, &rdesc);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucp_tag_unexp_recv(&worker->tm, rdesc, rts_hdr->tag.tag);
    }

    return status;
}

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq             = arg;
    ucp_rndv_rts_hdr_t *tag_rts_hdr = dest;

    tag_rts_hdr->tag.tag = sreq->send.msg_proto.tag.tag;

    return ucp_rndv_rts_pack(sreq, tag_rts_hdr, UCP_RNDV_RTS_FLAG_TAG);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    size_t packed_rkey_size;

    /* send the RTS. the pack_cb will pack all the necessary fields in the RTS */
    packed_rkey_size = ucp_ep_config(sreq->send.ep)->rndv.rkey_size;
    return ucp_do_am_single(self, UCP_AM_ID_RNDV_RTS, ucp_tag_rndv_rts_pack,
                            sizeof(ucp_rndv_rts_hdr_t) + packed_rkey_size);
}

ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *sreq)
{
    ucp_ep_h ep = sreq->send.ep;
    ucs_status_t status;

    ucp_trace_req(sreq, "start_rndv to %s buffer %p length %zu",
                  ucp_ep_peer_name(ep), sreq->send.buffer,
                  sreq->send.length);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    status = ucp_ep_resolve_remote_id(ep, sreq->send.lane);
    if (status != UCS_OK) {
        return status;
    }

    if (ucp_ep_is_tag_offload_enabled(ucp_ep_config(ep))) {
        status = ucp_tag_offload_start_rndv(sreq);
    } else {
        ucs_assert(sreq->send.lane == ucp_ep_get_am_lane(ep));
        sreq->send.uct.func = ucp_proto_progress_rndv_rts;
        status              = ucp_rndv_reg_send_buffer(sreq);
    }

    return status;
}

