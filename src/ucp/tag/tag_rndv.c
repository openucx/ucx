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
                          const ucp_tag_rndv_rts_hdr_t *rts_hdr)
{
    ucs_assert(rts_hdr->super.flags & UCP_RNDV_RTS_FLAG_TAG);

    /* rreq is the receive request on the receiver's side */
    rreq->recv.tag.info.sender_tag = rts_hdr->tag.tag;
    rreq->recv.tag.info.length     = rts_hdr->super.size;

    ucp_rndv_receive(worker, rreq, &rts_hdr->super, rts_hdr + 1);
}

ucs_status_t ucp_tag_rndv_process_rts(ucp_worker_h worker,
                                      ucp_rndv_rts_hdr_t *common_rts_hdr,
                                      size_t length, unsigned tl_flags)
{
    ucp_tag_rndv_rts_hdr_t *rts_hdr = ucs_derived_of(common_rts_hdr,
                                                     ucp_tag_rndv_rts_hdr_t);
    ucp_recv_desc_t *rdesc;
    ucp_tag_t *rdesc_hdr;
    ucp_request_t *rreq;
    ucs_status_t status;

    ucs_assert(rts_hdr->super.flags & UCP_RNDV_RTS_FLAG_TAG);

    rreq = ucp_tag_exp_search(&worker->tm, rts_hdr->tag.tag);
    if (rreq != NULL) {
        /* Cancel req in transport if it was offloaded, because it arrived
           as unexpected */
        ucp_tag_offload_try_cancel(worker, rreq, UCP_TAG_OFFLOAD_CANCEL_FORCE);
        ucp_tag_rndv_matched(worker, rreq, rts_hdr);

        UCP_WORKER_STAT_RNDV(worker, EXP, 1);
        return UCS_OK;
    }

    ucs_assert(length >= sizeof(*rts_hdr));

    /* Include tag before the header as well, to keep ucp_rdesc_get_tag() fast
     * (and therefore keep fast search by ucp_tag_unexp_search())
     */
    status = ucp_recv_desc_init(worker, rts_hdr, length, sizeof(*rdesc_hdr),
                                tl_flags, sizeof(*rts_hdr) + sizeof(*rdesc_hdr),
                                UCP_RECV_DESC_FLAG_RNDV,
                                sizeof(*rdesc_hdr), &rdesc);
    if (!UCS_STATUS_IS_ERR(status)) {
        rdesc_hdr  = (ucp_tag_t*)(rdesc + 1);
        *rdesc_hdr = rts_hdr->tag.tag;
        ucp_tag_unexp_recv(&worker->tm, rdesc, rts_hdr->tag.tag);
    }

    return status;
}

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq                 = arg;
    ucp_tag_rndv_rts_hdr_t *tag_rts_hdr = dest;

    tag_rts_hdr->tag.tag = sreq->send.msg_proto.tag.tag;

    return ucp_rndv_rts_pack(sreq, &tag_rts_hdr->super, sizeof(*tag_rts_hdr),
                             UCP_RNDV_RTS_FLAG_TAG);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    size_t packed_rkey_size;

    /* send the RTS. the pack_cb will pack all the necessary fields in the RTS */
    packed_rkey_size = ucp_ep_config(sreq->send.ep)->rndv.rkey_size;
    return ucp_do_am_single(self, UCP_AM_ID_RNDV_RTS, ucp_tag_rndv_rts_pack,
                            sizeof(ucp_tag_rndv_rts_hdr_t) + packed_rkey_size);
}

ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *sreq)
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

    ucp_send_request_set_id(sreq);

    if (ucp_ep_is_tag_offload_enabled(ucp_ep_config(ep))) {
        status = ucp_tag_offload_start_rndv(sreq);
    } else {
        ucs_assert(sreq->send.lane == ucp_ep_get_am_lane(ep));
        sreq->send.uct.func = ucp_proto_progress_rndv_rts;
        status              = ucp_rndv_reg_send_buffer(sreq);
    }

    return status;
}

