/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_am.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/tag/offload.h>


static inline size_t ucp_proto_max_packed_size()
{
    return ucs_max(sizeof(ucp_rndv_ack_hdr_t), sizeof(ucp_offload_ssend_hdr_t));
}

static size_t ucp_proto_pack(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    ucp_reply_hdr_t *rep_hdr;
    ucp_offload_ssend_hdr_t *off_rep_hdr;
    ucp_rndv_ack_hdr_t *ack_hdr;

    switch (req->send.proto.am_id) {
    case UCP_AM_ID_EAGER_SYNC_ACK:
        rep_hdr = dest;
        rep_hdr->req_id = req->send.proto.remote_req_id;
        rep_hdr->status = req->send.proto.status;
        return sizeof(*rep_hdr);
    case UCP_AM_ID_RNDV_ATS:
    case UCP_AM_ID_RNDV_ATP:
        ack_hdr               = dest;
        ack_hdr->super.req_id = req->send.proto.remote_req_id;
        ack_hdr->super.status = req->send.proto.status;
        ack_hdr->size         = req->send.length;
        return sizeof(*ack_hdr);
    case UCP_AM_ID_OFFLOAD_SYNC_ACK:
        off_rep_hdr = dest;
        off_rep_hdr->sender_tag = req->send.proto.sender_tag;
        off_rep_hdr->ep_id      = ucp_send_request_get_ep_remote_id(req);
        return sizeof(*off_rep_hdr);
    }

    ucs_fatal("unexpected am_id");
    return 0;
}

ucs_status_t
ucp_do_am_single(uct_pending_req_t *self, uint8_t am_id,
                 uct_pack_callback_t pack_cb, ssize_t max_packed_size)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;
    uint64_t *buffer;

    /* if packed data can fit short active message, use it, because it should
     * be faster than bcopy.
     */
    if ((max_packed_size <= UCS_ALLOCA_MAX_SIZE) &&
        (max_packed_size <= ucp_ep_config(ep)->am.max_short)) {
        req->send.lane = ucp_ep_get_am_lane(ep);
        buffer         = ucs_alloca(max_packed_size);
        packed_len     = pack_cb(buffer, req);
        ucs_assertv((packed_len >= 0) && (packed_len <= max_packed_size),
                    "packed_len=%zd max_packed_size=%zu", packed_len,
                    max_packed_size);

        return uct_ep_am_short(ep->uct_eps[req->send.lane], am_id, buffer[0],
                               &buffer[1], packed_len - sizeof(uint64_t));
    } else {
        return ucp_do_am_bcopy_single(self, am_id, pack_cb);
    }
}

ucs_status_t ucp_proto_progress_am_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    status = ucp_do_am_single(self, req->send.proto.am_id, ucp_proto_pack,
                              ucp_proto_max_packed_size());
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        return UCS_ERR_NO_RESOURCE;
    }

    /* TODO: handle failure */
    req->send.proto.comp_cb(req);
    return UCS_OK;
}

void ucp_proto_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(req->send.state.uct_comp.count == 0);
    ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
    ucp_request_complete_send(req, status);
}

void ucp_proto_am_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);

    if (req->send.state.dt.offset != req->send.length) {
        /* Cannot complete since not all fragments were posted yet */
        return;
    }

    ucp_proto_am_zcopy_req_complete(req, self->status);
}
