/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "proto.h"
#include "proto_am.inl"

#include <ucp/tag/offload.h>


static size_t ucp_proto_pack(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    ucp_reply_hdr_t *rep_hdr;
    ucp_offload_ssend_hdr_t *off_rep_hdr;

    switch (req->send.proto.am_id) {
    case UCP_AM_ID_EAGER_SYNC_ACK:
    case UCP_AM_ID_RNDV_ATS:
    case UCP_AM_ID_RNDV_ATP:
        rep_hdr = dest;
        rep_hdr->reqptr = req->send.proto.remote_request;
        rep_hdr->status = req->send.proto.status;
        return sizeof(*rep_hdr);
    case UCP_AM_ID_OFFLOAD_SYNC_ACK:
        off_rep_hdr = dest;
        off_rep_hdr->sender_tag  = req->send.proto.sender_tag;
        off_rep_hdr->sender_uuid = req->send.proto.sender_uuid;
        return sizeof(*off_rep_hdr);
    }

    ucs_bug("unexpected am_id");
    return 0;
}

ucs_status_t ucp_proto_progress_am_bcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    ucs_status_t status = ucp_do_am_bcopy_single(self, req->send.proto.am_id,
                                                 ucp_proto_pack);
    if (status == UCS_OK) {
        req->send.proto.comp_cb(req);
    }
    return status;
}

void ucp_proto_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(req->send.state.uct_comp.count == 0);
    ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
    ucp_request_complete_send(req, status);
}

void ucp_proto_am_zcopy_completion(uct_completion_t *self,
                                    ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    if (req->send.state.dt.offset == req->send.length) {
        ucp_proto_am_zcopy_req_complete(req, status);
    } else if (status != UCS_OK) {
        ucs_assert(req->send.state.uct_comp.count == 0);
        ucs_assert(status != UCS_INPROGRESS);

        /* NOTE: the request is in pending queue if data was not completely sent,
         *       just dereg the buffer here and complete request on purge
         *       pending later.
         */
        ucp_request_send_buffer_dereg(req);
        req->send.state.uct_comp.func = NULL;
    }
}
