/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "proto.h"
#include "proto_am.inl"

static size_t ucp_proto_pack(void *dest, void *arg)
{
    ucp_reply_hdr_t *rep_hdr = dest;
    ucp_request_t *req = arg;

    switch (req->send.proto.am_id) {
    case UCP_AM_ID_EAGER_SYNC_ACK:
        rep_hdr->reqptr = req->send.proto.remote_request;
        rep_hdr->status  = req->send.proto.status;
        return sizeof(*rep_hdr);
    }

    ucs_bug("unexpected am_id");
    return 0;
}

ucs_status_t ucp_proto_progress_am_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status;
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    status = ucp_do_am_bcopy_single(self, req->send.proto.am_id,
                                    ucp_proto_pack);
    if (status == UCS_OK) {
        ucs_mpool_put(req);
    }
    return status;
}

