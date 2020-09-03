/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"
#include "offload.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/proto/proto_am.inl>


/* packing start */

static UCS_F_ALWAYS_INLINE size_t
ucp_tag_pack_eager_common(ucp_request_t *req, void *dest,
                          size_t length, size_t hdr_length,
                          int UCS_V_UNUSED is_first)
{
    size_t packed_length;

    ucs_assert((length + hdr_length) <=
               ucp_ep_get_max_bcopy(req->send.ep, req->send.lane));
    ucs_assert(!is_first || (req->send.state.dt.offset == 0));

    packed_length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                                req->send.mem_type, dest, req->send.buffer,
                                &req->send.state.dt, length);
    return packed_length + hdr_length;
}

static size_t ucp_tag_pack_eager_only_dt(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;

    hdr->super.tag = req->send.msg_proto.tag.tag;

    return ucp_tag_pack_eager_common(req, hdr + 1, req->send.length,
                                     sizeof(*hdr), 1);
}

static size_t ucp_tag_pack_eager_sync_only_dt(void *dest, void *arg)
{
    ucp_eager_sync_hdr_t *hdr = dest;
    ucp_request_t *req = arg;

    hdr->super.super.tag = req->send.msg_proto.tag.tag;
    hdr->req.ep_id       = ucp_send_request_get_ep_remote_id(req);
    hdr->req.req_id      = ucp_send_request_get_id(req);

    return ucp_tag_pack_eager_common(req, hdr + 1, req->send.length,
                                     sizeof(*hdr), 1);
}

static size_t ucp_tag_pack_eager_first_dt(void *dest, void *arg)
{
    ucp_eager_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.lane == ucp_ep_get_am_lane(req->send.ep));

    length               = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                           sizeof(*hdr);
    length               = ucs_min(length, req->send.length);
    hdr->super.super.tag = req->send.msg_proto.tag.tag;
    hdr->total_len       = req->send.length;
    hdr->msg_id          = req->send.msg_proto.message_id;

    return ucp_tag_pack_eager_common(req, hdr + 1, length, sizeof(*hdr), 1);
}

static size_t ucp_tag_pack_eager_sync_first_dt(void *dest, void *arg)
{
    ucp_eager_sync_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.lane == ucp_ep_get_am_lane(req->send.ep));

    length                     = ucp_ep_get_max_bcopy(req->send.ep,
                                                      req->send.lane) -
                                 sizeof(*hdr);
    length                     = ucs_min(length, req->send.length);
    hdr->super.super.super.tag = req->send.msg_proto.tag.tag;
    hdr->super.total_len       = req->send.length;
    hdr->req.ep_id             = ucp_send_request_get_ep_remote_id(req);
    hdr->super.msg_id          = req->send.msg_proto.message_id;
    hdr->req.req_id            = ucp_send_request_get_id(req);

    return ucp_tag_pack_eager_common(req, hdr + 1, length, sizeof(*hdr), 1);
}

static size_t ucp_tag_pack_eager_middle_dt(void *dest, void *arg)
{
    ucp_eager_middle_hdr_t *hdr = dest;
    ucp_request_t *req          = arg;
    size_t length;

    length      = ucs_min(ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                          sizeof(*hdr),
                          req->send.length - req->send.state.dt.offset);
    hdr->msg_id = req->send.msg_proto.message_id;
    hdr->offset = req->send.state.dt.offset;

    return ucp_tag_pack_eager_common(req, hdr + 1, length, sizeof(*hdr), 0);
}

/* eager */

static ucs_status_t ucp_tag_eager_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);
    status = uct_ep_am_short(ep->uct_eps[req->send.lane], UCP_AM_ID_EAGER_ONLY,
                             req->send.msg_proto.tag.tag, req->send.buffer,
                             req->send.length);
    if (status != UCS_OK) {
        return status;
    }

    ucp_request_complete_send(req, UCS_OK);
    return UCS_OK;
}

static ucs_status_t ucp_tag_eager_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_EAGER_ONLY,
                                                 ucp_tag_pack_eager_only_dt);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 0, status);
}

static ucs_status_t ucp_tag_eager_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                ucp_tag_pack_eager_first_dt,
                                                ucp_tag_pack_eager_middle_dt, 1);

    return ucp_am_bcopy_handle_status_from_pending(self, 1, 0, status);
}

static ucs_status_t ucp_tag_eager_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_hdr_t hdr;

    hdr.super.tag = req->send.msg_proto.tag.tag;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_EAGER_ONLY, &hdr, sizeof(hdr),
                                  NULL, 0ul, ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_tag_eager_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_first_hdr_t first_hdr;
    ucp_eager_middle_hdr_t middle_hdr;

    first_hdr.super.super.tag = req->send.msg_proto.tag.tag;
    first_hdr.total_len       = req->send.length;
    first_hdr.msg_id          = req->send.msg_proto.message_id;
    middle_hdr.msg_id         = req->send.msg_proto.message_id;
    middle_hdr.offset         = req->send.state.dt.offset;

    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_EAGER_FIRST,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 &first_hdr, sizeof(first_hdr),
                                 &middle_hdr, sizeof(middle_hdr),
                                 NULL, 0ul, ucp_proto_am_zcopy_req_complete, 1);
}

ucs_status_t ucp_tag_send_start_rndv(uct_pending_req_t *self);

const ucp_request_send_proto_t ucp_tag_eager_proto = {
    .contig_short            = ucp_tag_eager_contig_short,
    .bcopy_single            = ucp_tag_eager_bcopy_single,
    .bcopy_multi             = ucp_tag_eager_bcopy_multi,
    .zcopy_single            = ucp_tag_eager_zcopy_single,
    .zcopy_multi             = ucp_tag_eager_zcopy_multi,
    .zcopy_completion        = ucp_proto_am_zcopy_completion,
    .only_hdr_size           = sizeof(ucp_eager_hdr_t)
};

/* eager sync */

void ucp_tag_eager_sync_completion(ucp_request_t *req, uint32_t flag,
                                   ucs_status_t status)
{
    static const uint16_t all_completed = UCP_REQUEST_FLAG_LOCAL_COMPLETED |
                                          UCP_REQUEST_FLAG_REMOTE_COMPLETED;

    ucs_assertv(!(req->flags & flag), "req->flags=%d flag=%d", req->flags, flag);
    req->flags |= flag;
    if (ucs_test_all_flags(req->flags, all_completed)) {
        ucp_request_complete_send(req, status);
    }
}

static ucs_status_t ucp_tag_eager_sync_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_EAGER_SYNC_ONLY,
                                                 ucp_tag_pack_eager_sync_only_dt);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 1, status);
}

static ucs_status_t ucp_tag_eager_sync_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_SYNC_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                ucp_tag_pack_eager_sync_first_dt,
                                                ucp_tag_pack_eager_middle_dt, 1);

    return ucp_am_bcopy_handle_status_from_pending(self, 1, 1, status);
}

void
ucp_tag_eager_sync_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucs_assert(req->send.state.uct_comp.count == 0);
    ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
    ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED,
                                  status);
}

void ucp_tag_eager_sync_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    ucs_status_t status = self->status;

    if (req->send.state.dt.offset == req->send.length) {
        ucp_tag_eager_sync_zcopy_req_complete(req, status);
    } else if (status != UCS_OK) {
        ucs_fatal("error handling is not supported with tag-sync protocol");
    }
}

static ucs_status_t ucp_tag_eager_sync_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_sync_hdr_t hdr;

    hdr.super.super.tag = req->send.msg_proto.tag.tag;
    hdr.req.ep_id       = ucp_send_request_get_ep_remote_id(req);
    hdr.req.req_id      = ucp_send_request_get_id(req);

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_EAGER_SYNC_ONLY, &hdr,
                                  sizeof(hdr), NULL, 0ul,
                                  ucp_tag_eager_sync_zcopy_req_complete);
}

static ucs_status_t ucp_tag_eager_sync_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_sync_first_hdr_t first_hdr;
    ucp_eager_middle_hdr_t middle_hdr;

    first_hdr.super.super.super.tag = req->send.msg_proto.tag.tag;
    first_hdr.super.total_len       = req->send.length;
    first_hdr.req.ep_id             = ucp_send_request_get_ep_remote_id(req);
    first_hdr.req.req_id            = ucp_send_request_get_id(req);
    first_hdr.super.msg_id          = req->send.msg_proto.message_id;
    middle_hdr.msg_id               = req->send.msg_proto.message_id;
    middle_hdr.offset               = req->send.state.dt.offset;

    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_EAGER_SYNC_FIRST,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 &first_hdr, sizeof(first_hdr),
                                 &middle_hdr, sizeof(middle_hdr), NULL, 0ul,
                                 ucp_tag_eager_sync_zcopy_req_complete, 1);
}

const ucp_request_send_proto_t ucp_tag_eager_sync_proto = {
    .contig_short            = NULL,
    .bcopy_single            = ucp_tag_eager_sync_bcopy_single,
    .bcopy_multi             = ucp_tag_eager_sync_bcopy_multi,
    .zcopy_single            = ucp_tag_eager_sync_zcopy_single,
    .zcopy_multi             = ucp_tag_eager_sync_zcopy_multi,
    .zcopy_completion        = ucp_tag_eager_sync_zcopy_completion,
    .only_hdr_size           = sizeof(ucp_eager_sync_hdr_t)
};

void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, void *hdr, uint16_t recv_flags)
{
    ucp_request_hdr_t *reqhdr;
    ucp_request_t *req;

    ucs_assert(recv_flags & UCP_RECV_DESC_FLAG_EAGER_SYNC);

    if (recv_flags & UCP_RECV_DESC_FLAG_EAGER_ONLY) {
        reqhdr = &((ucp_eager_sync_hdr_t*)hdr)->req;       /* only */
    } else {
        reqhdr = &((ucp_eager_sync_first_hdr_t*)hdr)->req; /* first */
    }

    if (recv_flags & UCP_RECV_DESC_FLAG_EAGER_OFFLOAD) {
        ucp_tag_offload_sync_send_ack(worker, reqhdr->ep_id,
                                      ((ucp_eager_sync_hdr_t*)hdr)->super.super.tag,
                                      recv_flags);
        return;
    }

    ucs_assert(reqhdr->req_id != UCP_REQUEST_ID_INVALID);
    req = ucp_proto_ssend_ack_request_alloc(worker, reqhdr->ep_id);
    if (req == NULL) {
        ucs_fatal("could not allocate request");
    }

    req->send.proto.am_id         = UCP_AM_ID_EAGER_SYNC_ACK;
    req->send.proto.remote_req_id = reqhdr->req_id;

    ucs_trace_req("send_sync_ack req %p ep %p", req, req->send.ep);

    ucp_request_send(req, 0);
}
