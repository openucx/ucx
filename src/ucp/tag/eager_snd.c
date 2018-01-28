/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"
#include "offload.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_am.inl>


/* packing  start */

static size_t ucp_tag_pack_eager_only_dt(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    hdr->super.tag = req->send.tag.tag;

    ucs_assert(req->send.state.dt.offset == 0);
    length = ucp_dt_pack(req->send.datatype, hdr + 1, req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_sync_only_dt(void *dest, void *arg)
{
    ucp_eager_sync_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    hdr->super.super.tag = req->send.tag.tag;
    hdr->req.sender_uuid = req->send.ep->worker->uuid;
    hdr->req.reqptr      = (uintptr_t)req;

    ucs_assert(req->send.state.dt.offset == 0);
    length = ucp_dt_pack(req->send.datatype, hdr + 1, req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_first_dt(void *dest, void *arg)
{
    ucp_eager_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.lane == ucp_ep_get_am_lane(req->send.ep));

    length               = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                           sizeof(*hdr);
    hdr->super.super.tag = req->send.tag.tag;
    hdr->total_len       = req->send.length;
    hdr->msg_id          = req->send.tag.message_id;

    ucs_assert(req->send.state.dt.offset == 0);
    ucs_assert(req->send.length > length);
    return sizeof(*hdr) + ucp_dt_pack(req->send.datatype, hdr + 1,
                                      req->send.buffer, &req->send.state.dt,
                                      length); 
}

static size_t ucp_tag_pack_eager_sync_first_dt(void *dest, void *arg)
{
    ucp_eager_sync_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.lane == ucp_ep_get_am_lane(req->send.ep));

    length                        = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                                    sizeof(*hdr);
    hdr->super.super.super.tag    = req->send.tag.tag;
    hdr->super.total_len          = req->send.length;
    hdr->req.sender_uuid          = req->send.ep->worker->uuid;
    hdr->super.msg_id             = req->send.tag.message_id;
    hdr->req.reqptr               = (uintptr_t)req;

    ucs_assert(req->send.state.dt.offset == 0);
    ucs_assert(req->send.length > length);
    return sizeof(*hdr) + ucp_dt_pack(req->send.datatype, hdr + 1,
                                      req->send.buffer, &req->send.state.dt,
                                      length);
}

static size_t ucp_tag_pack_eager_middle_dt(void *dest, void *arg)
{
    ucp_eager_middle_hdr_t *hdr = dest;
    ucp_request_t *req          = arg;
    size_t length;

    length          = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                      sizeof(*hdr);
    hdr->msg_id     = req->send.tag.message_id;
    hdr->offset     = req->send.state.dt.offset;
    return sizeof(*hdr) + ucp_dt_pack(req->send.datatype, hdr + 1,
                                      req->send.buffer, &req->send.state.dt,
                                      length);
}

static size_t ucp_tag_pack_eager_last_dt(void *dest, void *arg)
{
    ucp_eager_middle_hdr_t *hdr = dest;
    ucp_request_t *req          = arg;
    size_t length, ret_length;

    length         = req->send.length - req->send.state.dt.offset;
    hdr->msg_id    = req->send.tag.message_id;
    hdr->offset    = req->send.state.dt.offset;
    ret_length     = ucp_dt_pack(req->send.datatype, hdr + 1, req->send.buffer,
                                 &req->send.state.dt, length);
    ucs_assertv(ret_length == length, "length=%zu, max_length=%zu",
                ret_length, length);
    return sizeof(*hdr) + ret_length;
}

/* eager */

static ucs_status_t ucp_tag_eager_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);
    status = uct_ep_am_short(ep->uct_eps[req->send.lane], UCP_AM_ID_EAGER_ONLY,
                             req->send.tag.tag, req->send.buffer, req->send.length);
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
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    }
    return status;
}

static ucs_status_t ucp_tag_eager_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                sizeof(ucp_eager_middle_hdr_t),
                                                ucp_tag_pack_eager_first_dt,
                                                ucp_tag_pack_eager_middle_dt,
                                                ucp_tag_pack_eager_last_dt, 1);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }
    return status;
}

static ucs_status_t ucp_tag_eager_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_hdr_t hdr;

    hdr.super.tag = req->send.tag.tag;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_EAGER_ONLY, &hdr, sizeof(hdr),
                                  ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_tag_eager_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_first_hdr_t first_hdr;
    ucp_eager_middle_hdr_t middle_hdr;

    first_hdr.super.super.tag = req->send.tag.tag;
    first_hdr.total_len       = req->send.length;
    first_hdr.msg_id          = req->send.tag.message_id;
    middle_hdr.msg_id         = req->send.tag.message_id;
    middle_hdr.offset         = req->send.state.dt.offset;

    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_EAGER_FIRST,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 &first_hdr, sizeof(first_hdr),
                                 &middle_hdr, sizeof(middle_hdr),
                                 ucp_proto_am_zcopy_req_complete, 1);
}

ucs_status_t ucp_tag_send_start_rndv(uct_pending_req_t *self);

const ucp_proto_t ucp_tag_eager_proto = {
    .contig_short            = ucp_tag_eager_contig_short,
    .bcopy_single            = ucp_tag_eager_bcopy_single,
    .bcopy_multi             = ucp_tag_eager_bcopy_multi,
    .zcopy_single            = ucp_tag_eager_zcopy_single,
    .zcopy_multi             = ucp_tag_eager_zcopy_multi,
    .zcopy_completion        = ucp_proto_am_zcopy_completion,
    .only_hdr_size           = sizeof(ucp_eager_hdr_t),
    .first_hdr_size          = sizeof(ucp_eager_first_hdr_t),
    .mid_hdr_size            = sizeof(ucp_eager_hdr_t)
};

/* eager sync */

void ucp_tag_eager_sync_completion(ucp_request_t *req, uint16_t flag,
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
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED,
                                      UCS_OK);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }
    return status;
}

static ucs_status_t ucp_tag_eager_sync_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_SYNC_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                sizeof(ucp_eager_middle_hdr_t),
                                                ucp_tag_pack_eager_sync_first_dt,
                                                ucp_tag_pack_eager_middle_dt,
                                                ucp_tag_pack_eager_last_dt, 1);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED,
                                      UCS_OK);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }
    return status;
}

void
ucp_tag_eager_sync_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    if (req->send.state.dt.offset == req->send.length) {
        ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED,
                                      status);
    } else if (status != UCS_OK) {
        ucs_fatal("error handling is not supported with tag-sync protocol");
    }
}

static ucs_status_t ucp_tag_eager_sync_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_sync_hdr_t hdr;

    hdr.super.super.tag = req->send.tag.tag;
    hdr.req.sender_uuid = req->send.ep->worker->uuid;
    hdr.req.reqptr      = (uintptr_t)req;

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_EAGER_SYNC_ONLY, &hdr, sizeof(hdr),
                                  ucp_tag_eager_sync_zcopy_req_complete);
}

static ucs_status_t ucp_tag_eager_sync_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_sync_first_hdr_t first_hdr;
    ucp_eager_middle_hdr_t middle_hdr;

    first_hdr.super.super.super.tag = req->send.tag.tag;
    first_hdr.super.total_len       = req->send.length;
    first_hdr.req.sender_uuid       = req->send.ep->worker->uuid;
    first_hdr.req.reqptr            = (uintptr_t)req;
    first_hdr.super.msg_id          = req->send.tag.message_id;
    middle_hdr.msg_id               = req->send.tag.message_id;
    middle_hdr.offset               = req->send.state.dt.offset;

    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_EAGER_SYNC_FIRST,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 &first_hdr, sizeof(first_hdr),
                                 &middle_hdr, sizeof(middle_hdr),
                                 ucp_tag_eager_sync_zcopy_req_complete, 1);
}

void ucp_tag_eager_sync_zcopy_completion(uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *req;

    req = ucs_container_of(self, ucp_request_t, send.state.uct_comp);
    ucp_tag_eager_sync_zcopy_req_complete(req, status);
}

const ucp_proto_t ucp_tag_eager_sync_proto = {
    .contig_short            = NULL,
    .bcopy_single            = ucp_tag_eager_sync_bcopy_single,
    .bcopy_multi             = ucp_tag_eager_sync_bcopy_multi,
    .zcopy_single            = ucp_tag_eager_sync_zcopy_single,
    .zcopy_multi             = ucp_tag_eager_sync_zcopy_multi,
    .zcopy_completion        = ucp_tag_eager_sync_zcopy_completion,
    .only_hdr_size           = sizeof(ucp_eager_sync_hdr_t),
    .first_hdr_size          = sizeof(ucp_eager_sync_first_hdr_t),
    .mid_hdr_size            = sizeof(ucp_eager_hdr_t)
};

void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, void *hdr, uint16_t flags)
{
    ucp_eager_sync_hdr_t *eagers_hdr;
    ucp_request_hdr_t *reqhdr;
    ucp_request_t *req;

    ucs_assert(flags & UCP_RECV_DESC_FLAG_EAGER_SYNC);

    if (flags & UCP_RECV_DESC_FLAG_EAGER_OFFLOAD) {
        eagers_hdr = hdr;
        ucp_tag_offload_eager_sync_send_ack(worker,
                                            eagers_hdr->req.sender_uuid,
                                            eagers_hdr->super.super.tag);
        return;
    }

    if (flags & UCP_RECV_DESC_FLAG_EAGER_ONLY) {
        reqhdr = &((ucp_eager_sync_hdr_t*)hdr)->req;
    } else /* first */ {
        reqhdr = &((ucp_eager_sync_first_hdr_t*)hdr)->req;
    }

    ucs_assert(reqhdr->reqptr != 0);
    ucs_trace_req("send_sync_ack sender_uuid %"PRIx64" remote_request 0x%lx",
                  reqhdr->sender_uuid, reqhdr->reqptr);

    req = ucp_worker_allocate_reply(worker, reqhdr->sender_uuid);
    req->send.uct.func             = ucp_proto_progress_am_bcopy_single;
    req->send.proto.am_id          = UCP_AM_ID_EAGER_SYNC_ACK;
    req->send.proto.remote_request = reqhdr->reqptr;
    req->send.proto.status         = UCS_OK;
    req->send.proto.comp_cb        = ucp_request_put;

    ucp_request_send(req);
}
