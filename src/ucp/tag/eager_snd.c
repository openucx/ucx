/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_am.inl>


/* packing  start */

static size_t ucp_tag_pack_eager_only_contig(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length         = req->send.length;
    hdr->super.tag = req->send.tag;
    memcpy(hdr + 1, req->send.buffer, length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_sync_only_contig(void *dest, void *arg)
{
    ucp_eager_sync_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length               = req->send.length;
    hdr->super.super.tag = req->send.tag;
    hdr->req.sender_uuid = req->send.ep->worker->uuid;
    hdr->req.reqptr      = (uintptr_t)req;

    memcpy(hdr + 1, req->send.buffer, length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_first_contig(void *dest, void *arg)
{
    ucp_eager_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length               = ucp_ep_config(req->send.ep)->max_am_bcopy - sizeof(*hdr);
    hdr->super.super.tag = req->send.tag;
    hdr->total_len       = req->send.length;

    ucs_assert(req->send.state.offset == 0);
    ucs_assert(req->send.length > length);
    memcpy(hdr + 1, req->send.buffer, length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_sync_first_contig(void *dest, void *arg)
{
    ucp_eager_sync_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length                     = ucp_ep_config(req->send.ep)->max_am_bcopy - sizeof(*hdr);
    hdr->super.super.super.tag = req->send.tag;
    hdr->super.total_len       = req->send.length;
    hdr->req.sender_uuid       = req->send.ep->worker->uuid;
    hdr->req.reqptr            = (uintptr_t)req;

    ucs_debug("pack eager_sync_first paylen %zu", length);
    ucs_assert(req->send.state.offset == 0);
    ucs_assert(req->send.length > length);
    memcpy(hdr + 1, req->send.buffer, length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_middle_contig(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length         = ucp_ep_config(req->send.ep)->max_am_bcopy - sizeof(*hdr);
    ucs_debug("pack eager_middle paylen %zu", length);
    hdr->super.tag = req->send.tag;
    memcpy(hdr + 1, req->send.buffer + req->send.state.offset, length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_last_contig(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length         = req->send.length - req->send.state.offset;
    hdr->super.tag = req->send.tag;
    memcpy(hdr + 1, req->send.buffer + req->send.state.offset, length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_only_generic(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.offset == 0);
    hdr->super.tag = req->send.tag;
    length         = ucp_request_generic_dt_pack(req, hdr + 1, req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_first_generic(void *dest, void *arg)
{
    ucp_eager_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length, length;

    ucs_assert(req->send.state.offset == 0);

    max_length           = ucp_ep_config(req->send.ep)->max_am_bcopy - sizeof(*hdr);
    hdr->super.super.tag = req->send.tag;
    hdr->total_len       = req->send.length;

    ucs_assert(req->send.length > max_length);
    length = ucp_request_generic_dt_pack(req, hdr + 1, max_length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_middle_generic(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length;

    max_length     = ucp_ep_config(req->send.ep)->max_am_bcopy - sizeof(*hdr);
    hdr->super.tag = req->send.tag;
    return sizeof(*hdr) + ucp_request_generic_dt_pack(req, hdr + 1, max_length);
}

static size_t ucp_tag_pack_eager_last_generic(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length, length;

    max_length     = req->send.length - req->send.state.offset;
    hdr->super.tag = req->send.tag;
    length         = ucp_request_generic_dt_pack(req, hdr + 1, max_length);
    ucs_assertv(length == max_length, "length=%zu, max_length=%zu",
                length, max_length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_sync_only_generic(void *dest, void *arg)
{
    ucp_eager_sync_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.offset == 0);
    hdr->super.super.tag = req->send.tag;
    hdr->req.sender_uuid = req->send.ep->worker->uuid;
    hdr->req.reqptr      = (uintptr_t)req;
    length               = ucp_request_generic_dt_pack(req, hdr + 1,
                                                       req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_sync_first_generic(void *dest, void *arg)
{
    ucp_eager_sync_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length, length;

    ucs_assert(req->send.state.offset == 0);

    max_length                 = ucp_ep_config(req->send.ep)->max_am_bcopy -
                                 sizeof(*hdr);
    hdr->super.super.super.tag = req->send.tag;
    hdr->super.total_len       = req->send.length;
    hdr->req.sender_uuid       = req->send.ep->worker->uuid;
    hdr->req.reqptr            = (uintptr_t)req;

    ucs_assert(req->send.length > max_length);
    length = ucp_request_generic_dt_pack(req, hdr + 1, max_length);
    return sizeof(*hdr) + length;
}

/* eager */

static ucs_status_t ucp_tag_eager_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    ucs_status_t status;

    status = ucp_tag_send_eager_short(ep, req->send.tag, req->send.buffer,
                                      req->send.length);
    if (status != UCS_OK) {
        return status;
    }

    ucp_request_complete(req, req->cb.send, UCS_OK);
    return UCS_OK;
}

static ucs_status_t ucp_tag_eager_contig_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_EAGER_ONLY,
                                                 ucp_tag_pack_eager_only_contig);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_complete(req, req->cb.send, UCS_OK);
    }
    return status;
}

static ucs_status_t ucp_tag_eager_contig_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                UCP_AM_ID_EAGER_LAST,
                                                sizeof(ucp_eager_first_hdr_t),
                                                sizeof(ucp_eager_hdr_t),
                                                ucp_tag_pack_eager_first_contig,
                                                ucp_tag_pack_eager_middle_contig,
                                                ucp_tag_pack_eager_last_contig);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_complete(req, req->cb.send, UCS_OK);
    }
    return status;
}

static void ucp_tag_eager_contig_zcopy_req_complete(ucp_request_t *req)
{
    ucp_request_send_buffer_dereg(req);
    ucp_request_complete(req, req->cb.send, UCS_OK);
}

static ucs_status_t ucp_tag_eager_contig_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_hdr_t hdr;

    hdr.super.tag = req->send.tag;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_EAGER_ONLY, &hdr, sizeof(hdr),
                                  ucp_tag_eager_contig_zcopy_req_complete);
}

static ucs_status_t ucp_tag_eager_contig_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_first_hdr_t first_hdr;

    first_hdr.super.super.tag = req->send.tag;
    first_hdr.total_len       = req->send.length;
    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_EAGER_FIRST,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 UCP_AM_ID_EAGER_LAST,
                                 &first_hdr, sizeof(first_hdr),
                                 &first_hdr.super, sizeof(first_hdr.super),
                                 ucp_tag_eager_contig_zcopy_req_complete);
}

static void ucp_tag_eager_contig_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct_comp);
    ucp_tag_eager_contig_zcopy_req_complete(req);
}

static void ucp_tag_eager_generic_complere(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_generic_dt_finish(req);
    ucp_request_complete(req, req->cb.send, UCS_OK);
}

static ucs_status_t ucp_tag_eager_generic_single(uct_pending_req_t *self)
{
    ucs_status_t status;

    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_EAGER_ONLY,
                                    ucp_tag_pack_eager_only_generic);
    if (status != UCS_OK) {
        return status;
    }

    ucp_tag_eager_generic_complere(self);
    return UCS_OK;
}

static ucs_status_t ucp_tag_eager_generic_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                UCP_AM_ID_EAGER_LAST,
                                                sizeof(ucp_eager_first_hdr_t),
                                                sizeof(ucp_eager_hdr_t),
                                                ucp_tag_pack_eager_first_generic,
                                                ucp_tag_pack_eager_middle_generic,
                                                ucp_tag_pack_eager_last_generic);
    if (status == UCS_OK) {
        ucp_tag_eager_generic_complere(self);
    }
    return status;
}

const ucp_proto_t ucp_tag_eager_proto = {
    .contig_short            = ucp_tag_eager_contig_short,
    .contig_bcopy_single     = ucp_tag_eager_contig_bcopy_single,
    .contig_bcopy_multi      = ucp_tag_eager_contig_bcopy_multi,
    .contig_zcopy_single     = ucp_tag_eager_contig_zcopy_single,
    .contig_zcopy_multi      = ucp_tag_eager_contig_zcopy_multi,
    .contig_zcopy_completion = ucp_tag_eager_contig_zcopy_completion,
    .generic_single          = ucp_tag_eager_generic_single,
    .generic_multi           = ucp_tag_eager_generic_multi,
    .only_hdr_size           = sizeof(ucp_eager_hdr_t),
    .first_hdr_size          = sizeof(ucp_eager_first_hdr_t),
    .mid_hdr_size            = sizeof(ucp_eager_hdr_t)
};

/* eager sync */

void ucp_tag_eager_sync_completion(ucp_request_t *req, uint16_t flag)
{
    static const uint16_t all_completed = UCP_REQUEST_FLAG_LOCAL_COMPLETED |
                                          UCP_REQUEST_FLAG_REMOTE_COMPLETED;

    ucs_assertv(!(req->flags & flag), "req->flags=%d flag=%d", req->flags, flag);
    req->flags |= flag;
    if (ucs_test_all_flags(req->flags, all_completed)) {
        ucp_request_complete(req, req->cb.send, UCS_OK);
    }
}

static ucs_status_t ucp_tag_eager_sync_contig_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_EAGER_SYNC_ONLY,
                                                 ucp_tag_pack_eager_sync_only_contig);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED);
    }
    return status;
}

static ucs_status_t ucp_tag_eager_sync_contig_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_SYNC_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                UCP_AM_ID_EAGER_LAST,
                                                sizeof(ucp_eager_sync_first_hdr_t),
                                                sizeof(ucp_eager_hdr_t),
                                                ucp_tag_pack_eager_sync_first_contig,
                                                ucp_tag_pack_eager_middle_contig,
                                                ucp_tag_pack_eager_last_contig);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED);
    }
    return status;
}

static inline void ucp_tag_eager_sync_contig_zcopy_req_complete(ucp_request_t *req)
{
    ucp_request_send_buffer_dereg(req);
    ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED);
}

static ucs_status_t ucp_tag_eager_sync_contig_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_sync_hdr_t hdr;

    hdr.super.super.tag = req->send.tag;
    hdr.req.sender_uuid = req->send.ep->worker->uuid;
    hdr.req.reqptr      = (uintptr_t)req;

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_EAGER_SYNC_ONLY, &hdr, sizeof(hdr),
                                  ucp_tag_eager_sync_contig_zcopy_req_complete);
}

static ucs_status_t ucp_tag_eager_sync_contig_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_eager_sync_first_hdr_t first_hdr;

    first_hdr.super.super.super.tag = req->send.tag;
    first_hdr.super.total_len       = req->send.length;
    first_hdr.req.sender_uuid       = req->send.ep->worker->uuid;
    first_hdr.req.reqptr            = (uintptr_t)req;

    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_EAGER_SYNC_FIRST,
                                 UCP_AM_ID_EAGER_MIDDLE,
                                 UCP_AM_ID_EAGER_LAST,
                                 &first_hdr, sizeof(first_hdr),
                                 &first_hdr.super.super, sizeof(first_hdr.super.super),
                                 ucp_tag_eager_sync_contig_zcopy_req_complete);
}

static void ucp_tag_eager_sync_contig_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct_comp);
    ucp_tag_eager_sync_contig_zcopy_req_complete(req);
}

static inline void ucp_tag_eager_sync_generic_complete(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_generic_dt_finish(req);
    ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED);
}

static ucs_status_t ucp_tag_eager_sync_generic_single(uct_pending_req_t *self)
{
    ucs_status_t status;

    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_EAGER_SYNC_ONLY,
                                    ucp_tag_pack_eager_sync_only_generic);
    if (status == UCS_OK) {
        ucp_tag_eager_sync_generic_complete(self);
    }
    return UCS_OK;
}

static ucs_status_t ucp_tag_eager_sync_generic_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_EAGER_SYNC_FIRST,
                                                UCP_AM_ID_EAGER_MIDDLE,
                                                UCP_AM_ID_EAGER_LAST,
                                                sizeof(ucp_eager_sync_first_hdr_t),
                                                sizeof(ucp_eager_hdr_t),
                                                ucp_tag_pack_eager_sync_first_generic,
                                                ucp_tag_pack_eager_middle_generic,
                                                ucp_tag_pack_eager_last_generic);
    if (status == UCS_OK) {
        ucp_tag_eager_sync_generic_complete(self);
    }
    return status;
}

const ucp_proto_t ucp_tag_eager_sync_proto = {
    .contig_short            = NULL,
    .contig_bcopy_single     = ucp_tag_eager_sync_contig_bcopy_single,
    .contig_bcopy_multi      = ucp_tag_eager_sync_contig_bcopy_multi,
    .contig_zcopy_single     = ucp_tag_eager_sync_contig_zcopy_single,
    .contig_zcopy_multi      = ucp_tag_eager_sync_contig_zcopy_multi,
    .contig_zcopy_completion = ucp_tag_eager_sync_contig_zcopy_completion,
    .generic_single          = ucp_tag_eager_sync_generic_single,
    .generic_multi           = ucp_tag_eager_sync_generic_multi,
    .only_hdr_size           = sizeof(ucp_eager_sync_hdr_t),
    .first_hdr_size          = sizeof(ucp_eager_sync_first_hdr_t),
    .mid_hdr_size            = sizeof(ucp_eager_hdr_t)
};
