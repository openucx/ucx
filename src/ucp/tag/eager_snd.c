/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"

#include <ucp/core/ucp_worker.h>


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

static size_t ucp_tag_pack_eager_first_contig(void *dest, void *arg)
{
    ucp_eager_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length               = ucp_ep_config(req->send.ep)->eager.max_bcopy
                            + sizeof(ucp_eager_hdr_t) - sizeof(*hdr);
    hdr->super.super.tag = req->send.tag;
    hdr->total_len       = req->send.length;

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

    length         = ucp_ep_config(req->send.ep)->eager.max_bcopy;
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

ucs_status_t ucp_tag_send_eager_only_contig(ucp_ep_t *ep, ucp_tag_t tag,
                                            const void *buffer, size_t length)
{
    ucp_request_t req;
    return uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_ONLY,
                           ucp_tag_pack_eager_only_contig, &req);
}

ucs_status_t ucp_tag_progress_eager_contig(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    size_t max_bcopy_egr = ucp_ep_config(req->send.ep)->eager.max_bcopy;
    size_t length, offset;
    ssize_t packed_len;

    for (;;) {
        offset = req->send.state.offset;
        if (offset == 0) {
            /* First packet */
            length = req->send.length;
            if (length <= ucp_ep_config(req->send.ep)->eager.max_short) {
                /* Only packet */
                packed_len = ucp_tag_send_eager_short(ep, req->send.tag,
                                                      req->send.buffer, length);
                goto out_complete;
            } else if (length <= max_bcopy_egr) {
                /* Only packet */
                packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_ONLY,
                                             ucp_tag_pack_eager_only_contig, req);
                goto out_complete;
            } else {
                /* First of many */
                packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_FIRST,
                                             ucp_tag_pack_eager_first_contig, req);
                if (packed_len < 0) {
                    return packed_len; /* Failed */
                }

                req->send.state.offset += packed_len - sizeof(ucp_eager_first_hdr_t);
            }
        } else if (offset + max_bcopy_egr < req->send.length) {
            /* Middle packet */
            packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_MIDDLE,
                                         ucp_tag_pack_eager_middle_contig, req);
            if (packed_len < 0) {
                return packed_len; /* Failed */
            }

            req->send.state.offset += packed_len - sizeof(ucp_eager_hdr_t);
            ucs_assertv((packed_len < 0) || (packed_len == max_bcopy_egr + sizeof(ucp_eager_hdr_t)),
                        "packed_len=%zd max_bcopy_egr=%zu", packed_len, max_bcopy_egr);
        } else {
            /* Last packet */
            packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_LAST,
                                         ucp_tag_pack_eager_last_contig, req);
            goto out_complete;
        }
    }

out_complete:
    if (packed_len < 0) {
        return packed_len;
    }

    ucp_request_complete(req, req->cb.send, UCS_OK);
    return UCS_OK;
}

static inline size_t ucp_req_generic_dt_pack(ucp_request_t *req, void *dest,
                                             size_t length)
{
    ucp_dt_generic_t *dt = ucp_dt_generic(req->send.datatype);
    return dt->ops.pack(req->send.state.dt.generic.state,
                        req->send.state.offset, dest, length);
}

static inline void ucp_req_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt = ucp_dt_generic(req->send.datatype);
    return dt->ops.finish(req->send.state.dt.generic.state);
}

static size_t ucp_tag_pack_eager_only_generic(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.offset == 0);
    hdr->super.tag = req->send.tag;
    length         = ucp_req_generic_dt_pack(req, hdr + 1, req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_first_generic(void *dest, void *arg)
{
    ucp_eager_first_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length, length;

    ucs_assert(req->send.state.offset == 0);

    max_length           = ucp_ep_config(req->send.ep)->eager.max_bcopy
                            + sizeof(ucp_eager_hdr_t) - sizeof(*hdr);
    hdr->super.super.tag = req->send.tag;
    hdr->total_len       = req->send.length;

    ucs_assert(req->send.length > max_length);
    length = ucp_req_generic_dt_pack(req, hdr + 1, max_length);
    return sizeof(*hdr) + length;
}

static size_t ucp_tag_pack_eager_middle_generic(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length;

    max_length     = ucp_ep_config(req->send.ep)->eager.max_bcopy;
    hdr->super.tag = req->send.tag;
    return sizeof(*hdr) + ucp_req_generic_dt_pack(req, hdr + 1, max_length);
}

static size_t ucp_tag_pack_eager_last_generic(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t max_length, length;

    max_length     = req->send.length - req->send.state.offset;
    hdr->super.tag = req->send.tag;
    length         = ucp_req_generic_dt_pack(req, hdr + 1, max_length);
    ucs_assertv(length == max_length, "length=%zu, max_length=%zu",
                length, max_length);
    return sizeof(*hdr) + length;
}

ucs_status_t ucp_tag_progress_eager_generic(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    size_t max_bcopy_egr = ucp_ep_config(req->send.ep)->eager.max_bcopy;
    size_t length, offset;
    ssize_t packed_len;

    for (;;) {
        offset = req->send.state.offset;
        if (offset == 0) {
            /* First packet */
            length = req->send.length;
            if (length <= max_bcopy_egr) {
                /* Only packet */
                packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_ONLY,
                                             ucp_tag_pack_eager_only_generic, req);
                goto out_complete;
            } else {
                /* First of many */
                packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_FIRST,
                                             ucp_tag_pack_eager_first_generic, req);
                if (packed_len < 0) {
                    return packed_len; /* Failed */
                }

                req->send.state.offset += packed_len - sizeof(ucp_eager_first_hdr_t);
            }
        } else if (offset + max_bcopy_egr < req->send.length) {
            /* Middle packet */
            packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_MIDDLE,
                                         ucp_tag_pack_eager_middle_generic, req);
            if (packed_len < 0) {
                return packed_len; /* Failed */
            }

            req->send.state.offset += packed_len - sizeof(ucp_eager_hdr_t);
        } else {
            /* Last packet */
            packed_len = uct_ep_am_bcopy(ep->uct_ep, UCP_AM_ID_EAGER_LAST,
                                         ucp_tag_pack_eager_last_generic, req);
            goto out_complete;
        }
    }

out_complete:
    if (packed_len < 0) {
        return packed_len;
    }

    ucp_req_generic_dt_finish(req);
    ucp_request_complete(req, req->cb.send, UCS_OK);
    return UCS_OK;
}
