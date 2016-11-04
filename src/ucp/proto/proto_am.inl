/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.inl>


typedef void (*ucp_req_complete_func_t)(ucp_request_t *req);


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_do_am_bcopy_single(uct_pending_req_t *self, uint8_t am_id,
                       uct_pack_callback_t pack_cb)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len     = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], am_id, pack_cb, req);
    if (packed_len < 0) {
        return packed_len;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_bcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                                   uint8_t am_id_middle, uint8_t am_id_last,
                                   size_t hdr_size_middle,
                                   uct_pack_callback_t pack_first,
                                   uct_pack_callback_t pack_middle,
                                   uct_pack_callback_t pack_last)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    size_t max_middle  = ucp_ep_config(ep)->max_am_bcopy - hdr_size_middle;
    ssize_t packed_len;
    uct_ep_h uct_ep;
    size_t offset;

    offset         = req->send.state.offset;
    req->send.lane = ucp_ep_get_am_lane(ep);
    uct_ep         = ep->uct_eps[req->send.lane];

    if (offset == 0) {
        /* First */
        packed_len = uct_ep_am_bcopy(uct_ep, am_id_first, pack_first, req);
        if (packed_len < 0) {
            return packed_len; /* Failed */
        }

        return UCS_INPROGRESS;
    } else if (offset + max_middle < req->send.length) {
        /* Middle */
        packed_len = uct_ep_am_bcopy(uct_ep, am_id_middle, pack_middle, req);
        if (packed_len < 0) {
            return packed_len; /* Failed */
        }

        ucs_assertv((packed_len < 0) || (packed_len <= max_middle + hdr_size_middle),
                    "packed_len=%zd max_middle=%zu hdr_size_middle=%zu",
                    packed_len, max_middle, hdr_size_middle);
        return UCS_INPROGRESS;
    } else {
        /* Last */
        packed_len = uct_ep_am_bcopy(uct_ep, am_id_last, pack_last, req);
        if (packed_len < 0) {
            return packed_len; /* Failed */
        }

        return UCS_OK;
    }
}

static ucs_status_t UCS_F_ALWAYS_INLINE
ucp_do_am_zcopy_single(uct_pending_req_t *self, uint8_t am_id, const void *hdr,
                       size_t hdr_size, ucp_req_complete_func_t complete)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    ucs_status_t status;
    uct_iov_t iov[1];

    /* TODO fix UCT api to have header ptr as const */
    req->send.lane = ucp_ep_get_am_lane(ep);

    iov[0].buffer = (void*)req->send.buffer;
    iov[0].length = req->send.length;
    iov[0].memh   = req->send.state.dt.contig.memh;
    iov[0].count  = 1;
    iov[0].stride = 0;
    status = uct_ep_am_zcopy(ep->uct_eps[req->send.lane], am_id, (void*)hdr,
                             hdr_size, iov, 1, &req->send.uct_comp);
    if (status == UCS_OK) {
        complete(req);
    } else if (status < 0) {
        return status;
    } else {
        ucs_assert(status == UCS_INPROGRESS);
    }

    return UCS_OK;
}

static ucs_status_t UCS_F_ALWAYS_INLINE
ucp_do_am_zcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                      uint8_t am_id_middle, uint8_t am_id_last,
                      const void *hdr_first, size_t hdr_size_first,
                      const void *hdr_middle, size_t hdr_size_middle,
                      ucp_req_complete_func_t complete)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = req->send.ep;
    size_t max_middle = ucp_ep_config(ep)->max_am_zcopy - hdr_size_middle;
    ucs_status_t status;
    size_t offset;
    uct_ep_h uct_ep;
    uct_iov_t iov[1];

    offset         = req->send.state.offset;
    req->send.lane = ucp_ep_get_am_lane(ep);
    uct_ep         = ep->uct_eps[req->send.lane];
    iov[0].buffer  = (void *)(req->send.buffer + offset);
    iov[0].memh    = req->send.state.dt.contig.memh;
    iov[0].count   = 1;
    iov[0].stride  = 0;

    if (offset == 0) {
        /* First */
        iov[0].length = max_middle - hdr_size_first + hdr_size_middle;
        status = uct_ep_am_zcopy(uct_ep, am_id_first, (void*)hdr_first,
                                 hdr_size_first, iov, 1, &req->send.uct_comp);
        if (status < 0) {
            return status; /* Failed */
        }

        req->send.state.offset += iov[0].length;
        return UCS_INPROGRESS;
    } else if (offset + max_middle < req->send.length) {
        /* Middle */
        iov[0].length = max_middle;
        status = uct_ep_am_zcopy(uct_ep, am_id_middle, (void*)hdr_middle,
                                 hdr_size_middle, iov, 1, &req->send.uct_comp);
        if (status < 0) {
            return status; /* Failed */
        }

        req->send.state.offset += iov[0].length;
        return UCS_INPROGRESS;
    } else {
        /* Last */
        iov[0].length = req->send.length - offset;
        status = uct_ep_am_zcopy(uct_ep, am_id_last, (void*)hdr_middle,
                                 hdr_size_middle, iov, 1, &req->send.uct_comp);
        if (status < 0) {
            return status; /* Failed */
        }

        if (status == UCS_OK) {
            complete(req);
        } else {
            ucs_assert(status == UCS_INPROGRESS);
        }
        return UCS_OK;
    }
}
