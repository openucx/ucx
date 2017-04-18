/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/dt/dt.h>
#include <ucs/debug/profile.h>

typedef void (*ucp_req_complete_func_t)(ucp_request_t *req, ucs_status_t status);


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_do_am_bcopy_single(uct_pending_req_t *self, uint8_t am_id,
                       uct_pack_callback_t pack_cb)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len     = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], am_id, pack_cb,
                                     req, 0);
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
    size_t max_middle  = ucp_ep_config(ep)->am.max_bcopy - hdr_size_middle;
    ssize_t packed_len;
    uct_ep_h uct_ep;
    size_t offset;
    ucp_dt_state_t saved_state;

    saved_state    = req->send.state;
    offset         = req->send.state.offset;
    req->send.lane = ucp_ep_get_am_lane(ep);
    uct_ep         = ep->uct_eps[req->send.lane];

    if (offset == 0) {
        /* First */
        packed_len = uct_ep_am_bcopy(uct_ep, am_id_first, pack_first, req, 0);
        if (packed_len < 0) {
            goto err; /* Failed */
        }

        UCS_PROFILE_REQUEST_EVENT(req, "am_bcopy_first", packed_len);
        return UCS_INPROGRESS;
    } else if (offset + max_middle < req->send.length) {
        /* Middle */
        packed_len = uct_ep_am_bcopy(uct_ep, am_id_middle, pack_middle, req, 0);
        if (packed_len < 0) {
            goto err; /* Failed */
        }

        ucs_assertv((packed_len < 0) || (packed_len <= max_middle + hdr_size_middle),
                    "packed_len=%zd max_middle=%zu hdr_size_middle=%zu",
                    packed_len, max_middle, hdr_size_middle);
        UCS_PROFILE_REQUEST_EVENT(req, "am_bcopy_middle", packed_len);
        return UCS_INPROGRESS;
    } else {
        /* Last */
        packed_len = uct_ep_am_bcopy(uct_ep, am_id_last, pack_last, req, 0);
        if (packed_len < 0) {
            goto err; /* Failed */
        }

        UCS_PROFILE_REQUEST_EVENT(req, "am_bcopy_last", packed_len);
        return UCS_OK;
    }

err:
    req->send.state = saved_state; /* need to restore the offsets state */
    return packed_len;
}

static UCS_F_ALWAYS_INLINE
size_t ucp_dt_copy_uct(uct_iov_t *iov, size_t *iovcnt, size_t max_dst_iov,
                       ucp_dt_state_t *state, const ucp_dt_iov_t *src_iov,
                       ucp_datatype_t datatype, size_t length_max)
{
    size_t length_it = 0;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        iov[0].buffer = (void *)src_iov + state->offset;
        iov[0].length = length_max;
        iov[0].memh   = state->dt.contig.memh;
        iov[0].stride = 0;
        iov[0].count  = 1;

        *iovcnt   = 1;
        length_it = iov[0].length;
        break;
        
    case UCP_DATATYPE_STRIDE_R:
    case UCP_DATATYPE_STRIDE:
        length_it = ucp_dt_stride_copy_uct(iov, iovcnt, max_dst_iov, state,
                                           src_iov, datatype, length_max);
        break;
        
    case UCP_DATATYPE_IOV_R:
    case UCP_DATATYPE_IOV:
        length_it = ucp_dt_iov_copy_uct(iov, iovcnt, max_dst_iov, state,
                                        src_iov, datatype, length_max);
        break;
    default:
        ucs_error("Invalid data type");
    }

    return length_it;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_zcopy_single(uct_pending_req_t *self, uint8_t am_id,
                                    const void *hdr, size_t hdr_size,
                                    ucp_req_complete_func_t complete)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    size_t max_iov     = ucp_ep_config(ep)->am.max_iov;
    uct_iov_t *iov     = ucs_alloca(max_iov * sizeof(uct_iov_t));
    size_t iovcnt      = 0;
    ucp_dt_state_t saved_state;
    ucs_status_t status;

    saved_state    = req->send.state;
    req->send.lane = ucp_ep_get_am_lane(ep);

    ucp_dt_copy_uct(iov, &iovcnt, max_iov, &req->send.state, req->send.buffer,
                    req->send.datatype, req->send.length);

    status = uct_ep_am_zcopy(ep->uct_eps[req->send.lane], am_id, (void*)hdr,
                             hdr_size, iov, iovcnt, &req->send.uct_comp);
    if (status == UCS_OK) {
        complete(req, UCS_OK);
    } else if (status < 0) {
        req->send.state = saved_state; /* need to restore the offsets state */
        return status;
    } else {
        ucs_assert(status == UCS_INPROGRESS);
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_zcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                                   uint8_t am_id_middle, uint8_t am_id_last,
                                   const void *hdr_first, size_t hdr_size_first,
                                   const void *hdr_middle, size_t hdr_size_middle,
                                   ucp_req_complete_func_t complete)
{
    ucp_request_t *req      = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep            = req->send.ep;
    const size_t max_middle = ucp_ep_config(ep)->am.max_zcopy - hdr_size_middle;
    const size_t max_iov    = ucp_ep_config(ep)->am.max_iov;
    uct_iov_t *iov          = ucs_alloca(max_iov * sizeof(uct_iov_t));
    ucp_dt_state_t *state = &req->send.state;
    unsigned flag_iov_mid   = 0;
    size_t iovcnt           = 0;
    ucp_dt_state_t saved_state;
    size_t offset, length_it;
    ucs_status_t status;
    uct_ep_h uct_ep;

    saved_state             = req->send.state;
    offset                  = state->offset;
    req->send.lane          = ucp_ep_get_am_lane(ep);
    uct_ep                  = ep->uct_eps[req->send.lane];

    ucs_assert(max_iov > 0);
    if (UCP_DT_IS_IOV(req->send.datatype)) {
        /* This flag should guarantee middle stage usage if iovcnt exceeded */
        flag_iov_mid = ((state->dt.iov.iovcnt_offset + max_iov) <
                        state->dt.iov.iovcnt);
    }

    if (offset == 0) {
        /* First stage */
        length_it = ucp_dt_copy_uct(iov, &iovcnt, max_iov, state,
                                    req->send.buffer,  req->send.datatype,
                                    max_middle - hdr_size_first + hdr_size_middle);

        status = uct_ep_am_zcopy(uct_ep, am_id_first, (void*)hdr_first,
                                 hdr_size_first, iov, iovcnt, &req->send.uct_comp);
        if (status < 0) {
            goto err; /* Failed */
        }

        UCS_PROFILE_REQUEST_EVENT(req, "am_zcopy_first", iov[0].length);
        state->offset      += length_it;
        ++req->send.uct_comp.count;
        return UCS_INPROGRESS;
    } else if ((offset + max_middle < req->send.length) || flag_iov_mid) {
        /* Middle stage */
        length_it = ucp_dt_copy_uct(iov, &iovcnt, max_iov, state,
                                    req->send.buffer, req->send.datatype, max_middle);

        status = uct_ep_am_zcopy(uct_ep, am_id_middle, (void*)hdr_middle,
                                 hdr_size_middle, iov, iovcnt, &req->send.uct_comp);
        if (status < 0) {
            goto err; /* Failed */
        }

        UCS_PROFILE_REQUEST_EVENT(req, "am_zcopy_middle", iov[0].length);
        state->offset      += length_it;
        ++req->send.uct_comp.count;
        return UCS_INPROGRESS;
    } else {
        /* Last stage */
        length_it = ucp_dt_copy_uct(iov, &iovcnt, max_iov, state,
                                    req->send.buffer, req->send.datatype,
                                    req->send.length - offset);

        status = uct_ep_am_zcopy(uct_ep, am_id_last, (void*)hdr_middle,
                                 hdr_size_middle, iov, iovcnt, &req->send.uct_comp);
        if (status < 0) {
            goto err; /* Failed */
        }

        UCS_PROFILE_REQUEST_EVENT(req, "am_zcopy_last", iov[0].length);
        ucs_assert(req->send.length == (state->offset + length_it));
        if (status == UCS_OK) {
            complete(req, UCS_OK);
        } else {
            ucs_assert(status == UCS_INPROGRESS);
        }
        return UCS_OK;
    }

err:
    req->send.state = saved_state; /* need to restore the offsets state */
    return status;
}
