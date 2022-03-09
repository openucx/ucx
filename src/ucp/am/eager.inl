/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef AM_EAGER_INL_
#define AM_EAGER_INL_

#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_request.h>
#include <ucp/dt/datatype_iter.h>
#include <ucp/dt/datatype_iter.inl>


static UCS_F_ALWAYS_INLINE void
ucp_am_fill_header(ucp_am_hdr_t *hdr, ucp_request_t *req)
{
    hdr->am_id         = req->send.msg_proto.am.am_id;
    hdr->flags         = req->send.msg_proto.am.flags;
    hdr->header_length = req->send.msg_proto.am.header_length;
}

static UCS_F_ALWAYS_INLINE size_t ucp_am_send_req_total_size(ucp_request_t *req)
{
    return req->send.state.dt_iter.length +
           req->send.msg_proto.am.header_length;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_pack_user_header(void *buffer, ucp_request_t *req)
{
    ucp_dt_state_t hdr_state;

    hdr_state.offset = 0ul;

    ucp_dt_pack(req->send.ep->worker, ucp_dt_make_contig(1),
                UCS_MEMORY_TYPE_HOST, buffer, req->send.msg_proto.am.header,
                &hdr_state, req->send.msg_proto.am.header_length);
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_am_bcopy_pack_data(void *buffer, ucp_request_t *req, size_t length,
                       ucp_datatype_iter_t *next_iter)
{
    unsigned user_header_length = req->send.msg_proto.am.header_length;
    size_t total_length;
    void *user_hdr;

    ucs_assertv((req->send.state.dt_iter.length == 0) ||
                (length > user_header_length),
                "length %zu user_header length %u", length, user_header_length);

    total_length = ucp_datatype_iter_next_pack(
            &req->send.state.dt_iter, req->send.ep->worker,
            length - user_header_length, next_iter, buffer);
    if (user_header_length != 0) {
        /* Pack user header to the end of message/fragment */
        user_hdr = UCS_PTR_BYTE_OFFSET(buffer, total_length);
        ucp_am_pack_user_header(user_hdr, req);
        total_length += user_header_length;
    }

    return total_length;
}

#endif
