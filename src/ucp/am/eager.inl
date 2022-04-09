/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef AM_EAGER_INL_
#define AM_EAGER_INL_

#include "ucp_am.inl"

#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_request.h>
#include <ucp/dt/datatype_iter.h>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_common.inl>
#include <ucs/datastruct/mpool.inl>


static UCS_F_ALWAYS_INLINE size_t ucp_am_send_req_total_size(ucp_request_t *req)
{
    return req->send.state.dt_iter.length +
           req->send.msg_proto.am.header_length;
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_am_eager_bcopy_pack_data(void *buffer, ucp_request_t *req, size_t length,
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

static void ucp_am_eager_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucs_assert(req->send.msg_proto.am.reg_desc != NULL);
    ucs_mpool_put_inline(req->send.msg_proto.am.reg_desc);
    ucp_proto_request_zcopy_completion(self);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_eager_zcopy_pack_user_header(ucp_request_t *req)
{
    ucp_mem_desc_t *reg_desc;

    reg_desc = ucp_worker_mpool_get(&req->send.ep->worker->reg_mp);
    if (ucs_unlikely(reg_desc == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    if (req->send.msg_proto.am.header_length != 0) {
        ucs_assert(req->send.msg_proto.am.header != NULL);
        ucp_am_pack_user_header(reg_desc + 1, req);
    }

    req->send.msg_proto.am.reg_desc = reg_desc;

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void ucp_am_eager_zcopy_add_footer(
        ucp_request_t *req, size_t offset, ucp_rsc_index_t md_index,
        uct_iov_t *iov, size_t *iovcnt, size_t footer_size)
{
    ucp_mem_desc_t *reg_desc = req->send.msg_proto.am.reg_desc;
    void *buffer;
    uct_mem_h memh;

    if (footer_size == 0) {
        return;
    }

    ucs_assert(reg_desc != NULL);
    buffer = UCS_PTR_BYTE_OFFSET(reg_desc + 1, offset);

    if (md_index == UCP_NULL_RESOURCE) {
        memh = UCT_MEM_HANDLE_NULL;
    } else {
        memh = reg_desc->memh->uct[md_index];
    }

    ucp_add_uct_iov_elem(iov, buffer, footer_size, memh, iovcnt);
}

#endif
