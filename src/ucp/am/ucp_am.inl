/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCP_AM_INL_
#define UCP_AM_INL_

#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_request.h>


#define UCP_AM_OP_ID_MASK_ALL \
    (UCS_BIT(UCP_OP_ID_AM_SEND) | UCS_BIT(UCP_OP_ID_AM_SEND_REPLY))


static UCS_F_ALWAYS_INLINE void
ucp_am_fill_header(ucp_am_hdr_t *hdr, ucp_request_t *req)
{
    hdr->am_id         = req->send.msg_proto.am.am_id;
    hdr->flags         = req->send.msg_proto.am.flags;
    hdr->header_length = req->send.msg_proto.am.header_length;
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

static UCS_F_ALWAYS_INLINE int
ucp_am_check_init_params(const ucp_proto_init_params_t *init_params,
                         uint64_t op_id_mask, uint16_t exclude_flags)
{
    return (UCS_BIT(init_params->select_param->op_id) & op_id_mask) &&
           !(init_params->select_param->op_flags & exclude_flags);
}

#endif
