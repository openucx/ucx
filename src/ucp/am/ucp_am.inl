/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCP_AM_INL_
#define UCP_AM_INL_

#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_request.h>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_select.inl>


#define UCP_PROTO_AM_OP_ID_MASK \
    (UCS_BIT(UCP_OP_ID_AM_SEND) | UCS_BIT(UCP_OP_ID_AM_SEND_REPLY))


static UCS_F_ALWAYS_INLINE void
ucp_am_fill_header(ucp_am_hdr_t *hdr, ucp_request_t *req)
{
    hdr->am_id         = req->send.msg_proto.am.am_id;
    hdr->flags         = req->send.msg_proto.am.flags;
    hdr->header_length = req->send.msg_proto.am.header.length;
}

static UCS_F_ALWAYS_INLINE int
ucp_am_check_init_params(const ucp_proto_init_params_t *init_params,
                         uint64_t op_id_mask, uint16_t exclude_flags)
{
    return ucp_proto_init_check_op(init_params, op_id_mask) &&
           !(ucp_proto_select_op_flags(init_params->select_param) &
             exclude_flags);
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_config_is_am(const ucp_proto_config_t *proto_config)
{
    return ucp_proto_select_check_op(&proto_config->select_param,
                                     UCP_PROTO_AM_OP_ID_MASK);
}

#endif
