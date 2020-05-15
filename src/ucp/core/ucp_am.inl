/**
* Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucp/core/ucp_request.h>

static UCS_F_ALWAYS_INLINE ssize_t
ucp_am_get_short_max(const ucp_request_t *req,
                     const ucp_ep_msg_config_t *msg_config)
{
    return (!UCP_DT_IS_CONTIG(req->send.datatype) ||
            (req->flags & UCP_REQUEST_FLAG_SYNC) ||
            (!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(req->send.mem_type))) ||
           ((req->flags & UCP_REQUEST_FLAG_SEND_AM) &&
            (req->send.msg_proto.am.flags & UCP_AM_SEND_REPLY)) ?
           -1 : msg_config->max_short;
}
