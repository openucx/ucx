/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef PROTO_EAGER_INL_
#define PROTO_EAGER_INL_

#include "eager.h"

#include <ucp/proto/proto_common.inl>


static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_sync_send_completed_common(ucp_request_t *req)
{
    req->flags |= UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED;
    if (req->flags & UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED) {
        ucp_request_complete_send(req, UCS_OK);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_sync_bcopy_send_completed(ucp_request_t *req)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, UCP_DT_MASK_ALL);
    ucp_proto_eager_sync_send_completed_common(req);
    return UCS_OK;
}

#endif
