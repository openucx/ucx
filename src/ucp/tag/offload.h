/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_OFFLOAD_H_
#define UCP_TAG_OFFLOAD_H_

#include <ucp/dt/dt_contig.h>
#include <ucp/core/ucp_request.h>
#include <ucp/proto/proto.h>


extern const ucp_proto_t ucp_tag_offload_proto;

extern const ucp_proto_t ucp_tag_offload_sync_proto;

ucs_status_t ucp_tag_offload_unexp_eager(void *arg, void *data, size_t length,
                                         unsigned flags, uct_tag_t stag, uint64_t imm);

void ucp_tag_offload_cancel(ucp_context_t *context, ucp_request_t *req, int force);

int ucp_tag_offload_post(ucp_context_t *ctx, ucp_request_t *req);

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_try_post(ucp_context_t *ctx, ucp_request_t *req)
{
    if (ucs_unlikely(req->recv.length >= ctx->tm.post_thresh)) {
        if (ucp_tag_offload_post(ctx, req)) {
            return;
        }
    }
    req->flags |= UCP_REQUEST_FLAG_BLOCK_OFFLOAD;
    ++ctx->tm.sw_req_count;
}

#endif
