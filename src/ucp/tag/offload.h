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

/**
 * Header for SW RNDV request
 */
typedef struct ucp_sw_rndv_hdr {
    ucp_request_hdr_t super;
    size_t            length;
    uint16_t          flags;
} UCS_S_PACKED ucp_sw_rndv_hdr_t;

/**
 * Header for sync send acknowledgment
 */
typedef struct {
    uint64_t          sender_uuid;
    ucp_tag_t         sender_tag;
} UCS_S_PACKED ucp_offload_ssend_hdr_t;


extern const ucp_proto_t ucp_tag_offload_proto;

extern const ucp_proto_t ucp_tag_offload_sync_proto;

ucs_status_t ucp_tag_offload_rndv_zcopy(uct_pending_req_t *self);

void ucp_tag_offload_cancel_rndv(ucp_request_t *req);

ucs_status_t ucp_tag_offload_start_rndv(ucp_request_t *sreq);

void ucp_tag_offload_eager_sync_send_ack(ucp_worker_h worker,
                                         uint64_t sender_uuid,
                                         ucp_tag_t sender_tag);

ucs_status_t ucp_tag_offload_unexp_eager(void *arg, void *data, size_t length,
                                         unsigned flags, uct_tag_t stag, uint64_t imm);


ucs_status_t ucp_tag_offload_unexp_rndv(void *arg, unsigned flags, uint64_t stag,
                                        const void *hdr, unsigned hdr_length,
                                        uint64_t remote_addr, size_t length,
                                        const void *rkey_buf);

void ucp_tag_offload_cancel(ucp_context_t *context, ucp_request_t *req, int force);

int ucp_tag_offload_post(ucp_context_t *ctx, ucp_request_t *req);

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_try_post(ucp_context_t *ctx, ucp_request_t *req)
{
    if (ucs_unlikely((req->recv.length >= ctx->tm.offload.thresh) &&
                     (req->recv.state.offset == 0))) {
        if (ucp_tag_offload_post(ctx, req)) {
            return;
        }
    }
    req->flags |= UCP_REQUEST_FLAG_BLOCK_OFFLOAD;
    ++ctx->tm.offload.sw_req_count;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_try_cancel(ucp_context_t *ctx, ucp_request_t *req, int force)
{
    if (ucs_unlikely(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
        ucp_tag_offload_cancel(ctx, req, force);
    }
}

#endif
