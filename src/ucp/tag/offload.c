/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "offload.h"
#include "eager.h"
#include "rndv.h"
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_am.inl>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_request.h>
#include <ucp/tag/tag_match.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/sys.h>

/* Tag consumed by the transport - need to remove it from expected queue */
void ucp_tag_offload_tag_consumed(uct_tag_context_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, recv.uct_ctx);
    ucp_context_t *ctx = req->recv.worker->context;
    ucs_queue_head_t *queue;

    queue = ucp_tag_exp_get_req_queue(&ctx->tm, req);
    ucs_queue_remove(queue, &req->recv.queue);
}

/* Message is scattered to user buffer by the transport, complete the request */
void ucp_tag_offload_completed(uct_tag_context_t *self, uct_tag_t stag,
                               uint64_t imm, size_t length, ucs_status_t status)
{
    ucp_request_t *req        = ucs_container_of(self, ucp_request_t, recv.uct_ctx);
    ucp_context_t *ctx        = req->recv.worker->context;
    ucp_worker_iface_t *iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload_ifaces,
                                                              ucp_worker_iface_t, queue);

    req->recv.info.sender_tag = stag;
    req->recv.info.length     = length;

    ucp_request_memory_dereg(ctx, iface->rsc_index, req->recv.datatype,
                             &req->recv.state);
    ucp_request_complete_recv(req, status);
}

void ucp_tag_offload_cancel(ucp_context_t *ctx, ucp_request_t *req, int force)
{
    ucp_worker_iface_t *ucp_iface;
    ucs_status_t status;

    ucs_assert(req->flags & UCP_REQUEST_FLAG_OFFLOADED);

    ucp_iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload_ifaces,
                                              ucp_worker_iface_t, queue);
    ucp_request_memory_dereg(ctx, ucp_iface->rsc_index, req->recv.datatype,
                             &req->recv.state);
    status = uct_iface_tag_recv_cancel(ucp_iface->iface, &req->recv.uct_ctx, force);
    if (status != UCS_OK) {
        ucs_error("Failed to cancel recv in the transport: %s",
                  ucs_status_string(status));
    }
}

int ucp_tag_offload_post(ucp_context_t *ctx, ucp_request_t *req)
{
    size_t length = req->recv.length;
    ucp_worker_iface_t *ucp_iface;
    ucs_status_t status;
    uct_iov_t iov;

    if (!UCP_DT_IS_CONTIG(req->recv.datatype)) {
        /* Non-contig buffers not supported yet. */
        return 0;
    }

    if ((ctx->config.tag_sender_mask & req->recv.tag_mask) !=
         ctx->config.tag_sender_mask) {
        /* Wildcard.
         * TODO add check that only offload capable iface present. In
         * this case can post tag as well. */
        return 0;
    }

    if (ctx->tm.sw_req_count) {
        /* There are some requests which must be completed in SW. Do not post
         * tags to HW until they are completed. */
        return 0;
    }

    ucp_iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload_ifaces,
                                              ucp_worker_iface_t, queue);
    status = ucp_request_memory_reg(ctx, ucp_iface->rsc_index, req->recv.buffer,
                                    req->recv.length, req->recv.datatype,
                                    &req->recv.state);
    if (status != UCS_OK) {
        return 0;
    }

    req->recv.uct_ctx.tag_consumed_cb = ucp_tag_offload_tag_consumed;
    req->recv.uct_ctx.completed_cb    = ucp_tag_offload_completed;

    iov.buffer = (void*)req->recv.buffer;
    iov.length = length;
    iov.memh   = req->recv.state.dt.contig.memh;
    iov.count  = 1;
    iov.stride = 0;
    status = uct_iface_tag_recv_zcopy(ucp_iface->iface, req->recv.tag,
                                      req->recv.tag_mask, &iov, 1,
                                      &req->recv.uct_ctx);
    if (status != UCS_OK) {
        /* No more matching entries in the transport. */
        return 0;
    }
    req->flags |= UCP_REQUEST_FLAG_OFFLOADED;
    ucs_trace_req("recv request %p (%p) was posted to transport (rsc %d)",
                  req, req + 1, ucp_iface->rsc_index);
    return 1;
}

static size_t ucp_tag_offload_pack_eager(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    size_t length;

    length = ucp_dt_pack(req->send.datatype, dest, req->send.buffer,
                         &req->send.state, req->send.length);
    ucs_assert(length == req->send.length);
    return length;
}

static ucs_status_t ucp_tag_offload_eager_short(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_tag_lane(ep);
    status         = uct_ep_tag_eager_short(ep->uct_eps[req->send.lane],
                                            req->send.tag, req->send.buffer,
                                            req->send.length);
    if (status == UCS_OK) {
        ucp_request_complete_send(req, UCS_OK);
    }
    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_do_tag_offload_bcopy(uct_pending_req_t *self, uint64_t imm_data,
                         uct_pack_callback_t pack_cb)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_tag_lane(ep);
    packed_len     = uct_ep_tag_eager_bcopy(ep->uct_eps[req->send.lane],
                                            req->send.tag, imm_data,
                                            pack_cb, req);
    if (packed_len < 0) {
        return packed_len;
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_do_tag_offload_zcopy(uct_pending_req_t *self, uint64_t imm_data,
                         ucp_req_complete_func_t complete)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    size_t max_iov     = ucp_ep_config(ep)->tag.eager.max_iov;
    uct_iov_t *iov     = ucs_alloca(max_iov * sizeof(uct_iov_t));
    size_t iovcnt      = 0;
    ucs_status_t status;
    ucp_dt_state_t saved_state;

    saved_state    = req->send.state;
    req->send.lane = ucp_ep_get_tag_lane(ep);

    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iov, &req->send.state, req->send.buffer,
                        req->send.datatype, req->send.length);

    status = uct_ep_tag_eager_zcopy(ep->uct_eps[req->send.lane], req->send.tag,
                                    imm_data, iov, iovcnt, &req->send.uct_comp);
    if (status == UCS_OK) {
        complete(req);
    } else if (status < 0) {
        req->send.state = saved_state; /* need to restore the offsets state */
        return status;
    } else {
        ucs_assert(status == UCS_INPROGRESS);
    }

    return UCS_OK;
}

static ucs_status_t ucp_tag_offload_eager_bcopy(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_tag_offload_bcopy(self, 0ul,
                                                   ucp_tag_offload_pack_eager);

    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    }
    return status;
}

static ucs_status_t ucp_tag_offload_eager_zcopy(uct_pending_req_t *self)
{
    return ucp_do_tag_offload_zcopy(self, 0ul, ucp_tag_eager_zcopy_req_complete);
}

const ucp_proto_t ucp_tag_offload_proto = {
    .contig_short     = ucp_tag_offload_eager_short,
    .bcopy_single     = ucp_tag_offload_eager_bcopy,
    .bcopy_multi      = NULL,
    .zcopy_single     = ucp_tag_offload_eager_zcopy,
    .zcopy_multi      = NULL,
    .zcopy_completion = ucp_tag_eager_zcopy_completion,
    .only_hdr_size    = 0,
    .first_hdr_size   = 0,
    .mid_hdr_size     = 0
};

/* Eager sync */

const ucp_proto_t ucp_tag_offload_sync_proto = {
    .contig_short     = NULL,
    .bcopy_single     = ucs_empty_function_return_unsupported,
    .bcopy_multi      = NULL,
    .zcopy_single     = ucs_empty_function_return_unsupported,
    .zcopy_multi      = NULL,
    .zcopy_completion = NULL,
    .only_hdr_size    = 0,
    .first_hdr_size   = 0,
    .mid_hdr_size     = 0
};


