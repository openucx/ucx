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
#include <ucp/core/ucp_mm.h>
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
    ucp_worker_iface_t *iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload.ifaces,
                                                              ucp_worker_iface_t, queue);

    req->recv.info.sender_tag = stag;
    req->recv.info.length     = length;

    if (ucs_unlikely(imm)) {
        /* Sync send - need to send a reply */
        ucp_tag_offload_eager_sync_send_ack(req->recv.worker, imm, stag);
    }

    if (req->recv.rdesc != NULL) {
        status = ucp_dt_unpack(req->recv.datatype, req->recv.buffer, req->recv.length,
                               &req->recv.state, req->recv.rdesc + 1, length, 1);
        ucs_mpool_put_inline(req->recv.rdesc);
    } else {
        ucp_request_memory_dereg(ctx, iface->rsc_index, req->recv.datatype,
                                 &req->recv.state);
    }
    --ctx->tm.offload.post_count;
    ucp_request_complete_recv(req, status);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_release_buf(ucp_request_t *req, ucp_context_t *ctx,
                            ucp_rsc_index_t rsc_idx)
{
    if (req->recv.rdesc != NULL) {
        ucs_mpool_put_inline(req->recv.rdesc);
    } else {
        ucp_request_memory_dereg(ctx, rsc_idx, req->recv.datatype,
                                 &req->recv.state);
    }
}

static size_t ucp_tag_offload_fetch_rkey(ucp_ep_t *ep, ucp_sw_rndv_hdr_t *sw_hdr,
                                         ucp_rndv_rts_hdr_t *rts)
{
    uint64_t *addr;
    size_t rkey_size;

    if (sw_hdr->flags & UCP_RNDV_RTS_FLAG_PACKED_RKEY) {
        /* SW RNDV with valid remote addr and key. Must be large message which
         * does not fit offload RNDV proto. */
        rkey_size    = ucp_ep_md_attr(ep, ucp_ep_get_tag_lane(ep))->rkey_packed_size;
        addr         = (uint64_t*)(sw_hdr + 1);
        rts->address = *addr;
        rts->flags  |= sw_hdr->flags | UCP_RNDV_RTS_FLAG_OFFLOAD;
        memcpy(rts + 1, addr + 1, rkey_size);
        return rkey_size;
    }

    rts->address = 0;
    return 0;
}

/* RNDV request matched by the transport. Need to proceed with AM based RNDV */
void ucp_tag_offload_rndv_cb(uct_tag_context_t *self, uct_tag_t stag,
                             const void *header, unsigned header_length,
                             ucs_status_t status)
{
    ucp_request_t *req        = ucs_container_of(self, ucp_request_t, recv.uct_ctx);
    ucp_context_t *ctx        = req->recv.worker->context;
    ucp_sw_rndv_hdr_t *sreq   = (ucp_sw_rndv_hdr_t*)header;
    ucp_ep_t *ep              = ucp_worker_get_reply_ep(req->recv.worker,
                                                        sreq->super.sender_uuid);
    ucp_worker_iface_t *iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload.ifaces,
                                                              ucp_worker_iface_t, queue);
    unsigned length           = sizeof(ucp_rndv_rts_hdr_t);
    ucp_rndv_rts_hdr_t *rts;

    if (sreq->flags & UCP_RNDV_RTS_FLAG_PACKED_RKEY) {
        length += ucp_ep_md_attr(ep, req->send.lane)->rkey_packed_size;
    }
    rts = alloca(length);

    /* Emulate RTS without rkey (to be handled as AM-based RNDV). */
    rts->sreq      = sreq->super;
    rts->super.tag = stag;
    rts->flags     = 0;
    rts->size      = sreq->length;
    ucp_tag_offload_fetch_rkey(ep, sreq, rts);

    ucp_rndv_matched(req->recv.worker, req, rts);
    ucp_tag_offload_release_buf(req, ctx, iface->rsc_index);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_offload_unexp_rndv,
                 (arg, flags, stag, hdr, hdr_length, remote_addr, length, rkey_buf),
                 void *arg, unsigned flags, uint64_t stag, const void *hdr,
                 unsigned hdr_length, uint64_t remote_addr, size_t length,
                 const void *rkey_buf)
{
    ucp_worker_t *worker          = arg;
    ucp_request_hdr_t *rndv_hdr   = (ucp_request_hdr_t*)hdr;
    ucp_ep_t *ep                  = ucp_worker_get_reply_ep(worker, rndv_hdr->sender_uuid);
    const uct_md_attr_t *md_attr  = ucp_ep_md_attr(ep, ucp_ep_get_tag_lane(ep));
    size_t rkey_size        = rkey_buf ? md_attr->rkey_packed_size : 0;
    size_t len              = sizeof(ucp_rndv_rts_hdr_t) + rkey_size;
    ucp_rndv_rts_hdr_t *rts = ucs_alloca(len);
    ucp_sw_rndv_hdr_t  *sw_rndv_hdr;

    /* Fill RTS to emulate SW RNDV flow. */
    rts->super.tag = stag;
    rts->sreq      = *rndv_hdr;
    rts->address   = remote_addr;

    if (remote_addr) {
        rts->size  = length;
        rts->flags = UCP_RNDV_RTS_FLAG_OFFLOAD;
        if (rkey_buf) {
            memcpy(rts + 1, rkey_buf, rkey_size);
            len        += rkey_size;
            rts->flags |= UCP_RNDV_RTS_FLAG_PACKED_RKEY;
        }
    } else {
        /* This must be SW RNDV request. Take length from its header. */
        sw_rndv_hdr = ucs_derived_of(hdr, ucp_sw_rndv_hdr_t);
        rts->size   = sw_rndv_hdr->length;
        rts->flags  = 0;
        len        += ucp_tag_offload_fetch_rkey(ep, sw_rndv_hdr, rts);
    }

    /* Pass 0 as tl flags, because RTS needs to be stored in UCP pool. */
    ucp_rndv_process_rts(arg, rts, len, 0);

    /* Always return UCS_OK, since RNDV hdr should be stored in UCP mpool. */
    return UCS_OK;
}

void ucp_tag_offload_cancel(ucp_context_t *ctx, ucp_request_t *req, int force)
{
    ucp_worker_iface_t *ucp_iface;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
        return;
    }

    ucp_iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload.ifaces,
                                              ucp_worker_iface_t, queue);

    status = uct_iface_tag_recv_cancel(ucp_iface->iface, &req->recv.uct_ctx,
                                       force);
    if (status != UCS_OK) {
        ucs_error("Failed to cancel recv in the transport: %s",
                  ucs_status_string(status));
        return;
    }

    --ctx->tm.offload.post_count;
    ucp_tag_offload_release_buf(req, ctx, ucp_iface->rsc_index);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_offload_post_single(ucp_context_t *ctx, ucp_request_t *req)
{
    size_t length         = req->recv.length;
    ucp_mem_desc_t *rdesc = NULL;
    ucp_worker_iface_t *ucp_iface;
    ucs_status_t status;
    ucp_rsc_index_t mdi;
    uct_iov_t iov;
    ucp_iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload.ifaces,
                                              ucp_worker_iface_t, queue);

    if (ucs_unlikely(length >= ctx->tm.offload.zcopy_thresh)) {
        if (length > ucp_iface->attr.cap.tag.recv.max_zcopy) {
            /* Post maximum allowed length. If sender sends smaller message
             * (which is allowed per MPI standard), max recv should fit it.
             * Otherwise sender will send SW RNDV req, which is small enough. */
            ucs_assert(ucp_iface->attr.cap.tag.rndv.max_zcopy <=
                       ucp_iface->attr.cap.tag.recv.max_zcopy);

            length = ucp_iface->attr.cap.tag.recv.max_zcopy;
        }

        status = ucp_request_memory_reg(ctx, ucp_iface->rsc_index, req->recv.buffer,
                                        length, req->recv.datatype,
                                        &req->recv.state);
        if (status != UCS_OK) {
            return status;
        }

        req->recv.rdesc     = NULL;
        iov.buffer          = (void*)req->recv.buffer;
        iov.memh            = req->recv.state.dt.contig.memh;
    } else {
        rdesc = ucs_mpool_get_inline(&req->recv.worker->reg_mp);
        if (rdesc == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        mdi             = ctx->tl_rscs[ucp_iface->rsc_index].md_index;
        iov.memh        = ucp_memh2uct(rdesc->memh, mdi);
        iov.buffer      = rdesc + 1;
        req->recv.rdesc = rdesc;
    }

    iov.length = length;
    iov.count  = 1;
    iov.stride = 0;

    req->recv.uct_ctx.tag_consumed_cb = ucp_tag_offload_tag_consumed;
    req->recv.uct_ctx.completed_cb    = ucp_tag_offload_completed;
    req->recv.uct_ctx.rndv_cb         = ucp_tag_offload_rndv_cb;

    status = uct_iface_tag_recv_zcopy(ucp_iface->iface, req->recv.tag,
                                      req->recv.tag_mask, &iov, 1,
                                      &req->recv.uct_ctx);
    if (status != UCS_OK) {
        /* No more matching entries in the transport. */
        ucp_tag_offload_release_buf(req, ctx, ucp_iface->rsc_index);
        return status;
    }
    req->flags |= UCP_REQUEST_FLAG_OFFLOADED;
    ++ctx->tm.offload.post_count;

    return UCS_OK;
}


int ucp_tag_offload_post(ucp_context_t *ctx, ucp_request_t *req)
{
    size_t length = req->recv.length;
    ucp_worker_iface_t *ucp_iface;
    ucs_queue_head_t *exp_queue;
    ucp_request_t *req_e;
    ucs_status_t status;
    size_t bucket, max_post;

    if (!UCP_DT_IS_CONTIG(req->recv.datatype)) {
        /* Non-contig buffers not supported yet. */
        goto out_offload_blocked;
    }

    if ((ctx->config.tag_sender_mask & req->recv.tag_mask) !=
         ctx->config.tag_sender_mask) {
        /* Wildcard.
         * TODO add check that only offload capable iface present. In
         * this case can post tag as well. */
        goto out_offload_blocked;
    }


    if (ctx->tm.offload.sw_req_count) {
        ucp_iface = ucs_queue_head_elem_non_empty(&ctx->tm.offload.ifaces,
                    ucp_worker_iface_t, queue);
        max_post = ucp_iface->attr.cap.tag.recv.max_outstanding -
                   ctx->tm.offload.post_count;
        if ((length >= ctx->config.ext.tm_force_thresh) && !ctx->tm.offload.block_count &&
            (ctx->tm.offload.sw_req_count < max_post)) {
            for (bucket = 0; bucket < UCP_TAG_MATCH_QUEUES_NUM; ++bucket) {
                exp_queue = &ctx->tm.expected.hash[bucket];
                ucs_queue_for_each(req_e, exp_queue, recv.queue) {
                    if (!(req_e->flags & UCP_REQUEST_FLAG_OFFLOADED) && (req_e != req)) {
                        status = ucp_tag_offload_post_single(ctx, req_e);
                        if (status != UCS_OK) {
                            return 0;
                        }
                        --ctx->tm.offload.sw_req_count;
                    }
                }
            }
        } else {
            /* There are some requests which must be completed in SW. Do not post
             * tags to HW until they are completed. */
            return 0;
        }
    }

    status = ucp_tag_offload_post_single(ctx, req);
    if (status != UCS_OK) {
        return 0;
    }

    return 1;

out_offload_blocked:
    req->flags |= UCP_REQUEST_FLAG_BLOCK_OFFLOAD;
    ++ctx->tm.offload.block_count;
    return 0;
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
        complete(req, UCS_OK);
    } else if (status < 0) {
        req->send.state = saved_state; /* need to restore the offsets state */
        return status;
    } else {
        ucs_assert(status == UCS_INPROGRESS);
        ++req->send.uct_comp.count;
    }

    req->send.state.offset = req->send.length;

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

ucs_status_t ucp_tag_offload_sw_rndv(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucp_sw_rndv_hdr_t *rndv_hdr;
    unsigned rndv_hdr_len;
    size_t packed_rkey;
    uintptr_t *addr;

    if (UCP_DT_IS_CONTIG(req->send.datatype) &&
        (req->send.length > ucp_ep_config(ep)->tag.offload.max_rndv_zcopy)) {
        packed_rkey  = ucp_ep_md_attr(ep, req->send.lane)->rkey_packed_size;
        rndv_hdr_len = sizeof(ucp_sw_rndv_hdr_t) + packed_rkey + sizeof(uintptr_t);
        rndv_hdr     = ucs_alloca(rndv_hdr_len);
        addr         = (uintptr_t*)(rndv_hdr + 1);
        *addr        = (uintptr_t)req->send.buffer;
        rndv_hdr->flags = 0;
        ucp_tag_rndv_pack_rkey(req, req->send.lane, addr + 1, &rndv_hdr->flags);
    } else {
        rndv_hdr_len = sizeof(ucp_sw_rndv_hdr_t);
        rndv_hdr     = ucs_alloca(rndv_hdr_len);
        rndv_hdr->flags = 0;
    }
    rndv_hdr->super.sender_uuid = ep->worker->uuid;
    rndv_hdr->super.reqptr      = (uintptr_t)req;
    rndv_hdr->length            = req->send.length;

    return uct_ep_tag_rndv_request(ep->uct_eps[req->send.lane], req->send.tag,
                                   rndv_hdr, rndv_hdr_len);
}

static void ucp_tag_rndv_zcopy_completion(uct_completion_t *self,
                                          ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct_comp);
    ucp_tag_eager_zcopy_req_complete(req, status);
}

ucs_status_t ucp_tag_offload_rndv_zcopy(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    size_t max_iov     = ucp_ep_config(ep)->tag.eager.max_iov;
    uct_iov_t *iov     = ucs_alloca(max_iov * sizeof(uct_iov_t));
    size_t iovcnt      = 0;
    ucp_request_hdr_t rndv_hdr = {
        .sender_uuid = ep->worker->uuid,
        .reqptr      = (uintptr_t)req
    };
    void *rndv_op;

    req->send.uct_comp.count = 0;
    req->send.uct_comp.func  = ucp_tag_rndv_zcopy_completion;

    ucs_assert_always(UCP_DT_IS_CONTIG(req->send.datatype));
    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iov, &req->send.state, req->send.buffer,
                        req->send.datatype, req->send.length);

    rndv_op = uct_ep_tag_rndv_zcopy(ep->uct_eps[req->send.lane], req->send.tag,
                                    &rndv_hdr, sizeof(rndv_hdr), iov, iovcnt,
                                    &req->send.uct_comp);
    if (UCS_PTR_IS_ERR(rndv_op)) {
        return UCS_PTR_STATUS(rndv_op);
    }
    ++req->send.uct_comp.count;

    req->flags                   |= UCP_REQUEST_FLAG_OFFLOADED;
    req->send.tag_offload.rndv_op = rndv_op;
    return UCS_OK;
}

void ucp_tag_offload_cancel_rndv(ucp_request_t *req)
{
    ucp_ep_t *ep = req->send.ep;
    ucs_status_t status;

    status = uct_ep_tag_rndv_cancel(ep->uct_eps[ucp_ep_get_tag_lane(ep)],
                                    req->send.tag_offload.rndv_op);
    if (status != UCS_OK) {
        ucs_error("Failed to cancel tag rndv op %s", ucs_status_string(status));
    }
}

ucs_status_t ucp_tag_offload_start_rndv(ucp_request_t *sreq)
{
    ucs_status_t status;
    ucp_ep_t *ep          = sreq->send.ep;
    ucp_lane_index_t lane = ucp_ep_get_tag_lane(ep);

    sreq->send.lane = lane;
    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        (sreq->send.length <= ucp_ep_config(ep)->tag.offload.max_rndv_zcopy)) {
        status = ucp_request_send_buffer_reg(sreq, lane);
        if (status != UCS_OK) {
            return status;
        }
        sreq->send.uct.func = ucp_tag_offload_rndv_zcopy;
    } else {
        sreq->send.uct.func = ucp_tag_offload_sw_rndv;
    }
    return UCS_OK;
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
static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_sync_posted(ucp_worker_t *worker, ucp_request_t *req)
{
    req->send.tag_offload.ssend_tag = req->send.tag;
    ucs_queue_push(&worker->context->tm.offload.sync_reqs,
                   &req->send.tag_offload.queue);
}

static ucs_status_t ucp_tag_offload_eager_sync_bcopy(uct_pending_req_t *self)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_worker_t *worker = req->send.ep->worker;
    ucs_status_t status;

    status = ucp_do_tag_offload_bcopy(self, worker->uuid,
                                      ucp_tag_offload_pack_eager);
    if (status == UCS_OK) {
        ucp_tag_offload_sync_posted(worker, req);
        ucp_request_send_generic_dt_finish(req);
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED,
                                      UCS_OK);
    }
    return status;
}

static ucs_status_t ucp_tag_offload_eager_sync_zcopy(uct_pending_req_t *self)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_worker_t *worker = req->send.ep->worker;
    ucs_status_t status;

    status = ucp_do_tag_offload_zcopy(self, worker->uuid,
                                      ucp_tag_eager_sync_zcopy_req_complete);
    if (status == UCS_OK) {
        ucp_tag_offload_sync_posted(worker, req);
    }
    return status;
}

void ucp_tag_offload_eager_sync_send_ack(ucp_worker_h worker,
                                         uint64_t sender_uuid,
                                         ucp_tag_t sender_tag)
{
    ucp_request_t *req;

    ucs_trace_req("offload_send_sync_ack sender_uuid %"PRIx64" sender_tag %"PRIx64"",
                  sender_uuid, sender_tag);

    req = ucp_worker_allocate_reply(worker, sender_uuid);
    req->send.uct.func          = ucp_proto_progress_am_bcopy_single;
    req->send.proto.am_id       = UCP_AM_ID_OFFLOAD_SYNC_ACK;
    req->send.proto.sender_uuid = sender_uuid;
    req->send.proto.sender_tag  = sender_tag;
    ucp_request_start_send(req);
}

const ucp_proto_t ucp_tag_offload_sync_proto = {
    .contig_short     = NULL,
    .bcopy_single     = ucp_tag_offload_eager_sync_bcopy,
    .bcopy_multi      = NULL,
    .zcopy_single     = ucp_tag_offload_eager_sync_zcopy,
    .zcopy_multi      = NULL,
    .zcopy_completion = ucp_tag_eager_sync_zcopy_completion,
    .only_hdr_size    = 0,
    .first_hdr_size   = 0,
    .mid_hdr_size     = 0
};


