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
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/tag/tag_match.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/sys.h>


static UCS_F_ALWAYS_INLINE ucp_worker_iface_t*
ucp_tag_offload_iface(ucp_worker_t *worker)
{
    ucs_assert(!ucs_queue_is_empty(&worker->tm.offload.ifaces));

    return ucs_queue_head_elem_non_empty(&worker->tm.offload.ifaces,
                                         ucp_worker_iface_t, queue);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_release_buf(ucp_request_t *req, ucp_worker_t *worker)
{
    ucp_worker_iface_t *iface;

    if (req->recv.rdesc != NULL) {
        ucs_mpool_put_inline(req->recv.rdesc);
    } else {
        iface = ucp_tag_offload_iface(worker);
        ucp_request_memory_dereg(worker->context, iface->rsc_index,
                                 req->recv.datatype, &req->recv.state);
    }
}

/* Tag consumed by the transport - need to remove it from expected queue */
void ucp_tag_offload_tag_consumed(uct_tag_context_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, recv.uct_ctx);
    ucs_queue_head_t *queue;

    queue = ucp_tag_exp_get_req_queue(&req->recv.worker->tm, req);
    ucs_queue_remove(queue, &req->recv.queue);
}

/* Message is scattered to user buffer by the transport, complete the request */
void ucp_tag_offload_completed(uct_tag_context_t *self, uct_tag_t stag,
                               uint64_t imm, size_t length, ucs_status_t status)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, recv.uct_ctx);
    ucp_worker_t *worker = req->recv.worker;
    ucp_worker_iface_t *iface;

    req->recv.tag.info.sender_tag = stag;
    req->recv.tag.info.length     = length;

    if (ucs_unlikely(status != UCS_OK)) {
        ucp_tag_offload_release_buf(req, worker);
        goto out;
    }

    if (ucs_unlikely(imm)) {
        /* Sync send - need to send a reply */
        ucp_tag_offload_eager_sync_send_ack(req->recv.worker, imm, stag);
    }

    if (req->recv.rdesc != NULL) {
        status = ucp_dt_unpack(req->recv.datatype, req->recv.buffer,
                               req->recv.length, &req->recv.state,
                               req->recv.rdesc + 1, length,
                               UCP_RECV_DESC_FLAG_LAST);
        ucs_mpool_put_inline(req->recv.rdesc);
    } else {
        iface = ucp_tag_offload_iface(worker);
        ucp_request_memory_dereg(worker->context, iface->rsc_index,
                                 req->recv.datatype, &req->recv.state);
    }

    UCP_WORKER_STAT_TAG_OFFLOAD(req->recv.worker, MATCHED);
out:
    ucp_request_complete_tag_recv(req, status);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_tag_offload_rkey_size(ucp_worker_t *worker)
{
    ucp_worker_iface_t *iface = ucp_tag_offload_iface(worker);
    ucp_context_t *context    = worker->context;
    ucp_md_index_t md_idx;

    md_idx = context->tl_rscs[iface->rsc_index].md_index;
    return context->tl_mds[md_idx].attr.rkey_packed_size;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_tag_offload_copy_rkey(ucp_rndv_rts_hdr_t *rts, const void *rkey_buf,
                          size_t rkey_size)
{
    if (rkey_buf == NULL) {
        return 0;
    }

    ucs_assert(rts != NULL);
    memcpy(rts + 1, rkey_buf, rkey_size);
    rts->flags |= UCP_RNDV_RTS_FLAG_OFFLOAD | UCP_RNDV_RTS_FLAG_PACKED_RKEY;
    return rkey_size;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_fill_rts(ucp_rndv_rts_hdr_t *rts, const ucp_request_hdr_t *hdr,
                         uint64_t stag, uint64_t addr, size_t size,
                         uint16_t flags)
{
    rts->super.tag = stag;
    rts->sreq      = *hdr;
    rts->address   = addr;
    rts->size      = size;
    rts->flags     = flags;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_tag_offload_fill_sw_rts(ucp_rndv_rts_hdr_t *rts,
                            const ucp_sw_rndv_hdr_t *rndv_hdr,
                            unsigned header_length, uint64_t stag,
                            size_t rkey_size)
{
    size_t length = sizeof(*rts);
    ucp_sw_rndv_ext_hdr_t *ext_rndv_hdr;

    if (rndv_hdr->flags & UCP_RNDV_RTS_FLAG_PACKED_RKEY) {
        ucs_assert(rkey_size);
        ucs_assert((sizeof(*ext_rndv_hdr) + rkey_size) == header_length);

        ext_rndv_hdr = ucs_derived_of(rndv_hdr, ucp_sw_rndv_ext_hdr_t);

        ucp_tag_offload_fill_rts(rts, &rndv_hdr->super, stag,
                                 ext_rndv_hdr->address, rndv_hdr->length, 0);

        length += ucp_tag_offload_copy_rkey(rts, ext_rndv_hdr + 1, rkey_size);
    } else {
        ucs_assert(sizeof(*rndv_hdr) == header_length);
        ucp_tag_offload_fill_rts(rts, &rndv_hdr->super, stag, 0,
                                 rndv_hdr->length, 0);
    }

    return length;
}

/* RNDV request matched by the transport. Need to proceed with SW based RNDV */
void ucp_tag_offload_rndv_cb(uct_tag_context_t *self, uct_tag_t stag,
                             const void *header, unsigned header_length,
                             ucs_status_t status)
{
    ucp_request_t *req        = ucs_container_of(self, ucp_request_t, recv.uct_ctx);
    ucp_worker_t *worker      = req->recv.worker;
    ucp_sw_rndv_hdr_t *sw_hdr = (ucp_sw_rndv_hdr_t*)header;
    ucp_rndv_rts_hdr_t *rts;
    size_t rkey_size;

    UCP_WORKER_STAT_TAG_OFFLOAD(worker, MATCHED_SW_RNDV);

    if (ucs_unlikely(status != UCS_OK)) {
        ucp_tag_offload_release_buf(req, worker);
        ucp_request_complete_tag_recv(req, status);
        return;
    }

    rkey_size = (sw_hdr->flags & UCP_RNDV_RTS_FLAG_PACKED_RKEY) ?
                ucp_tag_offload_rkey_size(worker) : 0;

    rts = alloca(sizeof(*rts) + rkey_size);

    ucp_tag_offload_fill_sw_rts(rts, sw_hdr, header_length, stag, rkey_size);

    ucp_rndv_matched(req->recv.worker, req, rts);
    ucp_tag_offload_release_buf(req, worker);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_offload_unexp_rndv,
                 (arg, flags, stag, hdr, hdr_length, remote_addr, length, rkey_buf),
                 void *arg, unsigned flags, uint64_t stag, const void *hdr,
                 unsigned hdr_length, uint64_t remote_addr, size_t length,
                 const void *rkey_buf)
{
    ucp_worker_t *worker        = arg;
    ucp_request_hdr_t *rndv_hdr = (ucp_request_hdr_t*)hdr;
    ucp_rndv_rts_hdr_t *rts;
    size_t len;
    size_t rkey_size;

    rkey_size = ucp_tag_offload_rkey_size(worker);
    rts       = ucs_alloca(sizeof(*rts) + rkey_size); /* SW rndv req may also
                                                         carry a key */
    if (remote_addr) {
        /* Unexpected tag offload RNDV */
        ucp_tag_offload_fill_rts(rts, rndv_hdr, stag, remote_addr, length, 0);
        len = ucp_tag_offload_copy_rkey(rts, (void*)rkey_buf, rkey_size) +
              sizeof(*rts);

        UCP_WORKER_STAT_TAG_OFFLOAD(worker, RX_UNEXP_RNDV);
    } else {
        /* Unexpected tag offload rndv request. Sender buffer is either
           non-contig or it's length > rndv.max_zcopy capability of tag lane */
        len = ucp_tag_offload_fill_sw_rts(rts, (void*)hdr, hdr_length, stag,
                                          rkey_size);

        UCP_WORKER_STAT_TAG_OFFLOAD(worker, RX_UNEXP_SW_RNDV);
    }

    /* Pass 0 as tl flags, because RTS needs to be stored in UCP mpool. */
    ucp_rndv_process_rts(arg, rts, len, 0);

    return UCS_OK;
}

void ucp_tag_offload_cancel(ucp_worker_t *worker, ucp_request_t *req, int force)
{

    ucp_worker_iface_t *ucp_iface = ucp_tag_offload_iface(worker);
    ucs_status_t status;

    status = uct_iface_tag_recv_cancel(ucp_iface->iface, &req->recv.uct_ctx,
                                       force);
    if (status != UCS_OK) {
        ucs_error("Failed to cancel recv in the transport: %s",
                  ucs_status_string(status));
        return;
    }
    UCP_WORKER_STAT_TAG_OFFLOAD(worker, CANCELED);

    /* if cancel is not forced, need to wait its completion */
    if (force) {
        ucp_tag_offload_release_buf(req, worker);
    }
}

int ucp_tag_offload_post(ucp_request_t *req)
{
    size_t length          = req->recv.length;
    ucp_mem_desc_t *rdesc  = NULL;
    ucp_worker_t *worker   = req->recv.worker;
    ucp_context_t *context = worker->context;
    ucp_worker_iface_t *ucp_iface;
    ucs_status_t status;
    ucp_rsc_index_t mdi;
    uct_iov_t iov;

    if (!UCP_DT_IS_CONTIG(req->recv.datatype)) {
        /* Non-contig buffers not supported yet. */
        UCP_WORKER_STAT_TAG_OFFLOAD(worker, BLOCK_NON_CONTIG);
        return 0;
    }

    if ((context->config.tag_sender_mask & req->recv.tag.tag_mask) !=
         context->config.tag_sender_mask) {
        /* Wildcard.
         * TODO add check that only offload capable iface present. In
         * this case can post tag as well. */
        UCP_WORKER_STAT_TAG_OFFLOAD(worker, BLOCK_WILDCARD);
        return 0;
    }

    if (worker->tm.offload.sw_req_count) {
        /* There are some requests which must be completed in SW. Do not post
         * tags to HW until they are completed. */
        UCP_WORKER_STAT_TAG_OFFLOAD(worker, BLOCK_SW_PEND);
        return 0;
    }

    /* Request which was already matched in SW should not be
     * posted to the transport */
    ucs_assert(req->recv.state.offset == 0);

    ucp_iface = ucp_tag_offload_iface(worker);

    if (ucs_unlikely(length >= worker->tm.offload.zcopy_thresh)) {
        if (length > ucp_iface->attr.cap.tag.recv.max_zcopy) {
            /* Post maximum allowed length. If sender sends smaller message
             * (which is allowed per MPI standard), max recv should fit it.
             * Otherwise sender will send SW RNDV req, which is small enough. */
            ucs_assert(ucp_iface->attr.cap.tag.rndv.max_zcopy <=
                       ucp_iface->attr.cap.tag.recv.max_zcopy);

            length = ucp_iface->attr.cap.tag.recv.max_zcopy;
        }

        status = ucp_request_memory_reg(context, ucp_iface->rsc_index,
                                        req->recv.buffer, length,
                                        req->recv.datatype, &req->recv.state);
        if (status != UCS_OK) {
            return 0;
        }

        req->recv.rdesc = NULL;
        iov.buffer      = (void*)req->recv.buffer;
        iov.memh        = req->recv.state.dt.contig[0].memh;
    } else {
        rdesc = ucp_worker_mpool_get(worker);
        if (rdesc == NULL) {
            return 0;
        }

        mdi             = context->tl_rscs[ucp_iface->rsc_index].md_index;
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

    status = uct_iface_tag_recv_zcopy(ucp_iface->iface, req->recv.tag.tag,
                                      req->recv.tag.tag_mask, &iov, 1,
                                      &req->recv.uct_ctx);
    if (status != UCS_OK) {
        /* No more matching entries in the transport. */
        ucp_tag_offload_release_buf(req, worker);
        UCP_WORKER_STAT_TAG_OFFLOAD(worker, BLOCK_TAG_EXCEED);
        return 0;
    }
    UCP_WORKER_STAT_TAG_OFFLOAD(worker, POSTED);
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
                         &req->send.state.dt, req->send.length);
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
    ucp_request_t *req      = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t      *ep       = req->send.ep;
    ucp_dt_state_t dt_state = req->send.state.dt;
    size_t         max_iov  = ucp_ep_config(ep)->tag.eager.max_iov;
    uct_iov_t      *iov     = ucs_alloca(max_iov * sizeof(uct_iov_t));
    size_t         iovcnt   = 0;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_tag_lane(ep);

    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iov, &dt_state, req->send.buffer,
                        req->send.datatype, req->send.length);

    status = uct_ep_tag_eager_zcopy(ep->uct_eps[req->send.lane], req->send.tag,
                                    imm_data, iov, iovcnt,
                                    &req->send.state.uct_comp);
    if (status == UCS_OK) {
        complete(req, UCS_OK);
    } else if (status == UCS_INPROGRESS) {
        ucp_request_send_state_advance(req, &dt_state,
                                       UCP_REQUEST_SEND_PROTO_ZCOPY_AM, status);
    }

    return UCS_STATUS_IS_ERR(status) ? status : UCS_OK;
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
    return ucp_do_tag_offload_zcopy(self, 0ul,
                                    ucp_proto_am_zcopy_req_complete);
}

static size_t ucp_tag_offload_rndv_pack_rkey(ucp_request_t *sreq, ucp_lane_index_t lane,
                                             void *rkey_buf, uint16_t *flags)
{
    ucp_ep_h ep = sreq->send.ep;
    ucs_status_t status;

    ucs_assert(UCP_DT_IS_CONTIG(sreq->send.datatype));

    /* Check if the sender needs to register the send buffer -
     * is its datatype contiguous and does the receive side need it */
    if (ucp_ep_md_attr(ep, lane)->cap.flags & UCT_MD_FLAG_NEED_RKEY) {
        status = ucp_request_send_buffer_reg(sreq, lane);
        ucs_assert_always(status == UCS_OK);

        /* if the send buffer was registered, send the rkey */
        UCS_PROFILE_CALL(uct_md_mkey_pack, ucp_ep_md(ep, lane),
                         sreq->send.state.dt.dt.contig[0].memh, rkey_buf);
        *flags |= UCP_RNDV_RTS_FLAG_PACKED_RKEY;
        return ucp_ep_md_attr(ep, lane)->rkey_packed_size;
    }

    return 0;
}

ucs_status_t ucp_tag_offload_sw_rndv(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucp_sw_rndv_hdr_t *rndv_hdr;
    ucp_sw_rndv_ext_hdr_t *ext_rndv_hdr;
    unsigned rndv_hdr_len;
    size_t packed_rkey;

    if (UCP_DT_IS_CONTIG(req->send.datatype) &&
        (req->send.length > ucp_ep_config(ep)->tag.offload.max_rndv_zcopy)) {
        packed_rkey  = ucp_ep_md_attr(ep, req->send.lane)->rkey_packed_size;
        rndv_hdr_len = sizeof(ucp_sw_rndv_ext_hdr_t) + packed_rkey;
        ext_rndv_hdr = ucs_alloca(rndv_hdr_len);
        ext_rndv_hdr->address     = (uintptr_t)req->send.buffer;
        ext_rndv_hdr->super.flags = 0;
        ucp_tag_offload_rndv_pack_rkey(req, req->send.lane, ext_rndv_hdr + 1,
                                       &ext_rndv_hdr->super.flags);
        rndv_hdr = &ext_rndv_hdr->super;
    } else {
        rndv_hdr_len    = sizeof(ucp_sw_rndv_hdr_t);
        rndv_hdr        = ucs_alloca(rndv_hdr_len);
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
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucp_proto_am_zcopy_req_complete(req, status);
}

ucs_status_t ucp_tag_offload_rndv_zcopy(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    size_t max_iov     = ucp_ep_config(ep)->tag.eager.max_iov;
    uct_iov_t *iov     = ucs_alloca(max_iov * sizeof(uct_iov_t));
    size_t iovcnt      = 0;
    ucp_dt_state_t dt_state;
    void *rndv_op;

    ucp_request_hdr_t rndv_hdr = {
        .sender_uuid = ep->worker->uuid,
        .reqptr      = (uintptr_t)req
    };

    dt_state = req->send.state.dt;

    ucs_assert_always(UCP_DT_IS_CONTIG(req->send.datatype));
    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iov, &dt_state, req->send.buffer,
                        req->send.datatype, req->send.length);

    rndv_op = uct_ep_tag_rndv_zcopy(ep->uct_eps[req->send.lane], req->send.tag,
                                    &rndv_hdr, sizeof(rndv_hdr), iov, iovcnt,
                                    &req->send.state.uct_comp);
    if (UCS_PTR_IS_ERR(rndv_op)) {
        return UCS_PTR_STATUS(rndv_op);
    }
    ucp_request_send_state_advance(req, &dt_state,
                                   UCP_REQUEST_SEND_PROTO_RNDV_GET,
                                   UCS_INPROGRESS);

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
        ucp_request_send_state_reset(sreq, ucp_tag_rndv_zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_RNDV_GET);
        status = ucp_request_send_buffer_reg(sreq, lane);
        if (status != UCS_OK) {
            return status;
        }
        sreq->send.uct.func = ucp_tag_offload_rndv_zcopy;
    } else {
        ucp_request_send_state_reset(sreq, NULL, UCP_REQUEST_SEND_PROTO_RNDV_GET);
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
    .zcopy_completion = ucp_proto_am_zcopy_completion,
    .only_hdr_size    = 0,
    .first_hdr_size   = 0,
    .mid_hdr_size     = 0
};


/* Eager sync */
static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_sync_posted(ucp_worker_t *worker, ucp_request_t *req)
{
    req->send.tag_offload.ssend_tag = req->send.tag;
    ucs_queue_push(&worker->tm.offload.sync_reqs, &req->send.tag_offload.queue);
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
    ucp_request_send(req);
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


