/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "rndv.h"
#include "tag_match.inl"

#include "offload.h"
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_am.inl>
#include <ucs/datastruct/queue.h>

static int ucp_rndv_is_get_zcopy(ucp_request_t *sreq, ucp_rndv_mode_t rndv_mode)
{
    return ((rndv_mode == UCP_RNDV_MODE_GET_ZCOPY) ||
            ((rndv_mode == UCP_RNDV_MODE_AUTO) &&
             UCP_MEM_IS_HOST(sreq->send.mem_type)));
}

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq              = arg;   /* send request */
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = dest;
    ucp_worker_h worker              = sreq->send.ep->worker;
    ssize_t packed_rkey_size;

    rndv_rts_hdr->super.tag        = sreq->send.tag.tag;
    rndv_rts_hdr->sreq.reqptr      = (uintptr_t)sreq;
    rndv_rts_hdr->sreq.sender_uuid = worker->uuid;
    rndv_rts_hdr->size             = sreq->send.length;

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq, worker->context->config.ext.rndv_mode)) {
        /* pack rkey, ask target to do get_zcopy */
        rndv_rts_hdr->address = (uintptr_t)sreq->send.buffer;
        packed_rkey_size = ucp_rkey_pack_uct(worker->context,
                                             sreq->send.state.dt.dt.contig.md_map,
                                             sreq->send.state.dt.dt.contig.memh,
                                             rndv_rts_hdr + 1);
        if (packed_rkey_size < 0) {
            ucs_fatal("failed to pack rendezvous remote key: %s",
                      ucs_status_string(packed_rkey_size));
        }
    } else {
        rndv_rts_hdr->address = 0;
        packed_rkey_size      = 0;
    }

    return sizeof(*rndv_rts_hdr) + packed_rkey_size;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    /* send the RTS. the pack_cb will pack all the necessary fields in the RTS */
    return ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_RTS, ucp_tag_rndv_rts_pack);
}

static size_t ucp_tag_rndv_rtr_pack(void *dest, void *arg)
{
    ucp_request_t *rndv_req          = arg;
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = dest;
    ucp_request_t *rreq              = rndv_req->send.rndv_rtr.rreq;
    ssize_t packed_rkey_size;

    rndv_rtr_hdr->sreq_ptr = rndv_req->send.rndv_rtr.remote_request;
    rndv_rtr_hdr->rreq_ptr = (uintptr_t)rreq; /* request of receiver side */

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        rndv_rtr_hdr->address = (uintptr_t)rreq->recv.buffer;
        packed_rkey_size = ucp_rkey_pack_uct(rndv_req->send.ep->worker->context,
                                             rreq->recv.state.dt.contig.md_map,
                                             rreq->recv.state.dt.contig.memh,
                                             rndv_rtr_hdr + 1);
        if (packed_rkey_size < 0) {
            return packed_rkey_size;
        }
    } else {
        rndv_rtr_hdr->address = 0;
        packed_rkey_size      = 0;
    }

    return sizeof(*rndv_rtr_hdr) + packed_rkey_size;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rtr, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    /* send the RTR. the pack_cb will pack all the necessary fields in the RTR */
    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_RTR, ucp_tag_rndv_rtr_pack);
    if (status == UCS_OK) {
        ucp_request_put(rndv_req);
    }

    return status;
}

ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *sreq)
{
    ucp_ep_h ep = sreq->send.ep;
    ucp_md_map_t md_map;
    ucs_status_t status;

    ucp_trace_req(sreq, "start_rndv to %s buffer %p length %zu",
                  ucp_ep_peer_name(ep), sreq->send.buffer,
                  sreq->send.length);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    if (ep->flags & UCP_EP_FLAG_TAG_OFFLOAD_ENABLED) {
        status = ucp_tag_offload_start_rndv(sreq);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
            ucp_rndv_is_get_zcopy(sreq, ep->worker->context->config.ext.rndv_mode)) {
            /* register a contiguous buffer for rma_get */
            md_map = ucp_ep_config(ep)->key.rma_bw_md_map;
            status = ucp_request_send_buffer_reg(sreq, md_map);
            if (status != UCS_OK) {
                return status;
            }
        }

        ucs_assert(sreq->send.lane == ucp_ep_get_am_lane(ep));
        sreq->send.uct.func = ucp_proto_progress_rndv_rts;
    }

    ucp_ep_connect_remote(ep);
    return UCS_OK;
}

static void ucp_rndv_complete_send(ucp_request_t *sreq)
{
    ucp_request_send_generic_dt_finish(sreq);
    ucp_request_send_buffer_dereg(sreq);
    ucp_request_complete_send(sreq, UCS_OK);
}

static void ucp_rndv_req_send_ats(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                  uintptr_t remote_request)
{
    ucp_trace_req(rndv_req, "send ats remote_request 0x%lx", remote_request);
    UCS_PROFILE_REQUEST_EVENT(rreq, "send_ats", 0);

    rndv_req->send.lane         = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func     = ucp_proto_progress_am_bcopy_single;
    rndv_req->send.proto.am_id  = UCP_AM_ID_RNDV_ATS;
    rndv_req->send.proto.status = UCS_OK;
    rndv_req->send.proto.remote_request = remote_request;
    rndv_req->send.proto.comp_cb = ucp_request_put;

    ucp_request_send(rndv_req);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_complete_rma_put_zcopy, (sreq),
                      ucp_request_t *sreq)
{
    ucp_trace_req(sreq, "rndv_put completed");
    UCS_PROFILE_REQUEST_EVENT(sreq, "complete_rndv_put", 0);

    ucp_rkey_destroy(sreq->send.rndv_put.rkey);
    ucp_request_send_buffer_dereg(sreq);
    ucp_request_complete_send(sreq, UCS_OK);
}

static void ucp_rndv_send_atp(ucp_request_t *sreq, uintptr_t remote_request)
{
    ucs_assertv(sreq->send.state.dt.offset == sreq->send.length,
                "sreq=%p offset=%zu length=%zu", sreq,
                sreq->send.state.dt.offset, sreq->send.length);

    ucp_trace_req(sreq, "send atp remote_request 0x%lx", remote_request);
    UCS_PROFILE_REQUEST_EVENT(sreq, "send_atp", 0);

    sreq->send.lane                 = ucp_ep_get_am_lane(sreq->send.ep);
    sreq->send.uct.func             = ucp_proto_progress_am_bcopy_single;
    sreq->send.proto.am_id          = UCP_AM_ID_RNDV_ATP;
    sreq->send.proto.status         = UCS_OK;
    sreq->send.proto.remote_request = remote_request;
    sreq->send.proto.comp_cb        = ucp_rndv_complete_rma_put_zcopy;

    ucp_request_send(sreq);
}

static void ucp_rndv_zcopy_recv_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_request_recv_buffer_dereg(req);
    ucp_request_complete_tag_recv(req, status);
}

static void ucp_rndv_complete_rma_get_zcopy(ucp_request_t *rndv_req)
{
    ucp_request_t *rreq = rndv_req->send.rndv_get.rreq;

    ucs_assertv(rndv_req->send.state.dt.offset == rndv_req->send.length,
                "rndv_req=%p offset=%zu length=%zu", rndv_req,
                rndv_req->send.state.dt.offset, rndv_req->send.length);

    ucp_trace_req(rndv_req, "rndv_get completed");
    UCS_PROFILE_REQUEST_EVENT(rreq, "complete_rndv_get", 0);

    ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
    ucp_request_send_buffer_dereg(rndv_req);

    ucp_rndv_req_send_ats(rndv_req, rreq, rndv_req->send.rndv_get.remote_request);
    ucp_rndv_zcopy_recv_req_complete(rreq, UCS_OK);
}

static void ucp_rndv_recv_data_init(ucp_request_t *rreq, size_t size)
{
    rreq->status             = UCS_OK;
    rreq->recv.tag.remaining = size;
}

static void ucp_rndv_req_send_rtr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                   uintptr_t sender_reqptr)
{
    ucp_trace_req(rndv_req, "send rtr remote sreq 0x%lx rreq %p", sender_reqptr,
                  rreq);

    rndv_req->send.lane                    = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func                = ucp_proto_progress_rndv_rtr;
    rndv_req->send.rndv_rtr.remote_request = sender_reqptr;
    rndv_req->send.rndv_rtr.rreq           = rreq;

    ucp_request_send(rndv_req);
}

static void ucp_rndv_get_lanes_count(ucp_request_t *req)
{
    ucp_ep_h ep        = req->send.ep;
    ucp_lane_map_t map = 0;
    uct_rkey_t uct_rkey;
    ucp_lane_index_t lane;

    if (ucs_likely(req->send.rndv_get.lane_count != 0)) {
        return; /* already resolved */
    }

    while ((lane = ucp_rkey_get_rma_bw_lane(req->send.rndv_get.rkey, ep,
                                            &uct_rkey, map)) != UCP_NULL_LANE) {
        req->send.rndv_get.lane_count++;
        map |= UCS_BIT(lane);
    }

    req->send.rndv_get.lane_count = ucs_min(req->send.rndv_get.lane_count,
                                            ep->worker->context->config.ext.max_rndv_lanes);
}

static ucp_lane_index_t ucp_rndv_get_next_lane(ucp_request_t *rndv_req, uct_rkey_t *uct_rkey)
{
    /* get lane and mask it for next iteration.
     * next time this lane will not be selected & we continue
     * with another lane. After all lanes are masked - reset mask
     * to zero & start from scratch. this way allows to enumerate
     * all lanes */
    ucp_ep_h ep = rndv_req->send.ep;
    ucp_lane_index_t lane;

    lane = ucp_rkey_get_rma_bw_lane(rndv_req->send.rndv_get.rkey, ep,
                                    uct_rkey, rndv_req->send.rndv_get.lanes_map);

    if ((lane == UCP_NULL_LANE) && (rndv_req->send.rndv_get.lanes_map != 0)) {
        /* lanes_map != 0 - no more lanes (but BW lanes are exist because map
         * is not NULL - we found at least one lane on previous iteration).
         * reset used lanes map to NULL and iterate it again */
        rndv_req->send.rndv_get.lanes_map = 0;
        lane = ucp_rkey_get_rma_bw_lane(rndv_req->send.rndv_get.rkey, ep,
                                        uct_rkey, rndv_req->send.rndv_get.lanes_map);
    }

    if (ucs_unlikely(lane == UCP_NULL_LANE)) {
        /* there are no BW lanes */
        return UCP_NULL_LANE;
    }

    rndv_req->send.rndv_get.lanes_map |= UCS_BIT(lane);
    /* in case if masked too much lanes - reset mask to zero
     * to select first lane next time */
    if (ucs_count_one_bits(rndv_req->send.rndv_get.lanes_map) >=
        ep->worker->context->config.ext.max_rndv_lanes) {
        rndv_req->send.rndv_get.lanes_map = 0;
    }
    return lane;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_get_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep             = rndv_req->send.ep;
    const size_t max_iovcnt = 1;
    ucs_status_t status;
    size_t offset, length, ucp_mtu, remainder, align, chunk;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_rsc_index_t rsc_index;
    ucp_dt_state_t state;
    uct_rkey_t uct_rkey;
    size_t min_zcopy;
    size_t max_zcopy;
    size_t tail;

    if (ucp_ep_is_stub(ep)) {
        rndv_req->send.lane = 0;
        return UCS_ERR_NO_RESOURCE;
    }

    ucp_rndv_get_lanes_count(rndv_req);

    /* Figure out which lane to use for get operation */
    rndv_req->send.lane = ucp_rndv_get_next_lane(rndv_req, &uct_rkey);

    if (rndv_req->send.lane == UCP_NULL_LANE) {
        /* If can't perform get_zcopy - switch to active-message.
         * NOTE: we do not register memory and do not send our keys. */
        ucp_trace_req(rndv_req, "remote memory unreachable, switch to rtr");
        ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
        ucp_rndv_recv_data_init(rndv_req->send.rndv_get.rreq,
                                rndv_req->send.length);
        ucp_rndv_req_send_rtr(rndv_req, rndv_req->send.rndv_get.rreq,
                              rndv_req->send.rndv_get.remote_request);
        return UCS_OK;
    }

    if (!rndv_req->send.mdesc) {
        status = ucp_send_request_add_reg_lane(rndv_req, rndv_req->send.lane);
        ucs_assert_always(status == UCS_OK);
    }

    rsc_index = ucp_ep_get_rsc_index(rndv_req->send.ep, rndv_req->send.lane);
    align     = rndv_req->send.ep->worker->ifaces[rsc_index].attr.cap.get.opt_zcopy_align;
    ucp_mtu   = rndv_req->send.ep->worker->ifaces[rsc_index].attr.cap.get.align_mtu;
    min_zcopy = rndv_req->send.ep->worker->ifaces[rsc_index].attr.cap.get.min_zcopy;
    max_zcopy = ucp_ep_config(rndv_req->send.ep)->tag.rndv.max_get_zcopy;

    offset    = rndv_req->send.state.dt.offset;
    remainder = (uintptr_t)rndv_req->send.buffer % align;

    if ((offset == 0) && (remainder > 0) && (rndv_req->send.length > ucp_mtu)) {
        length = ucp_mtu - remainder;
    } else {
        chunk = ucs_align_up(rndv_req->send.length /
                             rndv_req->send.rndv_get.lane_count, align);
        length = ucs_min(ucs_min(chunk, rndv_req->send.length - offset), max_zcopy);
    }

    /* ensure that tail (rest of message) is over min_zcopy */
    tail = rndv_req->send.length - (offset + length);
    if (ucs_unlikely(tail && (tail < min_zcopy))) {
        /* ok, tail is less get zcopy minimal & could not be processed as
         * standalone operation */
        /* check if we have room to increase current part and not
         * step over max_zcopy */
        if (length < (max_zcopy - tail)) {
            /* if we can encrease length by min_zcopy - let's do it to
             * avoid small tail (we have limitation on minimal get zcopy) */
            length += tail;
        } else {
            /* reduce current length by align or min_zcopy value
             * to process it on next round */
            ucs_assert(length > ucs_max(min_zcopy, align));
            length -= ucs_max(min_zcopy, align);
        }
    }

    ucs_assert(length >= min_zcopy);
    ucs_assert((rndv_req->send.length - (offset + length) == 0) ||
               (rndv_req->send.length - (offset + length) >= min_zcopy));

    ucs_trace_data("req %p: offset %zu remainder %zu rma-get to %p len %zu lane %d",
                   rndv_req, offset, remainder, rndv_req->send.buffer + offset,
                   length, rndv_req->send.lane);

    state = rndv_req->send.state.dt;
    /* TODO: is this correct? memh array may skip MD's where
     * registration is not supported. for now SHM may avoid registration,
     * but it will work on single lane */
    ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iovcnt, &state,
                        rndv_req->send.buffer, ucp_dt_make_contig(1), length,
                        ucp_ep_md_index(rndv_req->send.ep, rndv_req->send.lane),
                        rndv_req->send.mdesc);

    for (;;) {
        status = uct_ep_get_zcopy(ep->uct_eps[rndv_req->send.lane],
                                  iov, iovcnt,
                                  rndv_req->send.rndv_get.remote_address + offset,
                                  uct_rkey,
                                  &rndv_req->send.state.uct_comp);
        ucp_request_send_state_advance(rndv_req, &state,
                                       UCP_REQUEST_SEND_PROTO_RNDV_GET,
                                       status);
        if (rndv_req->send.state.dt.offset == rndv_req->send.length) {
            if (rndv_req->send.state.uct_comp.count == 0) {
                ucp_rndv_complete_rma_get_zcopy(rndv_req);
            }
            return UCS_OK;
        } else if (!UCS_STATUS_IS_ERR(status)) {
            /* in case if not all chunks are transmitted - return in_progress
             * status */
            return UCS_INPROGRESS;
        } else {
            if (status == UCS_ERR_NO_RESOURCE) {
                if (rndv_req->send.lane != rndv_req->send.pending_lane) {
                    /* switch to new pending lane */
                    int pending_adde_res = ucp_request_pending_add(rndv_req, &status);
                    if (!pending_adde_res) {
                        /* failed to switch req to pending queue, try again */
                        continue;
                    }
                    ucs_assert(status == UCS_INPROGRESS);
                    return UCS_OK;
                }
            }
            return status;
        }
    }
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_get_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t,
                                               send.state.uct_comp);

    if (rndv_req->send.state.dt.offset == rndv_req->send.length) {
        ucp_rndv_complete_rma_get_zcopy(rndv_req);
    }
}

static void ucp_rndv_put_completion(uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);

    if (sreq->send.state.dt.offset == sreq->send.length) {
        ucp_rndv_send_atp(sreq, sreq->send.rndv_put.remote_request);
    }
}

static void ucp_rndv_req_send_rma_get(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucs_status_t status;

    ucp_trace_req(rndv_req, "start rma_get rreq %p", rreq);

    rndv_req->send.uct.func                = ucp_rndv_progress_rma_get_zcopy;
    rndv_req->send.buffer                  = rreq->recv.buffer;
    rndv_req->send.datatype                = ucp_dt_make_contig(1);
    rndv_req->send.length                  = rndv_rts_hdr->size;
    rndv_req->send.rndv_get.remote_request = rndv_rts_hdr->sreq.reqptr;
    rndv_req->send.rndv_get.remote_address = rndv_rts_hdr->address;
    rndv_req->send.rndv_get.rreq           = rreq;
    rndv_req->send.rndv_get.lanes_map      = 0;
    rndv_req->send.rndv_get.lane_count     = 0;
    rndv_req->send.datatype                = rreq->recv.datatype;

    status = ucp_ep_rkey_unpack(rndv_req->send.ep, rndv_rts_hdr + 1,
                                &rndv_req->send.rndv_get.rkey);
    if (status != UCS_OK) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(rndv_req->send.ep), ucs_status_string(status));
    }

    ucp_request_send_state_init(rndv_req, ucp_dt_make_contig(1), 0);
    ucp_request_send_state_reset(rndv_req, ucp_rndv_get_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_GET);

    ucp_request_send(rndv_req);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_matched, (worker, rreq, rndv_rts_hdr),
                      ucp_worker_h worker, ucp_request_t *rreq,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_rndv_mode_t rndv_mode;
    ucp_request_t *rndv_req;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_match", 0);

    /* rreq is the receive request on the receiver's side */
    rreq->recv.tag.info.sender_tag = rndv_rts_hdr->super.tag;
    rreq->recv.tag.info.length     = rndv_rts_hdr->size;

    /* the internal send request allocated on receiver side (to perform a "get"
     * operation, send "ATS" and "RTR") */
    rndv_req = ucp_worker_allocate_reply(worker, rndv_rts_hdr->sreq.sender_uuid);

    ucp_trace_req(rreq,
                  "rndv matched remote {address 0x%"PRIx64" size %zu sreq 0x%lx}"
                  " rndv_sreq %p", rndv_rts_hdr->address, rndv_rts_hdr->size,
                  rndv_rts_hdr->sreq.reqptr, rndv_req);

    if (ucs_unlikely(rreq->recv.length < rndv_rts_hdr->size)) {
        ucp_trace_req(rndv_req,
                      "rndv truncated remote size %zu local size %zu rreq %p",
                      rndv_rts_hdr->size, rreq->recv.length, rreq);
        ucp_rndv_req_send_ats(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr);
        ucp_request_recv_generic_dt_finish(rreq);
        ucp_rndv_zcopy_recv_req_complete(rreq, UCS_ERR_MESSAGE_TRUNCATED);
        goto out;
    }

    rndv_req->send.pending_lane = UCP_NULL_LANE;

    /* if the receive side is not connected yet then the RTS was received on a stub ep */
    ep = rndv_req->send.ep;
    if (ucp_ep_is_stub(ep)) {
        ucs_debug("received rts on a stub ep, ep %p am_lane %d", ep,
                  ucp_ep_get_am_lane(ep));
    }

    rndv_mode = worker->context->config.ext.rndv_mode;
    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        if (rndv_rts_hdr->address && (rndv_mode != UCP_RNDV_MODE_PUT_ZCOPY)) {
            /* try to fetch the data with a get_zcopy operation */
            ucp_rndv_req_send_rma_get(rndv_req, rreq, rndv_rts_hdr);
            goto out;
        } else if (rndv_mode != UCP_RNDV_MODE_GET_ZCOPY) {
            /* put protocol is allowed - register receive buffer memory for rma */
            ucp_request_recv_buffer_reg(rreq, ucp_ep_config(ep)->key.rma_bw_md_map);
        }
    }

    /* The sender didn't specify its address in the RTS, or the rndv mode was
     * configured to put - send an RTR and the sender will send the data with
     * active message or put_zcopy. */
    ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
    ucp_rndv_req_send_rtr(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_rndv_process_rts(void *arg, void *data, size_t length,
                                  unsigned tl_flags)
{
    ucp_worker_h worker                = arg;
    ucp_rndv_rts_hdr_t *rndv_rts_hdr   = data;
    ucp_recv_desc_t *rdesc;
    ucp_request_t *rreq;
    ucs_status_t status;

    rreq = ucp_tag_exp_search(&worker->tm, rndv_rts_hdr->super.tag);
    if (rreq != NULL) {
        ucp_rndv_matched(worker, rreq, rndv_rts_hdr);

        /* Cancel req in transport if it was offloaded, because it arrived
           as unexpected */
        ucp_tag_offload_try_cancel(worker, rreq, UCP_TAG_OFFLOAD_CANCEL_FORCE);

        UCP_WORKER_STAT_RNDV(worker, EXP);
        status = UCS_OK;
    } else {
        status = ucp_recv_desc_init(worker, data, length, tl_flags,
                                    sizeof(*rndv_rts_hdr),
                                    UCP_RECV_DESC_FLAG_RNDV, &rdesc);
        if (!UCS_STATUS_IS_ERR(status)) {
            ucp_tag_unexp_recv(&worker->tm, rdesc, rndv_rts_hdr->super.tag);
        }
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rts_handler,
                 (arg, data, length, tl_flags),
                 void *arg, void *data, size_t length, unsigned tl_flags)
{
    return ucp_rndv_process_rts(arg, data, length, tl_flags);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_ats_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *sreq = (ucp_request_t*) rep_hdr->reqptr;

    /* dereg the original send request and set it to complete */
    UCS_PROFILE_REQUEST_EVENT(sreq, "rndv_ats_recv", 0);
    if (sreq->flags & UCP_REQUEST_FLAG_OFFLOADED) {
        ucp_tag_offload_cancel_rndv(sreq);
    }
    ucp_rndv_complete_send(sreq);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_atp_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *rreq = (ucp_request_t*) rep_hdr->reqptr;

    /* dereg the original recv request and set it to complete */
    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_atp_recv", 0);
    ucp_rndv_zcopy_recv_req_complete(rreq, UCS_OK);
    return UCS_OK;
}

static size_t ucp_rndv_pack_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    hdr->rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    hdr->offset   = sreq->send.state.dt.offset;
    length        = ucp_ep_get_max_bcopy(sreq->send.ep, sreq->send.lane) - sizeof(*hdr);

    return sizeof(*hdr) + ucp_dt_pack(sreq->send.datatype, hdr + 1,
                                      sreq->send.buffer, &sreq->send.state.dt,
                                      length);
}

static size_t ucp_rndv_pack_data_last(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length, offset;

    offset        = sreq->send.state.dt.offset;
    hdr->rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    length        = sreq->send.length - offset;
    hdr->offset   = offset;

    return sizeof(*hdr) + ucp_dt_pack(sreq->send.datatype, hdr + 1,
                                      sreq->send.buffer, &sreq->send.state.dt,
                                      length);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_am_bcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = sreq->send.ep;
    ucs_status_t status;

    sreq->send.lane = ucp_ep_get_am_lane(ep);

    if (sreq->send.length <= ucp_ep_config(ep)->am.max_bcopy - sizeof(ucp_rndv_data_hdr_t)) {
        /* send a single bcopy message */
        status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_DATA,
                                        ucp_rndv_pack_data_last);
    } else {
        /* send multiple bcopy messages (fragments of the original send message) */
        status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       sizeof(ucp_rndv_data_hdr_t),
                                       ucp_rndv_pack_data,
                                       ucp_rndv_pack_data,
                                       ucp_rndv_pack_data_last, 1);
    }
    if (status == UCS_OK) {
        ucp_rndv_complete_send(sreq);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_put_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq     = ucs_container_of(self, ucp_request_t, send.uct);
    const size_t max_iovcnt = 1;
    ucp_ep_h ep             = sreq->send.ep;
    ucs_status_t status;
    size_t offset, ucp_mtu, align, remainder, length;
    ucp_rsc_index_t rsc_index;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_dt_state_t state;

    if (!sreq->send.mdesc) {
        status = ucp_request_send_buffer_reg_lane(sreq, sreq->send.lane);
        ucs_assert_always(status == UCS_OK);
    }

    rsc_index = ucp_ep_get_rsc_index(ep, sreq->send.lane);
    align     = ep->worker->ifaces[rsc_index].attr.cap.put.opt_zcopy_align;
    ucp_mtu   = ep->worker->ifaces[rsc_index].attr.cap.put.align_mtu;

    offset    = sreq->send.state.dt.offset;
    remainder = (uintptr_t)sreq->send.buffer % align;

    if ((offset == 0) && (remainder > 0) && (sreq->send.length > ucp_mtu)) {
        length = ucp_mtu - remainder;
    } else {
        length = ucs_min(sreq->send.length - offset,
                         ucp_ep_config(ep)->tag.rndv.max_put_zcopy);
    }

    ucs_trace_data("req %p: offset %zu remainder %zu. read to %p len %zu",
                   sreq, offset, (uintptr_t)sreq->send.buffer % align,
                   (void*)sreq->send.buffer + offset, length);

    state = sreq->send.state.dt;
    ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iovcnt, &state,
                        sreq->send.buffer, ucp_dt_make_contig(1), length, 0,
                        sreq->send.mdesc);
    status = uct_ep_put_zcopy(ep->uct_eps[sreq->send.lane],
                              iov, iovcnt,
                              sreq->send.rndv_put.remote_address + offset,
                              sreq->send.rndv_put.uct_rkey,
                              &sreq->send.state.uct_comp);
    ucp_request_send_state_advance(sreq, &state,
                                   UCP_REQUEST_SEND_PROTO_RNDV_PUT,
                                   status);
    if (sreq->send.state.dt.offset == sreq->send.length) {
        if (sreq->send.state.uct_comp.count == 0) {
            ucp_rndv_send_atp(sreq, sreq->send.rndv_put.remote_request);
        }
        return UCS_OK;
    } else if (!UCS_STATUS_IS_ERR(status)) {
        return UCS_INPROGRESS;
    } else {
        return status;
    }
}

static void ucp_rndv_am_zcopy_send_req_complete(ucp_request_t *req,
                                                ucs_status_t status)
{
    ucp_request_send_buffer_dereg(req);
    ucp_request_complete_send(req, status);
}

static void ucp_rndv_am_zcopy_completion(uct_completion_t *self,
                                         ucs_status_t status)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    if (sreq->send.state.dt.offset == sreq->send.length) {
        ucp_rndv_am_zcopy_send_req_complete(sreq, status);
    } else if (status != UCS_OK) {
        ucs_fatal("error handling is unsupported with rendezvous protocol");
    }
}

static ucs_status_t ucp_rndv_progress_am_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    hdr.offset   = 0;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_RNDV_DATA, &hdr, sizeof(hdr),
                                  ucp_rndv_am_zcopy_send_req_complete);
}

static ucs_status_t ucp_rndv_progress_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    hdr.offset   = sreq->send.state.dt.offset;
    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_rndv_am_zcopy_send_req_complete, 1);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_frag_put_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *frag_req = ucs_container_of(self, ucp_request_t, send.state.uct_comp);
    ucp_request_t *sreq     = frag_req->send.rndv_put.sreq;

    ucs_mpool_put_inline((void *)frag_req->send.mdesc);
    sreq->send.state.uct_comp.count--;
    if (0 == sreq->send.state.uct_comp.count) {
        ucp_rndv_send_atp(sreq, sreq->send.rndv_put.remote_request);
    }
    ucp_request_put(frag_req);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_frag_get_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *frag_req = ucs_container_of(self, ucp_request_t, send.state.uct_comp);
    ucp_request_t *sreq     = frag_req->send.rndv_get.rreq;
    size_t offset           = frag_req->send.rndv_get.remote_address - (uint64_t)(sreq->send.buffer);

    frag_req->send.ep                      = sreq->send.ep;
    ucp_request_send_state_reset(frag_req, ucp_rndv_frag_put_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    frag_req->send.uct.func                = ucp_rndv_progress_rma_put_zcopy;
    frag_req->send.rndv_put.sreq           = sreq;
    frag_req->send.rndv_put.rkey           = sreq->send.rndv_put.rkey;
    frag_req->send.rndv_put.uct_rkey       = sreq->send.rndv_put.uct_rkey;
    frag_req->send.rndv_put.remote_address = sreq->send.rndv_put.remote_address + offset;
    frag_req->send.lane                    = sreq->send.lane;
    frag_req->send.state.dt.dt.contig.md_map = 0;

    ucp_request_send(frag_req);
    sreq->send.state.dt.offset += frag_req->send.length;
}

static ucs_status_t ucp_rndv_pipeline(ucp_request_t *sreq, ucp_rndv_rtr_hdr_t *rndv_rtr_hdr)
{
    ucp_worker_h worker  = sreq->send.ep->worker;
    ucp_ep_h pipeline_ep = worker->mem_type_ep[sreq->send.mem_type];
    ucp_mem_desc_t *mdesc;
    ucp_request_t *frag_req;
    ucp_rsc_index_t md_index;
    ucs_status_t status;
    int i, num_frags;
    size_t frag_size, length;
    size_t offset;

    md_index =  ucp_ep_md_index(pipeline_ep,
                                ucp_ep_config(pipeline_ep)->key.rma_bw_lanes[0]);

    frag_size = worker->context->config.ext.rndv_frag_size;
    num_frags = (sreq->send.length + frag_size - 1) / frag_size;
    sreq->send.state.uct_comp.count = num_frags;
    sreq->send.state.dt.offset = 0;
    sreq->send.rndv_put.remote_request = rndv_rtr_hdr->rreq_ptr;
    sreq->send.rndv_put.remote_address = rndv_rtr_hdr->address;

    offset = 0;
    for (i = 0; i < num_frags; i++) {
        length = (i != (num_frags - 1)) ? frag_size : (sreq->send.length - offset);

        frag_req = ucp_request_get(worker);
        if (frag_req == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        mdesc = ucs_mpool_get_inline(&worker->rndv_frag_mp);
        if (mdesc == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        ucp_request_send_state_init(frag_req, ucp_dt_make_contig(1), 0);
        ucp_request_send_state_reset(frag_req, ucp_rndv_frag_get_completion,
                                     UCP_REQUEST_SEND_PROTO_RNDV_GET);
        frag_req->send.ep                        = pipeline_ep;
        frag_req->send.buffer                    = mdesc + 1;
        frag_req->send.datatype                  = ucp_dt_make_contig(1);
        frag_req->send.state.dt.dt.contig.memh[0]= ucp_memh2uct(mdesc->memh, md_index);
        frag_req->send.state.dt.dt.contig.md_map = UCS_BIT(md_index);
        frag_req->send.length                    = length;
        frag_req->send.uct.func                  = ucp_rndv_progress_rma_get_zcopy;
        frag_req->send.rndv_get.remote_address   = (uint64_t)(sreq->send.buffer + offset);
        frag_req->send.rndv_get.lanes_map        = 0;
        frag_req->send.rndv_get.lane_count       = 0;
        frag_req->send.rndv_get.rreq             = sreq;
        frag_req->send.mdesc                     = mdesc;

        ucp_request_send(frag_req);
        offset += length;
    }
    return UCS_OK;
 out:
    return status;;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rtr_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_request_t *sreq              = (ucp_request_t*)rndv_rtr_hdr->sreq_ptr;
    ucp_ep_h ep                      = sreq->send.ep;
    ucp_rndv_mode_t rndv_mode        = ep->worker->context->config.ext.rndv_mode;
    ucs_status_t status;

    ucp_trace_req(sreq, "received rtr address 0x%lx remote rreq 0x%lx",
                  rndv_rtr_hdr->address, rndv_rtr_hdr->rreq_ptr);
    UCS_PROFILE_REQUEST_EVENT(sreq, "rndv_rtr_recv", 0);

    if (sreq->flags & UCP_REQUEST_FLAG_OFFLOADED) {
        /* Do not deregister memory here, because am zcopy rndv may
         * need it registered (if am and tag is the same lane). */
        ucp_tag_offload_cancel_rndv(sreq);
    }

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) && rndv_rtr_hdr->address) {
        status = ucp_ep_rkey_unpack(ep, rndv_rtr_hdr + 1,
                                    &sreq->send.rndv_put.rkey);
        if (status != UCS_OK) {
            ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                      ucp_ep_peer_name(ep), ucs_status_string(status));
        }

        sreq->send.lane = ucp_rkey_get_rma_bw_lane(sreq->send.rndv_put.rkey, ep,
                                                   &sreq->send.rndv_put.uct_rkey, 0);
        if (sreq->send.lane != UCP_NULL_LANE) {
            if ((rndv_mode == UCP_RNDV_MODE_PUT_ZCOPY) ||
                ((rndv_mode == UCP_RNDV_MODE_AUTO) &&
                 UCP_MEM_IS_HOST(sreq->send.mem_type))) {
                ucp_request_send_state_reset(sreq, ucp_rndv_put_completion,
                                             UCP_REQUEST_SEND_PROTO_RNDV_PUT);
                sreq->send.uct.func                = ucp_rndv_progress_rma_put_zcopy;
                sreq->send.rndv_put.remote_request = rndv_rtr_hdr->rreq_ptr;
                sreq->send.rndv_put.remote_address = rndv_rtr_hdr->address;
                sreq->send.mdesc                   = NULL;
                goto out_send;
            } else {
                return ucp_rndv_pipeline(sreq, rndv_rtr_hdr);
            }
        } else {
            ucp_rkey_destroy(sreq->send.rndv_put.rkey);
        }
    }

    /* switch to AM */
    sreq->send.rndv_data.rreq_ptr = rndv_rtr_hdr->rreq_ptr;

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        (sreq->send.length >= ucp_ep_config(ep)->am.zcopy_thresh[0]))
    {
        status = ucp_request_send_buffer_reg_lane(sreq, ucp_ep_get_am_lane(ep));
        ucs_assert_always(status == UCS_OK);

        ucp_request_send_state_reset(sreq, ucp_rndv_am_zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_ZCOPY_AM);

        if ((sreq->send.length + sizeof(ucp_rndv_data_hdr_t)) <=
            ucp_ep_config(ep)->am.max_zcopy) {
            sreq->send.uct.func = ucp_rndv_progress_am_zcopy_single;
        } else {
            sreq->send.uct.func        = ucp_rndv_progress_am_zcopy_multi;
            sreq->send.tag.am_bw_index = 0;
        }
    } else {
        ucp_request_send_state_reset(sreq, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        sreq->send.uct.func        = ucp_rndv_progress_am_bcopy;
        sreq->send.tag.am_bw_index = 0;
    }

out_send:
    ucp_request_send(sreq);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_data_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq = (ucp_request_t*) rndv_data_hdr->rreq_ptr;
    size_t recv_len;

    recv_len = length - sizeof(*rndv_data_hdr);
    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_recv", recv_len);

    (void)ucp_tag_request_process_recv_data(rreq, rndv_data_hdr + 1, recv_len,
                                            rndv_data_hdr->offset, 1);
    return UCS_OK;
}

static void ucp_rndv_dump_rkey(const void *packed_rkey, char *buffer, size_t max)
{
    char *p    = buffer;
    char *endp = buffer + max;

    snprintf(p, endp - p, " rkey ");
    p += strlen(p);

    ucp_rkey_dump_packed(packed_rkey, p, endp - p);
}

static void ucp_rndv_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                          uint8_t id, const void *data, size_t length,
                          char *buffer, size_t max)
{

    const ucp_rndv_rts_hdr_t *rndv_rts_hdr = data;
    const ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    const ucp_rndv_data_hdr_t *rndv_data = data;
    const ucp_reply_hdr_t *rep_hdr = data;

    switch (id) {
    case UCP_AM_ID_RNDV_RTS:
        snprintf(buffer, max, "RNDV_RTS tag %"PRIx64" uuid %"PRIx64" sreq 0x%lx "
                 "address 0x%"PRIx64" size %zu", rndv_rts_hdr->super.tag,
                 rndv_rts_hdr->sreq.sender_uuid, rndv_rts_hdr->sreq.reqptr,
                 rndv_rts_hdr->address, rndv_rts_hdr->size);
        if (rndv_rts_hdr->address) {
            ucp_rndv_dump_rkey(rndv_rts_hdr + 1, buffer + strlen(buffer),
                               max - strlen(buffer));
        }
        break;
    case UCP_AM_ID_RNDV_ATS:
        snprintf(buffer, max, "RNDV_ATS sreq 0x%lx status '%s'",
                 rep_hdr->reqptr, ucs_status_string(rep_hdr->status));
        break;
    case UCP_AM_ID_RNDV_RTR:
        snprintf(buffer, max, "RNDV_RTR sreq 0x%lx rreq 0x%lx address 0x%lx",
                 rndv_rtr_hdr->sreq_ptr, rndv_rtr_hdr->rreq_ptr,
                 rndv_rtr_hdr->address);
        if (rndv_rtr_hdr->address) {
            ucp_rndv_dump_rkey(rndv_rtr_hdr + 1, buffer + strlen(buffer),
                               max - strlen(buffer));
        }
        break;
    case UCP_AM_ID_RNDV_DATA:
        snprintf(buffer, max, "RNDV_DATA rreq 0x%"PRIx64" offset %zu",
                 rndv_data->rreq_ptr, rndv_data->offset);
        break;
    case UCP_AM_ID_RNDV_ATP:
        snprintf(buffer, max, "RNDV_ATP sreq 0x%lx status '%s'",
                 rep_hdr->reqptr, ucs_status_string(rep_hdr->status));
        break;
    default:
        return;
    }
}

UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_RTS, ucp_rndv_rts_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_ATS, ucp_rndv_ats_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_ATP, ucp_rndv_atp_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_RTR, ucp_rndv_rtr_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA, ucp_rndv_data_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATP);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTR);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA);
