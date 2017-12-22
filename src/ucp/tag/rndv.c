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


size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq              = arg;   /* send request */
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = dest;
    ucp_worker_h worker              = sreq->send.ep->worker;
    ssize_t packed_rkey_size;

    rndv_rts_hdr->super.tag        = sreq->send.tag;
    rndv_rts_hdr->sreq.reqptr      = (uintptr_t)sreq;
    rndv_rts_hdr->sreq.sender_uuid = worker->uuid;
    rndv_rts_hdr->size             = sreq->send.length;

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype)) {
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

    ucp_trace_req(sreq, "start_rndv buffer %p length %zu", sreq->send.buffer,
                  sreq->send.length);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    if (ep->flags & UCP_EP_FLAG_TAG_OFFLOAD_ENABLED) {
        status = ucp_tag_offload_start_rndv(sreq);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
            (ep->worker->context->config.ext.rndv_mode != UCP_RNDV_MODE_PUT_ZCOPY))
        {
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

static void ucp_rndv_req_send_rtr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                   uintptr_t sender_reqptr)
{
    ucp_trace_req(rndv_req, "send rtr remote sreq 0x%lx rreq %p", sender_reqptr,
                  rreq);

    rreq->status                           = UCS_OK;
    rndv_req->send.lane                    = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func                = ucp_proto_progress_rndv_rtr;
    rndv_req->send.rndv_rtr.remote_request = sender_reqptr;
    rndv_req->send.rndv_rtr.rreq           = rreq;

    ucp_request_send(rndv_req);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_get_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep             = rndv_req->send.ep;
    ucs_status_t status;
    size_t offset, length, ucp_mtu, remainder, align;
    const size_t max_iovcnt = 1;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_rsc_index_t rsc_index;
    ucp_dt_state_t state;

    if (ucp_ep_is_stub(ep)) {
        rndv_req->send.lane = 0;
        return UCS_ERR_NO_RESOURCE;
    }

    /* Figure out which lane to use for get operation */
    rndv_req->send.lane = ucp_rkey_get_rma_bw_lane(rndv_req->send.rndv_get.rkey, ep,
                                                   &rndv_req->send.rndv_get.uct_rkey);
    if (rndv_req->send.lane == UCP_NULL_LANE) {
        /* If can't perform get_zcopy - switch to active-message.
         * NOTE: we do not register memory and do not send our keys. */
        ucp_trace_req(rndv_req, "remote memory unreachable, switch to rtr");
        ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
        ucp_rndv_req_send_rtr(rndv_req, rndv_req->send.rndv_get.rreq,
                              rndv_req->send.rndv_get.remote_request);
        return UCS_OK;
    }

    status = ucp_request_send_buffer_reg_lane(rndv_req, rndv_req->send.lane);
    ucs_assert_always(status == UCS_OK);

    rsc_index = ucp_ep_get_rsc_index(rndv_req->send.ep, rndv_req->send.lane);
    align     = rndv_req->send.ep->worker->ifaces[rsc_index].attr.cap.get.opt_zcopy_align;
    ucp_mtu   = rndv_req->send.ep->worker->ifaces[rsc_index].attr.cap.get.align_mtu;

    offset    = rndv_req->send.state.dt.offset;
    remainder = (uintptr_t)rndv_req->send.buffer % align;

    if ((offset == 0) && (remainder > 0) && (rndv_req->send.length > ucp_mtu)) {
        length = ucp_mtu - remainder;
    } else {
        length = ucs_min(rndv_req->send.length - offset,
                         ucp_ep_config(rndv_req->send.ep)->tag.rndv.max_get_zcopy);
    }

    ucs_trace_data("req %p: offset %zu remainder %zu rma-get to %p len %zu lane %d",
                   rndv_req, offset, remainder, rndv_req->send.buffer + offset,
                   length, rndv_req->send.lane);

    state = rndv_req->send.state.dt;
    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iovcnt, &state, rndv_req->send.buffer,
                        ucp_dt_make_contig(1), length);

    status = uct_ep_get_zcopy(ep->uct_eps[rndv_req->send.lane],
                              iov, iovcnt,
                              rndv_req->send.rndv_get.remote_address + offset,
                              rndv_req->send.rndv_get.uct_rkey,
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
        return status;
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
    ucp_rndv_req_send_rtr(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_rndv_process_rts(void *arg, void *data, size_t length,
                                  unsigned tl_flags)
{
    const unsigned recv_flags          = UCP_RECV_DESC_FLAG_FIRST |
                                         UCP_RECV_DESC_FLAG_LAST  |
                                         UCP_RECV_DESC_FLAG_RNDV;
    ucp_worker_h worker                = arg;
    ucp_rndv_rts_hdr_t *rndv_rts_hdr   = data;
    ucp_request_t *rreq;
    ucs_status_t status;

    rreq = ucp_tag_exp_search(&worker->tm, rndv_rts_hdr->super.tag,
                              rndv_rts_hdr->size, recv_flags);
    if (rreq != NULL) {
        ucp_rndv_matched(worker, rreq, rndv_rts_hdr);

        /* Cancel req in transport if it was offloaded, because it arrived
           as unexpected */
        ucp_tag_offload_try_cancel(worker, rreq, 1);

        UCP_WORKER_STAT_RNDV(worker, EXP);
        status = UCS_OK;
    } else {
        status = ucp_tag_unexp_recv(&worker->tm, worker, data, length, tl_flags,
                                    sizeof(*rndv_rts_hdr), recv_flags);
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

static size_t ucp_rndv_pack_single_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    ucs_assert(sreq->send.state.dt.offset == 0);

    hdr->rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    length = ucp_dt_pack(sreq->send.datatype, hdr + 1, sreq->send.buffer,
                         &sreq->send.state.dt, sreq->send.length);
    ucs_assert(length == sreq->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_rndv_pack_multi_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    hdr->rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    length        = ucp_ep_config(sreq->send.ep)->am.max_bcopy - sizeof(*hdr);

    return sizeof(*hdr) + ucp_dt_pack(sreq->send.datatype, hdr + 1,
                                      sreq->send.buffer, &sreq->send.state.dt,
                                      length);
}

static size_t ucp_rndv_pack_multi_data_last(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    hdr->rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    length        = sreq->send.length - sreq->send.state.dt.offset;

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
        status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_DATA_LAST,
                                        ucp_rndv_pack_single_data);
    } else {
        /* send multiple bcopy messages (fragments of the original send message) */
        status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA_LAST,
                                       sizeof(ucp_rndv_data_hdr_t),
                                       ucp_rndv_pack_multi_data,
                                       ucp_rndv_pack_multi_data,
                                       ucp_rndv_pack_multi_data_last);
    }
    if (status == UCS_OK) {
        ucp_rndv_complete_send(sreq);
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_put_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;
    size_t offset, ucp_mtu, align, remainder, length;
    ucp_rsc_index_t rsc_index;
    const size_t max_iovcnt = 1;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_dt_state_t state;

    status = ucp_request_send_buffer_reg_lane(sreq, sreq->send.lane);
    ucs_assert_always(status == UCS_OK);

    rsc_index = ucp_ep_get_rsc_index(sreq->send.ep, sreq->send.lane);
    align     = sreq->send.ep->worker->ifaces[rsc_index].attr.cap.put.opt_zcopy_align;
    ucp_mtu   = sreq->send.ep->worker->ifaces[rsc_index].attr.cap.put.align_mtu;

    offset    = sreq->send.state.dt.offset;
    remainder = (uintptr_t)sreq->send.buffer % align;

    if ((offset == 0) && (remainder > 0) && (sreq->send.length > ucp_mtu)) {
        length = ucp_mtu - remainder;
    } else {
        length = ucs_min(sreq->send.length - offset,
                         ucp_ep_config(sreq->send.ep)->tag.rndv.max_put_zcopy);
    }

    ucs_trace_data("req %p: offset %zu remainder %zu. read to %p len %zu",
                   sreq, offset, (uintptr_t)sreq->send.buffer % align,
                   (void*)sreq->send.buffer + offset, length);

    state = sreq->send.state.dt;
    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iovcnt, &state, sreq->send.buffer,
                        ucp_dt_make_contig(1), length);
    status = uct_ep_put_zcopy(sreq->send.ep->uct_eps[sreq->send.lane],
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
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_RNDV_DATA_LAST, &hdr, sizeof(hdr),
                                  ucp_rndv_am_zcopy_send_req_complete);
}

static ucs_status_t ucp_rndv_progress_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_ptr = sreq->send.rndv_data.rreq_ptr;
    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA_LAST,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_rndv_am_zcopy_send_req_complete);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rtr_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_request_t *sreq              = (ucp_request_t*)rndv_rtr_hdr->sreq_ptr;
    ucp_ep_h ep                      = sreq->send.ep;
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
                                                   &sreq->send.rndv_put.uct_rkey);
        if (sreq->send.lane != UCP_NULL_LANE) {
            ucp_request_send_state_reset(sreq, ucp_rndv_put_completion,
                                         UCP_REQUEST_SEND_PROTO_RNDV_PUT);
            sreq->send.uct.func                = ucp_rndv_progress_rma_put_zcopy;
            sreq->send.rndv_put.remote_request = rndv_rtr_hdr->rreq_ptr;
            sreq->send.rndv_put.remote_address = rndv_rtr_hdr->address;
            goto out_send;
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
            sreq->send.uct.func = ucp_rndv_progress_am_zcopy_multi;
        }
    } else {
        ucp_request_send_state_reset(sreq, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        sreq->send.uct.func = ucp_rndv_progress_am_bcopy;
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
    ucs_status_t status;
    uint8_t hdr_len = sizeof(*rndv_data_hdr);

    ucs_assert(length >= hdr_len);
    recv_len = length - hdr_len;

    if (rreq->status == UCS_OK) {
        ucs_trace_data("recv segment for rreq %p, offset %zu, length %zu",
                       rreq, rreq->recv.state.offset, recv_len);
    } else {
        ucs_trace_data("drop segment for rreq %p, length %zu, status %s",
                       rreq, recv_len, ucs_status_string(rreq->status));
        /* Drop the packet and return ok, so that the transport
         * would release the descriptor */
        return UCS_OK;
    }

    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_recv", recv_len);
    status = ucp_dt_unpack(rreq->recv.datatype, rreq->recv.buffer,
                           rreq->recv.length, &rreq->recv.state,
                           data + hdr_len, recv_len, 0);
    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        rreq->recv.state.offset += recv_len;
        return status;
    } else {
        rreq->status = status;
        return UCS_OK;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_data_last_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)

{
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq = (ucp_request_t*) rndv_data_hdr->rreq_ptr;
    size_t recv_len;
    ucs_status_t status;
    uint8_t hdr_len = sizeof(*rndv_data_hdr);

    ucs_assert(length >= hdr_len);
    recv_len = length - hdr_len;

    if (rreq->status == UCS_OK) {
        ucs_trace_data("recv last segment for rreq %p, length %zu",
                       rreq, recv_len);

        /* Check that total received length matches RTS->length */
        ucs_assert(rreq->recv.tag.info.length ==
                   rreq->recv.state.offset + recv_len);
        UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_last_recv", recv_len);
        status = ucp_dt_unpack(rreq->recv.datatype, rreq->recv.buffer,
                               rreq->recv.length, &rreq->recv.state,
                               data + hdr_len, recv_len,
                               UCP_RECV_DESC_FLAG_LAST);
    } else {
        ucs_trace_data("drop last segment for rreq %p, length %zu, status %s",
                       rreq, recv_len, ucs_status_string(rreq->status));
        status = rreq->status;
        ucp_request_recv_generic_dt_finish(rreq);
    }

    ucp_rndv_zcopy_recv_req_complete(rreq, status);
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
        snprintf(buffer, max, "RNDV_DATA rreq 0x%"PRIx64,
                 rndv_data->rreq_ptr);
        break;
    case UCP_AM_ID_RNDV_ATP:
        snprintf(buffer, max, "RNDV_ATP sreq 0x%lx status '%s'",
                 rep_hdr->reqptr, ucs_status_string(rep_hdr->status));
        break;
    case UCP_AM_ID_RNDV_DATA_LAST:
        snprintf(buffer, max, "RNDV_DATA_LAST rreq 0x%"PRIx64,
                 rndv_data->rreq_ptr);
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
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA_LAST, ucp_rndv_data_last_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATP);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTR);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA_LAST);
