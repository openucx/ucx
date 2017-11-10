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


static int ucp_tag_rndv_is_get_op_possible(ucp_ep_h ep, ucp_request_t *req)
{
    uint64_t md_flags;
    int i;
    uct_rkey_t rkey;

    ucs_assert(!ucp_ep_is_stub(ep));

    if (!req->send.rndv_get.rkey) {
        return 0;
    }

    for (i = 0; i < ucp_ep_rndv_num_lanes(ep); i++) {
        md_flags = ucp_ep_rndv_md_flags(ep, i);
        rkey = ucp_tag_rndv_rkey(req, i)->rkey;
        if ((md_flags & UCT_MD_FLAG_NEED_RKEY) && (rkey == UCT_INVALID_RKEY)) {
            return 0;
        }
    }

    return 1;
}

static void ucp_rndv_rma_request_send_buffer_dereg(ucp_request_t *sreq)
{
    /*
     * AM flow can come here with non registered memory.
     * TODO: remove extra check inside ucp_request_memory_dereg:
     *       (state->dt.contig.memh != UCT_MEM_HANDLE_NULL)
     */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype)) {
        ucp_request_rndv_mem_dereg(sreq);
    }
}

size_t ucp_tag_rndv_pack_rkey(ucp_request_t *sreq, ucp_lane_index_t lane,
                              void *rkey_buf, uint16_t *flags)
{
    ucp_ep_h ep = sreq->send.ep;
    ucs_status_t status;

    ucs_assert(UCP_DT_IS_CONTIG(sreq->send.datatype));

    /* Check if the sender needs to register the send buffer -
     * is its datatype contiguous and does the receive side need it */
    if (ucp_ep_rndv_md_flags(ep, 0) & UCT_MD_FLAG_NEED_RKEY) {
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

static size_t ucp_tag_rndv_pack_rkeys(ucp_request_t *sreq, void *rkey_buf, uint16_t *flags)
{
    ucp_ep_t        *ep     = sreq->send.ep;
    ucp_dt_state_t  *state  = &sreq->send.state.dt;
    size_t           packet = 0;
    int              i;
    ucp_lane_index_t lane;

    ucs_assert(UCP_DT_IS_CONTIG(sreq->send.datatype));

    ucp_request_rndv_mem_reg(sreq);

    for (i = 0; i < ucp_ep_rndv_num_lanes(ep); i++) {
        lane = ucp_ep_get_rndv_get_lane(ep, i);

        if (ucp_ep_rndv_md_flags(ep, i) & UCT_MD_FLAG_NEED_RKEY) {
            UCS_PROFILE_CALL(uct_md_mkey_pack, ucp_ep_md(ep, lane),
                             state->dt.contig[i].memh, rkey_buf + packet);
            packet += ucp_ep_md_attr(ep, lane)->rkey_packed_size;
        }
    }
    *flags |= UCP_RNDV_RTS_FLAG_PACKED_RKEY;

    ucs_assert_always(packet <= ucp_ep_config(ep)->am.max_bcopy);

    return packet;
}

static void ucp_tag_rndv_unpack_rkeys(ucp_request_t *req, void *rkey_buf, uint16_t flags)
{
    ucp_ep_t        *ep     = req->send.ep;
    size_t           packet = 0;
    ucp_lane_index_t rkeys;
    ucp_lane_index_t lane;
    int              i;

    ucs_assert(UCP_DT_IS_CONTIG(req->send.datatype));

    if (flags & (UCP_EP_FLAG_TAG_OFFLOAD_ENABLED | UCP_RNDV_RTS_FLAG_PACKED_RKEY)) {
        /* tag offload operates by single remote key only */
        rkeys = ucs_min(1, ucp_ep_rndv_num_lanes(ep));
    } else {
        rkeys = ucp_ep_rndv_num_lanes(ep);
    }

    ucp_request_rndv_get_create(req);

    req->send.rndv_get.rkey->num_lanes = rkeys;

    for (i = 0; i < rkeys; i++) {
        lane = ucp_ep_get_rndv_get_lane(ep, i);
        if (ucp_ep_rndv_md_flags(ep, i) & UCT_MD_FLAG_NEED_RKEY) {
            UCS_PROFILE_CALL(uct_rkey_unpack, rkey_buf + packet,
                             ucp_tag_rndv_rkey(req, i));
            packet += ucp_ep_md_attr(ep, lane)->rkey_packed_size;
        }
    }
}

static size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq = arg;   /* the sender's request */
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = dest;
    size_t packed_len = sizeof(*rndv_rts_hdr);
    ucp_ep_t *ep = sreq->send.ep;

    rndv_rts_hdr->flags            = 0;
    rndv_rts_hdr->super.tag        = sreq->send.tag;
    /* reqptr holds the original sreq */
    rndv_rts_hdr->sreq.reqptr      = (uintptr_t)sreq;
    rndv_rts_hdr->sreq.sender_uuid = ep->worker->uuid;
    rndv_rts_hdr->size             = sreq->send.length;
    if (UCP_DT_IS_CONTIG(sreq->send.datatype)) {
        rndv_rts_hdr->address = (uintptr_t) sreq->send.buffer;
        if (ucp_ep_is_rndv_lane_present(ep, 0)) {
            packed_len += ucp_tag_rndv_pack_rkeys(sreq, rndv_rts_hdr + 1,
                                                  &rndv_rts_hdr->flags);
        }
    } else if (UCP_DT_IS_GENERIC(sreq->send.datatype) ||
               UCP_DT_IS_IOV(sreq->send.datatype)) {
        rndv_rts_hdr->address = 0;
    }

    /* For rndv emulation based on AM rndv (send-recv), only the size of the rts
     * header is returned */
    return packed_len;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    /* send the RTS. the pack_cb will pack all the necessary fields in the RTS */
    return ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_RTS, ucp_tag_rndv_rts_pack);
}

static size_t ucp_tag_rndv_rtr_pack(void *dest, void *arg)
{
    ucp_request_t *rndv_req = arg;   /* the receive's rndv_req */
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = dest;

    /* sreq_ptr holds the sender's send req */
    rndv_rtr_hdr->sreq_ptr = rndv_req->send.proto.remote_request;

    /* rreq_ptr holds the recv req on the recv side */
    rndv_rtr_hdr->rreq_ptr = rndv_req->send.proto.rreq_ptr;

    return sizeof(*rndv_rtr_hdr);
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
    ucs_status_t status;

    ucs_trace_req("starting rndv. sreq: %p. buffer: %p, length: %zu",
                  sreq, sreq->send.buffer, sreq->send.length);
    sreq->flags |= UCP_REQUEST_FLAG_RNDV;

    ucp_ep_connect_remote(sreq->send.ep);

    if (sreq->send.ep->flags & UCP_EP_FLAG_TAG_OFFLOAD_ENABLED) {
        status = ucp_tag_offload_start_rndv(sreq);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        ucp_request_send_state_reset(sreq, NULL,
                                     UCP_REQUEST_SEND_PROTO_RNDV_GET);
        sreq->send.uct.func = ucp_proto_progress_rndv_rts;
    }
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);
    return UCS_OK;
}

static void ucp_rndv_send_ats(ucp_request_t *rndv_req, uintptr_t remote_request)
{
    ucs_trace_req("send ats ep %p rndv_req %p remote_request 0x%lx",
                  rndv_req->send.ep, rndv_req, remote_request);

    UCS_PROFILE_REQUEST_EVENT(rndv_req->send.rndv_get.rreq, "send_ats", 0);

    rndv_req->send.lane         = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func     = ucp_proto_progress_am_bcopy_single;
    rndv_req->send.proto.am_id  = UCP_AM_ID_RNDV_ATS;
    rndv_req->send.proto.status = UCS_OK;
    rndv_req->send.proto.remote_request = remote_request;

    ucp_request_send(rndv_req);
}

static void ucp_rndv_complete_rndv_get(ucp_request_t *rndv_req)
{
    ucp_request_t *rreq = rndv_req->send.rndv_get.rreq;

    ucs_assertv(rndv_req->send.state.dt.offset == rndv_req->send.length,
                "rndv_req=%p offset=%zu length=%zu", rndv_req,
                rndv_req->send.state.dt.offset, rndv_req->send.length);

    ucs_trace_data("ep: %p rndv get completed", rndv_req->send.ep);

    UCS_PROFILE_REQUEST_EVENT(rreq, "complete_rndv_get", 0); // TODO
    ucp_request_complete_recv(rreq, UCS_OK);

    ucp_rndv_rma_request_send_buffer_dereg(rndv_req);

    ucp_request_rndv_get_release(rndv_req);

    ucp_rndv_send_ats(rndv_req, rndv_req->send.rndv_get.remote_request);
}

static ucs_status_t ucp_rndv_truncated(uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_t *rreq     = (ucp_request_t*) rndv_req->send.proto.rreq_ptr;

    /* if the recv request has a generic datatype, need to finish it */
    ucp_request_recv_generic_dt_finish(rreq);

    ucp_request_complete_recv(rreq, UCS_ERR_MESSAGE_TRUNCATED);
    ucp_rndv_send_ats(rndv_req, rndv_req->send.proto.remote_request);

    return UCS_OK;
}

static void ucp_rndv_recv_am(ucp_request_t *rndv_req, ucp_request_t *rreq,
                             uintptr_t sender_reqptr, size_t total_size)
{
    ucs_trace_req("handle generic datatype on rndv receive. local rndv_req: %p, "
                  "recv request: %p", rndv_req, rreq);

    /* rndv_req is the request that would send the RTR message to the sender */
    rndv_req->send.lane     = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func = ucp_proto_progress_rndv_rtr;
    /* save the sender's send request and send it in the RTR */
    rndv_req->send.proto.remote_request = sender_reqptr;
    rndv_req->send.proto.status         = UCS_OK;
    rndv_req->send.proto.rreq_ptr       = (uintptr_t) rreq;

    if (ucs_unlikely(rreq->recv.length < total_size)) {
        ucs_trace_req("rndv msg truncated: rndv_req: %p. received %zu. "
                      "expected %zu on rreq: %p ",
                      rndv_req, total_size, rreq->recv.length, rreq);
        rndv_req->send.uct.func = ucp_rndv_truncated;
    } else {
        rreq->status = UCS_OK;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_get_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep             = rndv_req->send.ep;
    ucs_status_t status;
    size_t offset, length, ucp_mtu, align;
    const size_t max_iovcnt = 1;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_rsc_index_t rsc_index;
    ucp_dt_state_t state;
    ucp_lane_index_t lane;
    ucp_lane_index_t lane_idx;
    uct_rkey_t rkey;
    int lanes;

    if (ucp_ep_is_stub(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (!(ucp_tag_rndv_is_get_op_possible(ep, rndv_req))) {
        /* can't perform get_zcopy - switch to AM rndv */
        ucp_request_rndv_get_release(rndv_req);
        ucp_rndv_recv_am(rndv_req, rndv_req->send.rndv_get.rreq,
                         rndv_req->send.rndv_get.remote_request,
                         rndv_req->send.length);

        return UCS_INPROGRESS;
    }

    /* rndv_req is the internal request to perform the get operation */
    if (!ucp_dt_have_rndv_lanes(&rndv_req->send.state.dt)) {
        /* TODO Not all UCTs need registration on the recv side */
        ucp_request_rndv_mem_reg(rndv_req);
        rndv_req->send.rndv_get.rkey->lane_idx = 0;
    }

    lane_idx  = rndv_req->send.rndv_get.rkey->lane_idx;
    rsc_index = ucp_ep_get_rsc_index(ep, ucp_ep_get_rndv_get_lane(ep, lane_idx));
    align     = ep->worker->ifaces[rsc_index].attr.cap.get.opt_zcopy_align;
    ucp_mtu   = ep->worker->ifaces[rsc_index].attr.cap.get.align_mtu;
    lanes     = rndv_req->send.rndv_get.rkey->num_lanes;

    ucs_trace_data("ep: %p try to progress get_zcopy for rndv get. rndv_req: %p. lane: %d",
                   ep, rndv_req, rndv_req->send.lane);

    offset = rndv_req->send.state.dt.offset;

    if ((offset == 0) && ((uintptr_t)rndv_req->send.buffer % align) &&
        (rndv_req->send.length > ucp_mtu )) {
        length = ucp_mtu - ((uintptr_t)rndv_req->send.buffer % align);
    } else {
        length = ucs_min(rndv_req->send.length - offset,
                         ucs_min(ucp_ep_config(ep)->tag.rndv.max_get_zcopy,
                                 ((rndv_req->send.length) - 1) /
                                  lanes));
    }

    if (length == 0) {
        return UCS_OK;
    }

    ucs_trace_data("offset %zu remainder %zu. read to %p len %zu",
                   offset, (uintptr_t)rndv_req->send.buffer % align,
                   (void*)rndv_req->send.buffer + offset, length);

    state = rndv_req->send.state.dt;
    ucp_dt_iov_copy_uct(iov, &iovcnt, max_iovcnt, &state, rndv_req->send.buffer,
                        ucp_dt_make_contig(1), length);

    iov[0].memh = rndv_req->send.state.dt.dt.contig[lane_idx].memh;
    lane        = ucp_ep_get_rndv_get_lane(ep, lane_idx);
    rkey        = ucp_tag_rndv_rkey(rndv_req, lane_idx)->rkey;
    rndv_req->send.rndv_get.rkey->lane_idx = (lane_idx + 1) % lanes;

    status = uct_ep_get_zcopy(ep->uct_eps[lane],
                              iov, iovcnt,
                              rndv_req->send.rndv_get.remote_address + offset,
                              rkey,
                              &rndv_req->send.state.uct_comp);
    ucp_request_send_state_advance(rndv_req, &state,
                                   UCP_REQUEST_SEND_PROTO_RNDV_GET,
                                   status);
    if (rndv_req->send.state.dt.offset == rndv_req->send.length) {
        if (rndv_req->send.state.uct_comp.count == 0) {
            ucp_rndv_complete_rndv_get(rndv_req);
        }
    } else if (status == UCS_OK) {
            /* in case if not all chunks are transmitted - return in_progress
             * status */
            return UCS_INPROGRESS;
    }

    return status;
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_get_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t,
                                               send.state.uct_comp);

    if (rndv_req->send.state.dt.offset == rndv_req->send.length) {
        ucs_trace_req("completed rndv get operation rndv_req: %p", rndv_req);
        ucp_rndv_complete_rndv_get(rndv_req);
    }
}

static void ucp_rndv_handle_recv_contig(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                        ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucs_trace_req("ucp_rndv_handle_recv_contig rndv_req %p rreq %p", rndv_req,
                  rreq);

    if (ucs_unlikely(rreq->recv.length < rndv_rts_hdr->size)) {
        ucs_trace_req("rndv msg truncated: rndv_req: %p. received %zu. "
                      "expected %zu on rreq: %p ",
                      rndv_req, rndv_rts_hdr->size, rreq->recv.length, rreq);
        rndv_req->send.uct.func  = ucp_rndv_truncated;
        rndv_req->send.lane      = ucp_ep_get_am_lane(rndv_req->send.ep);
        /* prepare for handling a truncated message. the rndv_get struct isn't needed anymore */
        rndv_req->send.proto.remote_request = rndv_rts_hdr->sreq.reqptr;
        rndv_req->send.proto.rreq_ptr       = (uintptr_t) rreq;
    } else {
        ucp_tag_rndv_unpack_rkeys(rndv_req, rndv_rts_hdr + 1, rndv_rts_hdr->flags);
        /* rndv_req is the request that would perform the get operation */
        rndv_req->send.uct.func     = ucp_proto_progress_rndv_get_zcopy;
        rndv_req->send.buffer       = rreq->recv.buffer;
        rndv_req->send.datatype     = ucp_dt_make_contig(1);
        rndv_req->send.length       = rndv_rts_hdr->size;
        ucp_request_send_state_reset(rndv_req, ucp_rndv_get_completion,
                                     UCP_REQUEST_SEND_PROTO_RNDV_GET);
        if (rndv_rts_hdr->flags & UCP_RNDV_RTS_FLAG_OFFLOAD) {
            rndv_req->send.lane     = ucp_ep_get_tag_lane(rndv_req->send.ep);
        } else {
            rndv_req->send.lane     = ucp_ep_get_rndv_get_lane(rndv_req->send.ep, 0);
        }
        rndv_req->send.rndv_get.remote_request = rndv_rts_hdr->sreq.reqptr;
        rndv_req->send.rndv_get.remote_address = rndv_rts_hdr->address;
        rndv_req->send.rndv_get.rreq = rreq;
    }
    ucp_request_send(rndv_req);
}

static void ucp_rndv_handle_recv_am(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                    ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucs_trace_req("ucp_rndv_handle_recv_am rndv_req %p rreq %p", rndv_req, rreq);

    ucp_rndv_recv_am(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr,
                     rndv_rts_hdr->size);

    ucp_request_send(rndv_req);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_matched, (worker, rreq, rndv_rts_hdr),
                      ucp_worker_h worker, ucp_request_t *rreq,
                      ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_request_t *rndv_req;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_match", 0);

    /* rreq is the receive request on the receiver's side */
    rreq->recv.info.sender_tag = rndv_rts_hdr->super.tag;
    rreq->recv.info.length     = rndv_rts_hdr->size;

    /* the internal send request allocated on receiver side (to perform a "get"
     * operation, send "ATS" and "RTR") */
    rndv_req = ucp_worker_allocate_reply(worker, rndv_rts_hdr->sreq.sender_uuid);
    ep = rndv_req->send.ep;
    rndv_req->send.rndv_get.rkey = NULL;
    rndv_req->send.datatype = rreq->recv.datatype;

    ucs_trace_req("ucp_rndv_matched remote_address 0x%"PRIx64" remote_req 0x%lx "
                  "rndv_req %p", rndv_rts_hdr->address, rndv_rts_hdr->sreq.reqptr,
                  rndv_req);

    /* if the receive side is not connected yet then the RTS was received on a stub ep */
    if (ucp_ep_is_stub(ep)) {
        ucs_debug("received rts on a stub ep, ep=%p, rndv_lane=%d, "
                  "am_lane=%d", ep, ucp_ep_is_rndv_lane_present(ep, 0) ?
                  ucp_ep_get_rndv_get_lane(ep, 0): UCP_NULL_LANE,
                  ucp_ep_get_am_lane(ep));
    }

    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        if ((rndv_rts_hdr->address != 0) && (ucp_ep_is_rndv_lane_present(ep, 0) ||
            (rndv_rts_hdr->flags & UCP_RNDV_RTS_FLAG_OFFLOAD))) {
            /* read the data from the sender with a get_zcopy operation on the
             * rndv lane */
            ucp_rndv_handle_recv_contig(rndv_req, rreq, rndv_rts_hdr);
        } else {
            /* if the sender didn't specify its address in the RTS, can't do a
             * get operation, so send an RTR and the sender will send the data
             * with AM messages */
            ucp_rndv_handle_recv_am(rndv_req, rreq, rndv_rts_hdr);
        }
    } else if (UCP_DT_IS_GENERIC(rreq->recv.datatype) ||
               UCP_DT_IS_IOV(rreq->recv.datatype)) {
        /* if the recv side has a generic datatype,
         * send an RTR and the sender will send the data with AM messages */
        ucp_rndv_handle_recv_am(rndv_req, rreq, rndv_rts_hdr);
    } else {
        ucs_fatal("datatype isn't implemented");
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_rndv_process_rts(void *arg, void *data, size_t length,
                                  unsigned tl_flags)
{
    const unsigned recv_flags = UCP_RECV_DESC_FLAG_FIRST |
                                UCP_RECV_DESC_FLAG_LAST  |
                                UCP_RECV_DESC_FLAG_RNDV;
    ucp_worker_h worker = arg;
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = data;
    ucp_context_h context = worker->context;
    ucp_request_t *rreq;
    ucs_status_t status;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&context->mt_lock);

    rreq = ucp_tag_exp_search(&context->tm, rndv_rts_hdr->super.tag,
                              rndv_rts_hdr->size, recv_flags);
    if (rreq != NULL) {
        ucp_rndv_matched(worker, rreq, rndv_rts_hdr);

        /* Cancel req in transport if it was offloaded, because it arrived
           as unexpected */
        ucp_tag_offload_try_cancel(context, rreq, 1);

        UCP_WORKER_STAT_RNDV(worker, EXP);
        status = UCS_OK;
    } else {
        status = ucp_tag_unexp_recv(&context->tm, worker, data, length, tl_flags,
                                    sizeof(*rndv_rts_hdr), recv_flags);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&context->mt_lock);
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
        ucp_request_send_buffer_dereg(sreq, ucp_ep_get_tag_lane(sreq->send.ep));
    } else {
        ucp_rndv_rma_request_send_buffer_dereg(sreq);
    }
    ucp_request_send_generic_dt_finish(sreq);
    ucp_request_complete_send(sreq, UCS_OK);
    return UCS_OK;
}

static size_t ucp_rndv_pack_single_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    ucs_assert(sreq->send.state.dt.offset == 0);

    hdr->rreq_ptr = sreq->send.proto.rreq_ptr;
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

    hdr->rreq_ptr = sreq->send.proto.rreq_ptr;
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

    hdr->rreq_ptr = sreq->send.proto.rreq_ptr;
    length        = sreq->send.length - sreq->send.state.dt.offset;

    return sizeof(*hdr) + ucp_dt_pack(sreq->send.datatype, hdr + 1,
                                      sreq->send.buffer, &sreq->send.state.dt,
                                      length);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_bcopy_send, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = sreq->send.ep;
    ucs_status_t status;

    sreq->send.lane = ucp_ep_get_am_lane(ep);

    if (sreq->send.length <= ucp_ep_config(ep)->am.max_bcopy - sizeof(ucp_rndv_data_hdr_t)) {
        /* send a single bcopy message */
        ucs_trace_data("send on sreq %p, am lane: %d, datatype: %zu. single message "
                       "(bcopy), size: %zu", sreq, sreq->send.lane,
                       (sreq->send.datatype & UCP_DATATYPE_CLASS_MASK), sreq->send.length);
        status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_DATA_LAST,
                                        ucp_rndv_pack_single_data);
    } else {
        /* send multiple bcopy messages (fragments of the original send message) */
        ucs_trace_data("send on sreq %p, size: %zu ,am lane: %d, offset: %zu, datatype: %zu. "
                       "multi message (bcopy)",
                       sreq, sreq->send.length, sreq->send.lane,
                       sreq->send.state.dt.offset,
                       (sreq->send.datatype & UCP_DATATYPE_CLASS_MASK));
        status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA_LAST,
                                       sizeof(ucp_rndv_data_hdr_t),
                                       ucp_rndv_pack_multi_data,
                                       ucp_rndv_pack_multi_data,
                                       ucp_rndv_pack_multi_data_last);
    }
    if (status == UCS_OK) {
        ucp_request_send_generic_dt_finish(sreq);
        ucp_request_complete_send(sreq, UCS_OK);
    }

    return status;
}

static void ucp_rndv_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_request_send_buffer_dereg(req, ucp_ep_get_am_lane(req->send.ep));
    ucp_request_complete_send(req, status);
}

static void ucp_rndv_contig_zcopy_completion(uct_completion_t *self,
                                             ucs_status_t status)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    if (sreq->send.state.dt.offset == sreq->send.length) {
        ucp_rndv_zcopy_req_complete(sreq, status);
    } else if (status != UCS_OK) {
        ucs_fatal("error handling is unsupported with rendezvous protocol");
    }
}

static ucs_status_t ucp_rndv_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_ptr = sreq->send.proto.rreq_ptr;

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_RNDV_DATA_LAST, &hdr, sizeof(hdr),
                                  ucp_rndv_zcopy_req_complete);
}

static ucs_status_t ucp_rndv_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_ptr = sreq->send.proto.rreq_ptr;

    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA_LAST,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_rndv_zcopy_req_complete);
}

static void ucp_rndv_prepare_zcopy_send_buffer(ucp_request_t *sreq, ucp_ep_h ep)
{
    ucs_status_t status;

    if ((sreq->flags & UCP_REQUEST_FLAG_OFFLOADED) &&
        (ucp_ep_get_am_lane(ep) != ucp_ep_get_tag_lane(ep))) {
        ucp_request_send_buffer_dereg(sreq, ucp_ep_get_tag_lane(sreq->send.ep));
    } else {
        /* dereg the original send request since we are going to send on the AM lane next */
        ucp_rndv_rma_request_send_buffer_dereg(sreq);
    }
    if (sreq->send.state.dt.dt.contig[0].memh == UCT_MEM_HANDLE_NULL) {
        /* register the send buffer for the zcopy operation */
        status = ucp_request_send_buffer_reg(sreq, ucp_ep_get_am_lane(ep));
        ucs_assert_always(status == UCS_OK);
    }
}

static void ucp_rndv_prepare_zcopy(ucp_request_t *sreq, ucp_ep_h ep)
{
    ucs_trace_data("send on sreq %p with zcopy, am lane: %d, datatype: %zu, "
                   "size: %zu", sreq, sreq->send.lane,
                   (sreq->send.datatype & UCP_DATATYPE_CLASS_MASK),
                   sreq->send.length);

    ucp_rndv_prepare_zcopy_send_buffer(sreq, ep);

    ucp_request_send_state_reset(sreq, ucp_rndv_contig_zcopy_completion,
                                 UCP_REQUEST_SEND_PROTO_ZCOPY_AM);

    if (sreq->send.length <= ucp_ep_config(ep)->am.max_zcopy -
        sizeof(ucp_rndv_data_hdr_t)) {
        sreq->send.uct.func   = ucp_rndv_zcopy_single;
    } else {
        sreq->send.uct.func   = ucp_rndv_zcopy_multi;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rtr_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_request_t *sreq = (ucp_request_t*) rndv_rtr_hdr->sreq_ptr;
    ucp_ep_h ep = sreq->send.ep;

    /* make sure that the ep on which the rtr was received on is connected */
    ucs_assert_always(!ucp_ep_is_stub(ep));
    ucs_trace_req("RTR received. start sending on sreq %p", sreq);

    if (sreq->flags & UCP_REQUEST_FLAG_OFFLOADED) {
        /* Do not deregister memory here, because am zcopy rndv may
         * need it registered (if am and tag is the same lane). */
        ucp_tag_offload_cancel_rndv(sreq);
    }

    if ((UCP_DT_IS_CONTIG(sreq->send.datatype)) &&
        (sreq->send.length >= ucp_ep_config(ep)->am.zcopy_thresh[0])) {
        /* send with zcopy */
        ucp_rndv_prepare_zcopy(sreq, ep);
    } else {
        /* send with bcopy */
        /* deregister the sender's buffer if it was registered */
        ucp_rndv_rma_request_send_buffer_dereg(sreq);
        ucp_request_send_state_reset(sreq, NULL,
                                     UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        sreq->send.uct.func = ucp_rndv_progress_bcopy_send;
    }

    UCS_PROFILE_REQUEST_EVENT(sreq, "rndv_rtr_recv", 0);
    sreq->send.proto.rreq_ptr = rndv_rtr_hdr->rreq_ptr;
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
        ucs_assert(rreq->recv.info.length == rreq->recv.state.offset + recv_len);
        UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_last_recv", recv_len);
        status = ucp_dt_unpack(rreq->recv.datatype, rreq->recv.buffer,
                               rreq->recv.length, &rreq->recv.state,
                               data + hdr_len, recv_len, 1);
    } else {
        ucs_trace_data("drop last segment for rreq %p, length %zu, status %s",
                       rreq, recv_len, ucs_status_string(rreq->status));
        status = rreq->status;
        ucp_request_recv_generic_dt_finish(rreq);
    }

    ucp_request_complete_recv(rreq, status);

    return UCS_OK;
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
                 "address 0x%"PRIx64" size %zu rkey ", rndv_rts_hdr->super.tag,
                 rndv_rts_hdr->sreq.sender_uuid, rndv_rts_hdr->sreq.reqptr,
                 rndv_rts_hdr->address, rndv_rts_hdr->size);
        ucs_log_dump_hex((void*)rndv_rts_hdr + sizeof(*rndv_rts_hdr), length,
                         buffer + strlen(buffer), max - strlen(buffer));
        break;
    case UCP_AM_ID_RNDV_ATS:
        snprintf(buffer, max, "RNDV_ATS sreq 0x%lx status '%s'",
                 rep_hdr->reqptr, ucs_status_string(rep_hdr->status));
        break;
    case UCP_AM_ID_RNDV_RTR:
        snprintf(buffer, max, "RNDV_RTR sreq 0x%lx rreq 0x%lx",
                 rndv_rtr_hdr->sreq_ptr, rndv_rtr_hdr->rreq_ptr);
        break;
    case UCP_AM_ID_RNDV_DATA:
        snprintf(buffer, max, "RNDV_DATA rreq 0x%"PRIx64,
                 rndv_data->rreq_ptr);
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
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_RTR, ucp_rndv_rtr_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA, ucp_rndv_data_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA_LAST, ucp_rndv_data_last_handler,
              ucp_rndv_dump, UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTR);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA_LAST);
