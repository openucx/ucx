/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "rndv.h"
#include <ucp/proto/proto_am.inl>
#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/queue.h>

#define UCP_ALIGN 256
#define UCP_MTU_SIZE 4096

static size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq = arg;   /* the sender's request */
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = dest;
    ucp_lane_index_t rndv_lane = ucp_ep_get_rndv_get_lane(sreq->send.ep);

    rndv_rts_hdr->super.tag        = sreq->send.tag;
    /* reqptr holds the original sreq */
    rndv_rts_hdr->sreq.reqptr      = (uintptr_t)sreq;
    rndv_rts_hdr->sreq.sender_uuid = sreq->send.ep->worker->uuid;
    rndv_rts_hdr->address          = (uintptr_t)sreq->send.buffer;
    rndv_rts_hdr->size             = sreq->send.length;
    if (sreq->send.state.dt.contig.memh != UCT_INVALID_MEM_HANDLE) {
        uct_md_mkey_pack(ucp_ep_md(sreq->send.ep, rndv_lane),
                         sreq->send.state.dt.contig.memh,
                         rndv_rts_hdr + 1);
        return sizeof(*rndv_rts_hdr) +
                      ucp_ep_md_attr(sreq->send.ep, rndv_lane)->rkey_packed_size;
    } else {
        /* For rndv emulation based on send-recv */
        return sizeof(*rndv_rts_hdr);
    }
}

static ucs_status_t ucp_proto_progress_rndv_rts(uct_pending_req_t *self)
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

    /* For rndv emulation based on send-recv */
    return sizeof(*rndv_rtr_hdr);
}

static ucs_status_t ucp_proto_progress_rndv_rtr(uct_pending_req_t *self)
{
    ucp_request_t *op_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    /* send the RTR. the pack_cb will pack all the necessary fields in the RTR */
    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_RTR, ucp_tag_rndv_rtr_pack);
    if (status == UCS_OK) {
        ucs_mpool_put(op_req);
    }

    return status;
}

ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *sreq)
{
    ucs_status_t status;

    ucs_trace_req("starting rndv. sreq: %p. buffer: %p, length: %zu",
                  sreq, sreq->send.buffer, sreq->send.length);

    ucp_ep_connect_remote(sreq->send.ep);

    /* zcopy */
    status = ucp_request_send_buffer_reg(sreq, ucp_ep_get_rndv_get_lane(sreq->send.ep));
    if (status != UCS_OK) {
        return status;
    }

    sreq->send.uct.func = ucp_proto_progress_rndv_rts;
    return UCS_OK;
}

static void ucp_rndv_send_ats(ucp_request_t *rndv_req, uintptr_t remote_request)
{
    ucs_trace_req("ep: %p send ats. rndv_req: %p, remote_request: %zu",
                  rndv_req->send.ep, rndv_req, remote_request);

    rndv_req->send.lane         = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func     = ucp_proto_progress_am_bcopy_single;
    rndv_req->send.proto.am_id  = UCP_AM_ID_RNDV_ATS;
    rndv_req->send.proto.status = UCS_OK;
    rndv_req->send.proto.remote_request = remote_request;

    ucp_request_start_send(rndv_req);
}

static void ucp_rndv_complete_rndv_get(ucp_request_t *rndv_req)
{
    ucp_request_t *rreq = rndv_req->send.rndv_get.rreq;

    ucs_trace_data("ep: %p rndv get completed", rndv_req->send.ep);

    ucp_request_complete_recv(rreq, UCS_OK, &rreq->recv.info);
    uct_rkey_release(&rndv_req->send.rndv_get.rkey_bundle);
    ucp_request_send_buffer_dereg(rndv_req,
                                  ucp_ep_get_rndv_get_lane(rndv_req->send.ep));

    ucp_rndv_send_ats(rndv_req, rndv_req->send.rndv_get.remote_request);
}

ucs_status_t ucp_proto_progress_rndv_get(uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;
    size_t offset, length;
    uct_iov_t iov[1];

    if (ucp_ep_is_stub(rndv_req->send.ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    /* reset the lane to rndv since it might have been set to 0 since it was stub on RTS receive */
    rndv_req->send.lane = ucp_ep_get_rndv_get_lane(rndv_req->send.ep);

    ucs_trace_data("ep: %p try to progress get_zcopy for rndv get. rndv_req: %p. lane: %d",
                   rndv_req->send.ep, rndv_req, rndv_req->send.lane);

    /* rndv_req is the internal request to perform the get operation */
    if (rndv_req->send.state.dt.contig.memh == UCT_INVALID_MEM_HANDLE) {
        /* TODO Not all UCTs need registration on the recv side */
        status = ucp_request_send_buffer_reg(rndv_req, rndv_req->send.lane);
        ucs_assert_always(status == UCS_OK);

        {
            size_t max_get_zcopy = ucp_ep_config(rndv_req->send.ep)->max_rndv_get_zcopy;
            size_t remainder = (uintptr_t) rndv_req->send.buffer % UCP_ALIGN; /* TODO make UCP_ALIGN come from the transport */

            if (remainder && (rndv_req->send.length > UCP_MTU_SIZE )) {
                rndv_req->send.uct_comp.count = 1 +
                                               (rndv_req->send.length - (UCP_MTU_SIZE - remainder) +
                                               max_get_zcopy - 1) / max_get_zcopy;
            } else {
                rndv_req->send.uct_comp.count = (rndv_req->send.length + max_get_zcopy - 1) / max_get_zcopy;
            }
        }
    }

    offset = rndv_req->send.state.offset;

    if ((offset == 0) && ((uintptr_t)rndv_req->send.buffer % UCP_ALIGN) &&
        (rndv_req->send.length > UCP_MTU_SIZE )) {
        length = UCP_MTU_SIZE - ((uintptr_t)rndv_req->send.buffer % UCP_ALIGN);
    } else {
        length = ucs_min(rndv_req->send.length - offset,
                         ucp_ep_config(rndv_req->send.ep)->max_rndv_get_zcopy);
    }

    ucs_trace_data("offset %zu remainder %zu. read to %p len %zu",
                   offset, (uintptr_t)rndv_req->send.buffer % UCP_ALIGN,
                   (void*)rndv_req->send.buffer + offset, length);

    iov[0].buffer = (void*)rndv_req->send.buffer + offset;
    iov[0].length = length;
    iov[0].memh   = rndv_req->send.state.dt.contig.memh;
    iov[0].count  = 1;
    iov[0].stride = 0;
    status = uct_ep_get_zcopy(ucp_ep_get_rndv_data_uct_ep(rndv_req->send.ep),
                              iov, 1,
                              rndv_req->send.rndv_get.remote_address + offset,
                              rndv_req->send.rndv_get.rkey_bundle.rkey,
                              &rndv_req->send.uct_comp);

    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        rndv_req->send.state.offset += length;
        if (rndv_req->send.state.offset == rndv_req->send.length) {
            if (status == UCS_OK) {
                /* if the zcopy operation was locally-completed, the uct_comp callback
                 * won't be called, so do the completion procedure here */
                ucp_rndv_complete_rndv_get(rndv_req);
            }
            return UCS_OK;
        } else {
            return UCS_INPROGRESS;
        }
    } else {
       return status;
    }
}

static ucs_status_t ucp_rndv_truncated(uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_t *rreq     = (ucp_request_t*) rndv_req->send.proto.rreq_ptr;

    /* if the recv request has a generic datatype, need to finish it */
    ucp_request_recv_generic_dt_finish(rreq);

    ucp_request_complete_recv(rreq, UCS_ERR_MESSAGE_TRUNCATED, &rreq->recv.info);
    ucp_rndv_send_ats(rndv_req, rndv_req->send.proto.remote_request);

    return UCS_OK;
}

static void ucp_rndv_get_completion(uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct_comp);

    ucs_trace_req("rndv: completed get operation. rndv_req: %p", rndv_req);
    ucp_rndv_complete_rndv_get(rndv_req);
}

static void ucp_rndv_handle_recv_contig(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                        ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    size_t recv_size;

    ucs_trace_req("handle contig datatype on rndv receive. local rndv_req: %p, "
                  "recv request: %p", rndv_req, rreq);

    /* rndv_req is the request that would perform the get operation */
    rndv_req->send.uct.func     = ucp_proto_progress_rndv_get;
    rndv_req->send.buffer       = rreq->recv.buffer;

    uct_rkey_unpack(rndv_rts_hdr + 1, &rndv_req->send.rndv_get.rkey_bundle);
    rndv_req->send.rndv_get.remote_request = rndv_rts_hdr->sreq.reqptr;
    rndv_req->send.rndv_get.remote_address = rndv_rts_hdr->address;
    rndv_req->send.rndv_get.rreq = rreq;

    recv_size = ucp_contig_dt_length(rreq->recv.datatype, rreq->recv.count);
    if (ucs_unlikely(recv_size < rndv_rts_hdr->size)) {
        ucs_trace_req("rndv msg truncated: rndv_req: %p. received %zu. "
                      "expected %zu on rreq: %p ",
                      rndv_req, rndv_rts_hdr->size, recv_size, rreq);
        rndv_req->send.uct.func  = ucp_rndv_truncated;
        rndv_req->send.lane      = ucp_ep_get_am_lane(rndv_req->send.ep);
        /* prepare for handling a truncated message. the rndv_get struct isn't needed anymore */
        rndv_req->send.proto.remote_request = rndv_rts_hdr->sreq.reqptr;
        rndv_req->send.proto.rreq_ptr       = (uintptr_t) rreq;
    } else {
        rndv_req->send.length         = rndv_rts_hdr->size;
        rndv_req->send.uct_comp.func  = ucp_rndv_get_completion;
        rndv_req->send.state.offset   = 0;
        rndv_req->send.lane           = ucp_ep_get_rndv_get_lane(rndv_req->send.ep);
        rndv_req->send.state.dt.contig.memh = UCT_INVALID_MEM_HANDLE;
    }
    ucp_request_start_send(rndv_req);
}

static void ucp_rndv_handle_recv_generic(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                         ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_dt_generic_t *dt_gen;
    size_t recv_size;

    ucs_trace_req("handle generic datatype on rndv receive. local rndv_req: %p, "
                  "recv request: %p", rndv_req, rreq);

    /* rndv_req is the request that would send the RTR message to the sender */
    rndv_req->send.lane     = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func = ucp_proto_progress_rndv_rtr;
    /* save the sender's send request and send it in the RTR */
    rndv_req->send.proto.remote_request = rndv_rts_hdr->sreq.reqptr;
    rndv_req->send.proto.status         = UCS_OK;
    rndv_req->send.proto.rreq_ptr       = (uintptr_t) rreq;

    dt_gen = ucp_dt_generic(rreq->recv.datatype);
    recv_size = dt_gen->ops.packed_size(rreq->recv.state.dt.generic.state);
    if (ucs_unlikely(recv_size < rndv_rts_hdr->size)) {
        ucs_trace_req("rndv msg truncated: rndv_req: %p. received %zu. "
                      "expected %zu on rreq: %p ",
                      rndv_req, rndv_rts_hdr->size, recv_size, rreq);
        rndv_req->send.uct.func = ucp_rndv_truncated;
    }

    ucp_request_start_send(rndv_req);
}

void ucp_rndv_matched(ucp_worker_h worker, ucp_request_t *rreq,
                      ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_request_t *rndv_req;

    UCS_ASYNC_BLOCK(&worker->async);

    /* rreq is the receive request on the receiver's side */
    rreq->recv.info.sender_tag = rndv_rts_hdr->super.tag;
    rreq->recv.info.length     = rndv_rts_hdr->size;

    ucs_assert_always(rreq->recv.count != 0);

    /* the internal send request allocated on receiver side (to perform a "get"
     * operation, send "ATS" and "RTR") */
    rndv_req = ucp_worker_allocate_reply(worker, rndv_rts_hdr->sreq.sender_uuid);

    ucs_trace_req("ucp_rndv_matched. remote data address: %zu. remote request: %zu. "
                  "local rndv_req: %p", rndv_rts_hdr->address,
                  rndv_rts_hdr->sreq.reqptr, rndv_req);

    /* if the receive side is not connected yet then the RTS was received on a stub ep */
    if (ucp_ep_is_stub(rndv_req->send.ep)) {
        ucs_debug("received rts on a stub ep, ep=%p, rndv_lane=%d, am_lane=%d",
                   rndv_req->send.ep, ucp_ep_get_rndv_get_lane(rndv_req->send.ep),
                   ucp_ep_get_am_lane(rndv_req->send.ep));
    }

    /* if on the recv side there is a contig datatype, read the data with a get operation */
    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
         ucp_rndv_handle_recv_contig(rndv_req, rreq, rndv_rts_hdr);
    } else if (UCP_DT_IS_GENERIC(rreq->recv.datatype)) {
        /* if on the recv side there is a generic datatype, send an RTR */
        ucp_rndv_handle_recv_generic(rndv_req, rreq, rndv_rts_hdr);
    } else {
        ucs_fatal("datatype isn't implemented");
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
}

static ucs_status_t
ucp_rndv_rts_handler(void *arg, void *data, size_t length, void *desc)
{
    ucp_worker_h worker = arg;
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = data;
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc = desc;
    ucp_tag_t recv_tag = rndv_rts_hdr->super.tag;
    ucp_request_t *rreq;
    ucs_queue_iter_t iter;

    /* Search in expected queue */
    ucs_queue_for_each_safe(rreq, iter, &context->tag.expected, recv.queue) {
        rreq = ucs_container_of(*iter, ucp_request_t, recv.queue);
        if (ucp_tag_recv_is_match(recv_tag, UCP_RECV_DESC_FLAG_FIRST, rreq->recv.tag,
                                  rreq->recv.tag_mask, rreq->recv.state.offset,
                                  rreq->recv.info.sender_tag))
        {
            ucp_tag_log_match(recv_tag, rreq, rreq->recv.tag, rreq->recv.tag_mask,
                              rreq->recv.state.offset, "expected-rndv");
            ucs_queue_del_iter(&context->tag.expected, iter);
            ucp_rndv_matched(worker, rreq, rndv_rts_hdr);
            return UCS_OK;
        }
    }

    ucs_trace_req("unexp rndv recv tag %"PRIx64" length %zu desc %p",
                  recv_tag, length, rdesc);
    if (data != rdesc + 1) {
        memcpy(rdesc + 1, data, length);
    }

    rdesc->length  = length;
    rdesc->hdr_len = sizeof(*rndv_rts_hdr);
    rdesc->flags   = UCP_RECV_DESC_FLAG_FIRST | UCP_RECV_DESC_FLAG_LAST |
                     UCP_RECV_DESC_FLAG_RNDV;
    ucs_queue_push(&context->tag.unexpected, &rdesc->queue);
    return UCS_INPROGRESS;
}

static ucs_status_t
ucp_rndv_ats_handler(void *arg, void *data, size_t length, void *desc)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *sreq = (ucp_request_t*) rep_hdr->reqptr;

    /* dereg the original send request and set it to complete */
    ucp_request_send_buffer_dereg(sreq, ucp_ep_get_rndv_get_lane(sreq->send.ep));
    ucp_request_complete_send(sreq, UCS_OK);
    return UCS_OK;
}

static size_t ucp_rndv_pack_single_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    ucs_assert(sreq->send.state.offset == 0);

    hdr->rreq_ptr = sreq->send.proto.rreq_ptr;
    length = ucp_tag_pack_dt_copy(hdr + 1, sreq->send.buffer,
                                  &sreq->send.state, sreq->send.length,
                                  sreq->send.datatype);
    ucs_assert(length == sreq->send.length);
    return sizeof(*hdr) + length;
}

static size_t ucp_rndv_pack_multi_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    hdr->rreq_ptr = sreq->send.proto.rreq_ptr;
    length        = ucp_ep_config(sreq->send.ep)->max_am_bcopy - sizeof(*hdr);

    return sizeof(*hdr) + ucp_tag_pack_dt_copy(hdr + 1, sreq->send.buffer,
                                               &sreq->send.state, length,
                                               sreq->send.datatype);
}

static size_t ucp_rndv_pack_multi_data_last(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length;

    hdr->rreq_ptr = sreq->send.proto.rreq_ptr;
    length        = sreq->send.length - sreq->send.state.offset;

    return sizeof(*hdr) + ucp_tag_pack_dt_copy(hdr + 1, sreq->send.buffer,
                                               &sreq->send.state, length,
                                               sreq->send.datatype);
}

static ucs_status_t ucp_rndv_progress_bcopy_send(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep = sreq->send.ep;
    ucs_status_t status;

    sreq->send.lane = ucp_ep_get_am_lane(ep);

    if (sreq->send.length <= ucp_ep_config(ep)->max_am_bcopy - sizeof(ucp_rndv_data_hdr_t)) {
        /* send a single bcopy message */
        ucs_trace_data("send on sreq %p, am lane: %d. single message "
                       "(bcopy), size: %zu", sreq, sreq->send.lane, sreq->send.length);
        status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_DATA_LAST,
                                        ucp_rndv_pack_single_data);
    } else {
        /* send multiple bcopy messages (fragments of the original send message) */
        ucs_trace_data("send on sreq %p, am lane: %d, offset: %zu. multi message (bcopy)",
                       sreq, sreq->send.lane, sreq->send.state.offset);
        status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA_LAST,
                                       sizeof(ucp_rndv_data_hdr_t),
                                       ucp_rndv_pack_multi_data,
                                       ucp_rndv_pack_multi_data,
                                       ucp_rndv_pack_multi_data_last);
    }
    if (status == UCS_OK) {
        ucp_request_complete_send(sreq, UCS_OK);
    }

    return status;
}

static void ucp_rndv_zcopy_req_complete(ucp_request_t *req)
{
    ucp_request_send_buffer_dereg(req, ucp_ep_get_am_lane(req->send.ep));
    ucp_request_complete_send(req, UCS_OK);
}

static void ucp_rndv_contig_zcopy_completion(uct_completion_t *self,
                                             ucs_status_t status)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct_comp);
    ucp_rndv_zcopy_req_complete(sreq);
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

static ucs_status_t
ucp_rndv_rtr_handler(void *arg, void *data, size_t length, void *desc)
{
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_request_t *sreq = (ucp_request_t*) rndv_rtr_hdr->sreq_ptr;
    ucp_ep_h ep = sreq->send.ep;

    /* make sure that the ep on which the rtr was received on is connected */
    ucs_assert_always(!ucp_ep_is_stub(ep));
    ucs_trace_req("RTR received. start sending on sreq %p", sreq);

    if (sreq->send.length >= ucp_ep_config(ep)->zcopy_thresh) {
        /* send with zcopy */
        size_t max_zcopy;
        ucs_status_t status;

        ucs_trace_data("send on sreq %p with zcopy, am lane: %d, size: %zu",
                       sreq, sreq->send.lane, sreq->send.length);

        if (ucp_ep_get_am_lane(ep) != ucp_ep_get_rndv_get_lane(ep)) {
            /* dereg the original send request since we are going to send on the AM lane next */
            ucp_request_send_buffer_dereg(sreq, ucp_ep_get_rndv_get_lane(ep));
            status = ucp_request_send_buffer_reg(sreq, ucp_ep_get_am_lane(ep));
            ucs_assert_always(status == UCS_OK);
        }

        sreq->send.uct_comp.func = ucp_rndv_contig_zcopy_completion;

        max_zcopy = ucp_ep_config(ep)->max_am_zcopy;
        if (sreq->send.length <= max_zcopy - sizeof(ucp_rndv_data_hdr_t)) {
            sreq->send.uct_comp.count = 1;
            sreq->send.uct.func = ucp_rndv_zcopy_single;
        } else {
            /* calculate number of zcopy fragments */
            sreq->send.uct_comp.count = 1 + (sreq->send.length - 1) /
                                        (max_zcopy - sizeof(ucp_rndv_data_hdr_t));
            sreq->send.uct.func = ucp_rndv_zcopy_multi;
        }
    } else {
        /* send with bcopy */
        /* dereg the original send request since we are going to send with bcopy */
        ucp_request_send_buffer_dereg(sreq, ucp_ep_get_rndv_get_lane(ep));
        sreq->send.uct.func = ucp_rndv_progress_bcopy_send;
    }

    sreq->send.proto.rreq_ptr = rndv_rtr_hdr->rreq_ptr;
    ucp_request_start_send(sreq);
    return UCS_OK;
}

static ucs_status_t
ucp_rndv_data_handler(void *arg, void *data, size_t length, void *desc)
{
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq = (ucp_request_t*) rndv_data_hdr->rreq_ptr;
    size_t recv_len;
    ucs_status_t status;
    uint8_t hdr_len = sizeof(*rndv_data_hdr);

    ucs_trace_data("recv segment for rreq %p. offset %zu ",
                   rreq, rreq->recv.state.offset);

    ucs_assert(length >= hdr_len);
    recv_len = length - hdr_len;
    status = ucp_tag_process_recv(rreq->recv.buffer, rreq->recv.count,
                                  rreq->recv.datatype, &rreq->recv.state,
                                  data + hdr_len, recv_len, 0);
    rreq->recv.state.offset += recv_len;

    return status;
}

static ucs_status_t
ucp_rndv_data_last_handler(void *arg, void *data, size_t length, void *desc)
{
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq = (ucp_request_t*) rndv_data_hdr->rreq_ptr;
    size_t recv_len;
    ucs_status_t status;
    uint8_t hdr_len = sizeof(*rndv_data_hdr);

    ucs_trace_data("recv last segment for rreq %p ", rreq);

    ucs_assert(length >= hdr_len);
    recv_len = length - hdr_len;
    status = ucp_tag_process_recv(rreq->recv.buffer, rreq->recv.count,
                                  rreq->recv.datatype, &rreq->recv.state,
                                  data + hdr_len, recv_len, UCP_RECV_DESC_FLAG_LAST);

    ucp_request_complete_recv(rreq, status, &rreq->recv.info);

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
        snprintf(buffer, max, "RNDV_RTS tag %"PRIx64" uuid %"PRIx64
                 "address 0x%"PRIx64" size %zu rkey ", rndv_rts_hdr->super.tag,
                 rndv_rts_hdr->sreq.sender_uuid,
                 rndv_rts_hdr->sreq.reqptr, rndv_rts_hdr->size);

        ucs_log_dump_hex((void*)rndv_rts_hdr + sizeof(*rndv_rts_hdr), length,
                         buffer + strlen(buffer), max - strlen(buffer));
        break;
    case UCP_AM_ID_RNDV_ATS:
        snprintf(buffer, max, "RNDV_ATS request %0"PRIx64" status '%s'", rep_hdr->reqptr,
                 ucs_status_string(rep_hdr->status));
        break;
    case UCP_AM_ID_RNDV_RTR:
        snprintf(buffer, max, "RNDV_RTR sreq_ptr 0x%"PRIx64" rreq_ptr 0x%"PRIx64,
                 rndv_rtr_hdr->sreq_ptr, rndv_rtr_hdr->rreq_ptr);
        break;
    case UCP_AM_ID_RNDV_DATA:
        snprintf(buffer, max, "RNDV_DATA rreq_ptr 0x%"PRIx64,
                 rndv_data->rreq_ptr);
        break;
    case UCP_AM_ID_RNDV_DATA_LAST:
        snprintf(buffer, max, "RNDV_DATA_LAST rreq_ptr 0x%"PRIx64,
                 rndv_data->rreq_ptr);
        break;
    default:
        return;
    }
}

UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_RTS, ucp_rndv_rts_handler,
              ucp_rndv_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_ATS, ucp_rndv_ats_handler,
              ucp_rndv_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_RTR, ucp_rndv_rtr_handler,
              ucp_rndv_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA, ucp_rndv_data_handler,
              ucp_rndv_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA_LAST, ucp_rndv_data_last_handler,
              ucp_rndv_dump, UCT_AM_CB_FLAG_SYNC);
