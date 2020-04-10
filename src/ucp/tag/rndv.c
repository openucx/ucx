/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rndv.h"
#include "tag_match.inl"
#include "offload.h"

#include <ucp/proto/proto_am.inl>
#include <ucs/datastruct/queue.h>

static int ucp_rndv_is_recv_pipeline_needed(ucp_request_t *rndv_req,
                                            ucs_memory_type_t mem_type)
{
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;

    /* no bw lanes */
    if (!ucp_ep_config(rndv_req->send.ep)->key.rma_bw_md_map) {
        return 0;
    }

    /* check if there is a bw lane to register mem type */
    ucs_for_each_bit(md_index,
                     ucp_ep_config(rndv_req->send.ep)->key.rma_bw_md_map) {
        md_attr = &rndv_req->send.ep->worker->context->tl_mds[md_index].attr;
        if (md_attr->cap.reg_mem_types & UCS_BIT(mem_type)) {
            return 0;
        }
    }

    return 1;
}

static ucp_lane_index_t
ucp_rndv_req_get_zcopy_rma_lane(ucp_request_t *rndv_req, ucp_lane_map_t ignore,
                                uct_rkey_t *uct_rkey_p)
{
    ucp_ep_h ep                = rndv_req->send.ep;
    ucp_ep_config_t *ep_config = ucp_ep_config(ep);

   return ucp_rkey_find_rma_lane(ep->worker->context, ep_config,
                                 rndv_req->send.mem_type,
                                 ep_config->tag.rndv.get_zcopy_lanes,
                                 rndv_req->send.rndv_get.rkey, ignore, uct_rkey_p);
}

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq              = arg;   /* send request */
    ucp_rndv_rts_hdr_t *rndv_rts_hdr = dest;
    ucp_worker_h worker              = sreq->send.ep->worker;
    ssize_t packed_rkey_size;

    rndv_rts_hdr->super.tag        = sreq->send.msg_proto.tag.tag;
    rndv_rts_hdr->sreq.reqptr      = (uintptr_t)sreq;
    rndv_rts_hdr->sreq.ep_ptr      = ucp_request_get_dest_ep_ptr(sreq);
    rndv_rts_hdr->size             = sreq->send.length;

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq->send.mem_type,
                              worker->context->config.ext.rndv_mode)) {
        /* pack rkey, ask target to do get_zcopy */
        rndv_rts_hdr->address = (uintptr_t)sreq->send.buffer;
        packed_rkey_size = ucp_rkey_pack_uct(worker->context,
                                             sreq->send.state.dt.dt.contig.md_map,
                                             sreq->send.state.dt.dt.contig.memh,
                                             sreq->send.mem_type,
                                             rndv_rts_hdr + 1);
        if (packed_rkey_size < 0) {
            ucs_fatal("failed to pack rendezvous remote key: %s",
                      ucs_status_string((ucs_status_t)packed_rkey_size));
        }

        ucs_assert(packed_rkey_size <=
                   ucp_ep_config(sreq->send.ep)->tag.rndv.rkey_size);
    } else {
        rndv_rts_hdr->address = 0;
        packed_rkey_size      = 0;
    }

    return sizeof(*rndv_rts_hdr) + packed_rkey_size;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    size_t packed_rkey_size;

    /* send the RTS. the pack_cb will pack all the necessary fields in the RTS */
    packed_rkey_size = ucp_ep_config(sreq->send.ep)->tag.rndv.rkey_size;
    return ucp_do_am_single(self, UCP_AM_ID_RNDV_RTS, ucp_tag_rndv_rts_pack,
                            sizeof(ucp_rndv_rts_hdr_t) + packed_rkey_size);
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
        rndv_rtr_hdr->size    = rndv_req->send.rndv_rtr.length;
        rndv_rtr_hdr->offset  = rreq->recv.frag.offset;

        packed_rkey_size = ucp_rkey_pack_uct(rndv_req->send.ep->worker->context,
                                             rreq->recv.state.dt.contig.md_map,
                                             rreq->recv.state.dt.contig.memh,
                                             rreq->recv.mem_type,
                                             rndv_rtr_hdr + 1);
        if (packed_rkey_size < 0) {
            return packed_rkey_size;
        }
    } else {
        rndv_rtr_hdr->address = 0;
        rndv_rtr_hdr->size    = 0;
        rndv_rtr_hdr->offset  = 0;
        packed_rkey_size      = 0;
    }

    return sizeof(*rndv_rtr_hdr) + packed_rkey_size;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_rndv_rtr, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    size_t packed_rkey_size;
    ucs_status_t status;

    /* send the RTR. the pack_cb will pack all the necessary fields in the RTR */
    packed_rkey_size = ucp_ep_config(rndv_req->send.ep)->tag.rndv.rkey_size;
    status = ucp_do_am_single(self, UCP_AM_ID_RNDV_RTR, ucp_tag_rndv_rtr_pack,
                              sizeof(ucp_rndv_rtr_hdr_t) + packed_rkey_size);
    if (status == UCS_OK) {
        /* release rndv request */
        ucp_request_put(rndv_req);
    }

    return status;
}

ucs_status_t ucp_tag_rndv_reg_send_buffer(ucp_request_t *sreq)
{
    ucp_ep_h ep = sreq->send.ep;
    ucp_md_map_t md_map;
    ucs_status_t status;

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq->send.mem_type,
                              ep->worker->context->config.ext.rndv_mode)) {

        /* register a contiguous buffer for rma_get */
        md_map = ucp_ep_config(ep)->key.rma_bw_md_map;

        /* Pass UCT_MD_MEM_FLAG_HIDE_ERRORS flag, because registration may fail
         * if md does not support send memory type (e.g. CUDA memory). In this
         * case RTS will be sent with empty key, and sender will fallback to
         * PUT or pipeline protocols. */
        status = ucp_request_send_buffer_reg(sreq, md_map,
                                             UCT_MD_MEM_FLAG_HIDE_ERRORS);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *sreq)
{
    ucp_ep_h ep = sreq->send.ep;
    ucs_status_t status;

    ucp_trace_req(sreq, "start_rndv to %s buffer %p length %zu",
                  ucp_ep_peer_name(ep), sreq->send.buffer,
                  sreq->send.length);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    status = ucp_ep_resolve_dest_ep_ptr(ep, sreq->send.lane);
    if (status != UCS_OK) {
        return status;
    }

    if (ucp_ep_is_tag_offload_enabled(ucp_ep_config(ep))) {
        status = ucp_tag_offload_start_rndv(sreq);
    } else {
        ucs_assert(sreq->send.lane == ucp_ep_get_am_lane(ep));
        sreq->send.uct.func = ucp_proto_progress_rndv_rts;
        status              = ucp_tag_rndv_reg_send_buffer(sreq);
    }

    return status;
}

static void ucp_rndv_complete_send(ucp_request_t *sreq, ucs_status_t status)
{
    ucp_request_send_generic_dt_finish(sreq);
    ucp_request_send_buffer_dereg(sreq);
    ucp_request_complete_send(sreq, status);
}

static void ucp_rndv_req_send_ats(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                  uintptr_t remote_request, ucs_status_t status)
{
    ucp_trace_req(rndv_req, "send ats remote_request 0x%lx", remote_request);
    UCS_PROFILE_REQUEST_EVENT(rreq, "send_ats", 0);

    rndv_req->send.lane                 = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func             = ucp_proto_progress_am_single;
    rndv_req->send.proto.am_id          = UCP_AM_ID_RNDV_ATS;
    rndv_req->send.proto.status         = status;
    rndv_req->send.proto.remote_request = remote_request;
    rndv_req->send.proto.comp_cb        = ucp_request_put;

    ucp_request_send(rndv_req, 0);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_complete_rma_put_zcopy, (sreq),
                      ucp_request_t *sreq)
{
    ucp_trace_req(sreq, "rndv_put completed");
    UCS_PROFILE_REQUEST_EVENT(sreq, "complete_rndv_put", 0);

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

    /* destroy rkey before it gets overridden by ATP protocol data */
    ucp_rkey_destroy(sreq->send.rndv_put.rkey);

    sreq->send.lane                 = ucp_ep_get_am_lane(sreq->send.ep);
    sreq->send.uct.func             = ucp_proto_progress_am_single;
    sreq->send.proto.am_id          = UCP_AM_ID_RNDV_ATP;
    sreq->send.proto.status         = UCS_OK;
    sreq->send.proto.remote_request = remote_request;
    sreq->send.proto.comp_cb        = ucp_rndv_complete_rma_put_zcopy;

    ucp_request_send(sreq, 0);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_complete_frag_rma_put_zcopy, (fsreq),
                      ucp_request_t *fsreq)
{
    ucp_request_t *sreq = fsreq->send.proto.sreq;

    sreq->send.state.dt.offset += fsreq->send.length;

    /* delete fragments send request */
    ucp_request_put(fsreq);

    /* complete send request after put completions of all fragments */
    if (sreq->send.state.dt.offset == sreq->send.length) {
        ucp_rndv_complete_rma_put_zcopy(sreq);
    }
}

static void ucp_rndv_send_frag_atp(ucp_request_t *fsreq, uintptr_t remote_request)
{
    ucp_trace_req(fsreq, "send frag atp remote_request 0x%lx", remote_request);
    UCS_PROFILE_REQUEST_EVENT(fsreq, "send_frag_atp", 0);

    /* destroy rkey before it gets overridden by ATP protocol data */
    ucp_rkey_destroy(fsreq->send.rndv_put.rkey);

    fsreq->send.lane                 = ucp_ep_get_am_lane(fsreq->send.ep);
    fsreq->send.uct.func             = ucp_proto_progress_am_single;
    fsreq->send.proto.sreq           = fsreq->send.rndv_put.sreq;
    fsreq->send.proto.am_id          = UCP_AM_ID_RNDV_ATP;
    fsreq->send.proto.status         = UCS_OK;
    fsreq->send.proto.remote_request = remote_request;
    fsreq->send.proto.comp_cb        = ucp_rndv_complete_frag_rma_put_zcopy;

    ucp_request_send(fsreq, 0);
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

    ucp_rndv_req_send_ats(rndv_req, rreq, rndv_req->send.rndv_get.remote_request,
                          UCS_OK);
    ucp_rndv_zcopy_recv_req_complete(rreq, UCS_OK);
}

static void ucp_rndv_recv_data_init(ucp_request_t *rreq, size_t size)
{
    rreq->status             = UCS_OK;
    rreq->recv.tag.remaining = size;
    rreq->recv.frag.rreq     = NULL;
    rreq->recv.frag.offset   = 0;
}

static void ucp_rndv_req_send_rtr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                  uintptr_t sender_reqptr, size_t recv_length)
{
    ucp_trace_req(rndv_req, "send rtr remote sreq 0x%lx rreq %p", sender_reqptr,
                  rreq);

    rndv_req->send.lane                    = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func                = ucp_proto_progress_rndv_rtr;
    rndv_req->send.rndv_rtr.remote_request = sender_reqptr;
    rndv_req->send.rndv_rtr.rreq           = rreq;
    rndv_req->send.rndv_rtr.length         = recv_length;

    ucp_request_send(rndv_req, 0);
}

static void ucp_rndv_get_lanes_count(ucp_request_t *rndv_req)
{
    ucp_ep_h ep        = rndv_req->send.ep;
    ucp_lane_map_t map = 0;
    uct_rkey_t uct_rkey;
    ucp_lane_index_t lane;

    if (ucs_likely(rndv_req->send.rndv_get.lane_count != 0)) {
        return; /* already resolved */
    }

    while ((lane = ucp_rndv_req_get_zcopy_rma_lane(rndv_req, map, &uct_rkey))
            != UCP_NULL_LANE) {
        rndv_req->send.rndv_get.lane_count++;
        map |= UCS_BIT(lane);
    }

    rndv_req->send.rndv_get.lane_count = ucs_min(rndv_req->send.rndv_get.lane_count,
                                                 ep->worker->context->config.ext.max_rndv_lanes);
}

static ucp_lane_index_t
ucp_rndv_get_zcopy_get_lane(ucp_request_t *rndv_req, uct_rkey_t *uct_rkey)
{
    ucp_lane_index_t lane;

    lane = ucp_rndv_req_get_zcopy_rma_lane(rndv_req,
                                           rndv_req->send.rndv_get.lanes_map,
                                           uct_rkey);

    if ((lane == UCP_NULL_LANE) && (rndv_req->send.rndv_get.lanes_map != 0)) {
        /* lanes_map != 0 - no more lanes (but BW lanes are exist because map
         * is not NULL - we found at least one lane on previous iteration).
         * reset used lanes map to NULL and iterate it again */
        rndv_req->send.rndv_get.lanes_map = 0;
        lane = ucp_rndv_req_get_zcopy_rma_lane(rndv_req,
                                               rndv_req->send.rndv_get.lanes_map,
                                               uct_rkey);
    }

    return lane;
}

static void ucp_rndv_get_zcopy_next_lane(ucp_request_t *rndv_req)
{
    /* mask lane for next iteration.
     * next time this lane will not be selected & we continue
     * with another lane */
    ucp_ep_h ep = rndv_req->send.ep;

    rndv_req->send.rndv_get.lanes_map |= UCS_BIT(rndv_req->send.lane);

    /* in case if masked too much lanes - reset mask to zero
     * to select first lane next time */
    if (ucs_popcount(rndv_req->send.rndv_get.lanes_map) >=
        ep->worker->context->config.ext.max_rndv_lanes) {
        rndv_req->send.rndv_get.lanes_map = 0;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_get_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep             = rndv_req->send.ep;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    const size_t max_iovcnt = 1;
    uct_iface_attr_t* attrs;
    ucs_status_t status;
    size_t offset, length, ucp_mtu, remaining, align, chunk;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_rsc_index_t rsc_index;
    ucp_dt_state_t state;
    uct_rkey_t uct_rkey;
    size_t min_zcopy;
    size_t max_zcopy;
    size_t tail;
    int pending_add_res;
    ucp_lane_index_t lane;

    ucp_rndv_get_lanes_count(rndv_req);

    /* Figure out which lane to use for get operation */
    rndv_req->send.lane = lane = ucp_rndv_get_zcopy_get_lane(rndv_req, &uct_rkey);

    if (lane == UCP_NULL_LANE) {
        /* If can't perform get_zcopy - switch to active-message.
         * NOTE: we do not register memory and do not send our keys. */
        ucp_trace_req(rndv_req, "remote memory unreachable, switch to rtr");
        ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
        ucp_rndv_recv_data_init(rndv_req->send.rndv_get.rreq,
                                rndv_req->send.length);
        ucp_rndv_req_send_rtr(rndv_req, rndv_req->send.rndv_get.rreq,
                              rndv_req->send.rndv_get.remote_request,
                              rndv_req->send.length);
        return UCS_OK;
    }

    if (!rndv_req->send.mdesc) {
        status = ucp_send_request_add_reg_lane(rndv_req, lane);
        ucs_assert_always(status == UCS_OK);
    }

    rsc_index = ucp_ep_get_rsc_index(ep, lane);
    attrs     = ucp_worker_iface_get_attr(ep->worker, rsc_index);
    align     = attrs->cap.get.opt_zcopy_align;
    ucp_mtu   = attrs->cap.get.align_mtu;
    min_zcopy = config->tag.rndv.min_get_zcopy;
    max_zcopy = config->tag.rndv.max_get_zcopy;

    offset    = rndv_req->send.state.dt.offset;
    remaining = (uintptr_t)rndv_req->send.buffer % align;

    if ((offset == 0) && (remaining > 0) && (rndv_req->send.length > ucp_mtu)) {
        length = ucp_mtu - remaining;
    } else {
        chunk = ucs_align_up((size_t)(ucs_min(rndv_req->send.length /
                                              rndv_req->send.rndv_get.lane_count,
                                              max_zcopy) * config->tag.rndv.scale[lane]),
                             align);
        length = ucs_min(chunk, rndv_req->send.length - offset);
    }

    /* ensure that the current length is over min_zcopy */
    length = ucs_max(length, min_zcopy);

    /* ensure that tail (rest of message) is over min_zcopy */
    ucs_assertv(rndv_req->send.length >= (offset + length),
                "send_length=%zu, offset=%zu, length=%zu",
                rndv_req->send.length, offset, length);
    tail = rndv_req->send.length - (offset + length);
    if (ucs_unlikely((tail != 0) && (tail < min_zcopy))) {
        /* ok, tail is less get zcopy minimal & could not be processed as
         * standalone operation */
        /* check if we have room to increase current part and not
         * step over max_zcopy */
        if (length < (max_zcopy - tail)) {
            /* if we can increase length by min_zcopy - let's do it to
             * avoid small tail (we have limitation on minimal get zcopy) */
            length += tail;
        } else {
            /* reduce current length by align or min_zcopy value
             * to process it on next round */
            ucs_assert(length > ucs_max(min_zcopy, align));
            length -= ucs_max(min_zcopy, align);
        }
    }

    ucs_assertv(length >= min_zcopy, "length=%zu, min_zcopy=%zu",
                length, min_zcopy);
    ucs_assertv(((rndv_req->send.length - (offset + length)) == 0) ||
                ((rndv_req->send.length - (offset + length)) >= min_zcopy),
                "send_length=%zu, offset=%zu, length=%zu, min_zcopy=%zu",
                rndv_req->send.length, offset, length, min_zcopy);

    ucs_trace_data("req %p: offset %zu remainder %zu rma-get to %p len %zu lane %d",
                   rndv_req, offset, remaining,
                   UCS_PTR_BYTE_OFFSET(rndv_req->send.buffer, offset),
                   length, lane);

    state = rndv_req->send.state.dt;
    /* TODO: is this correct? memh array may skip MD's where
     * registration is not supported. for now SHM may avoid registration,
     * but it will work on single lane */
    ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iovcnt, &state,
                        rndv_req->send.buffer, ucp_dt_make_contig(1), length,
                        ucp_ep_md_index(ep, lane),
                        rndv_req->send.mdesc);

    for (;;) {
        status = uct_ep_get_zcopy(ep->uct_eps[lane],
                                  iov, iovcnt,
                                  rndv_req->send.rndv_get.remote_address + offset,
                                  uct_rkey,
                                  &rndv_req->send.state.uct_comp);
        ucp_request_send_state_advance(rndv_req, &state,
                                       UCP_REQUEST_SEND_PROTO_RNDV_GET,
                                       status);
        if (rndv_req->send.state.dt.offset == rndv_req->send.length) {
            if (rndv_req->send.state.uct_comp.count == 0) {
                rndv_req->send.state.uct_comp.func(&rndv_req->send.state.uct_comp, status);
            }
            return UCS_OK;
        } else if (!UCS_STATUS_IS_ERR(status)) {
            /* in case if not all chunks are transmitted - return in_progress
             * status */
            ucp_rndv_get_zcopy_next_lane(rndv_req);
            return UCS_INPROGRESS;
        } else {
            if (status == UCS_ERR_NO_RESOURCE) {
                if (lane != rndv_req->send.pending_lane) {
                    /* switch to new pending lane */
                    pending_add_res = ucp_request_pending_add(rndv_req, &status, 0);
                    if (!pending_add_res) {
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
    rndv_req->send.mem_type                = rreq->recv.mem_type;
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

    ucp_request_send(rndv_req, 0);
}

static void ucp_rndv_send_frag_rtr(ucp_worker_h worker, ucp_request_t *rndv_req,
                                   ucp_request_t *rreq,
                                   const ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    size_t max_frag_size = worker->context->config.ext.rndv_frag_size;
    int i, num_frags;
    size_t frag_size;
    size_t offset;
    ucp_mem_desc_t *mdesc;
    ucp_request_t *freq;
    ucp_request_t *frndv_req;
    unsigned md_index;
    unsigned memh_index;

    ucp_trace_req(rreq, "using rndv pipeline protocol rndv_req %p", rndv_req);

    offset    = 0;
    num_frags = ucs_div_round_up(rndv_rts_hdr->size, max_frag_size);

    for (i = 0; i < num_frags; i++) {
        frag_size = ucs_min(max_frag_size, (rndv_rts_hdr->size - offset));

        /* internal fragment recv request allocated on receiver side to receive
         *  put fragment from sender and to perform a put to recv buffer */
        freq = ucp_request_get(worker);
        if (freq == NULL) {
            ucs_fatal("failed to allocate fragment receive request");
        }

        /* internal rndv request to send RTR */
        frndv_req = ucp_request_get(worker);
        if (frndv_req == NULL) {
            ucs_fatal("failed to allocate fragment rendezvous reply");
        }

        /* allocate fragment recv buffer desc*/
        mdesc = ucp_worker_mpool_get(&worker->rndv_frag_mp);
        if (mdesc == NULL) {
            ucs_fatal("failed to allocate fragment memory buffer");
        }

        freq->recv.buffer                 = mdesc + 1;
        freq->recv.datatype               = ucp_dt_make_contig(1);
        freq->recv.mem_type               = UCS_MEMORY_TYPE_HOST;
        freq->recv.length                 = frag_size;
        freq->recv.state.dt.contig.md_map = 0;
        freq->recv.frag.rreq              = rreq;
        freq->recv.frag.offset            = offset;
        freq->flags                      |= UCP_REQUEST_DEBUG_RNDV_FRAG;

        memh_index = 0;
        ucs_for_each_bit(md_index,
                         (ucp_ep_config(rndv_req->send.ep)->key.rma_bw_md_map &
                          mdesc->memh->md_map)) {
            freq->recv.state.dt.contig.memh[memh_index++] = ucp_memh2uct(mdesc->memh, md_index);
            freq->recv.state.dt.contig.md_map            |= UCS_BIT(md_index);
        }
        ucs_assert(memh_index <= UCP_MAX_OP_MDS);

        frndv_req->send.ep           = rndv_req->send.ep;
        frndv_req->send.pending_lane = UCP_NULL_LANE;

        ucp_rndv_req_send_rtr(frndv_req, freq, rndv_rts_hdr->sreq.reqptr,
                              freq->recv.length);
        offset += frag_size;
    }

    /* release original rndv reply request */
    ucp_request_put(rndv_req);
}

static UCS_F_ALWAYS_INLINE int
ucp_rndv_is_rkey_ptr(const ucp_rndv_rts_hdr_t *rndv_rts_hdr, ucp_ep_h ep,
                     ucs_memory_type_t recv_mem_type, ucp_rndv_mode_t rndv_mode)
{
    const ucp_ep_config_t *ep_config = ucp_ep_config(ep);

    return /* must have remote address */
           (rndv_rts_hdr->address != 0) &&
           /* remote key must be on a memory domain for which we support rkey_ptr */
           (ucp_rkey_packed_md_map(rndv_rts_hdr + 1) &
            ep_config->tag.rndv.rkey_ptr_dst_mds) &&
           /* rendezvous mode must not be forced to put/get */
           (rndv_mode == UCP_RNDV_MODE_AUTO) &&
           /* need local memory access for data unpack */
           UCP_MEM_IS_ACCESSIBLE_FROM_CPU(recv_mem_type);
}

static UCS_F_ALWAYS_INLINE void
ucp_rndv_rkey_ptr_rreq_advance(ucp_request_t *req, size_t length)
{
    /* advance `offset` only in case of contiguous datatype, `state` for other
     * datatypes is advanced in ucp_request_recv_data_unpack() */
    if ((req->recv.datatype & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_CONTIG) {
        req->recv.state.offset += length;
    }
}

static unsigned ucp_rndv_progress_rkey_ptr(void *arg)
{
    ucp_worker_h worker     = (ucp_worker_h)arg;
    ucp_request_t *rndv_req = ucs_queue_head_elem_non_empty(&worker->rkey_ptr_reqs,
                                                            ucp_request_t,
                                                            send.rkey_ptr.queue_elem);
    ucp_request_t *rreq     = rndv_req->send.rndv_get.rreq;
    size_t seg_size         = ucs_min(worker->context->config.ext.rkey_ptr_seg_size,
                                      rndv_req->send.length - rreq->recv.state.offset);
    ucs_status_t status;

    status = ucp_request_recv_data_unpack(rreq, rndv_req->send.buffer, seg_size,
                                          rreq->recv.state.offset, 1);
    ucp_rndv_rkey_ptr_rreq_advance(rreq, seg_size);
    if (ucs_unlikely(status != UCS_OK) ||
        (rreq->recv.state.offset == rndv_req->send.length)) {
        ucp_request_complete_tag_recv(rreq, status);
        ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
        ucp_rndv_req_send_ats(rndv_req, rreq,
                              rndv_req->send.rndv_get.remote_request, status);

        ucs_queue_pull_non_empty(&worker->rkey_ptr_reqs);
        if (ucs_queue_is_empty(&worker->rkey_ptr_reqs)) {
            uct_worker_progress_unregister_safe(worker->uct,
                                                &worker->rkey_ptr_cb_id);
        }
    }

    return 1;
}

static void ucp_rndv_do_rkey_ptr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                 const ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_ep_h ep                      = rndv_req->send.ep;
    const ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    ucp_worker_h worker              = rreq->recv.worker;
    ucp_md_index_t dst_md_index;
    ucp_lane_index_t i, lane;
    ucs_status_t status;
    unsigned rkey_index;
    void *local_ptr;
    ucp_rkey_h rkey;

    ucp_trace_req(rndv_req, "start rkey_ptr rndv rreq %p", rreq);

    status = ucp_ep_rkey_unpack(ep, rndv_rts_hdr + 1, &rkey);
    if (status != UCS_OK) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(ep), ucs_status_string(status));
    }

    /* Find a lane which is capable of accessing the destination memory */
    lane = UCP_NULL_LANE;
    for (i = 0; i < ep_config->key.num_lanes; ++i) {
        dst_md_index = ep_config->key.lanes[i].dst_md_index;
        if (UCS_BIT(dst_md_index) & rkey->md_map) {
            lane = i;
            break;
        }
    }

    if (ucs_unlikely(lane == UCP_NULL_LANE)) {
        /* We should be able to find a lane, because ucp_rndv_is_rkey_ptr()
         * already checked that (rkey->md_map & ep_config->rkey_ptr_dst_mds) != 0
         */
        ucs_fatal("failed to find a lane to access remote memory domains 0x%lx",
                  rkey->md_map);
    }

    rkey_index = ucs_bitmap2idx(rkey->md_map, dst_md_index);
    status     = uct_rkey_ptr(rkey->tl_rkey[rkey_index].cmpt,
                              &rkey->tl_rkey[rkey_index].rkey,
                              rndv_rts_hdr->address, &local_ptr);
    if (status != UCS_OK) {
        ucp_request_complete_tag_recv(rreq, status);
        ucp_rkey_destroy(rkey);
        ucp_rndv_req_send_ats(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr, status);
        return;
    }

    rreq->recv.state.offset = 0;

    ucp_trace_req(rndv_req, "obtained a local pointer to remote buffer: %p",
                  local_ptr);
    rndv_req->send.buffer                  = local_ptr;
    rndv_req->send.length                  = rndv_rts_hdr->size;
    rndv_req->send.rkey_ptr.rkey           = rkey;
    rndv_req->send.rkey_ptr.remote_request = rndv_rts_hdr->sreq.reqptr;
    rndv_req->send.rkey_ptr.rreq           = rreq;

    ucs_queue_push(&worker->rkey_ptr_reqs, &rndv_req->send.rkey_ptr.queue_elem);
    uct_worker_progress_register_safe(worker->uct,
                                      ucp_rndv_progress_rkey_ptr,
                                      rreq->recv.worker,
                                      UCS_CALLBACKQ_FLAG_FAST,
                                      &worker->rkey_ptr_cb_id);
}

static UCS_F_ALWAYS_INLINE int
ucp_rndv_test_zcopy_scheme_support(size_t length, size_t min_zcopy,
                                   size_t max_zcopy, int split)
{
    return /* is the current message greater than the minimal GET/PUT Zcopy? */
           (length >= min_zcopy) &&
           /* is the current message less than the maximal GET/PUT Zcopy? */
           ((length <= max_zcopy) ||
            /* or can the message be split? */ split);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_matched, (worker, rreq, rndv_rts_hdr),
                      ucp_worker_h worker, ucp_request_t *rreq,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_rndv_mode_t rndv_mode;
    ucp_request_t *rndv_req;
    ucp_ep_h ep;
    ucp_ep_config_t *ep_config;

    UCS_ASYNC_BLOCK(&worker->async);

    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_match", 0);

    /* rreq is the receive request on the receiver's side */
    rreq->recv.tag.info.sender_tag = rndv_rts_hdr->super.tag;
    rreq->recv.tag.info.length     = rndv_rts_hdr->size;

    /* the internal send request allocated on receiver side (to perform a "get"
     * operation, send "ATS" and "RTR") */
    rndv_req = ucp_request_get(worker);
    if (rndv_req == NULL) {
        ucs_error("failed to allocate rendezvous reply");
        goto out;
    }

    rndv_req->send.ep           = ucp_worker_get_ep_by_ptr(worker,
                                                           rndv_rts_hdr->sreq.ep_ptr);
    rndv_req->flags             = 0;
    rndv_req->send.mdesc        = NULL;
    rndv_req->send.pending_lane = UCP_NULL_LANE;

    ucp_trace_req(rreq,
                  "rndv matched remote {address 0x%"PRIx64" size %zu sreq 0x%lx}"
                  " rndv_sreq %p", rndv_rts_hdr->address, rndv_rts_hdr->size,
                  rndv_rts_hdr->sreq.reqptr, rndv_req);

    if (ucs_unlikely(rreq->recv.length < rndv_rts_hdr->size)) {
        ucp_trace_req(rndv_req,
                      "rndv truncated remote size %zu local size %zu rreq %p",
                      rndv_rts_hdr->size, rreq->recv.length, rreq);
        ucp_rndv_req_send_ats(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr, UCS_OK);
        ucp_request_recv_generic_dt_finish(rreq);
        ucp_rndv_zcopy_recv_req_complete(rreq, UCS_ERR_MESSAGE_TRUNCATED);
        goto out;
    }

    /* if the receive side is not connected yet then the RTS was received on a stub ep */
    ep        = rndv_req->send.ep;
    ep_config = ucp_ep_config(ep);
    rndv_mode = worker->context->config.ext.rndv_mode;

    if (ucp_rndv_is_rkey_ptr(rndv_rts_hdr, ep, rreq->recv.mem_type, rndv_mode)) {
        ucp_rndv_do_rkey_ptr(rndv_req, rreq, rndv_rts_hdr);
        goto out;
    }

    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        if ((rndv_rts_hdr->address != 0) &&
            (ucp_rndv_is_get_zcopy(rreq->recv.mem_type, rndv_mode)) &&
            ucp_rndv_test_zcopy_scheme_support(rndv_rts_hdr->size,
                                               ep_config->tag.rndv.min_get_zcopy,
                                               ep_config->tag.rndv.max_get_zcopy,
                                               ep_config->tag.rndv.get_zcopy_split)) {
            /* try to fetch the data with a get_zcopy operation */
            ucp_rndv_req_send_rma_get(rndv_req, rreq, rndv_rts_hdr);
            goto out;
        } else if (rndv_mode == UCP_RNDV_MODE_AUTO) {
            /* check if we need pipelined memtype staging */
            if (UCP_MEM_IS_CUDA(rreq->recv.mem_type) &&
                ucp_rndv_is_recv_pipeline_needed(rndv_req,
                                                 rreq->recv.mem_type)) {
                ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
                ucp_rndv_send_frag_rtr(worker, rndv_req, rreq, rndv_rts_hdr);
                goto out;
            }
        }
        /* put protocol is allowed - register receive buffer memory for rma */
        ucs_assert(rndv_rts_hdr->size <= rreq->recv.length);
        ucp_request_recv_buffer_reg(rreq, ep_config->key.rma_bw_md_map,
                                    rndv_rts_hdr->size);
    }

    /* The sender didn't specify its address in the RTS, or the rndv mode was
     * configured to PUT, or GET rndv mode is unsupported - send an RTR and
     * the sender will send the data with active message or put_zcopy. */
    ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
    ucp_rndv_req_send_rtr(rndv_req, rreq, rndv_rts_hdr->sreq.reqptr,
                          rndv_rts_hdr->size);

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
        status = ucp_recv_desc_init(worker, data, length, 0, tl_flags,
                                    sizeof(*rndv_rts_hdr),
                                    UCP_RECV_DESC_FLAG_RNDV, 0, &rdesc);
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
    ucp_rndv_complete_send(sreq, rep_hdr->status);
    return UCS_OK;
}

static size_t ucp_rndv_pack_data(void *dest, void *arg)
{
    ucp_rndv_data_hdr_t *hdr = dest;
    ucp_request_t *sreq = arg;
    size_t length, offset;

    offset        = sreq->send.state.dt.offset;
    hdr->rreq_ptr = sreq->send.msg_proto.tag.rreq_ptr;
    hdr->offset   = offset;
    length        = ucs_min(sreq->send.length - offset,
                            ucp_ep_get_max_bcopy(sreq->send.ep, sreq->send.lane) - sizeof(*hdr));

    return sizeof(*hdr) + ucp_dt_pack(sreq->send.ep->worker, sreq->send.datatype,
                                      sreq->send.mem_type, hdr + 1, sreq->send.buffer,
                                      &sreq->send.state.dt, length);
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
                                        ucp_rndv_pack_data);
    } else {
        status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       ucp_rndv_pack_data,
                                       ucp_rndv_pack_data, 1);
    }
    if (status == UCS_OK) {
        ucp_rndv_complete_send(sreq, UCS_OK);
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
    size_t offset, ucp_mtu, align, remaining, length;
    uct_iface_attr_t *attrs;
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    ucp_dt_state_t state;

    if (!sreq->send.mdesc) {
        status = ucp_request_send_buffer_reg_lane(sreq, sreq->send.lane, 0);
        ucs_assert_always(status == UCS_OK);
    }

    attrs     = ucp_worker_iface_get_attr(ep->worker,
                                          ucp_ep_get_rsc_index(ep, sreq->send.lane));
    align     = attrs->cap.put.opt_zcopy_align;
    ucp_mtu   = attrs->cap.put.align_mtu;

    offset    = sreq->send.state.dt.offset;
    remaining = (uintptr_t)sreq->send.buffer % align;

    if ((offset == 0) && (remaining > 0) && (sreq->send.length > ucp_mtu)) {
        length = ucp_mtu - remaining;
    } else {
        length = ucs_min(sreq->send.length - offset,
                         ucp_ep_config(ep)->tag.rndv.max_put_zcopy);
    }

    ucs_trace_data("req %p: offset %zu remainder %zu. read to %p len %zu",
                   sreq, offset, (uintptr_t)sreq->send.buffer % align,
                   UCS_PTR_BYTE_OFFSET(sreq->send.buffer, offset), length);

    state = sreq->send.state.dt;
    ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iovcnt, &state,
                        sreq->send.buffer, ucp_dt_make_contig(1), length,
                        ucp_ep_md_index(ep, sreq->send.lane), sreq->send.mdesc);
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
            sreq->send.state.uct_comp.func(&sreq->send.state.uct_comp, status);
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
    ucs_assert(req->send.state.uct_comp.count == 0);
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

    hdr.rreq_ptr = sreq->send.msg_proto.tag.rreq_ptr;
    hdr.offset   = 0;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_RNDV_DATA, &hdr, sizeof(hdr),
                                  ucp_rndv_am_zcopy_send_req_complete);
}

static ucs_status_t ucp_rndv_progress_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_ptr = sreq->send.msg_proto.tag.rreq_ptr;
    hdr.offset   = sreq->send.state.dt.offset;
    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_rndv_am_zcopy_send_req_complete, 1);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_frag_send_put_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *freq = ucs_container_of(self, ucp_request_t, send.state.uct_comp);
    ucp_request_t *req  = freq->send.rndv_put.sreq;

    /* release memory descriptor */
    if (freq->send.mdesc) {
        ucs_mpool_put_inline((void *)freq->send.mdesc);
    }

    req->send.state.dt.offset += freq->send.length;
    ucs_assert(req->send.state.dt.offset <= req->send.length);

    /* send ATP for last fragment of the rndv request */
    if (req->send.length == req->send.state.dt.offset) {
        ucp_rndv_send_frag_atp(req, req->send.rndv_put.remote_request);
    }

    ucp_request_put(freq);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_frag_recv_put_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *freq = ucs_container_of(self, ucp_request_t, send.state.uct_comp);
    ucp_request_t *req  = freq->send.rndv_put.sreq;

    /* release memory descriptor */
    if (freq->send.mdesc) {
        ucs_mpool_put_inline((void *)freq->send.mdesc);
    }

    /* put completion on mem type endpoint to stage data to recv buffer */
    req->recv.tag.remaining -= freq->send.length;

    ucp_request_put(freq);

    if (req->recv.tag.remaining == 0) {
        ucp_request_complete_tag_recv(req, UCS_OK);
    }
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_frag_get_completion, (self, status),
                      uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *freq  = ucs_container_of(self, ucp_request_t, send.state.uct_comp);
    ucp_request_t *fsreq = freq->send.rndv_get.rreq;

    /* get completed on memtype endpoint to stage on host. send put request to receiver*/
    ucp_request_send_state_reset(freq, ucp_rndv_frag_send_put_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    freq->send.rndv_put.remote_address   = fsreq->send.rndv_put.remote_address +
        (freq->send.rndv_get.remote_address - (uint64_t)fsreq->send.buffer);
    freq->send.ep                        = fsreq->send.ep;
    freq->send.uct.func                  = ucp_rndv_progress_rma_put_zcopy;
    freq->send.rndv_put.sreq             = fsreq;
    freq->send.rndv_put.rkey             = fsreq->send.rndv_put.rkey;
    freq->send.rndv_put.uct_rkey         = fsreq->send.rndv_put.uct_rkey;
    freq->send.lane                      = fsreq->send.lane;
    freq->send.state.dt.dt.contig.md_map = 0;

    ucp_request_send(freq, 0);
}

static ucs_status_t ucp_rndv_pipeline(ucp_request_t *sreq,
                                      ucp_rndv_rtr_hdr_t *rndv_rtr_hdr)
{
    ucp_worker_h worker   = sreq->send.ep->worker;
    ucp_context_h context = worker->context;
    const uct_md_attr_t *md_attr;
    ucp_lane_index_t mem_type_rma_lane;
    ucp_ep_h mem_type_ep;
    ucp_mem_desc_t *mdesc;
    ucp_request_t *freq;
    ucp_request_t *fsreq;
    ucp_md_index_t md_index;
    ucs_status_t status;
    int i, num_frags;
    size_t max_frag_size, rndv_size, length;
    size_t offset, rndv_base_offset;

    ucp_trace_req(sreq, "using rndv pipeline protocol");

    /* check if lane supports host memory, to stage sends through host memory */
    md_attr = ucp_ep_md_attr(sreq->send.ep, sreq->send.lane);
    if (!(md_attr->cap.reg_mem_types & UCS_BIT(UCS_MEMORY_TYPE_HOST))) {
        return UCS_ERR_UNSUPPORTED;
    }

    rndv_size        = ucs_min(rndv_rtr_hdr->size, sreq->send.length);
    max_frag_size    = context->config.ext.rndv_frag_size;
    rndv_base_offset = rndv_rtr_hdr->offset;
    num_frags        = ucs_div_round_up(rndv_size, max_frag_size);

    /* initialize send req state on first fragment rndv request */
    if (rndv_base_offset == 0) {
         ucp_request_send_state_reset(sreq, NULL, UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    }

    /* internal send request allocated on sender side to handle send fragments for RTR */
    fsreq = ucp_request_get(worker);
    if (fsreq == NULL) {
        ucs_fatal("failed to allocate fragment receive request");
    }

    ucp_request_send_state_init(fsreq, ucp_dt_make_contig(1), 0);
    fsreq->send.buffer                  = UCS_PTR_BYTE_OFFSET(sreq->send.buffer,
                                                              rndv_base_offset);
    fsreq->send.length                  = rndv_size;
    fsreq->send.ep                      = sreq->send.ep;
    fsreq->send.lane                    = sreq->send.lane;
    fsreq->send.rndv_put.rkey           = sreq->send.rndv_put.rkey;
    fsreq->send.rndv_put.uct_rkey       = sreq->send.rndv_put.uct_rkey;
    fsreq->send.rndv_put.remote_request = rndv_rtr_hdr->rreq_ptr;
    fsreq->send.rndv_put.remote_address = rndv_rtr_hdr->address;
    fsreq->send.rndv_put.sreq           = sreq;
    fsreq->send.state.dt.offset         = 0;

    offset = 0;
    for (i = 0; i < num_frags; i++) {
        length = (i == (num_frags - 1)) ? (rndv_size - offset) : max_frag_size;

        /* internal fragment send request allocated on sender side to receive
         *  mem type fragment stage to host and to perform a put to receiver */
        freq = ucp_request_get(worker);
        if (freq == NULL) {
            ucs_fatal("failed to allocate fragment receive request");
        }

        if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(sreq->send.mem_type)) {
            /* sbuf is in host, directly do put */
            ucp_request_send_state_reset(freq, ucp_rndv_frag_send_put_completion,
                                         UCP_REQUEST_SEND_PROTO_RNDV_PUT);
            md_index                              = ucp_ep_md_index(sreq->send.ep,
                                                                    sreq->send.lane);
            freq->send.ep                         = fsreq->send.ep;
            freq->send.buffer                     = UCS_PTR_BYTE_OFFSET(fsreq->send.buffer,
                                                                        offset);
            freq->send.datatype                   = ucp_dt_make_contig(1);
            freq->send.mem_type                   = UCS_MEMORY_TYPE_HOST;
            freq->send.state.dt.dt.contig.memh[0] =
                        ucp_memh_map2uct(sreq->send.state.dt.dt.contig.memh,
                                         sreq->send.state.dt.dt.contig.md_map, md_index);
            freq->send.state.dt.dt.contig.md_map  = UCS_BIT(md_index);
            freq->send.length                     = length;
            freq->send.uct.func                   = ucp_rndv_progress_rma_put_zcopy;
            freq->send.rndv_put.sreq              = fsreq;
            freq->send.rndv_put.rkey              = fsreq->send.rndv_put.rkey;
            freq->send.rndv_put.uct_rkey          = fsreq->send.rndv_put.uct_rkey;
            freq->send.rndv_put.remote_address    = rndv_rtr_hdr->address + offset;
            freq->send.rndv_put.remote_request    = rndv_rtr_hdr->rreq_ptr;
            freq->send.lane                       = fsreq->send.lane;
            freq->send.mdesc                      = NULL;
        } else {
            /* perform get on memtype endpoint to stage data to host memory */
            mem_type_ep       = worker->mem_type_ep[sreq->send.mem_type];
            mem_type_rma_lane = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
            if (mem_type_rma_lane == UCP_NULL_LANE) {
                return UCS_ERR_UNSUPPORTED;
            }

            mdesc = ucp_worker_mpool_get(&worker->rndv_frag_mp);
            if (mdesc == NULL) {
                status = UCS_ERR_NO_MEMORY;
                goto out;
            }

            ucp_request_send_state_init(freq, ucp_dt_make_contig(1), 0);
            ucp_request_send_state_reset(freq, ucp_rndv_frag_get_completion,
                                         UCP_REQUEST_SEND_PROTO_RNDV_GET);
            md_index                              = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);
            freq->send.ep                         = mem_type_ep;
            freq->send.buffer                     = mdesc + 1;
            freq->send.datatype                   = ucp_dt_make_contig(1);
            freq->send.mem_type                   = sreq->send.mem_type;
            freq->send.state.dt.dt.contig.memh[0] = ucp_memh2uct(mdesc->memh, md_index);
            freq->send.state.dt.dt.contig.md_map  = UCS_BIT(md_index);
            freq->send.length                     = length;
            freq->send.uct.func                   = ucp_rndv_progress_rma_get_zcopy;
            freq->send.rndv_get.rkey              = NULL;
            freq->send.rndv_get.remote_address    =
                    (uint64_t)UCS_PTR_BYTE_OFFSET(fsreq->send.buffer, offset);
            freq->send.rndv_get.lanes_map         = 0;
            freq->send.rndv_get.lane_count        = 0;
            freq->send.rndv_get.rreq              = fsreq;
            freq->send.mdesc                      = mdesc;

        }

        ucp_request_send(freq, 0);
        offset += length;
    }

    return UCS_OK;
 out:
    return status;;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_atp_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *req       = (ucp_request_t*) rep_hdr->reqptr;
    ucp_request_t *rreq;
    ucp_worker_h worker;
    ucp_lane_index_t mem_type_rma_lane;
    ucp_mem_desc_t *mdesc;
    ucp_md_index_t md_index;
    ucp_ep_h mem_type_ep;
    size_t frag_size, frag_offset;

    if (req->recv.frag.rreq) {
        /* atp for fragmented rndv request */
        rreq        = req->recv.frag.rreq;
        worker      = rreq->recv.worker;
        frag_size   = req->recv.length;
        frag_offset = req->recv.frag.offset;
        ucs_assert_always(!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(rreq->recv.mem_type));

        /* perform a put zcopy on memtype endpoint to stage from
         * frag recv buffer to memtype recv buffer */
        mem_type_ep       = worker->mem_type_ep[rreq->recv.mem_type];
        mem_type_rma_lane = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
        if (mem_type_rma_lane == UCP_NULL_LANE) {
            ucs_fatal("no rma bw lane to stage from stage buffer to"
                       " memory type recv buffer");
        }
        md_index  = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);
        mdesc     = (ucp_mem_desc_t*) req->recv.buffer - 1;

        ucp_request_send_state_init(req, ucp_dt_make_contig(1), 0);
        ucp_request_send_state_reset(req, ucp_rndv_frag_recv_put_completion,
                                     UCP_REQUEST_SEND_PROTO_RNDV_PUT);
        req->send.ep                         = mem_type_ep;
        req->send.lane                       = mem_type_rma_lane;
        req->send.buffer                     = mdesc + 1;
        req->send.datatype                   = ucp_dt_make_contig(1);
        req->send.mem_type                   = rreq->recv.mem_type;
        req->send.state.dt.dt.contig.memh[0] = ucp_memh2uct(mdesc->memh, md_index);
        req->send.state.dt.dt.contig.md_map  = UCS_BIT(md_index);
        req->send.length                     = frag_size;
        req->send.uct.func                   = ucp_rndv_progress_rma_put_zcopy;
        req->send.rndv_put.sreq              = rreq;
        req->send.rndv_put.rkey              = NULL;
        req->send.rndv_put.remote_address    =
                    (uint64_t)UCS_PTR_BYTE_OFFSET(rreq->recv.buffer, frag_offset);
        req->send.mdesc                      = mdesc;

        ucp_request_send(req, 0);
    } else {
        UCS_PROFILE_REQUEST_EVENT(req, "rndv_atp_recv", 0);
        ucp_rndv_zcopy_recv_req_complete(req, UCS_OK);
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rtr_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_request_t *sreq              = (ucp_request_t*)rndv_rtr_hdr->sreq_ptr;
    ucp_ep_h ep                      = sreq->send.ep;
    ucp_ep_config_t *ep_config       = ucp_ep_config(ep);
    ucp_context_h context            = ep->worker->context;
    ucs_status_t status;
    int is_pipeline_rndv;

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

        is_pipeline_rndv = ((!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(sreq->send.mem_type) ||
                             (sreq->send.length != rndv_rtr_hdr->size)) &&
                            (context->config.ext.rndv_mode != UCP_RNDV_MODE_PUT_ZCOPY));

        sreq->send.lane = ucp_rkey_find_rma_lane(ep->worker->context, ep_config,
                                                 (is_pipeline_rndv ?
                                                  sreq->send.rndv_put.rkey->mem_type :
                                                  sreq->send.mem_type),
                                                 ep_config->tag.rndv.put_zcopy_lanes,
                                                 sreq->send.rndv_put.rkey, 0,
                                                 &sreq->send.rndv_put.uct_rkey);
        if (sreq->send.lane != UCP_NULL_LANE) {
            /*
             * Try pipeline protocol for non-host memory, if PUT_ZCOPY protocol is
             * not explicitly required. If pipeline is UNSUPPORTED, fallback to
             * PUT_ZCOPY anyway.
             */
            if (is_pipeline_rndv) {
                status = ucp_rndv_pipeline(sreq, rndv_rtr_hdr);
                if (status != UCS_ERR_UNSUPPORTED) {
                    return status;
                }
                /* If we get here, it means that RNDV pipeline protocol is
                 * unsupported and we have to use PUT_ZCOPY RNDV scheme instead */
            }

            if ((context->config.ext.rndv_mode != UCP_RNDV_MODE_GET_ZCOPY) &&
                ucp_rndv_test_zcopy_scheme_support(sreq->send.length,
                                                   ep_config->tag.rndv.min_put_zcopy,
                                                   ep_config->tag.rndv.max_put_zcopy,
                                                   ep_config->tag.rndv.put_zcopy_split)) {
                ucp_request_send_state_reset(sreq, ucp_rndv_put_completion,
                                             UCP_REQUEST_SEND_PROTO_RNDV_PUT);
                sreq->send.uct.func                = ucp_rndv_progress_rma_put_zcopy;
                sreq->send.rndv_put.remote_request = rndv_rtr_hdr->rreq_ptr;
                sreq->send.rndv_put.remote_address = rndv_rtr_hdr->address;
                sreq->send.mdesc                   = NULL;
                goto out_send;
            } else {
                ucp_rkey_destroy(sreq->send.rndv_put.rkey);
            }
        } else {
            ucp_rkey_destroy(sreq->send.rndv_put.rkey);
        }
    }

    ucp_trace_req(sreq, "using rdnv_data protocol");

    /* switch to AM */
    sreq->send.msg_proto.tag.rreq_ptr = rndv_rtr_hdr->rreq_ptr;

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        (sreq->send.length >=
         ep_config->am.mem_type_zcopy_thresh[sreq->send.mem_type]))
    {
        status = ucp_request_send_buffer_reg_lane(sreq, ucp_ep_get_am_lane(ep), 0);
        ucs_assert_always(status == UCS_OK);

        ucp_request_send_state_reset(sreq, ucp_rndv_am_zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_ZCOPY_AM);

        if ((sreq->send.length + sizeof(ucp_rndv_data_hdr_t)) <=
            ep_config->am.max_zcopy) {
            sreq->send.uct.func = ucp_rndv_progress_am_zcopy_single;
        } else {
            sreq->send.uct.func              = ucp_rndv_progress_am_zcopy_multi;
            sreq->send.msg_proto.am_bw_index = 1;
        }
    } else {
        ucp_request_send_state_reset(sreq, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        sreq->send.uct.func              = ucp_rndv_progress_am_bcopy;
        sreq->send.msg_proto.am_bw_index = 1;
    }

out_send:
    ucp_request_send(sreq, 0);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_data_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq = (ucp_request_t*) rndv_data_hdr->rreq_ptr;
    size_t recv_len;

    ucs_assert(!(rreq->flags & UCP_REQUEST_DEBUG_RNDV_FRAG));

    recv_len = length - sizeof(*rndv_data_hdr);
    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_recv", recv_len);

    (void)ucp_tag_request_process_recv_data(rreq, rndv_data_hdr + 1, recv_len,
                                            rndv_data_hdr->offset, 1, 0);
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
        ucs_assert(rndv_rts_hdr->sreq.ep_ptr != 0);
        snprintf(buffer, max, "RNDV_RTS tag %"PRIx64" ep_ptr %lx sreq 0x%lx "
                 "address 0x%"PRIx64" size %zu", rndv_rts_hdr->super.tag,
                 rndv_rts_hdr->sreq.ep_ptr, rndv_rts_hdr->sreq.reqptr,
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
              ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_ATS, ucp_rndv_ats_handler,
              ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_ATP, ucp_rndv_atp_handler,
              ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_RTR, ucp_rndv_rtr_handler,
              ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_RNDV_DATA, ucp_rndv_data_handler,
              ucp_rndv_dump, 0);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATP);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTR);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA);
