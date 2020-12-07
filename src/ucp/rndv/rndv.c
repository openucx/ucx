/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rndv.h"

/* TODO: Avoid dependency on tag (or other API) specifics, since this is common
 * basic rendezvous implementation.
 */
#include <ucp/tag/tag_rndv.h>
#include <ucp/tag/tag_match.inl>
#include <ucp/tag/offload.h>
#include <ucp/proto/proto_am.inl>
#include <ucs/datastruct/queue.h>


static UCS_F_ALWAYS_INLINE int
ucp_rndv_is_get_zcopy(ucp_request_t *req, ucp_context_h context)
{
    return ((context->config.ext.rndv_mode == UCP_RNDV_MODE_GET_ZCOPY) ||
            ((context->config.ext.rndv_mode == UCP_RNDV_MODE_AUTO) &&
             (!UCP_MEM_IS_GPU(req->send.mem_type) ||
              (req->send.length < context->config.ext.rndv_pipeline_send_thresh))));
}

static int ucp_rndv_is_recv_pipeline_needed(ucp_request_t *rndv_req,
                                            const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                                            const void *rkey_buf,
                                            ucs_memory_type_t mem_type,
                                            int is_get_zcopy_failed)
{
    const ucp_ep_config_t *ep_config = ucp_ep_config(rndv_req->send.ep);
    ucp_context_h context            = rndv_req->send.ep->worker->context;
    int found                        = 0;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    uint64_t mem_types;
    int i;

    for (i = 0;
         (i < UCP_MAX_LANES) &&
         (ep_config->key.rma_bw_lanes[i] != UCP_NULL_LANE); i++) {
        md_index = ep_config->md_index[ep_config->key.rma_bw_lanes[i]];
        if (context->tl_mds[md_index].attr.cap.access_mem_types
            & UCS_BIT(UCS_MEMORY_TYPE_HOST)) {
            found = 1;
            break;
        }
    }

    /* no host bw lanes for pipeline staging */
    if (!found) {
        return 0;
    }

    if (is_get_zcopy_failed) {
        return 1;
    }

    /* disqualify recv side pipeline if
     * a mem_type bw lane exist AND
     * lane can do RMA on remote mem_type
     */
    mem_types = UCS_BIT(mem_type);
    if (rndv_rts_hdr->address) {
        mem_types |= UCS_BIT(ucp_rkey_packed_mem_type(rkey_buf));
    }

    ucs_for_each_bit(md_index, ep_config->key.rma_bw_md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if (ucs_test_all_flags(md_attr->cap.reg_mem_types, mem_types)) {
            return 0;
        }
    }

    return 1;
}

static int ucp_rndv_is_put_pipeline_needed(uintptr_t remote_address,
                                           size_t length, size_t min_get_zcopy,
                                           size_t max_get_zcopy,
                                           int is_get_zcopy_failed)
{
    /* fallback to PUT pipeline if remote mem type is non-HOST memory OR
     * can't do GET ZCOPY */
    return ((remote_address == 0) || (max_get_zcopy == 0) ||
            (length < min_get_zcopy) || is_get_zcopy_failed);
}

size_t ucp_rndv_rts_pack(ucp_request_t *sreq, ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                         size_t rndv_rts_hdr_size, uint16_t flags)
{
    ucp_worker_h worker = sreq->send.ep->worker;
    ssize_t packed_rkey_size;
    void *rkey_buf;

    rndv_rts_hdr->sreq.ep_id  = ucp_send_request_get_ep_remote_id(sreq);
    rndv_rts_hdr->sreq.req_id = ucp_send_request_get_id(sreq);
    rndv_rts_hdr->size        = sreq->send.length;
    rndv_rts_hdr->flags       = flags;

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq, worker->context)) {
        /* pack rkey, ask target to do get_zcopy */
        rndv_rts_hdr->address = (uintptr_t)sreq->send.buffer;
        rkey_buf              = UCS_PTR_BYTE_OFFSET(rndv_rts_hdr,
                                                    rndv_rts_hdr_size);
        packed_rkey_size = ucp_rkey_pack_uct(worker->context,
                                             sreq->send.state.dt.dt.contig.md_map,
                                             sreq->send.state.dt.dt.contig.memh,
                                             sreq->send.mem_type, rkey_buf);
        if (packed_rkey_size < 0) {
            ucs_fatal("failed to pack rendezvous remote key: %s",
                      ucs_status_string((ucs_status_t)packed_rkey_size));
        }

        ucs_assert(packed_rkey_size <=
                   ucp_ep_config(sreq->send.ep)->rndv.rkey_size);
    } else {
        rndv_rts_hdr->address = 0;
        packed_rkey_size      = 0;
    }

    return rndv_rts_hdr_size + packed_rkey_size;
}

static size_t ucp_rndv_rtr_pack(void *dest, void *arg)
{
    ucp_request_t *rndv_req          = arg;
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = dest;
    ucp_request_t *rreq              = rndv_req->super_req;
    ucp_ep_h ep                      = rndv_req->send.ep;
    ssize_t packed_rkey_size;

    rndv_rtr_hdr->sreq_id = rndv_req->send.rndv_rtr.req_id;
    /* request of receiver side */
    rndv_rtr_hdr->rreq_id = ucp_worker_get_request_id(ep->worker, rreq,
                                                      ucp_ep_use_indirect_id(ep));

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        rndv_rtr_hdr->address = (uintptr_t)rreq->recv.buffer;
        rndv_rtr_hdr->size    = rndv_req->send.rndv_rtr.length;
        rndv_rtr_hdr->offset  = rndv_req->send.rndv_rtr.offset;

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
    packed_rkey_size = ucp_ep_config(rndv_req->send.ep)->rndv.rkey_size;
    status = ucp_do_am_single(self, UCP_AM_ID_RNDV_RTR, ucp_rndv_rtr_pack,
                              sizeof(ucp_rndv_rtr_hdr_t) + packed_rkey_size);
    if (status == UCS_OK) {
        /* release rndv request */
        ucp_request_put(rndv_req);
    }

    return status;
}

ucs_status_t ucp_rndv_reg_send_buffer(ucp_request_t *sreq)
{
    ucp_ep_h ep = sreq->send.ep;
    ucp_md_map_t md_map;
    ucs_status_t status;

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq, ep->worker->context)) {

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

static UCS_F_ALWAYS_INLINE size_t
ucp_rndv_adjust_zcopy_length(size_t min_zcopy, size_t max_zcopy, size_t align,
                             size_t send_length, size_t offset, size_t length)
{
    size_t result_length, tail;

    ucs_assert(length > 0);

    /* ensure that the current length is over min_zcopy */
    result_length = ucs_max(length, min_zcopy);

    /* ensure that the current length is less than max_zcopy */
    result_length = ucs_min(result_length, max_zcopy);

    /* ensure that tail (rest of message) is over min_zcopy */
    ucs_assertv(send_length >= (offset + result_length),
                "send_length=%zu, offset=%zu, length=%zu",
                send_length, offset, result_length);
    tail = send_length - (offset + result_length);
    if (ucs_unlikely((tail != 0) && (tail < min_zcopy))) {
        /* ok, tail is less zcopy minimal & could not be processed as
         * standalone operation */
        /* check if we have room to increase current part and not
         * step over max_zcopy */
        if (result_length < (max_zcopy - tail)) {
            /* if we can increase length by min_zcopy - let's do it to
             * avoid small tail (we have limitation on minimal get zcopy) */
            result_length += tail;
        } else {
            /* reduce current length by align or min_zcopy value
             * to process it on next round */
            ucs_assert(result_length > ucs_max(min_zcopy, align));
            result_length -= ucs_max(min_zcopy, align);
        }
    }

    ucs_assertv(result_length >= min_zcopy, "length=%zu, min_zcopy=%zu",
                result_length, min_zcopy);
    ucs_assertv(((send_length - (offset + result_length)) == 0) ||
                ((send_length - (offset + result_length)) >= min_zcopy),
                "send_length=%zu, offset=%zu, length=%zu, min_zcopy=%zu",
                send_length, offset, result_length, min_zcopy);

    return result_length;
}

static void ucp_rndv_complete_send(ucp_request_t *sreq, ucs_status_t status)
{
    ucp_request_send_generic_dt_finish(sreq);
    ucp_request_send_buffer_dereg(sreq);
    ucp_request_complete_send(sreq, status);
}

void ucp_rndv_req_send_ats(ucp_request_t *rndv_req, ucp_request_t *rreq,
                           ucs_ptr_map_key_t remote_req_id, ucs_status_t status)
{
    ucp_trace_req(rndv_req, "send ats remote_req_id 0x%"PRIxPTR, remote_req_id);
    UCS_PROFILE_REQUEST_EVENT(rreq, "send_ats", 0);

    rndv_req->send.lane                = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func            = ucp_proto_progress_am_single;
    rndv_req->send.proto.am_id         = UCP_AM_ID_RNDV_ATS;
    rndv_req->send.proto.status        = status;
    rndv_req->send.proto.remote_req_id = remote_req_id;
    rndv_req->send.proto.comp_cb       = ucp_request_put;

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

static void ucp_rndv_send_atp(ucp_request_t *sreq,
                              ucs_ptr_map_key_t remote_req_id)
{
    ucs_assertv(sreq->send.state.dt.offset == sreq->send.length,
                "sreq=%p offset=%zu length=%zu", sreq,
                sreq->send.state.dt.offset, sreq->send.length);

    ucp_trace_req(sreq, "send atp remote_req_id 0x%"PRIxPTR, remote_req_id);
    UCS_PROFILE_REQUEST_EVENT(sreq, "send_atp", 0);

    /* destroy rkey before it gets overridden by ATP protocol data */
    ucp_rkey_destroy(sreq->send.rndv_put.rkey);

    sreq->send.lane                = ucp_ep_get_am_lane(sreq->send.ep);
    sreq->send.uct.func            = ucp_proto_progress_am_single;
    sreq->send.proto.am_id         = UCP_AM_ID_RNDV_ATP;
    sreq->send.proto.status        = UCS_OK;
    sreq->send.proto.remote_req_id = remote_req_id;
    sreq->send.proto.comp_cb       = ucp_rndv_complete_rma_put_zcopy;

    ucp_request_send(sreq, 0);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_complete_frag_rma_put_zcopy, (fsreq),
                      ucp_request_t *fsreq)
{
    ucp_request_t *sreq = fsreq->super_req;

    sreq->send.state.dt.offset += fsreq->send.length;

    /* delete fragments send request */
    ucp_request_put(fsreq);

    /* complete send request after put completions of all fragments */
    if (sreq->send.state.dt.offset == sreq->send.length) {
        ucp_rndv_complete_rma_put_zcopy(sreq);
    }
}

static void ucp_rndv_send_frag_atp(ucp_request_t *fsreq,
                                   ucs_ptr_map_key_t req_id)
{
    ucp_trace_req(fsreq, "send frag atp remote req_id 0x%"PRIxPTR, req_id);
    UCS_PROFILE_REQUEST_EVENT(fsreq, "send_frag_atp", 0);

    /* destroy rkey before it gets overridden by ATP protocol data */
    ucp_rkey_destroy(fsreq->send.rndv_put.rkey);

    fsreq->send.lane                = ucp_ep_get_am_lane(fsreq->send.ep);
    fsreq->send.uct.func            = ucp_proto_progress_am_single;
    fsreq->send.proto.am_id         = UCP_AM_ID_RNDV_ATP;
    fsreq->send.proto.status        = UCS_OK;
    fsreq->send.proto.remote_req_id = req_id;
    fsreq->send.proto.comp_cb       = ucp_rndv_complete_frag_rma_put_zcopy;

    ucp_request_send(fsreq, 0);
}

static UCS_F_ALWAYS_INLINE void
ucp_rndv_recv_req_complete(ucp_request_t *req, ucs_status_t status)
{
    if (req->flags & UCP_REQUEST_FLAG_RECV_AM) {
        ucp_request_complete_am_recv(req, status);
    } else {
        ucs_assert(req->flags & UCP_REQUEST_FLAG_RECV_TAG);
        ucp_request_complete_tag_recv(req, status);
    }
}

static void ucp_rndv_zcopy_recv_req_complete(ucp_request_t *req,
                                             ucs_status_t status)
{
    ucp_request_recv_buffer_dereg(req);
    ucp_rndv_recv_req_complete(req, status);
}

static void ucp_rndv_complete_rma_get_zcopy(ucp_request_t *rndv_req,
                                            ucs_status_t status)
{
    ucp_request_t *rreq = rndv_req->super_req;

    ucs_assertv(rndv_req->send.state.dt.offset == rndv_req->send.length,
                "rndv_req=%p offset=%zu length=%zu", rndv_req,
                rndv_req->send.state.dt.offset, rndv_req->send.length);

    ucp_trace_req(rndv_req, "rndv_get completed with status %s",
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(rreq, "complete_rndv_get", 0);

    ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
    ucp_request_send_buffer_dereg(rndv_req);

    if (status == UCS_OK) {
        ucp_rndv_req_send_ats(rndv_req, rreq,
                              rndv_req->send.rndv_get.remote_req_id, UCS_OK);
    } else {
        /* if completing RNDV with the error, just release RNDV request */
        ucp_request_put(rndv_req);
    }

    ucs_assert(rreq->recv.state.dt.contig.md_map == 0);
    ucp_rndv_recv_req_complete(rreq, status);
}

static void ucp_rndv_recv_data_init(ucp_request_t *rreq, size_t size)
{
    rreq->status         = UCS_OK;
    rreq->recv.remaining = size;
}

static void ucp_rndv_req_send_rtr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                  ucs_ptr_map_key_t sender_req_id,
                                  size_t recv_length, size_t offset)
{
    ucp_trace_req(rndv_req, "send rtr remote sreq_id 0x%"PRIxPTR" rreq %p",
                  sender_req_id, rreq);

    rndv_req->super_req            = rreq;
    rndv_req->send.lane            = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func        = ucp_proto_progress_rndv_rtr;
    rndv_req->send.rndv_rtr.req_id = sender_req_id;
    rndv_req->send.rndv_rtr.length = recv_length;
    rndv_req->send.rndv_rtr.offset = offset;

    ucp_request_send(rndv_req, 0);
}

static ucp_lane_index_t
ucp_rndv_get_zcopy_get_lane(ucp_request_t *rndv_req, uct_rkey_t *uct_rkey)
{
    ucp_lane_index_t lane_idx;
    ucp_ep_config_t *ep_config;
    ucp_rkey_h rkey;
    uint8_t rkey_index;

    if (ucs_unlikely(!rndv_req->send.rndv_get.lanes_map_all)) {
        return UCP_NULL_LANE;
    }

    lane_idx   = ucs_ffs64_safe(rndv_req->send.lanes_map_avail);
    ucs_assert(lane_idx < UCP_MAX_LANES);
    rkey       = rndv_req->send.rndv_get.rkey;
    rkey_index = rndv_req->send.rndv_get.rkey_index[lane_idx];
    *uct_rkey  = (rkey_index != UCP_NULL_RESOURCE) ?
                 rkey->tl_rkey[rkey_index].rkey.rkey : UCT_INVALID_RKEY;
    ep_config  = ucp_ep_config(rndv_req->send.ep);
    return ep_config->rndv.get_zcopy_lanes[lane_idx];
}

static void ucp_rndv_get_zcopy_next_lane(ucp_request_t *rndv_req)
{
    rndv_req->send.lanes_map_avail    &= rndv_req->send.lanes_map_avail - 1;
    if (!rndv_req->send.lanes_map_avail) {
        rndv_req->send.lanes_map_avail = rndv_req->send.rndv_get.lanes_map_all;
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
    int pending_add_res;
    ucp_lane_index_t lane;

    /* Figure out which lane to use for get operation */
    rndv_req->send.lane = lane = ucp_rndv_get_zcopy_get_lane(rndv_req, &uct_rkey);

    if (lane == UCP_NULL_LANE) {
        /* If can't perform get_zcopy - switch to active-message.
         * NOTE: we do not register memory and do not send our keys. */
        ucp_trace_req(rndv_req, "remote memory unreachable, switch to rtr");
        ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
        ucp_rndv_recv_data_init(rndv_req->super_req,
                                rndv_req->send.length);
        /* Update statistics counters from get_zcopy to rtr */
        UCP_WORKER_STAT_RNDV(ep->worker, GET_ZCOPY, -1);
        UCP_WORKER_STAT_RNDV(ep->worker, SEND_RTR,  +1);
        ucp_rndv_req_send_rtr(rndv_req, rndv_req->super_req,
                              rndv_req->send.rndv_get.remote_req_id,
                              rndv_req->send.length, 0ul);
        return UCS_OK;
    }

    ucs_assert_always(rndv_req->send.rndv_get.lanes_count > 0);

    if (!rndv_req->send.mdesc) {
        status = ucp_send_request_add_reg_lane(rndv_req, lane);
        ucs_assert_always(status == UCS_OK);
    }

    rsc_index = ucp_ep_get_rsc_index(ep, lane);
    attrs     = ucp_worker_iface_get_attr(ep->worker, rsc_index);
    align     = attrs->cap.get.opt_zcopy_align;
    ucp_mtu   = attrs->cap.get.align_mtu;
    min_zcopy = config->rndv.min_get_zcopy;
    max_zcopy = config->rndv.max_get_zcopy;

    offset    = rndv_req->send.state.dt.offset;
    remaining = (uintptr_t)rndv_req->send.buffer % align;

    if ((offset == 0) && (remaining > 0) && (rndv_req->send.length > ucp_mtu)) {
        length = ucp_mtu - remaining;
    } else {
        chunk = ucs_align_up((size_t)(rndv_req->send.length /
                                      rndv_req->send.rndv_get.lanes_count
                                      * config->rndv.scale[lane]),
                             align);
        length = ucs_min(chunk, rndv_req->send.length - offset);
    }

    length = ucp_rndv_adjust_zcopy_length(min_zcopy, max_zcopy, align,
                                          rndv_req->send.length, offset,
                                          length);

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
                uct_completion_update_status(&rndv_req->send.state.uct_comp,
                                             status);
                rndv_req->send.state.uct_comp.func(&rndv_req->send.state.uct_comp);
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
                    pending_add_res = ucp_request_pending_add(rndv_req, 0);
                    if (!pending_add_res) {
                        /* failed to switch req to pending queue, try again */
                        continue;
                    }
                    return UCS_OK;
                }
            }
            return status;
        }
    }
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_get_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t,
                                               send.state.uct_comp);

    if (rndv_req->send.state.dt.offset == rndv_req->send.length) {
        ucp_rndv_complete_rma_get_zcopy(rndv_req, self->status);
    }
}

static void ucp_rndv_put_completion(uct_completion_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);

    if (sreq->send.state.dt.offset == sreq->send.length) {
        ucp_rndv_send_atp(sreq, sreq->send.rndv_put.rreq_remote_id);
    }
}

static void ucp_rndv_req_init_get_zcopy_lane_map(ucp_request_t *rndv_req)
{
    ucp_ep_h ep                = rndv_req->send.ep;
    ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    ucp_context_h context      = ep->worker->context;
    ucs_memory_type_t mem_type = rndv_req->send.mem_type;
    ucp_rkey_h rkey            = rndv_req->send.rndv_get.rkey;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane, lane_idx;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    ucp_md_index_t dst_md_index;
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *iface_attr;
    double max_lane_bw, lane_bw;
    int i;

    max_lane_bw = 0;
    lane_map    = 0;
    for (i = 0; i < UCP_MAX_LANES; i++) {
        lane = ep_config->rndv.get_zcopy_lanes[i];
        if (lane == UCP_NULL_LANE) {
            break; /* no more lanes */
        }

        md_index   = ep_config->md_index[lane];
        md_attr    = &context->tl_mds[md_index].attr;
        rsc_index  = ep_config->key.lanes[lane].rsc_index;
        iface_attr = ucp_worker_iface_get_attr(ep->worker, rsc_index);
        lane_bw    = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);

        if (ucs_unlikely((md_index != UCP_NULL_RESOURCE) &&
                         !(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY))) {
            /* Lane does not need rkey, can use the lane with invalid rkey  */
            if (!rkey || ((md_attr->cap.access_mem_types & UCS_BIT(mem_type)) &&
                          (mem_type == rkey->mem_type))) {
                rndv_req->send.rndv_get.rkey_index[i] = UCP_NULL_RESOURCE;
                lane_map                             |= UCS_BIT(i);
                max_lane_bw                           = ucs_max(max_lane_bw, lane_bw);
                continue;
            }
        }

        if (ucs_unlikely((md_index != UCP_NULL_RESOURCE) &&
                         (!(md_attr->cap.reg_mem_types & UCS_BIT(mem_type))))) {
            continue;
        }

        dst_md_index = ep_config->key.lanes[lane].dst_md_index;
        if (rkey && ucs_likely(rkey->md_map & UCS_BIT(dst_md_index))) {
            /* Return first matching lane */
            rndv_req->send.rndv_get.rkey_index[i] = ucs_bitmap2idx(rkey->md_map,
                                                                   dst_md_index);
            lane_map                             |= UCS_BIT(i);
            max_lane_bw                           = ucs_max(max_lane_bw, lane_bw);
        }
    }

    if (ucs_popcount(lane_map) > 1) {
        /* remove lanes if bandwidth is too less compare to best lane */
        ucs_for_each_bit(lane_idx, lane_map) {
            ucs_assert(lane_idx < UCP_MAX_LANES);
            lane       = ep_config->rndv.get_zcopy_lanes[lane_idx];
            rsc_index  = ep_config->key.lanes[lane].rsc_index;
            iface_attr = ucp_worker_iface_get_attr(ep->worker, rsc_index);
            lane_bw    = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);

            if ((lane_bw/max_lane_bw) <
                (1. / context->config.ext.multi_lane_max_ratio)) {
                lane_map                                    &= ~UCS_BIT(lane_idx);
                rndv_req->send.rndv_get.rkey_index[lane_idx] = UCP_NULL_RESOURCE;
            }
        }
    }

    rndv_req->send.lanes_map_avail          =
    rndv_req->send.rndv_get.lanes_map_all   = lane_map;
    rndv_req->send.rndv_get.lanes_count     = ucs_popcount(lane_map);
}

static ucs_status_t ucp_rndv_req_send_rma_get(ucp_request_t *rndv_req,
                                              ucp_request_t *rreq,
                                              const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                                              const void *rkey_buf)
{
    ucp_ep_h ep = rndv_req->send.ep;
    ucs_status_t status;
    uct_rkey_t uct_rkey;

    ucp_trace_req(rndv_req, "start rma_get rreq %p", rreq);

    rndv_req->super_req                    = rreq;
    rndv_req->send.uct.func                = ucp_rndv_progress_rma_get_zcopy;
    rndv_req->send.buffer                  = rreq->recv.buffer;
    rndv_req->send.mem_type                = rreq->recv.mem_type;
    rndv_req->send.datatype                = ucp_dt_make_contig(1);
    rndv_req->send.length                  = rndv_rts_hdr->size;
    rndv_req->send.rndv_get.remote_req_id  = rndv_rts_hdr->sreq.req_id;
    rndv_req->send.rndv_get.remote_address = rndv_rts_hdr->address;

    status = ucp_ep_rkey_unpack(ep, rkey_buf, &rndv_req->send.rndv_get.rkey);
    if (status != UCS_OK) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(ep), ucs_status_string(status));
    }

    ucp_request_send_state_init(rndv_req, ucp_dt_make_contig(1), 0);
    ucp_request_send_state_reset(rndv_req, ucp_rndv_get_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_GET);

    ucp_rndv_req_init_get_zcopy_lane_map(rndv_req);

    rndv_req->send.lane = ucp_rndv_get_zcopy_get_lane(rndv_req, &uct_rkey);
    if (rndv_req->send.lane == UCP_NULL_LANE) {
        return UCS_ERR_UNREACHABLE;
    }

    UCP_WORKER_STAT_RNDV(ep->worker, GET_ZCOPY, 1);
    ucp_request_send(rndv_req, 0);

    return UCS_OK;
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_recv_frag_put_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq              = ucs_container_of(self, ucp_request_t,
                                                        send.state.uct_comp);
    ucp_worker_h worker              = freq->send.ep->worker;
    ucs_ptr_map_key_t rreq_remote_id = freq->send.rndv_put.rreq_remote_id;
    int is_put_proto                 = (rreq_remote_id == UCP_REQUEST_ID_INVALID);
    ucp_request_t *req               = freq->super_req;
    ucp_request_t *rndv_req;

    ucs_trace_req("freq:%p: recv_frag_put done. rreq:%p ", freq, req);

    /* release memory descriptor */
    ucs_mpool_put_inline((void *)freq->send.mdesc);

    /* rndv_req is NULL in case of put protocol */
    if (!is_put_proto) {
        rndv_req = ucp_worker_get_request_by_id(worker, rreq_remote_id);
        /* pipeline recv get protocol */
        rndv_req->send.state.dt.offset += freq->send.length;

        /* send ATS for fragment get rndv completion */
        if (rndv_req->send.length == rndv_req->send.state.dt.offset) {
            ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
            ucp_rndv_req_send_ats(rndv_req, req,
                                  rndv_req->send.rndv_get.remote_req_id,
                                  UCS_OK);
        }
    }

    ucs_assertv(req->recv.remaining >= freq->send.length,
                "req->recv.remaining %zu, freq->send.length %zu",
                req->recv.remaining, freq->send.length);
    req->recv.remaining -= freq->send.length;
    if (req->recv.remaining == 0) {
        ucp_request_complete_tag_recv(req, UCS_OK);
        if (!is_put_proto) {
            ucp_worker_del_request_id(worker, rreq_remote_id);
        }
    }

    ucp_request_put(freq);
}

static UCS_F_ALWAYS_INLINE void
ucp_rndv_init_mem_type_frag_req(ucp_worker_h worker, ucp_request_t *freq, int rndv_op,
                                uct_completion_callback_t comp_cb, ucp_mem_desc_t *mdesc,
                                ucs_memory_type_t mem_type, size_t length,
                                uct_pending_callback_t uct_func)
{
    ucp_ep_h mem_type_ep;
    ucp_md_index_t md_index;
    ucp_lane_index_t mem_type_rma_lane;

    ucp_request_send_state_init(freq, ucp_dt_make_contig(1), 0);
    ucp_request_send_state_reset(freq, comp_cb, rndv_op);

    freq->send.buffer   = mdesc + 1;
    freq->send.length   = length;
    freq->send.datatype = ucp_dt_make_contig(1);
    freq->send.mem_type = mem_type;
    freq->send.mdesc    = mdesc;
    freq->send.uct.func = uct_func;

    if (mem_type != UCS_MEMORY_TYPE_HOST) {
        mem_type_ep       = worker->mem_type_ep[mem_type];
        mem_type_rma_lane = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
        md_index          = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);
        ucs_assert(mem_type_rma_lane != UCP_NULL_LANE);

        freq->send.lane                       = mem_type_rma_lane;
        freq->send.ep                         = mem_type_ep;
        freq->send.state.dt.dt.contig.memh[0] = ucp_memh2uct(mdesc->memh, md_index);
        freq->send.state.dt.dt.contig.md_map  = UCS_BIT(md_index);
    }
}

static void
ucp_rndv_recv_frag_put_mem_type(ucp_request_t *rreq, ucp_request_t *rndv_req,
                                ucp_request_t *freq, ucp_mem_desc_t *mdesc,
                                size_t length, size_t offset)
{

    ucs_assert_always(!UCP_MEM_IS_HOST(rreq->recv.mem_type));

    /* PUT on memtype endpoint to stage from
     * frag recv buffer to memtype recv buffer
     */

    ucp_rndv_init_mem_type_frag_req(rreq->recv.worker, freq, UCP_REQUEST_SEND_PROTO_RNDV_PUT,
                                    ucp_rndv_recv_frag_put_completion, mdesc, rreq->recv.mem_type,
                                    length, ucp_rndv_progress_rma_put_zcopy);

    freq->super_req                    = rreq;
    freq->send.rndv_put.rkey           = NULL;
    freq->send.rndv_put.remote_address = (uintptr_t)rreq->recv.buffer + offset;
    if (rndv_req == NULL) {
        freq->send.rndv_put.rreq_remote_id = UCP_REQUEST_ID_INVALID;
    } else {
        freq->send.rndv_put.rreq_remote_id =
            ucp_worker_get_request_id(rreq->recv.worker, rndv_req,
                                      ucp_ep_use_indirect_id(freq->send.ep));
    }

    ucp_request_send(freq, 0);
}

static void
ucp_rndv_send_frag_update_get_rkey(ucp_worker_h worker, ucp_request_t *freq,
                                   ucp_mem_desc_t *mdesc,
                                   ucs_memory_type_t mem_type)
{
    ucp_rkey_h *rkey_p  = &freq->send.rndv_get.rkey;
    uint8_t *rkey_index = freq->send.rndv_get.rkey_index;
    void *rkey_buffer;
    size_t rkey_size;
    ucs_status_t status;
    ucp_ep_h mem_type_ep;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    ucp_lane_index_t mem_type_rma_lane;

    mem_type_ep       = worker->mem_type_ep[mem_type];
    mem_type_rma_lane = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
    ucs_assert(mem_type_rma_lane != UCP_NULL_LANE);

    md_index = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);
    md_attr  = &mem_type_ep->worker->context->tl_mds[md_index].attr;

    if (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY)) {
        return;
    }

    status = ucp_rkey_pack(mem_type_ep->worker->context, mdesc->memh,
                           &rkey_buffer, &rkey_size);
    ucs_assert_always(status == UCS_OK);

    status = ucp_ep_rkey_unpack(mem_type_ep, rkey_buffer, rkey_p);
    ucs_assert_always(status == UCS_OK);
    ucp_rkey_buffer_release(rkey_buffer);

    memset(rkey_index, 0, UCP_MAX_LANES * sizeof(uint8_t));
}

static void
ucp_rndv_send_frag_get_mem_type(ucp_request_t *sreq, ucs_ptr_map_key_t rreq_id,
                                size_t length, uint64_t remote_address,
                                ucs_memory_type_t remote_mem_type, ucp_rkey_h rkey,
                                uint8_t *rkey_index, ucp_lane_map_t lanes_map,
                                int update_get_rkey,
                                uct_completion_callback_t comp_cb)
{
    ucp_worker_h worker = sreq->send.ep->worker;
    ucp_request_t *freq;
    ucp_mem_desc_t *mdesc;
    ucp_lane_index_t i;

    /* GET fragment to stage buffer */

    freq = ucp_request_get(worker);
    if (ucs_unlikely(freq == NULL)) {
        ucs_fatal("failed to allocate fragment receive request");
    }

    mdesc = ucp_worker_mpool_get(&worker->rndv_frag_mp);
    if (ucs_unlikely(mdesc == NULL)) {
        ucs_fatal("failed to allocate fragment memory desc");
    }

    freq->send.ep = sreq->send.ep;

    ucp_rndv_init_mem_type_frag_req(worker, freq, UCP_REQUEST_SEND_PROTO_RNDV_GET,
                                    comp_cb, mdesc, remote_mem_type, length,
                                    ucp_rndv_progress_rma_get_zcopy);

    freq->super_req                     = sreq;
    freq->send.lanes_map_avail          =
    freq->send.rndv_get.lanes_map_all   = lanes_map;    
    freq->send.rndv_get.lanes_count     = ucs_popcount(lanes_map);
    freq->send.rndv_get.rkey            = rkey;
    freq->send.rndv_get.remote_address  = remote_address;
    freq->send.rndv_get.remote_req_id   = rreq_id;

    for (i = 0; i < UCP_MAX_LANES; i++) {
        freq->send.rndv_get.rkey_index[i] = rkey_index ? rkey_index[i]
                                                       : UCP_NULL_RESOURCE;
    }

    if (update_get_rkey) {
        ucp_rndv_send_frag_update_get_rkey(worker, freq, mdesc, remote_mem_type);
    }

    freq->status = UCS_INPROGRESS;
    ucp_request_send(freq, 0);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_recv_frag_get_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq     = ucs_container_of(self, ucp_request_t,
                                               send.state.uct_comp);
    ucp_request_t *rndv_req = freq->super_req;
    ucp_request_t *rreq     = rndv_req->super_req;

    ucs_trace_req("freq:%p: recv_frag_get done. rreq:%p length:%ld"
                  " offset:%"PRIu64,
                  freq, rndv_req, freq->send.length,
                  freq->send.rndv_get.remote_address - rndv_req->send.rndv_get.remote_address);

    /* fragment GET completed from remote to staging buffer, issue PUT from
     * staging buffer to recv buffer */
    ucp_rndv_recv_frag_put_mem_type(rreq, rndv_req, freq,
                                    (ucp_mem_desc_t *)freq->send.buffer -1,
                                    freq->send.length, (freq->send.rndv_get.remote_address -
                                    rndv_req->send.rndv_get.remote_address));
}

static ucs_status_t
ucp_rndv_recv_start_get_pipeline(ucp_worker_h worker, ucp_request_t *rndv_req,
                                 ucp_request_t *rreq,
                                 ucs_ptr_map_key_t remote_req_id,
                                 const void *rkey_buffer,
                                 uint64_t remote_address, size_t size,
                                 size_t base_offset)
{
    ucp_ep_h ep             = rndv_req->send.ep;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    ucp_context_h context   = worker->context;
    ucs_status_t status;
    size_t max_frag_size, offset, length;
    size_t min_zcopy, max_zcopy;

    min_zcopy                              = config->rndv.min_get_zcopy;
    max_zcopy                              = config->rndv.max_get_zcopy;
    max_frag_size                          = ucs_min(context->config.ext.rndv_frag_size,
                                                     max_zcopy);
    rndv_req->super_req                    = rreq;
    rndv_req->send.rndv_get.remote_req_id  = remote_req_id;
    rndv_req->send.rndv_get.remote_address = remote_address - base_offset;
    rndv_req->send.length                  = size;
    rndv_req->send.state.dt.offset         = 0;
    rndv_req->send.mem_type                = rreq->recv.mem_type;

    /* Protocol:
     * Step 1: GET remote fragment into HOST fragment buffer
     * Step 2: PUT from fragment buffer to MEM TYPE destination
     * Step 3: Send ATS for RNDV request
     */

    status = ucp_ep_rkey_unpack(rndv_req->send.ep, rkey_buffer,
                                &rndv_req->send.rndv_get.rkey);
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(rndv_req->send.ep), ucs_status_string(status));
    }

    ucp_rndv_req_init_get_zcopy_lane_map(rndv_req);

    offset = 0;
    while (offset != size) {
        length = ucp_rndv_adjust_zcopy_length(min_zcopy, max_frag_size, 0,
                                              size, offset, size - offset);

        /* GET remote fragment into HOST fragment buffer */
        ucp_rndv_send_frag_get_mem_type(rndv_req, remote_req_id, length,
                                        remote_address + offset, UCS_MEMORY_TYPE_HOST,
                                        rndv_req->send.rndv_get.rkey,
                                        rndv_req->send.rndv_get.rkey_index,
                                        rndv_req->send.rndv_get.lanes_map_all, 0,
                                        ucp_rndv_recv_frag_get_completion);

        offset += length;
    }

    return UCS_OK;
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
        freq->recv.frag.offset            = offset;
        freq->super_req                   = rreq;
        freq->flags                      |= UCP_REQUEST_FLAG_RNDV_FRAG;

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

        ucp_rndv_req_send_rtr(frndv_req, freq, rndv_rts_hdr->sreq.req_id,
                              freq->recv.length, offset);
        offset += frag_size;
    }

    /* release original rndv reply request */
    ucp_request_put(rndv_req);
}

static UCS_F_ALWAYS_INLINE int
ucp_rndv_is_rkey_ptr(const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                     const void *rkey_buffer, ucp_ep_h ep,
                     ucs_memory_type_t recv_mem_type, ucp_rndv_mode_t rndv_mode)
{
    const ucp_ep_config_t *ep_config = ucp_ep_config(ep);

    return /* must have remote address */
           (rndv_rts_hdr->address != 0) &&
           /* remote key must be on a memory domain for which we support rkey_ptr */
           (ucp_rkey_packed_md_map(rkey_buffer) &
            ep_config->rndv.rkey_ptr_dst_mds) &&
           /* rendezvous mode must not be forced to put/get */
           (rndv_mode == UCP_RNDV_MODE_AUTO) &&
           /* need local memory access for data unpack */
           UCP_MEM_IS_ACCESSIBLE_FROM_CPU(recv_mem_type);
}

static unsigned ucp_rndv_progress_rkey_ptr(void *arg)
{
    ucp_worker_h worker     = (ucp_worker_h)arg;
    ucp_request_t *rndv_req = ucs_queue_head_elem_non_empty(&worker->rkey_ptr_reqs,
                                                            ucp_request_t,
                                                            send.rkey_ptr.queue_elem);
    ucp_request_t *rreq     = rndv_req->super_req;
    size_t seg_size         = ucs_min(worker->context->config.ext.rkey_ptr_seg_size,
                                      rndv_req->send.length - rreq->recv.state.offset);
    ucs_status_t status;
    size_t offset, new_offset;
    int last;

    offset     = rreq->recv.state.offset;
    new_offset = offset + seg_size;
    last       = new_offset == rndv_req->send.length;
    status     = ucp_request_recv_data_unpack(rreq,
                                              UCS_PTR_BYTE_OFFSET(rndv_req->send.buffer,
                                                                  offset),
                                              seg_size, offset, last);
    if (ucs_unlikely(status != UCS_OK) || last) {
        ucs_queue_pull_non_empty(&worker->rkey_ptr_reqs);
        ucp_rndv_recv_req_complete(rreq, status);
        ucp_rkey_destroy(rndv_req->send.rkey_ptr.rkey);
        ucp_rndv_req_send_ats(rndv_req, rreq,
                              rndv_req->send.rkey_ptr.req_id, status);
        if (ucs_queue_is_empty(&worker->rkey_ptr_reqs)) {
            uct_worker_progress_unregister_safe(worker->uct,
                                                &worker->rkey_ptr_cb_id);
        }
    } else {
        rreq->recv.state.offset = new_offset;
    }

    return 1;
}

static void ucp_rndv_do_rkey_ptr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                 const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                                 const void *rkey_buf)
{
    ucp_ep_h ep                      = rndv_req->send.ep;
    const ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    ucp_worker_h worker              = rreq->recv.worker;
    ucp_md_index_t dst_md_index      = 0;
    ucp_lane_index_t i, lane;
    ucs_status_t status;
    unsigned rkey_index;
    void *local_ptr;
    ucp_rkey_h rkey;

    ucp_trace_req(rndv_req, "start rkey_ptr rndv rreq %p", rreq);

    status = ucp_ep_rkey_unpack(ep, rkey_buf, &rkey);
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
        ucs_fatal("failed to find a lane to access remote memory domains "
                  "0x%"PRIx64, rkey->md_map);
    }

    rkey_index = ucs_bitmap2idx(rkey->md_map, dst_md_index);
    status     = uct_rkey_ptr(rkey->tl_rkey[rkey_index].cmpt,
                              &rkey->tl_rkey[rkey_index].rkey,
                              rndv_rts_hdr->address, &local_ptr);
    if (status != UCS_OK) {
        ucp_request_complete_tag_recv(rreq, status);
        ucp_rkey_destroy(rkey);
        ucp_rndv_req_send_ats(rndv_req, rreq, rndv_rts_hdr->sreq.req_id, status);
        return;
    }

    rreq->recv.state.offset = 0;

    ucp_trace_req(rndv_req, "obtained a local pointer to remote buffer: %p",
                  local_ptr);
    rndv_req->super_req            = rreq;
    rndv_req->send.buffer          = local_ptr;
    rndv_req->send.length          = rndv_rts_hdr->size;
    rndv_req->send.rkey_ptr.rkey   = rkey;
    rndv_req->send.rkey_ptr.req_id = rndv_rts_hdr->sreq.req_id;

    UCP_WORKER_STAT_RNDV(ep->worker, RKEY_PTR, 1);

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

UCS_PROFILE_FUNC_VOID(ucp_rndv_receive, (worker, rreq, rndv_rts_hdr, rkey_buf),
                      ucp_worker_h worker, ucp_request_t *rreq,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                      const void *rkey_buf)
{
    ucp_rndv_mode_t rndv_mode;
    ucp_request_t *rndv_req;
    ucp_ep_h ep;
    ucp_ep_config_t *ep_config;
    ucs_status_t status;
    int is_get_zcopy_failed;

    UCS_ASYNC_BLOCK(&worker->async);

    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_receive", 0);

    /* the internal send request allocated on receiver side (to perform a "get"
     * operation, send "ATS" and "RTR") */
    rndv_req = ucp_request_get(worker);
    if (rndv_req == NULL) {
        ucs_error("failed to allocate rendezvous reply");
        goto out;
    }

    rndv_req->send.ep           = ucp_worker_get_ep_by_id(worker,
                                                    rndv_rts_hdr->sreq.ep_id);
    rndv_req->flags             = 0;
    rndv_req->send.mdesc        = NULL;
    rndv_req->send.pending_lane = UCP_NULL_LANE;
    is_get_zcopy_failed         = 0;

    ucp_trace_req(rreq,
                  "rndv matched remote {address 0x%"PRIx64" size %zu sreq_id "
                  "0x%"PRIx64"} rndv_sreq %p", rndv_rts_hdr->address,
                  rndv_rts_hdr->size, rndv_rts_hdr->sreq.req_id, rndv_req);

    if (ucs_unlikely(rreq->recv.length < rndv_rts_hdr->size)) {
        ucp_trace_req(rndv_req,
                      "rndv truncated remote size %zu local size %zu rreq %p",
                      rndv_rts_hdr->size, rreq->recv.length, rreq);
        ucp_rndv_req_send_ats(rndv_req, rreq, rndv_rts_hdr->sreq.req_id, UCS_OK);
        ucp_request_recv_generic_dt_finish(rreq);
        ucp_rndv_zcopy_recv_req_complete(rreq, UCS_ERR_MESSAGE_TRUNCATED);
        goto out;
    }

    /* if the receive side is not connected yet then the RTS was received on a stub ep */
    ep        = rndv_req->send.ep;
    ep_config = ucp_ep_config(ep);
    rndv_mode = worker->context->config.ext.rndv_mode;

    if (ucp_rndv_is_rkey_ptr(rndv_rts_hdr, rkey_buf, ep, rreq->recv.mem_type,
                             rndv_mode)) {
        ucp_rndv_do_rkey_ptr(rndv_req, rreq, rndv_rts_hdr, rkey_buf);
        goto out;
    }

    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        if ((rndv_rts_hdr->address != 0) &&
            ucp_rndv_test_zcopy_scheme_support(rndv_rts_hdr->size,
                                               ep_config->rndv.min_get_zcopy,
                                               ep_config->rndv.max_get_zcopy,
                                               ep_config->rndv.get_zcopy_split)) {
            /* try to fetch the data with a get_zcopy operation */
            status = ucp_rndv_req_send_rma_get(rndv_req, rreq, rndv_rts_hdr,
                                               rkey_buf);
            if (status == UCS_OK) {
                goto out;
            }

            /* fallback to non get zcopy protocol */
            ucp_rkey_destroy(rndv_req->send.rndv_get.rkey);
            is_get_zcopy_failed = 1;
        }

        if (rndv_mode == UCP_RNDV_MODE_AUTO) {
            /* check if we need pipelined memtype staging */
            if (UCP_MEM_IS_GPU(rreq->recv.mem_type) &&
                ucp_rndv_is_recv_pipeline_needed(rndv_req, rndv_rts_hdr,
                                                 rkey_buf, rreq->recv.mem_type,
                                                 is_get_zcopy_failed)) {
                ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
                if (ucp_rndv_is_put_pipeline_needed(rndv_rts_hdr->address,
                                                    rndv_rts_hdr->size,
                                                    ep_config->rndv.min_get_zcopy,
                                                    ep_config->rndv.max_get_zcopy,
                                                    is_get_zcopy_failed)) {
                    /* send FRAG RTR for sender to PUT the fragment. */
                    ucp_rndv_send_frag_rtr(worker, rndv_req, rreq, rndv_rts_hdr);
                } else {
                    /* sender address is present. do GET pipeline */
                    ucp_rndv_recv_start_get_pipeline(worker, rndv_req, rreq,
                                                     rndv_rts_hdr->sreq.req_id,
                                                     rkey_buf,
                                                     rndv_rts_hdr->address,
                                                     rndv_rts_hdr->size, 0);
                }
                goto out;
            }
        }

        if (!is_get_zcopy_failed) {
            /* put protocol is allowed - register receive buffer memory for rma */
            ucs_assert(rndv_rts_hdr->size <= rreq->recv.length);
            ucp_request_recv_buffer_reg(rreq, ep_config->key.rma_bw_md_map,
                                        rndv_rts_hdr->size);
        }
    }

    /* The sender didn't specify its address in the RTS, or the rndv mode was
     * configured to PUT, or GET rndv mode is unsupported - send an RTR and
     * the sender will send the data with active message or put_zcopy. */
    ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
    UCP_WORKER_STAT_RNDV(ep->worker, SEND_RTR, 1);
    ucp_rndv_req_send_rtr(rndv_req, rreq, rndv_rts_hdr->sreq.req_id,
                          rndv_rts_hdr->size, 0ul);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rts_handler,
                 (arg, data, length, tl_flags),
                 void *arg, void *data, size_t length, unsigned tl_flags)
{
    ucp_worker_h worker         = arg;
    ucp_rndv_rts_hdr_t *rts_hdr = data;

    if (rts_hdr->flags & UCP_RNDV_RTS_FLAG_TAG) {
        return ucp_tag_rndv_process_rts(worker, rts_hdr, length, tl_flags);
    } else {
        ucs_assert(rts_hdr->flags & UCP_RNDV_RTS_FLAG_AM);
        return ucp_am_rndv_process_rts(arg, data, length, tl_flags);
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_ats_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker      = arg;
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *sreq      = ucp_worker_extract_request_by_id(worker,
                                                                rep_hdr->req_id);

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

    offset       = sreq->send.state.dt.offset;
    hdr->rreq_id = sreq->send.msg_proto.rreq_id;
    hdr->offset  = offset;
    length       = ucs_min(sreq->send.length - offset,
                           ucp_ep_get_max_bcopy(sreq->send.ep, sreq->send.lane) - sizeof(*hdr));

    return sizeof(*hdr) + ucp_dt_pack(sreq->send.ep->worker, sreq->send.datatype,
                                      sreq->send.mem_type, hdr + 1, sreq->send.buffer,
                                      &sreq->send.state.dt, length);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_am_bcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep        = sreq->send.ep;
    ucs_status_t status;

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
                         ucp_ep_config(ep)->rndv.max_put_zcopy);
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
            uct_completion_update_status(&sreq->send.state.uct_comp, status);
            sreq->send.state.uct_comp.func(&sreq->send.state.uct_comp);
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

static void ucp_rndv_am_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    ucs_status_t status = self->status;

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

    hdr.rreq_id = sreq->send.msg_proto.rreq_id;
    hdr.offset  = 0;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_RNDV_DATA, &hdr, sizeof(hdr),
                                  NULL, 0ul,
                                  ucp_rndv_am_zcopy_send_req_complete);
}

static ucs_status_t ucp_rndv_progress_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rndv_data_hdr_t hdr;

    hdr.rreq_id = sreq->send.msg_proto.rreq_id;
    hdr.offset  = sreq->send.state.dt.offset;
    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_RNDV_DATA,
                                 UCP_AM_ID_RNDV_DATA,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 NULL, 0ul,
                                 ucp_rndv_am_zcopy_send_req_complete, 1);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_send_frag_put_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    ucp_request_t *req  = freq->super_req;

    /* release memory descriptor */
    if (freq->send.mdesc) {
        ucs_mpool_put_inline((void *)freq->send.mdesc);
    }

    req->send.state.dt.offset += freq->send.length;
    ucs_assert(req->send.state.dt.offset <= req->send.length);

    /* send ATP for last fragment of the rndv request */
    if (req->send.length == req->send.state.dt.offset) {
        ucp_rndv_send_frag_atp(req, req->send.rndv_put.rreq_remote_id);
    }

    ucp_request_put(freq);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_put_pipeline_frag_get_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq  = ucs_container_of(self, ucp_request_t,
                                            send.state.uct_comp);
    ucp_request_t *fsreq = freq->super_req;

    /* get rkey can be NULL if memtype ep doesn't need RKEY */
    if (freq->send.rndv_get.rkey != NULL) {
        ucp_rkey_destroy(freq->send.rndv_get.rkey);
    }

    /* get completed on memtype endpoint to stage on host. send put request to receiver*/
    ucp_request_send_state_reset(freq, ucp_rndv_send_frag_put_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    freq->super_req                      = fsreq;
    freq->send.rndv_put.remote_address   = fsreq->send.rndv_put.remote_address +
        (freq->send.rndv_get.remote_address - (uint64_t)fsreq->send.buffer);
    freq->send.ep                        = fsreq->send.ep;
    freq->send.uct.func                  = ucp_rndv_progress_rma_put_zcopy;
    freq->send.rndv_put.rkey             = fsreq->send.rndv_put.rkey;
    freq->send.rndv_put.uct_rkey         = fsreq->send.rndv_put.uct_rkey;
    freq->send.lane                      = fsreq->send.lane;
    freq->send.state.dt.dt.contig.md_map = 0;

    ucp_request_send(freq, 0);
}

static ucs_status_t ucp_rndv_send_start_put_pipeline(ucp_request_t *sreq,
                                                     ucp_rndv_rtr_hdr_t *rndv_rtr_hdr)
{
    ucp_ep_h ep             = sreq->send.ep;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    ucp_worker_h worker     = sreq->send.ep->worker;
    ucp_context_h context   = worker->context;
    const uct_md_attr_t *md_attr;
    ucp_request_t *freq;
    ucp_request_t *fsreq;
    ucp_md_index_t md_index;
    size_t max_frag_size, rndv_size, length;
    size_t offset, rndv_base_offset;
    size_t min_zcopy, max_zcopy;

    ucp_trace_req(sreq, "using put rndv pipeline protocol");

    /* Protocol:
     * Step 1: GET fragment from send buffer to HOST fragment buffer
     * Step 2: PUT from fragment HOST buffer to remote HOST fragment buffer
     * Step 3: send ATP for each fragment request
     */

    /* check if lane supports host memory, to stage sends through host memory */
    md_attr = ucp_ep_md_attr(sreq->send.ep, sreq->send.lane);
    if (!(md_attr->cap.reg_mem_types & UCS_BIT(UCS_MEMORY_TYPE_HOST))) {
        return UCS_ERR_UNSUPPORTED;
    }

    min_zcopy        = config->rndv.min_put_zcopy;
    max_zcopy        = config->rndv.max_put_zcopy;
    rndv_size        = ucs_min(rndv_rtr_hdr->size, sreq->send.length);
    max_frag_size    = ucs_min(context->config.ext.rndv_frag_size, max_zcopy);
    rndv_base_offset = rndv_rtr_hdr->offset;

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
    fsreq->super_req                    = sreq;
    fsreq->send.buffer                  = UCS_PTR_BYTE_OFFSET(sreq->send.buffer,
                                                              rndv_base_offset);
    fsreq->send.length                  = rndv_size;
    fsreq->send.mem_type                = sreq->send.mem_type;
    fsreq->send.ep                      = sreq->send.ep;
    fsreq->send.lane                    = sreq->send.lane;
    fsreq->send.rndv_put.rkey           = sreq->send.rndv_put.rkey;
    fsreq->send.rndv_put.uct_rkey       = sreq->send.rndv_put.uct_rkey;
    fsreq->send.rndv_put.rreq_remote_id = rndv_rtr_hdr->rreq_id;
    fsreq->send.rndv_put.remote_address = rndv_rtr_hdr->address;
    fsreq->send.state.dt.offset         = 0;

    offset = 0;
    while (offset != rndv_size) {
        length = ucp_rndv_adjust_zcopy_length(min_zcopy, max_frag_size, 0,
                                              rndv_size, offset, rndv_size - offset);

        if (UCP_MEM_IS_HOST(sreq->send.mem_type)) {
            /* sbuf is in host, directly do put */
            freq = ucp_request_get(worker);
            if (ucs_unlikely(freq == NULL)) {
                ucs_error("failed to allocate fragment receive request");
                return UCS_ERR_NO_MEMORY;
            }

            ucp_request_send_state_reset(freq, ucp_rndv_send_frag_put_completion,
                                         UCP_REQUEST_SEND_PROTO_RNDV_PUT);
            md_index                              = ucp_ep_md_index(sreq->send.ep,
                                                                    sreq->send.lane);
            freq->super_req                       = fsreq;
            freq->send.ep                         = fsreq->send.ep;
            freq->send.buffer                     = UCS_PTR_BYTE_OFFSET(fsreq->send.buffer,
                                                                        offset);
            freq->send.datatype                   = ucp_dt_make_contig(1);
            freq->send.mem_type                   = UCS_MEMORY_TYPE_HOST;
            freq->send.state.dt.dt.contig.memh[0] =
                        ucp_memh_map2uct(sreq->send.state.dt.dt.contig.memh,
                                         sreq->send.state.dt.dt.contig.md_map, md_index);
            freq->send.state.dt.dt.contig.md_map = UCS_BIT(md_index);
            freq->send.length                    = length;
            freq->send.uct.func                  = ucp_rndv_progress_rma_put_zcopy;
            freq->send.rndv_put.rkey             = fsreq->send.rndv_put.rkey;
            freq->send.rndv_put.uct_rkey         = fsreq->send.rndv_put.uct_rkey;
            freq->send.rndv_put.remote_address   = rndv_rtr_hdr->address + offset;
            freq->send.rndv_put.rreq_remote_id   = rndv_rtr_hdr->rreq_id;
            freq->send.lane                      = fsreq->send.lane;
            freq->send.mdesc                     = NULL;

            ucp_request_send(freq, 0);
        } else {
            ucp_rndv_send_frag_get_mem_type(fsreq, 0, length,
                                            (uint64_t)UCS_PTR_BYTE_OFFSET(fsreq->send.buffer, offset),
                                            fsreq->send.mem_type, NULL, NULL, UCS_BIT(0), 1,
                                            ucp_rndv_put_pipeline_frag_get_completion);
        }

        offset += length;
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_atp_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *req       = ucp_worker_get_request_by_id(arg,
                                                            rep_hdr->req_id);

    if (req->flags & UCP_REQUEST_FLAG_RNDV_FRAG) {
        /* received ATP for frag RTR request */
        ucs_assert(req->super_req != NULL);
        UCS_PROFILE_REQUEST_EVENT(req, "rndv_frag_atp_recv", 0);
        ucp_rndv_recv_frag_put_mem_type(req->super_req, NULL, req,
                                        ((ucp_mem_desc_t*) req->recv.buffer - 1),
                                        req->recv.length, req->recv.frag.offset);
    } else {
        UCS_PROFILE_REQUEST_EVENT(req, "rndv_atp_recv", 0);
        ucp_worker_del_request_id(arg, rep_hdr->req_id);
        ucp_rndv_zcopy_recv_req_complete(req, UCS_OK);
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rtr_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_request_t *sreq              = ucp_worker_get_request_by_id(arg,
                                                                    rndv_rtr_hdr->sreq_id);
    ucp_ep_h ep                      = sreq->send.ep;
    ucp_ep_config_t *ep_config       = ucp_ep_config(ep);
    ucp_context_h context            = ep->worker->context;
    ucs_status_t status;
    int is_pipeline_rndv;

    ucp_trace_req(sreq, "received rtr address 0x%"PRIx64" remote rreq_id"
                  "0x%"PRIx64, rndv_rtr_hdr->address, rndv_rtr_hdr->rreq_id);
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

        is_pipeline_rndv = ((!UCP_MEM_IS_HOST(sreq->send.mem_type) ||
                             (sreq->send.length != rndv_rtr_hdr->size)) &&
                            (context->config.ext.rndv_mode != UCP_RNDV_MODE_PUT_ZCOPY));

        sreq->send.lane = ucp_rkey_find_rma_lane(ep->worker->context, ep_config,
                                                 (is_pipeline_rndv ?
                                                  sreq->send.rndv_put.rkey->mem_type :
                                                  sreq->send.mem_type),
                                                 ep_config->rndv.put_zcopy_lanes,
                                                 sreq->send.rndv_put.rkey, 0,
                                                 &sreq->send.rndv_put.uct_rkey);
        if (sreq->send.lane != UCP_NULL_LANE) {
            /*
             * Try pipeline protocol for non-host memory, if PUT_ZCOPY protocol is
             * not explicitly required. If pipeline is UNSUPPORTED, fallback to
             * PUT_ZCOPY anyway.
             */
            if (is_pipeline_rndv) {
                status = ucp_rndv_send_start_put_pipeline(sreq, rndv_rtr_hdr);
                if (status != UCS_ERR_UNSUPPORTED) {
                    return status;
                }
                /* If we get here, it means that RNDV pipeline protocol is
                 * unsupported and we have to use PUT_ZCOPY RNDV scheme instead */
            }

            if ((context->config.ext.rndv_mode != UCP_RNDV_MODE_GET_ZCOPY) &&
                ucp_rndv_test_zcopy_scheme_support(sreq->send.length,
                                                   ep_config->rndv.min_put_zcopy,
                                                   ep_config->rndv.max_put_zcopy,
                                                   ep_config->rndv.put_zcopy_split)) {
                ucp_request_send_state_reset(sreq, ucp_rndv_put_completion,
                                             UCP_REQUEST_SEND_PROTO_RNDV_PUT);
                sreq->send.uct.func                = ucp_rndv_progress_rma_put_zcopy;
                sreq->send.rndv_put.rreq_remote_id = rndv_rtr_hdr->rreq_id;
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
    sreq->send.msg_proto.rreq_id = rndv_rtr_hdr->rreq_id;

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
            sreq->send.uct.func    = ucp_rndv_progress_am_zcopy_single;
        } else {
            sreq->send.uct.func    = ucp_rndv_progress_am_zcopy_multi;
            sreq->send.am_bw_index = 1;
        }
    } else {
        ucp_request_send_state_reset(sreq, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        sreq->send.uct.func        = ucp_rndv_progress_am_bcopy;
        sreq->send.am_bw_index     = 1;
    }

out_send:
    ucp_request_send(sreq, 0);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_data_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker                = arg;
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq;
    ucs_status_t status;
    size_t recv_len;

    rreq = ucp_worker_get_request_by_id(worker, rndv_data_hdr->rreq_id);
    ucs_assert(!(rreq->flags & UCP_REQUEST_FLAG_RNDV_FRAG) &&
               (rreq->flags & (UCP_REQUEST_FLAG_RECV_AM |
                               UCP_REQUEST_FLAG_RECV_TAG)));

    recv_len = length - sizeof(*rndv_data_hdr);
    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_recv", recv_len);

    status = ucp_request_process_recv_data(rreq, rndv_data_hdr + 1, recv_len,
                                           rndv_data_hdr->offset, 1,
                                           rreq->flags & UCP_REQUEST_FLAG_RECV_AM);
    if (status != UCS_INPROGRESS) {
        ucp_worker_del_request_id(worker, rndv_data_hdr->rreq_id);
    }
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
    const ucp_rndv_data_hdr_t *rndv_data   = data;
    const ucp_reply_hdr_t *rep_hdr         = data;
    UCS_STRING_BUFFER_ONSTACK(rts_info, 64);
    ucp_tag_rndv_rts_hdr_t *tag_rts;
    ucp_am_rndv_rts_hdr_t *am_rts;
    void *rkey_buf;

    switch (id) {
    case UCP_AM_ID_RNDV_RTS:
        ucs_assert(rndv_rts_hdr->sreq.ep_id != UCP_EP_ID_INVALID);

        if (rndv_rts_hdr->flags & UCP_RNDV_RTS_FLAG_AM) {
            am_rts   = ucs_derived_of(rndv_rts_hdr, ucp_am_rndv_rts_hdr_t);
            rkey_buf = am_rts + 1;
            ucs_string_buffer_appendf(&rts_info, "AM am_id %u",
                                      am_rts->am.am_id);
        } else {
            ucs_assert(rndv_rts_hdr->flags & UCP_RNDV_RTS_FLAG_TAG);

            tag_rts  = ucs_derived_of(rndv_rts_hdr, ucp_tag_rndv_rts_hdr_t);
            rkey_buf = tag_rts + 1;

            ucs_string_buffer_appendf(&rts_info, "TAG tag %"PRIx64"",
                                      tag_rts->tag.tag);
        }

        snprintf(buffer, max, "RNDV_RTS %s ep_id 0x%"PRIx64" sreq_id"
                 " 0x%"PRIx64" address 0x%"PRIx64" size %zu",
                 ucs_string_buffer_cstr(&rts_info), rndv_rts_hdr->sreq.ep_id,
                 rndv_rts_hdr->sreq.req_id, rndv_rts_hdr->address,
                 rndv_rts_hdr->size);

        if (rndv_rts_hdr->address) {
            ucp_rndv_dump_rkey(rkey_buf, buffer + strlen(buffer),
                               max - strlen(buffer));
        }
        break;
    case UCP_AM_ID_RNDV_ATS:
        snprintf(buffer, max, "RNDV_ATS sreq_id 0x%"PRIx64" status '%s'",
                 rep_hdr->req_id, ucs_status_string(rep_hdr->status));
        break;
    case UCP_AM_ID_RNDV_RTR:
        snprintf(buffer, max, "RNDV_RTR sreq_id 0x%"PRIx64" rreq_id 0x%"PRIx64
                 " address 0x%"PRIx64, rndv_rtr_hdr->sreq_id,
                 rndv_rtr_hdr->rreq_id, rndv_rtr_hdr->address);
        if (rndv_rtr_hdr->address) {
            ucp_rndv_dump_rkey(rndv_rtr_hdr + 1, buffer + strlen(buffer),
                               max - strlen(buffer));
        }
        break;
    case UCP_AM_ID_RNDV_DATA:
        snprintf(buffer, max, "RNDV_DATA rreq_id 0x%"PRIx64" offset %zu",
                 rndv_data->rreq_id, rndv_data->offset);
        break;
    case UCP_AM_ID_RNDV_ATP:
        snprintf(buffer, max, "RNDV_ATP sreq_id 0x%"PRIx64" status '%s'",
                 rep_hdr->req_id, ucs_status_string(rep_hdr->status));
        break;
    default:
        return;
    }
}

UCP_DEFINE_AM(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_RTS,
              ucp_rndv_rts_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_ATS,
              ucp_rndv_ats_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_ATP,
              ucp_rndv_atp_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_RTR,
              ucp_rndv_rtr_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_DATA,
              ucp_rndv_data_handler, ucp_rndv_dump, 0);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATS);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_ATP);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_RTR);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_RNDV_DATA);
