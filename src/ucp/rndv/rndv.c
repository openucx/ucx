/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rndv.inl"
#include "proto_rndv.inl"

/* TODO: Avoid dependency on tag (or other API) specifics, since this is common
 * basic rendezvous implementation.
 */
#include <ucp/tag/tag_rndv.h>
#include <ucp/tag/tag_match.inl>
#include <ucp/tag/offload.h>
#include <ucp/proto/proto_am.inl>
#include <ucs/datastruct/queue.h>


static UCS_F_ALWAYS_INLINE int
ucp_rndv_memtype_direct_support(ucp_context_h context, size_t reg_length,
                                uint64_t reg_mem_types)
{
    /* if the message size is rndv_memtype_direct_size or larger,
    ** disable zero-copy rndv on lanes that also support host memory.
    */
    return !(reg_mem_types & UCS_BIT(UCS_MEMORY_TYPE_HOST)) ||
            (reg_length < context->config.ext.rndv_memtype_direct_size);
}

static UCS_F_ALWAYS_INLINE void
ucp_rndv_memtype_direct_update_md_map(ucp_context_h context,
                                      ucp_request_t *sreq, ucp_md_map_t *md_map)
{
    unsigned md_index;
    uct_md_attr_t *md_attr;

    if (UCP_MEM_IS_HOST(sreq->send.mem_type)) {
        return;
    }

    ucs_for_each_bit(md_index, *md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if (!ucp_rndv_memtype_direct_support(context, sreq->send.length,
                                             md_attr->cap.reg_mem_types)) {
            *md_map &= ~UCS_BIT(md_index);
        }
    }
}

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
    ucs_memory_type_t frag_mem_type  = context->config.ext.rndv_frag_mem_type;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    uint64_t mem_types;
    int i;

    for (i = 0;
         (i < UCP_MAX_LANES) &&
         (ep_config->key.rma_bw_lanes[i] != UCP_NULL_LANE); i++) {
        md_index = ep_config->md_index[ep_config->key.rma_bw_lanes[i]];
        if (context->tl_mds[md_index].attr.cap.reg_mem_types &
            UCS_BIT(frag_mem_type)) {
            found = 1;
            break;
        }
    }

    /* no bounce buffer mem_type bw lanes for pipeline staging */
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

static UCS_F_ALWAYS_INLINE int
ucp_rndv_is_put_pipeline_needed(uintptr_t remote_address, size_t length,
                                const void *rkey_buf,
                                const ucp_ep_rndv_zcopy_config_t *get_zcopy,
                                const ucp_ep_rndv_zcopy_config_t *put_zcopy,
                                int is_get_zcopy_failed)
{
    if (ucp_rkey_packed_mem_type(rkey_buf) == UCS_MEMORY_TYPE_HOST) {
        return 0;
    }

    /* Fallback to PUT pipeline if: */
    return /* Remote mem type is non-HOST memory OR can't do GET ZCOPY */
           ((remote_address == 0) || (get_zcopy->max == 0) ||
            (length < get_zcopy->min) || is_get_zcopy_failed) &&
           /* AND can do PUT assuming that configurations are symmetric */
           ((put_zcopy->max != 0) && (length >= put_zcopy->min));
}

size_t ucp_rndv_rts_pack(ucp_request_t *sreq, ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                         ucp_rndv_rts_opcode_t opcode)
{
    ucp_worker_h worker = sreq->send.ep->worker;
    ucp_memory_info_t mem_info;
    ssize_t packed_rkey_size;
    void *rkey_buf;

    rndv_rts_hdr->sreq.ep_id  = ucp_send_request_get_ep_remote_id(sreq);
    rndv_rts_hdr->sreq.req_id = ucp_send_request_get_id(sreq);
    rndv_rts_hdr->size        = sreq->send.length;
    rndv_rts_hdr->opcode      = opcode;

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq, worker->context)) {
        /* pack rkey, ask target to do get_zcopy */
        mem_info.type         = sreq->send.mem_type;
        mem_info.sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;
        rndv_rts_hdr->address = (uintptr_t)sreq->send.buffer;
        rkey_buf              = UCS_PTR_BYTE_OFFSET(rndv_rts_hdr,
                                                    sizeof(*rndv_rts_hdr));
        packed_rkey_size      = ucp_rkey_pack_uct(
                worker->context, sreq->send.state.dt.dt.contig.md_map,
                sreq->send.state.dt.dt.contig.memh, &mem_info, 0,
                ucp_ep_config(sreq->send.ep)->uct_rkey_pack_flags, NULL,
                rkey_buf);
        if (packed_rkey_size < 0) {
            ucs_fatal("failed to pack rendezvous remote key: %s",
                      ucs_status_string((ucs_status_t)packed_rkey_size));
        }

        ucs_assert(packed_rkey_size <=
                   ucp_ep_config(sreq->send.ep)->rndv.rkey_size);
        sreq->flags |= UCP_REQUEST_FLAG_RKEY_INUSE;
    } else {
        rndv_rts_hdr->address = 0;
        packed_rkey_size      = 0;
    }

    return sizeof(*rndv_rts_hdr) + packed_rkey_size;
}

static size_t ucp_rndv_rtr_pack(void *dest, void *arg)
{
    ucp_request_t *rndv_req          = arg;
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = dest;
    ucp_request_t *rreq              = ucp_request_get_super(rndv_req);
    ucp_ep_h ep                      = rndv_req->send.ep;
    ucp_memory_info_t mem_info;
    ssize_t packed_rkey_size;

    /* Request ID of sender side (remote) */
    rndv_rtr_hdr->sreq_id = rreq->recv.remote_req_id;
    /* Request ID of receiver side (local) */
    rndv_rtr_hdr->rreq_id = ucp_send_request_get_id(rndv_req);

    /* Pack remote keys (which can be empty list) */
    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        rndv_rtr_hdr->address = (uintptr_t)rreq->recv.buffer;
        rndv_rtr_hdr->size    = rndv_req->send.rndv_rtr.length;
        rndv_rtr_hdr->offset  = rndv_req->send.rndv_rtr.offset;
        mem_info.type         = rreq->recv.mem_type;
        mem_info.sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;

        packed_rkey_size = ucp_rkey_pack_uct(
                ep->worker->context, rreq->recv.state.dt.contig.md_map,
                rreq->recv.state.dt.contig.memh, &mem_info, 0,
                ucp_ep_config(ep)->uct_rkey_pack_flags, NULL, rndv_rtr_hdr + 1);
        if (packed_rkey_size < 0) {
            return packed_rkey_size;
        }

        rreq->flags |= UCP_REQUEST_FLAG_RKEY_INUSE;
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
    ucp_request_t *rreq     = ucp_request_get_super(rndv_req);
    ucp_md_map_t md_map     = UCP_DT_IS_CONTIG(rreq->recv.datatype) ?
                              rreq->recv.state.dt.contig.md_map : 0;
    size_t packed_rkey_size;
    ucs_status_t status;

    /* Send the RTR. The pack_cb will pack all the necessary fields in the RTR */
    packed_rkey_size = ucp_rkey_packed_size(rndv_req->send.ep->worker->context,
                                            md_map, UCS_SYS_DEVICE_ID_UNKNOWN,
                                            0);
    status           = ucp_do_am_single(
                           self, UCP_AM_ID_RNDV_RTR, ucp_rndv_rtr_pack,
                           sizeof(ucp_rndv_rtr_hdr_t) + packed_rkey_size);

    return ucp_rndv_send_handle_status_from_pending(rndv_req, status);
}

ucs_status_t
ucp_rndv_reg_send_buffer(ucp_request_t *sreq, const ucp_request_param_t *param)
{
    ucp_ep_h ep = sreq->send.ep;
    ucp_md_map_t md_map;
    ucs_status_t status;

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        ucp_rndv_is_get_zcopy(sreq, ep->worker->context)) {

        /* register a contiguous buffer for rma_get */
        md_map = ucp_ep_config(ep)->key.rma_bw_md_map;
        ucp_rndv_memtype_direct_update_md_map(ep->worker->context, sreq, &md_map);

        status = ucp_send_request_set_user_memh(sreq, md_map, param);
        if (status != UCS_OK) {
            return status;
        }

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

void ucp_rndv_req_send_ack(ucp_request_t *ack_req, size_t ack_size,
                           ucs_ptr_map_key_t remote_req_id, ucs_status_t status,
                           ucp_am_id_t am_id, const char *ack_str)
{
    ucp_trace_req(ack_req, "%s remote_req_id 0x%" PRIxPTR " size %zu", ack_str,
                  remote_req_id, ack_size);
    UCS_PROFILE_REQUEST_EVENT(ack_req, ack_str, 0);

    ack_req->send.lane                = ucp_ep_get_am_lane(ack_req->send.ep);
    ack_req->send.uct.func            = ucp_proto_progress_am_single;
    ack_req->send.length              = ack_size;
    ack_req->send.proto.am_id         = am_id;
    ack_req->send.proto.status        = status;
    ack_req->send.proto.remote_req_id = remote_req_id;
    ack_req->send.proto.comp_cb       = ucp_request_put;
    ucp_request_send_state_reset(ack_req, NULL,
                                 UCP_REQUEST_SEND_PROTO_BCOPY_AM);
    ucp_request_send(ack_req);
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

static void ucp_rndv_complete_rma_put_zcopy(ucp_request_t *sreq, int is_frag_put)
{
    ucs_status_t status = sreq->send.state.uct_comp.status;
    ucp_request_t *atp_req;

    ucs_assertv(sreq->send.state.dt.offset <= sreq->send.length,
                "sreq=%p offset=%zu length=%zu", sreq,
                sreq->send.state.dt.offset, sreq->send.length);

    /* complete send request after PUT completions of all fragments */
    if (sreq->send.state.dt.offset != sreq->send.length) {
        return;
    }

    ucp_trace_req(sreq, "rndv_put completed with status %s",
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(sreq, "complete_rndv_put", 0);

    if (is_frag_put) {
        ucp_send_request_id_release(sreq);
    } else {
        ucp_rkey_destroy(sreq->send.rndv.rkey);

        if (status == UCS_OK) {
            atp_req = ucp_request_get(sreq->send.ep->worker);
            if (ucs_unlikely(atp_req == NULL)) {
                ucs_fatal("failed to allocate request for sending ATP");
            }

            ucs_assertv(sreq->send.state.dt.offset == sreq->send.length,
                        "req=%p offset=%zu length=%zu", sreq,
                        sreq->send.state.dt.offset, sreq->send.length);
            atp_req->send.ep = sreq->send.ep;
            atp_req->flags   = 0;
            ucp_rndv_req_send_ack(atp_req, sreq->send.length,
                                  sreq->send.rndv.remote_req_id, status,
                                  UCP_AM_ID_RNDV_ATP, "send_atp");
        }
    }

    ucp_request_send_buffer_dereg(sreq);
    ucp_request_complete_send(sreq, status);
}

static void ucp_rndv_recv_data_init(ucp_request_t *rreq, size_t size)
{
    rreq->status         = UCS_OK;
    rreq->recv.remaining = size;
}

ucs_status_t ucp_rndv_send_rts(ucp_request_t *sreq, uct_pack_callback_t pack_cb,
                               size_t rts_size)
{
    size_t max_rts_size = ucp_ep_config(sreq->send.ep)->rndv.rkey_size +
                          rts_size;
    ucs_status_t status;

    status = ucp_do_am_single(&sreq->send.uct, UCP_AM_ID_RNDV_RTS, pack_cb,
                              max_rts_size);
    return ucp_rndv_send_handle_status_from_pending(sreq, status);
}

static void ucp_rndv_req_send_rtr(ucp_request_t *rndv_req, ucp_request_t *rreq,
                                  ucs_ptr_map_key_t sender_req_id,
                                  size_t recv_length, size_t offset)
{
    ucp_trace_req(rndv_req, "send rtr remote sreq_id 0x%"PRIxPTR" rreq %p",
                  sender_req_id, rreq);

    /* Reset super request and send state, since it may be set by the previous
     * protocol (e.g. RNDV GET Zcopy) */
    ucp_request_reset_super(rndv_req);
    ucp_request_send_state_reset(rndv_req, NULL,
                                 UCP_REQUEST_SEND_PROTO_BCOPY_AM);
    UCP_WORKER_STAT_RNDV(rndv_req->send.ep->worker, SEND_RTR, +1);

    rreq->recv.remote_req_id       = sender_req_id;
    rndv_req->send.lane            = ucp_ep_get_am_lane(rndv_req->send.ep);
    rndv_req->send.uct.func        = ucp_proto_progress_rndv_rtr;
    rndv_req->send.rndv_rtr.length = recv_length;
    rndv_req->send.rndv_rtr.offset = offset;

    ucp_request_set_super(rndv_req, rreq);
    ucp_send_request_id_alloc(rndv_req);

    ucp_request_send(rndv_req);
}

static ucp_lane_index_t ucp_rndv_zcopy_get_lane(ucp_request_t *rndv_req,
                                                uct_rkey_t *uct_rkey,
                                                unsigned proto)
{
    ucp_lane_index_t lane_idx;
    ucp_ep_config_t *ep_config;
    ucp_rkey_h rkey;
    uint8_t rkey_index;

    ucs_assert((proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) ||
               (proto == UCP_REQUEST_SEND_PROTO_RNDV_PUT));

    if (ucs_unlikely(!rndv_req->send.rndv.lanes_map_all)) {
        return UCP_NULL_LANE;
    }

    lane_idx   = ucs_ffs64_safe(rndv_req->send.lanes_map_avail);
    ucs_assert(lane_idx < UCP_MAX_LANES);
    rkey       = rndv_req->send.rndv.rkey;
    rkey_index = rndv_req->send.rndv.rkey_index[lane_idx];
    *uct_rkey  = ucp_rkey_get_tl_rkey(rkey, rkey_index);
    ep_config  = ucp_ep_config(rndv_req->send.ep);
    return (proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) ?
           ep_config->rndv.get_zcopy.lanes[lane_idx] :
           ep_config->rndv.put_zcopy.lanes[lane_idx];
}

static void ucp_rndv_zcopy_next_lane(ucp_request_t *rndv_req)
{
    rndv_req->send.lanes_map_avail    &= rndv_req->send.lanes_map_avail - 1;
    if (!rndv_req->send.lanes_map_avail) {
        rndv_req->send.lanes_map_avail = rndv_req->send.rndv.lanes_map_all;
    }
}

static ucs_status_t
ucp_rndv_progress_rma_zcopy_common(ucp_request_t *req, ucp_lane_index_t lane,
                                   uct_rkey_t uct_rkey, unsigned proto)
{
    const size_t max_iovcnt = 1;
    ucp_ep_h ep             = req->send.ep;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    uct_iov_t iov[max_iovcnt];
    size_t iovcnt;
    uct_iface_attr_t *attrs;
    ucs_status_t status;
    size_t offset, length, ucp_mtu, remaining, align, chunk;
    ucp_dt_state_t state;
    ucp_rsc_index_t rsc_index;
    size_t min_zcopy;
    size_t max_zcopy;
    double scale;
    int pending_add_res;

    ucs_assert_always(req->send.lane != UCP_NULL_LANE);
    ucs_assert_always(req->send.rndv.lanes_count > 0);

    if (req->send.rndv.mdesc == NULL) {
        status = ucp_send_request_add_reg_lane(req, lane);
        ucs_assert_always(status == UCS_OK);
    }

    rsc_index = ucp_ep_get_rsc_index(ep, lane);
    attrs     = ucp_worker_iface_get_attr(ep->worker, rsc_index);

    if (proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) {
        align     = attrs->cap.get.opt_zcopy_align;
        ucp_mtu   = attrs->cap.get.align_mtu;
        min_zcopy = config->rndv.get_zcopy.min;
        max_zcopy = config->rndv.get_zcopy.max;
        scale     = config->rndv.get_zcopy.scale[lane];
    } else {
        align     = attrs->cap.put.opt_zcopy_align;
        ucp_mtu   = attrs->cap.put.align_mtu;
        min_zcopy = config->rndv.put_zcopy.min;
        max_zcopy = config->rndv.put_zcopy.max;
        scale     = config->rndv.put_zcopy.scale[lane];
    }

    offset    = req->send.state.dt.offset;
    remaining = (uintptr_t)req->send.buffer % align;

    if ((offset == 0) && (remaining > 0) && (req->send.length > ucp_mtu)) {
        length = ucp_mtu - remaining;
    } else {
        chunk  = ucs_align_up((size_t)(req->send.length /
                                       req->send.rndv.lanes_count * scale),
                              align);
        length = ucs_min(chunk, req->send.length - offset);
    }

    length = ucp_rndv_adjust_zcopy_length(min_zcopy, max_zcopy, align,
                                          req->send.length, offset, length);

    ucs_trace_data("req %p: offset %zu remain %zu RMA-%s to %p len %zu lane %d",
                   req, offset, remaining,
                   (proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) ? "GET" : "PUT",
                   UCS_PTR_BYTE_OFFSET(req->send.buffer, offset), length, lane);

    state = req->send.state.dt;
    /* TODO: is this correct? memh array may skip MD's where
     * registration is not supported. for now SHM may avoid registration,
     * but it will work on single lane */
    ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iovcnt, &state,
                        req->send.buffer, ucp_dt_make_contig(1), length,
                        ucp_ep_md_index(ep, lane), req->send.rndv.mdesc);

    for (;;) {
        if (proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) {
            status = uct_ep_get_zcopy(ep->uct_eps[lane], iov, iovcnt,
                                      req->send.rndv.remote_address + offset,
                                      uct_rkey, &req->send.state.uct_comp);
        } else {
            status = uct_ep_put_zcopy(ep->uct_eps[lane], iov, iovcnt,
                                      req->send.rndv.remote_address + offset,
                                      uct_rkey, &req->send.state.uct_comp);
        }

        ucp_request_send_state_advance(req, &state, proto, status);
        if (ucs_likely(!UCS_STATUS_IS_ERR(status))) {
            if (req->send.state.dt.offset == req->send.length) {
                ucp_send_request_invoke_uct_completion(req);
                return UCS_OK;
            }

            /* Return in_progress status in case if not all chunks are
             * transmitted */
            ucp_rndv_zcopy_next_lane(req);
            return UCS_INPROGRESS;
        } else if (status == UCS_ERR_NO_RESOURCE) {
            if (lane != req->send.pending_lane) {
                /* switch to new pending lane */
                pending_add_res = ucp_request_pending_add(req);
                if (!pending_add_res) {
                    /* failed to switch req to pending queue, try again */
                    continue;
                }
                return UCS_OK;
            }
            return UCS_ERR_NO_RESOURCE;
        } else {
            return UCS_OK;
        }
    }
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_get_completion, (self), uct_completion_t *self)
{
    ucp_request_t *rndv_req = ucs_container_of(self, ucp_request_t,
                                               send.state.uct_comp);
    ucp_ep_h UCS_V_UNUSED ep;
    ucp_request_t *rreq;
    ucs_status_t status;

    if (rndv_req->send.state.dt.offset != rndv_req->send.length) {
        return;
    }

    rreq   = ucp_request_get_super(rndv_req);
    status = rndv_req->send.state.uct_comp.status;
    ep     = rndv_req->send.ep;

    ucs_assertv(rndv_req->send.state.dt.offset == rndv_req->send.length,
                "rndv_req=%p offset=%zu length=%zu", rndv_req,
                rndv_req->send.state.dt.offset, rndv_req->send.length);

    ucp_trace_req(rndv_req, "rndv_get completed with status %s",
                  ucs_status_string(status));
    UCS_PROFILE_REQUEST_EVENT(rreq, "complete_rndv_get", 0);

    ucp_rkey_destroy(rndv_req->send.rndv.rkey);
    ucp_request_send_buffer_dereg(rndv_req);

    if (status == UCS_OK) {
        ucp_rndv_req_send_ack(rndv_req, rndv_req->send.length,
                              rndv_req->send.rndv.remote_req_id, UCS_OK,
                              UCP_AM_ID_RNDV_ATS, "send_ats");
    } else {
        /* if completing RNDV with the error, just release RNDV request */
        ucp_request_put(rndv_req);
    }

    /* Check for possible leak of memory registration: we should either not have
     * any memory registration, or not own it, or be called from the context of
     * ucp_request_send_state_ff() */
    ucs_assert((rreq->recv.state.dt.contig.md_map == 0) ||
               (rreq->flags & UCP_REQUEST_FLAG_USER_MEMH) ||
               ((ep->flags & UCP_EP_FLAG_FAILED) && (status != UCS_OK)));
    ucp_rndv_recv_req_complete(rreq, status);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_put_completion, (self), uct_completion_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    ucp_rndv_complete_rma_put_zcopy(sreq, 0);
}

static void ucp_rndv_req_init_lanes(ucp_request_t *req,
                                    ucp_lane_map_t lanes_map,
                                    uint8_t lanes_count)
{
    req->send.lanes_map_avail    = lanes_map;
    req->send.rndv.lanes_map_all = lanes_map;
    req->send.rndv.lanes_count   = lanes_count;
}

static void ucp_rndv_req_init_zcopy_lane_map(ucp_request_t *rndv_req,
                                             ucs_memory_type_t mem_type,
                                             size_t length, unsigned proto)
{
    ucp_ep_h ep                = rndv_req->send.ep;
    ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    ucp_context_h context      = ep->worker->context;
    ucp_rkey_h rkey            = rndv_req->send.rndv.rkey;
    ucp_lane_index_t *lanes;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane, lane_idx;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    ucp_md_index_t dst_md_index;
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *iface_attr;
    double max_lane_bw, lane_bw;
    int i;

    ucs_assert((proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) ||
               (proto == UCP_REQUEST_SEND_PROTO_RNDV_PUT));

    lanes = (proto == UCP_REQUEST_SEND_PROTO_RNDV_GET) ?
            ep_config->rndv.get_zcopy.lanes :
            ep_config->rndv.put_zcopy.lanes;

    max_lane_bw = 0;
    lane_map    = 0;
    for (i = 0; i < UCP_MAX_LANES; i++) {
        lane = lanes[i];
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
                rndv_req->send.rndv.rkey_index[i] = UCP_NULL_RESOURCE;
                lane_map                         |= UCS_BIT(i);
                max_lane_bw                       = ucs_max(max_lane_bw, lane_bw);
                continue;
            }
        }

        if (ucs_unlikely((md_index != UCP_NULL_RESOURCE) &&
                         (!(md_attr->cap.reg_mem_types & UCS_BIT(mem_type))))) {
            continue;
        }

        if (!UCP_MEM_IS_HOST(mem_type) &&
            !ucp_rndv_memtype_direct_support(context, length,
                                             md_attr->cap.reg_mem_types)) {
            continue;
        }

        dst_md_index = ep_config->key.lanes[lane].dst_md_index;
        if (rkey && ucs_likely(rkey->md_map & UCS_BIT(dst_md_index))) {
            /* Return first matching lane */
            rndv_req->send.rndv.rkey_index[i] = ucs_bitmap2idx(rkey->md_map,
                                                               dst_md_index);
            lane_map                         |= UCS_BIT(i);
            max_lane_bw                       = ucs_max(max_lane_bw, lane_bw);
        }
    }

    if (ucs_popcount(lane_map) > 1) {
        /* remove lanes if bandwidth is too less compare to best lane */
        ucs_for_each_bit(lane_idx, lane_map) {
            ucs_assert(lane_idx < UCP_MAX_LANES);
            lane       = lanes[lane_idx];
            rsc_index  = ep_config->key.lanes[lane].rsc_index;
            iface_attr = ucp_worker_iface_get_attr(ep->worker, rsc_index);
            lane_bw    = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);

            if ((lane_bw / max_lane_bw) <
                (1. / context->config.ext.multi_lane_max_ratio)) {
                lane_map                                &= ~UCS_BIT(lane_idx);
                rndv_req->send.rndv.rkey_index[lane_idx] = UCP_NULL_RESOURCE;
            }
        }
    }

    ucp_rndv_req_init_lanes(rndv_req, lane_map, ucs_popcount(lane_map));
}

static void ucp_rndv_req_init(ucp_request_t *req, ucp_request_t *super_req,
                              ucp_lane_map_t lanes_map, uint8_t lanes_count,
                              ucp_rkey_h rkey, uint64_t remote_address,
                              uint8_t *rkey_index)
{
    ucp_lane_index_t i;

    req->send.rndv.rkey           = rkey;
    req->send.rndv.remote_address = remote_address;
    req->send.pending_lane        = UCP_NULL_LANE;

    ucp_request_set_super(req, super_req);
    ucp_rndv_req_init_lanes(req, lanes_map, lanes_count);

    if (rkey_index != NULL) {
        memcpy(req->send.rndv.rkey_index, rkey_index,
               sizeof(*req->send.rndv.rkey_index) * UCP_MAX_LANES);
    } else {
        for (i = 0; i < UCP_MAX_LANES; i++) {
            req->send.rndv.rkey_index[i] = UCP_NULL_RESOURCE;
        }
    }
}

static void
ucp_rndv_rkey_ptr_get_mem_type(ucp_request_t *sreq, size_t length,
                               uint64_t remote_address,
                               void *remote_frag_rkey_ptr,
                               ucs_memory_type_t remote_mem_type,
                               uct_completion_callback_t comp_cb)
{
    ucp_lane_map_t lanes_map = UCS_BIT(0);
    ucp_worker_h worker      = sreq->send.ep->worker;
    ucp_request_t *freq;
    ucp_ep_h mem_type_ep;
    ucp_md_index_t md_index;
    ucp_lane_index_t mem_type_rma_lane;

    /* GET fragment to remote stage buffer */

    freq = ucp_request_get(worker);
    if (ucs_unlikely(freq == NULL)) {
        ucs_fatal("failed to allocate fragment receive request");
    }

    ucp_request_send_state_init(freq, ucp_dt_make_contig(1), 0);
    ucp_request_send_state_reset(freq, comp_cb,
                                 UCP_REQUEST_SEND_PROTO_RNDV_GET);

    freq->flags             = 0;
    freq->send.buffer       = remote_frag_rkey_ptr;
    freq->send.length       = length;
    freq->send.datatype     = ucp_dt_make_contig(1);
    freq->send.mem_type     = remote_mem_type;
    freq->send.rndv.mdesc   = NULL;
    freq->send.uct.func     = ucp_rndv_progress_rma_get_zcopy;
    freq->send.pending_lane = UCP_NULL_LANE;

    mem_type_ep             = worker->mem_type_ep[remote_mem_type];
    mem_type_rma_lane       = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
    md_index                = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);

    ucs_assert(mem_type_rma_lane != UCP_NULL_LANE);

    freq->send.lane                       = mem_type_rma_lane;
    freq->send.ep                         = mem_type_ep;
    freq->send.state.dt.dt.contig.memh[0] = NULL;
    freq->send.state.dt.dt.contig.md_map  = UCS_BIT(md_index);

    ucp_rndv_req_init(freq, sreq, lanes_map, ucs_popcount(lanes_map), NULL,
                      remote_address, NULL);

    UCP_WORKER_STAT_RNDV(freq->send.ep->worker, GET_ZCOPY, 1);

    freq->status = UCS_INPROGRESS;
    ucp_request_send(freq);
}

static void
ucp_rndv_req_init_remote_from_super_req(ucp_request_t *req,
                                        ucp_request_t *super_req,
                                        size_t remote_address_offset)
{
    req->flags   = 0;
    req->send.ep = super_req->send.ep;

    ucp_rndv_req_init(req, super_req, super_req->send.rndv.lanes_map_all,
                      super_req->send.rndv.lanes_count,
                      super_req->send.rndv.rkey,
                      super_req->send.rndv.remote_address +
                      remote_address_offset,
                      super_req->send.rndv.rkey_index);
}

static void ucp_rndv_req_init_from_super_req(ucp_request_t *req,
                                             ucp_request_t *super_req,
                                             size_t length,
                                             size_t send_buffer_offset,
                                             size_t remote_address_offset,
                                             ucs_ptr_map_key_t remote_req_id)
{
    ucs_assert(length > 0);

    req->send.length = length;
    req->send.buffer = UCS_PTR_BYTE_OFFSET(super_req->send.buffer,
                                           send_buffer_offset);

    ucp_rndv_req_init_remote_from_super_req(req, super_req,
                                            remote_address_offset);

    req->send.rndv.remote_req_id = remote_req_id;
}

static ucs_status_t ucp_rndv_req_send_rma_get(ucp_request_t *rndv_req,
                                              ucp_request_t *rreq,
                                              const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                                              const void *rkey_buf)
{
    ucp_ep_h ep = rndv_req->send.ep;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    ucp_lane_index_t lane;
    ucp_md_map_t md_map;

    ucp_trace_req(rndv_req, "start rma_get rreq %p", rreq);

    rndv_req->send.uct.func            = ucp_rndv_progress_rma_get_zcopy;
    rndv_req->send.buffer              = rreq->recv.buffer;
    rndv_req->send.mem_type            = rreq->recv.mem_type;
    rndv_req->send.datatype            = ucp_dt_make_contig(1);
    rndv_req->send.length              = rndv_rts_hdr->size;
    rndv_req->send.rndv.remote_req_id  = rndv_rts_hdr->sreq.req_id;
    rndv_req->send.rndv.remote_address = rndv_rts_hdr->address;
    rndv_req->send.pending_lane        = UCP_NULL_LANE;

    ucp_request_set_super(rndv_req, rreq);

    status = ucp_ep_rkey_unpack(ep, rkey_buf, &rndv_req->send.rndv.rkey);
    if (status != UCS_OK) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(ep), ucs_status_string(status));
    }

    ucp_request_send_state_init(rndv_req, ucp_dt_make_contig(1), 0);
    ucp_request_send_state_reset(rndv_req, ucp_rndv_get_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_GET);

    ucp_rndv_req_init_zcopy_lane_map(rndv_req, rndv_req->send.mem_type,
                                     rndv_req->send.length,
                                     UCP_REQUEST_SEND_PROTO_RNDV_GET);

    /* Copy user registration from receive request to rndv send request */
    if (rreq->flags & UCP_REQUEST_FLAG_USER_MEMH) {
        md_map = 0;
        ucs_for_each_bit(lane, rndv_req->send.rndv.lanes_map_all) {
            ucs_assert(lane < UCP_MAX_LANES);
            md_map |= ucp_ep_md_index(ep, lane);
        }

        ucp_request_init_dt_reg_from_memh(rndv_req, md_map,
                                          rreq->recv.user_memh,
                                          &rndv_req->send.state.dt.dt.contig);
    }

    rndv_req->send.lane =
        ucp_rndv_zcopy_get_lane(rndv_req, &uct_rkey,
                                UCP_REQUEST_SEND_PROTO_RNDV_GET);
    if (rndv_req->send.lane == UCP_NULL_LANE) {
        goto err;
    }

    UCP_WORKER_STAT_RNDV(ep->worker, GET_ZCOPY, 1);
    ucp_request_send(rndv_req);

    return UCS_OK;

err:
    ucp_request_reset_super(rndv_req);
    ucp_rkey_destroy(rndv_req->send.rndv.rkey);
    return UCS_ERR_UNREACHABLE;
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_recv_frag_put_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq     = ucs_container_of(self, ucp_request_t,
                                               send.state.uct_comp);
    /* if the super request is a receive request, it means that's used RNDV
     * scheme is PUT pipeline protocol, otherwise - GET pipeline protocol (where
     * the super request is an intermediate RNDV request) */
    int is_put_proto        = ucp_request_get_super(freq)->flags &
                              (UCP_REQUEST_FLAG_RECV_TAG |
                               UCP_REQUEST_FLAG_RECV_AM);
    ucp_request_t *rreq;
    ucp_request_t *rndv_req;

    /* release memory descriptor */
    ucs_mpool_put_inline((void*)freq->send.rndv.mdesc);

    /* rndv_req is NULL in case of put protocol */
    if (!is_put_proto) {
        rndv_req = ucp_request_get_super(freq);
        rreq     = ucp_request_get_super(rndv_req);

        ucs_trace_req("freq:%p: recv_frag_put done, nrdv_req:%p rreq:%p ", freq,
                      rndv_req, rreq);

        /* pipeline recv get protocol */
        rndv_req->send.state.dt.offset += freq->send.length;

        /* send ATS for fragment get rndv completion */
        if (rndv_req->send.length == rndv_req->send.state.dt.offset) {
            ucp_rkey_destroy(rndv_req->send.rndv.rkey);
            ucp_rndv_req_send_ack(rndv_req, rndv_req->send.length,
                                  rndv_req->send.rndv.remote_req_id, UCS_OK,
                                  UCP_AM_ID_RNDV_ATS, "send_ats");
        }
    } else {
        rreq = ucp_request_get_super(freq);
        ucs_trace_req("freq:%p: recv_frag_put done, rreq:%p ", freq, rreq);
    }

    ucs_assertv(rreq->recv.remaining >= freq->send.length,
                "rreq->recv.remaining %zu, freq->send.length %zu",
                rreq->recv.remaining, freq->send.length);
    rreq->recv.remaining -= freq->send.length;
    if (rreq->recv.remaining == 0) {
        ucp_rndv_recv_req_complete(rreq, UCS_OK);
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

    freq->flags             = 0;
    freq->send.buffer       = mdesc->ptr;
    freq->send.length       = length;
    freq->send.datatype     = ucp_dt_make_contig(1);
    freq->send.mem_type     = mem_type;
    freq->send.rndv.mdesc   = mdesc;
    freq->send.uct.func     = uct_func;
    freq->send.pending_lane = UCP_NULL_LANE;

    if (mem_type != UCS_MEMORY_TYPE_HOST) {
        mem_type_ep       = worker->mem_type_ep[mem_type];
        mem_type_rma_lane = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
        md_index          = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);
        ucs_assert(mem_type_rma_lane != UCP_NULL_LANE);

        freq->send.lane                       = mem_type_rma_lane;
        freq->send.ep                         = mem_type_ep;
        freq->send.state.dt.dt.contig.memh[0] = mdesc->memh->uct[md_index];
        freq->send.state.dt.dt.contig.md_map  = UCS_BIT(md_index);
    }
}

static void
ucp_rndv_recv_frag_put_mem_type(ucp_request_t *rreq, ucp_request_t *freq,
                                ucp_mem_desc_t *mdesc, size_t length,
                                size_t offset)
{

    ucs_assert_always(!UCP_MEM_IS_HOST(rreq->recv.mem_type));

    /* PUT on memtype endpoint to stage from
     * frag recv buffer to memtype recv buffer
     */

    ucp_rndv_init_mem_type_frag_req(rreq->recv.worker, freq,
                                    UCP_REQUEST_SEND_PROTO_RNDV_PUT,
                                    ucp_rndv_recv_frag_put_completion, mdesc,
                                    rreq->recv.mem_type, length,
                                    ucp_rndv_progress_rma_put_zcopy);

    ucp_rndv_req_init(freq, rreq, 0, 0, NULL,
                      (uintptr_t)UCS_PTR_BYTE_OFFSET(rreq->recv.buffer, offset),
                      NULL);

    ucp_rndv_req_init_zcopy_lane_map(freq, freq->send.mem_type,
                                     freq->send.length,
                                     UCP_REQUEST_SEND_PROTO_RNDV_PUT);

    ucp_request_send(freq);
}

static void
ucp_rndv_send_frag_update_get_rkey(ucp_worker_h worker, ucp_request_t *freq,
                                   ucp_mem_desc_t *mdesc,
                                   ucs_memory_type_t mem_type)
{
    ucp_rkey_h *rkey_p  = &freq->send.rndv.rkey;
    uint8_t *rkey_index = freq->send.rndv.rkey_index;
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

ucs_mpool_ops_t ucp_frag_mpool_ops = {
    .chunk_alloc   = ucp_frag_mpool_malloc,
    .chunk_release = ucp_frag_mpool_free,
    .obj_init      = ucp_frag_mpool_obj_init,
    .obj_cleanup   = ucs_empty_function
};

ucp_mem_desc_t *
ucp_rndv_mpool_get(ucp_worker_h worker, ucs_memory_type_t mem_type,
                   ucs_sys_device_t sys_dev)
{
    ucp_rndv_mpool_priv_t *mpriv;
    ucp_worker_mpool_key_t key;
    ucs_status_t status;
    unsigned num_frags;
    ucs_mpool_t *mpool;
    khiter_t khiter;
    int khret;

    key.sys_dev  = sys_dev;
    key.mem_type = mem_type;

    khiter = kh_get(ucp_worker_mpool_hash, &worker->mpool_hash, key);
    if (ucs_likely(khiter != kh_end(&worker->mpool_hash))) {
        mpool = &kh_val(&worker->mpool_hash, khiter);
        goto out_mp_get;
    }

    khiter = kh_put(ucp_worker_mpool_hash, &worker->mpool_hash, key, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        return NULL;
    }

    ucs_assert_always(khret != UCS_KH_PUT_KEY_PRESENT);

    mpool     = &kh_value(&worker->mpool_hash, khiter);
    num_frags = worker->context->config.ext.rndv_num_frags[key.mem_type];
    status = ucs_mpool_init(mpool, sizeof(ucp_rndv_mpool_priv_t),
                            sizeof(ucp_mem_desc_t), 0, 1, num_frags, UINT_MAX,
                            &ucp_frag_mpool_ops, "ucp_rndv_frags");
    if (status != UCS_OK) {
        return NULL;
    }

    mpriv            = ucs_mpool_priv(mpool);
    mpriv->worker    = worker;
    mpriv->mem_type  = key.mem_type;

out_mp_get:
    return ucp_worker_mpool_get(mpool);
}

static void
ucp_rndv_send_frag_get_mem_type(ucp_request_t *sreq, size_t length,
                                uint64_t remote_address,
                                ucs_memory_type_t remote_mem_type,
                                ucp_rkey_h rkey, uint8_t *rkey_index,
                                ucp_lane_map_t lanes_map, int update_get_rkey,
                                uct_completion_callback_t comp_cb)
{
    ucp_worker_h worker             = sreq->send.ep->worker;
    ucs_memory_type_t frag_mem_type = worker->context->config.ext.rndv_frag_mem_type;
    ucp_request_t *freq;
    ucp_mem_desc_t *mdesc;

    /* GET fragment to stage buffer */

    freq = ucp_request_get(worker);
    if (ucs_unlikely(freq == NULL)) {
        ucs_fatal("failed to allocate fragment receive request");
    }

    mdesc = ucp_rndv_mpool_get(worker, frag_mem_type,
                               UCS_SYS_DEVICE_ID_UNKNOWN);
    if (ucs_unlikely(mdesc == NULL)) {
        ucs_fatal("failed to allocate fragment memory desc");
    }

    freq->send.ep         = sreq->send.ep;
    freq->send.rndv.mdesc = mdesc;

    ucp_rndv_init_mem_type_frag_req(worker, freq, UCP_REQUEST_SEND_PROTO_RNDV_GET,
                                    comp_cb, mdesc, remote_mem_type, length,
                                    ucp_rndv_progress_rma_get_zcopy);
    ucp_rndv_req_init(freq, sreq, lanes_map, ucs_popcount(lanes_map), rkey,
                      remote_address, rkey_index);

    if (update_get_rkey) {
        ucp_rndv_send_frag_update_get_rkey(worker, freq, mdesc, remote_mem_type);
    }

    UCP_WORKER_STAT_RNDV(freq->send.ep->worker, GET_ZCOPY, 1);

    freq->status = UCS_INPROGRESS;
    ucp_request_send(freq);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_recv_frag_get_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);
    ucp_request_t *rndv_req, *rreq;
    uint64_t offset;

    if (freq->send.state.dt.offset != freq->send.length) {
        return;
    }

    rndv_req = ucp_request_get_super(freq);
    rreq     = ucp_request_get_super(rndv_req);
    offset   = freq->send.rndv.remote_address -
               rndv_req->send.rndv.remote_address;

    ucs_trace_req("freq:%p: recv_frag_get done. rreq:%p length:%"PRIu64
                  " offset:%"PRIu64,
                  freq, rndv_req, freq->send.length, offset);

    /* fragment GET completed from remote to staging buffer, issue PUT from
     * staging buffer to recv buffer */
    ucp_rndv_recv_frag_put_mem_type(rreq, freq,
                                    freq->send.rndv.mdesc,
                                    freq->send.length, offset);
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
    uct_rkey_t uct_rkey;
    size_t max_frag_size, offset, length;
    size_t min_zcopy, max_zcopy;
    ucp_lane_index_t lane;
    ucs_memory_type_t frag_mem_type;
    size_t frag_size;

    /* use ucp_rkey_packed_mem_type(rkey_buffer) with non-host fragments */
    frag_mem_type = context->config.ext.rndv_frag_mem_type;
    frag_size     = context->config.ext.rndv_frag_size[frag_mem_type];

    min_zcopy                          = config->rndv.get_zcopy.min;
    max_zcopy                          = config->rndv.get_zcopy.max;
    max_frag_size                      = ucs_min(frag_size, max_zcopy);
    rndv_req->send.rndv.remote_req_id  = remote_req_id;
    rndv_req->send.rndv.remote_address = remote_address - base_offset;
    rndv_req->send.length              = size;
    rndv_req->send.state.dt.offset     = 0;
    rndv_req->send.mem_type            = rreq->recv.mem_type;
    rndv_req->send.pending_lane        = UCP_NULL_LANE;

    ucp_request_set_super(rndv_req, rreq);

    /* Protocol:
     * Step 1: GET remote fragment into HOST fragment buffer
     * Step 2: PUT from fragment buffer to MEM TYPE destination
     * Step 3: Send ATS for RNDV request
     */

    status = ucp_ep_rkey_unpack(rndv_req->send.ep, rkey_buffer,
                                &rndv_req->send.rndv.rkey);
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(rndv_req->send.ep), ucs_status_string(status));
    }

    ucp_rndv_req_init_zcopy_lane_map(rndv_req, rndv_req->send.mem_type,
                                     rndv_req->send.length,
                                     UCP_REQUEST_SEND_PROTO_RNDV_GET);

    lane = ucp_rndv_zcopy_get_lane(rndv_req, &uct_rkey,
                                   UCP_REQUEST_SEND_PROTO_RNDV_GET);
    if (lane == UCP_NULL_LANE) {
        goto err;
    }

    offset = 0;
    while (offset != size) {
        length = ucp_rndv_adjust_zcopy_length(min_zcopy, max_frag_size, 0,
                                              size, offset, size - offset);

        /* GET remote fragment into HOST fragment buffer */
        ucp_rndv_send_frag_get_mem_type(rndv_req, length,
                                        remote_address + offset,
                                        UCS_MEMORY_TYPE_HOST, /* TODO: find bounce buffer memory type */
                                        rndv_req->send.rndv.rkey,
                                        rndv_req->send.rndv.rkey_index,
                                        rndv_req->send.rndv.lanes_map_all, 0,
                                        ucp_rndv_recv_frag_get_completion);

        offset += length;
    }

    return UCS_OK;

err:
    ucp_request_reset_super(rndv_req);
    ucp_rkey_destroy(rndv_req->send.rndv.rkey);
    return UCS_ERR_UNREACHABLE;
}

static void ucp_rndv_send_frag_rtr(ucp_worker_h worker, ucp_request_t *rndv_req,
                                   ucp_request_t *rreq,
                                   const ucp_rndv_rts_hdr_t *rndv_rts_hdr)
{
    ucp_context_h context = worker->context;
    size_t max_frag_size;
    ucs_memory_type_t frag_mem_type;
    int i, num_frags;
    size_t frag_size;
    size_t offset;
    ucp_mem_desc_t *mdesc;
    ucp_request_t *freq;
    ucp_request_t *frndv_req;
    unsigned md_index;
    unsigned memh_index;
    const uct_md_attr_t *md_attr;
    ucp_md_map_t alloc_md_map;

    ucp_trace_req(rreq, "using rndv pipeline protocol rndv_req %p", rndv_req);

    offset        = 0;
    frag_mem_type = worker->context->config.ext.rndv_frag_mem_type;
    max_frag_size = worker->context->config.ext.rndv_frag_size[frag_mem_type];
    num_frags     = ucs_div_round_up(rndv_rts_hdr->size, max_frag_size);

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
        mdesc = ucp_rndv_mpool_get(worker, frag_mem_type,
                                   UCS_SYS_DEVICE_ID_UNKNOWN);
        if (mdesc == NULL) {
            ucs_fatal("failed to allocate fragment memory buffer");
        }

        freq->recv.buffer                 = mdesc->ptr;
        freq->recv.datatype               = ucp_dt_make_contig(1);
        freq->recv.mem_type               = mdesc->memh->mem_type;
        freq->recv.length                 = frag_size;
        freq->recv.state.dt.contig.md_map = 0;
        freq->recv.frag.offset            = offset;
        freq->flags                       = UCP_REQUEST_FLAG_RNDV_FRAG;

        ucp_request_set_super(freq, rreq);

        alloc_md_map = 0;
        /* TODO: Check if can avoid for inter-node eps */
        ucs_for_each_bit(md_index, mdesc->memh->md_map) {
            if (md_index == mdesc->memh->alloc_md_index) {
                md_attr = &context->tl_mds[md_index].attr;
                if (md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR) {
                    alloc_md_map |= UCS_BIT(md_index);
                }
                break;
            }
        }

        memh_index = 0;
        ucs_for_each_bit(md_index,
                         (ucp_ep_config(rndv_req->send.ep)->key.rma_bw_md_map &
                          mdesc->memh->md_map) | alloc_md_map) {
            freq->recv.state.dt.contig.memh[memh_index++] = mdesc->memh->uct[md_index];
            freq->recv.state.dt.contig.md_map            |= UCS_BIT(md_index);
        }
        ucs_assert(memh_index <= UCP_MAX_OP_MDS);

        frndv_req->flags             = 0;
        frndv_req->send.ep           = rndv_req->send.ep;
        frndv_req->send.rndv.mdesc   = mdesc;
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
           ((rndv_mode == UCP_RNDV_MODE_AUTO) ||
           /*  or be forced to rkey_ptr */
            (rndv_mode == UCP_RNDV_MODE_RKEY_PTR)) &&
           /* need local memory access for data unpack */
           UCP_MEM_IS_ACCESSIBLE_FROM_CPU(recv_mem_type);
}

static unsigned ucp_rndv_progress_rkey_ptr(void *arg)
{
    ucp_worker_h worker     = (ucp_worker_h)arg;
    ucp_request_t *rndv_req = ucs_queue_head_elem_non_empty(&worker->rkey_ptr_reqs,
                                                            ucp_request_t,
                                                            send.rkey_ptr.queue_elem);
    ucp_request_t *rreq     = ucp_request_get_super(rndv_req);
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
        ucp_rndv_req_send_ack(rndv_req, rndv_req->send.length,
                              rndv_req->send.rkey_ptr.remote_req_id, status,
                              UCP_AM_ID_RNDV_ATS, "send_ats");
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
        ucp_rndv_recv_req_complete(rreq, status);
        ucp_rkey_destroy(rkey);
        ucp_rndv_req_send_ack(rndv_req, rndv_req->send.length,
                              rndv_rts_hdr->sreq.req_id, status,
                              UCP_AM_ID_RNDV_ATS, "send_ats");
        return;
    }

    rreq->recv.state.offset = 0;

    ucp_trace_req(rndv_req, "obtained a local pointer to remote buffer: %p",
                  local_ptr);
    rndv_req->send.buffer                 = local_ptr;
    rndv_req->send.length                 = rndv_rts_hdr->size;
    rndv_req->send.rkey_ptr.rkey          = rkey;
    rndv_req->send.rkey_ptr.remote_req_id = rndv_rts_hdr->sreq.req_id;

    ucp_request_set_super(rndv_req, rreq);
    UCP_WORKER_STAT_RNDV(ep->worker, RKEY_PTR, 1);

    ucs_queue_push(&worker->rkey_ptr_reqs, &rndv_req->send.rkey_ptr.queue_elem);
    uct_worker_progress_register_safe(worker->uct,
                                      ucp_rndv_progress_rkey_ptr,
                                      rreq->recv.worker,
                                      UCS_CALLBACKQ_FLAG_FAST,
                                      &worker->rkey_ptr_cb_id);
}

static UCS_F_ALWAYS_INLINE void *
ucp_rndv_get_frag_rkey_ptr(ucp_worker_h worker, ucp_ep_h ep,
                           ucp_rndv_rtr_hdr_t *rtr_hdr,
                           ucs_memory_type_t local_mem_type,
                           ucp_rkey_h *rkey_p)
{
    ucp_context_h context            = worker->context;
    ucp_md_index_t rkey_ptr_md_index = UCP_NULL_RESOURCE;
    ucp_ep_peer_mem_data_t *ppln_data;
    ucp_md_map_t md_map;
    ucp_md_index_t md_index;
    const uct_md_attr_t *md_attr;
    ucp_ep_h mem_type_ep;
    ucp_lane_index_t mem_type_rma_lane;
    ucs_memory_type_t remote_mem_type;
    unsigned rkey_index;
    void *local_ptr;
    ucs_status_t status;

    if (local_mem_type == UCS_MEMORY_TYPE_UNKNOWN) {
        return NULL;
    }

    mem_type_ep = worker->mem_type_ep[local_mem_type];
    if (mem_type_ep == NULL) {
        return NULL;
    }

    remote_mem_type = ucp_rkey_packed_mem_type(rtr_hdr + 1);
    md_map          = ucp_rkey_packed_md_map(rtr_hdr + 1) &
                      ucp_ep_config(ep)->key.reachable_md_map;

    ucs_for_each_bit(md_index, md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if ((md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR) &&
            /* Do not use xpmem, because cuda_copy registration will fail and
             * performance will not be optimal. */
            !(md_attr->cap.flags & UCT_MD_FLAG_REG) &&
            (md_attr->cap.access_mem_types & UCS_BIT(remote_mem_type))) {
            rkey_ptr_md_index = md_index;
            break;
        }
    }

    if (rkey_ptr_md_index == UCP_NULL_RESOURCE) {
        return NULL;
    }

    ppln_data  = ucp_ep_peer_mem_get(context, ep, rtr_hdr->address,
                                     rtr_hdr->size, rtr_hdr + 1,
                                     rkey_ptr_md_index);
    rkey_index = ucs_bitmap2idx(ppln_data->rkey->md_map, rkey_ptr_md_index);
    status     = uct_rkey_ptr(ppln_data->rkey->tl_rkey[rkey_index].cmpt,
                              &ppln_data->rkey->tl_rkey[rkey_index].rkey,
                              rtr_hdr->address, &local_ptr);
    if (status != UCS_OK) {
        ucp_rkey_destroy(ppln_data->rkey);
        ppln_data->size = 0; /* Make sure hash element is updated next time */
        return NULL;
    }

    if (ppln_data->uct_memh != NULL) {
        goto out;
    }

    /* Register remote memory segment with memtype ep MD. Without
     * registration fetching data from GPU to CPU will be performance
     * inefficient. */
    md_map              = 0;
    mem_type_rma_lane   = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
    ppln_data->md_index = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);
    status              = ucp_mem_rereg_mds(
                           context, UCS_BIT(ppln_data->md_index), local_ptr,
                           ppln_data->size,
                           UCT_MD_MEM_ACCESS_RMA | UCT_MD_MEM_FLAG_HIDE_ERRORS,
                           NULL, remote_mem_type, NULL, &ppln_data->uct_memh,
                           &md_map);
    if (status != UCS_OK) {
        ppln_data->md_index = UCP_NULL_RESOURCE;
    } else {
        ucs_assertv(md_map == UCS_BIT(ppln_data->md_index),
                    "mdmap=0x%lx, md_index=%u", md_map,
                    ppln_data->md_index);
    }

out:
    *rkey_p = ppln_data->rkey;

    return local_ptr;
}

static UCS_F_ALWAYS_INLINE int
ucp_rndv_test_zcopy_scheme_support(size_t length,
                                   const ucp_ep_rndv_zcopy_config_t *zcopy)
{
    return /* is the current message greater than the minimal GET/PUT Zcopy? */
           (length >= zcopy->min) &&
           /* is the current message less than the maximal GET/PUT Zcopy? */
           ((length <= zcopy->max) ||
            /* or can the message be split? */ zcopy->split);
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
    int is_get_zcopy_supported;
    int is_get_zcopy_failed;
    ucp_ep_rndv_zcopy_config_t *get_zcopy;
    ucp_ep_rndv_zcopy_config_t *put_zcopy;
    ucs_memory_type_t src_mem_type;

    UCS_ASYNC_BLOCK(&worker->async);

    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_receive", 0);

    /* if receiving a message on an already closed endpoint, stop processing */
    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, rndv_rts_hdr->sreq.ep_id,
                                  { status = UCS_ERR_CANCELED; goto err; },
                                  "RNDV rts");

    /* the internal send request allocated on receiver side (to perform a "get"
     * operation, send "ATS" and "RTR") */
    rndv_req = ucp_request_get(worker);
    if (rndv_req == NULL) {
        ucs_error("failed to allocate rendezvous reply");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rndv_req->flags           = 0;
    rndv_req->send.ep         = ep;
    rndv_req->send.rndv.mdesc = NULL;
    is_get_zcopy_failed       = 0;
    src_mem_type              = UCS_MEMORY_TYPE_HOST;

    ucp_trace_req(rreq,
                  "rndv matched remote {address 0x%"PRIx64" size %zu sreq_id "
                  "0x%"PRIx64"} rndv_sreq %p", rndv_rts_hdr->address,
                  rndv_rts_hdr->size, rndv_rts_hdr->sreq.req_id, rndv_req);

    if (ucs_unlikely(rreq->recv.length < rndv_rts_hdr->size)) {
        ucp_trace_req(rndv_req,
                      "rndv truncated remote size %zu local size %zu rreq %p",
                      rndv_rts_hdr->size, rreq->recv.length, rreq);
        ucp_rndv_req_send_ack(rndv_req, rndv_rts_hdr->size,
                              rndv_rts_hdr->sreq.req_id, UCS_OK,
                              UCP_AM_ID_RNDV_ATS, "send_ats");
        ucp_request_recv_generic_dt_finish(rreq);
        ucp_rndv_zcopy_recv_req_complete(rreq, UCS_ERR_MESSAGE_TRUNCATED);
        goto out;
    }

    /* if the receive side is not connected yet then the RTS was received on a stub ep */
    ep_config = ucp_ep_config(ep);
    get_zcopy = &ep_config->rndv.get_zcopy;
    rndv_mode = worker->context->config.ext.rndv_mode;

    if (ucp_rndv_is_rkey_ptr(rndv_rts_hdr, rkey_buf, ep, rreq->recv.mem_type,
                             rndv_mode)) {
        ucp_rndv_do_rkey_ptr(rndv_req, rreq, rndv_rts_hdr, rkey_buf);
        goto out;
    }

    if (UCP_DT_IS_CONTIG(rreq->recv.datatype)) {
        is_get_zcopy_supported =
                (rndv_rts_hdr->address != 0) &&
                ucp_rndv_test_zcopy_scheme_support(rndv_rts_hdr->size,
                                                   get_zcopy);
        if (is_get_zcopy_supported) {
            /* try to fetch the data with a get_zcopy operation */
            status = ucp_rndv_req_send_rma_get(rndv_req, rreq, rndv_rts_hdr,
                                               rkey_buf);
            if (status == UCS_OK) {
                goto out;
            }

            /* fallback to non get zcopy protocol */
            is_get_zcopy_failed = 1;
            src_mem_type        = ucp_rkey_packed_mem_type(rkey_buf);
        }

        /* Check if pipelined memtype staging is needed */
        if ((rndv_mode == UCP_RNDV_MODE_AUTO) &&
            UCP_MEM_IS_GPU(rreq->recv.mem_type) &&
            ucp_rndv_is_recv_pipeline_needed(rndv_req, rndv_rts_hdr, rkey_buf,
                                             rreq->recv.mem_type,
                                             is_get_zcopy_failed)) {
            put_zcopy = &ep_config->rndv.put_zcopy;
            ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
            if (ucp_rndv_is_put_pipeline_needed(rndv_rts_hdr->address,
                                                rndv_rts_hdr->size, rkey_buf,
                                                get_zcopy, put_zcopy,
                                                is_get_zcopy_failed)) {
                /* send FRAG RTR for sender to PUT the fragment. */
                ucp_rndv_send_frag_rtr(worker, rndv_req, rreq, rndv_rts_hdr);
                goto out;
            } else if (is_get_zcopy_supported) {
                ucs_assert(is_get_zcopy_failed);

                /* sender address is present. do GET pipeline */
                status = ucp_rndv_recv_start_get_pipeline(
                        worker, rndv_req, rreq, rndv_rts_hdr->sreq.req_id,
                        rkey_buf, rndv_rts_hdr->address, rndv_rts_hdr->size, 0);
                if (status == UCS_OK) {
                    goto out;
                }
            }
        }

        if (!is_get_zcopy_failed || !UCP_MEM_IS_HOST(src_mem_type)) {
            if (rreq->flags & UCP_REQUEST_FLAG_USER_MEMH) {
                /* At this point we know the datatype is contig */
                ucp_request_init_dt_reg_from_memh(rreq,
                                                  ep_config->key.rma_bw_md_map,
                                                  rreq->recv.user_memh,
                                                  &rreq->recv.state.dt.contig);
            }

            /* register receive buffer for
             * put protocol (or) pipeline rndv for non-host memory type
             */
            ucs_assert(rndv_rts_hdr->size <= rreq->recv.length);
            ucp_request_recv_buffer_reg(rreq, ep_config->key.rma_bw_md_map,
                                        rndv_rts_hdr->size);
        }
    }

    /* The sender didn't specify its address in the RTS, or the rndv mode was
     * configured to PUT, or GET rndv mode is unsupported - send an RTR and
     * the sender will send the data with active message or put_zcopy. */
    ucp_rndv_recv_data_init(rreq, rndv_rts_hdr->size);
    ucp_rndv_req_send_rtr(rndv_req, rreq, rndv_rts_hdr->sreq.req_id,
                          rndv_rts_hdr->size, 0ul);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return;

err:
    ucp_rndv_recv_req_complete(rreq, status);
    goto out;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_rts_handler,
                 (arg, data, length, tl_flags),
                 void *arg, void *data, size_t length, unsigned tl_flags)
{
    ucp_worker_h worker         = arg;
    ucp_rndv_rts_hdr_t *rts_hdr = data;

    if (ucp_rndv_rts_is_am(rts_hdr)) {
        return ucp_am_rndv_process_rts(arg, data, length, tl_flags);
    } else {
        ucs_assert(ucp_rndv_rts_is_tag(rts_hdr));
        return ucp_tag_rndv_process_rts(worker, rts_hdr, length, tl_flags);
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_ats_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker      = arg;
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *sreq;

    if (worker->context->config.ext.proto_enable) {
        return ucp_proto_rndv_ats_handler(arg, data, length, flags);
    }

    UCP_SEND_REQUEST_GET_BY_ID(&sreq, worker, rep_hdr->req_id, 1, return UCS_OK,
                               "RNDV ATS %p", rep_hdr);

    /* dereg the original send request and set it to complete */
    UCS_PROFILE_REQUEST_EVENT(sreq, "rndv_ats_recv", 0);
    if (sreq->flags & UCP_REQUEST_FLAG_OFFLOADED) {
        ucp_tag_offload_cancel_rndv(sreq);
    }

    ucp_request_complete_and_dereg_send(sreq, rep_hdr->status);
    return UCS_OK;
}

ucs_status_t ucp_rndv_send_handle_status_from_pending(ucp_request_t *sreq,
                                                      ucs_status_t status)
{
    /* We rely on the fact that the RTS and RTR should not be sent by AM bcopy
     * multi */
    ucs_assert((status != UCP_STATUS_PENDING_SWITCH) &&
               (status != UCS_INPROGRESS));

    if (ucs_unlikely(status != UCS_OK)) {
        if (status == UCS_ERR_NO_RESOURCE) {
            return UCS_ERR_NO_RESOURCE;
        }

        ucp_ep_req_purge(sreq->send.ep, sreq, status, 0);
    }

    /* Don't release RNDV send request in case of success, since it was sent to
     * a peer as a remote request ID */
    return UCS_OK;
}

static size_t ucp_rndv_pack_data(void *dest, void *arg)
{
    ucp_request_data_hdr_t *hdr = dest;
    ucp_request_t *sreq         = arg;
    size_t length, offset;

    offset       = sreq->send.state.dt.offset;
    hdr->req_id  = sreq->send.rndv_data.remote_req_id;
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
    int single          = (sreq->send.length + sizeof(ucp_request_data_hdr_t)) <=
                          ucp_ep_config(ep)->am.max_bcopy;
    ucs_status_t status;

    if (single) {
        /* send a single bcopy message */
        status = ucp_do_am_bcopy_single(self, UCP_AM_ID_RNDV_DATA,
                                        ucp_rndv_pack_data);
        ucs_assert(status != UCS_INPROGRESS);
    } else {
        status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_RNDV_DATA,
                                       UCP_AM_ID_RNDV_DATA,
                                       ucp_rndv_pack_data,
                                       ucp_rndv_pack_data, 1);

        if (status == UCS_INPROGRESS) {
            return UCS_INPROGRESS;
        } else if (ucs_unlikely(status == UCP_STATUS_PENDING_SWITCH)) {
            return UCS_OK;
        }
    }

    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        return UCS_ERR_NO_RESOURCE;
    }

    ucp_request_complete_and_dereg_send(sreq, status);

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_put_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    uct_rkey_t uct_rkey;

    ucs_assert_always(sreq->send.rndv.lanes_count > 0);

    /* Figure out which lane to use for put operation */
    sreq->send.lane = ucp_rndv_zcopy_get_lane(sreq, &uct_rkey,
                                              UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    if (sreq->send.lane == UCP_NULL_LANE) {
        /* Unexpected behavior */
        ucs_fatal("sreq %p: unable to get PUT Zcopy lane", sreq);
    }

    return ucp_rndv_progress_rma_zcopy_common(sreq, sreq->send.lane, uct_rkey,
                                              UCP_REQUEST_SEND_PROTO_RNDV_PUT);
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
    }
}

static ucs_status_t ucp_rndv_progress_am_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_data_hdr_t hdr;

    hdr.req_id  = sreq->send.rndv_data.remote_req_id;
    hdr.offset  = 0;
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_RNDV_DATA, &hdr, sizeof(hdr),
                                  NULL, 0ul,
                                  ucp_rndv_am_zcopy_send_req_complete);
}

static ucs_status_t ucp_rndv_progress_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_data_hdr_t hdr;

    hdr.req_id  = sreq->send.rndv_data.remote_req_id;
    hdr.offset  = sreq->send.state.dt.offset;
    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_RNDV_DATA, UCP_AM_ID_RNDV_DATA,
                                 &hdr, sizeof(hdr), &hdr, sizeof(hdr), NULL,
                                 0ul, 0ul, ucp_rndv_am_zcopy_send_req_complete,
                                 1);
}


static UCS_F_ALWAYS_INLINE void
ucp_rndv_send_frag_completion_common(uct_completion_t *comp, int is_rkey_ptr)
{
    ucp_request_t *freq = ucs_container_of(comp, ucp_request_t,
                                           send.state.uct_comp);
    ucp_request_t *fsreq, *sreq;

    if (freq->send.state.dt.offset != freq->send.length) {
        return;
    }

    if (freq->send.rndv.mdesc != NULL) {
        ucs_mpool_put_inline((void*)freq->send.rndv.mdesc);
    }

    fsreq                        = ucp_request_get_super(freq);
    sreq                         = ucp_request_get_super(fsreq);
    fsreq->send.state.dt.offset += freq->send.length;
    ucs_assert(fsreq->send.state.dt.offset <= fsreq->send.length);

    /* Send ATP for last fragment of the rndv request */
    if (fsreq->send.length == fsreq->send.state.dt.offset) {
        if (!is_rkey_ptr) {
            ucp_rkey_destroy(fsreq->send.rndv.rkey);
        }

        sreq->send.state.dt.offset += fsreq->send.length;

        /* Keep a status of a send request up to date updating it by a status
         * from a request created for tracking a UCT PUT Zcopy operation */
        uct_completion_update_status(&sreq->send.state.uct_comp, comp->status);
        ucp_rndv_complete_rma_put_zcopy(sreq, 1);

        ucp_rndv_req_send_ack(fsreq, fsreq->send.length,
                              fsreq->send.rndv.remote_req_id, comp->status,
                              UCP_AM_ID_RNDV_ATP, "send_frag_atp");
    }

   if (!is_rkey_ptr) {
      /* Release registered memory during doing PUT operation for a
       * given fragment */
       ucp_request_send_buffer_dereg(freq);
   }

   ucp_request_put(freq);
}


UCS_PROFILE_FUNC_VOID(ucp_rndv_rkey_ptr_frag_completion, (self),
                      uct_completion_t *self)
{
    ucp_rndv_send_frag_completion_common(self, 1);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_send_frag_put_completion, (self),
                      uct_completion_t *self)
{
    ucp_rndv_send_frag_completion_common(self, 0);
}

UCS_PROFILE_FUNC_VOID(ucp_rndv_put_pipeline_frag_get_completion, (self),
                      uct_completion_t *self)
{
    ucp_request_t *freq  = ucs_container_of(self, ucp_request_t,
                                            send.state.uct_comp);
    ucp_request_t *fsreq = ucp_request_get_super(freq);

    /* get rkey can be NULL if memtype ep doesn't need RKEY */
    if (freq->send.rndv.rkey != NULL) {
        ucp_rkey_destroy(freq->send.rndv.rkey);
    }

    /* get completed on memtype endpoint to stage on host. send put request to receiver*/
    ucp_request_send_state_reset(freq, ucp_rndv_send_frag_put_completion,
                                 UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    ucp_rndv_req_init_remote_from_super_req(freq, fsreq,
                                            freq->send.rndv.remote_address -
                                            (uint64_t)fsreq->send.buffer);

    freq->send.uct.func                  = ucp_rndv_progress_rma_put_zcopy;
    freq->send.lane                      = fsreq->send.lane;
    freq->send.state.dt.dt.contig.md_map = 0;

    ucp_request_send(freq);
}

static UCS_F_ALWAYS_INLINE ucp_request_t*
ucp_rndv_frag_req_get(ucp_worker_h worker, ucp_request_t *super_req,
                      size_t length, size_t send_buffer_offset,
                      size_t remote_address_offset,
                      ucs_ptr_map_key_t remote_req_id)
{
    ucp_request_t *freq;

    freq = ucp_request_get(worker);
    if (freq == NULL) {
        ucs_fatal("failed to allocate rndv fragment request");
    }

    ucp_request_send_state_init(freq, ucp_dt_make_contig(1), 0);
    ucp_rndv_req_init_from_super_req(freq, super_req, length,
                                     send_buffer_offset, remote_address_offset,
                                     remote_req_id);
    freq->send.mem_type = super_req->send.mem_type;

    return freq;
}

static ucs_status_t
ucp_rndv_send_start_peer_seg_pipeline(ucp_worker_h worker, ucp_request_t *sreq,
                                      ucp_rndv_rtr_hdr_t *rndv_rtr_hdr,
                                      size_t rndv_size)
{
    ucp_request_t *fsreq;
    ucp_ep_config_t *memtype_ep_config;
    void *remote_frag_rkey_ptr;
    size_t offset, length;

    if (!worker->context->config.ext.rndv_shm_ppln_enable) {
        return UCS_ERR_UNSUPPORTED;
    }

    remote_frag_rkey_ptr = ucp_rndv_get_frag_rkey_ptr(
                             worker, sreq->send.ep, rndv_rtr_hdr,
                             sreq->send.mem_type, &sreq->send.rndv.rkey);
    if (remote_frag_rkey_ptr == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }
    ucs_assert(!UCP_MEM_IS_HOST(sreq->send.mem_type));

    if (rndv_rtr_hdr->offset == 0) {
        ucp_request_send_state_reset(sreq, NULL,
                                     UCP_REQUEST_SEND_PROTO_RNDV_PUT);
    }

    /* Internal send request allocated on sender side to handle send
     * fragments for RTR */
    fsreq = ucp_rndv_frag_req_get(worker, sreq, rndv_size, rndv_rtr_hdr->offset,
                                  0, rndv_rtr_hdr->rreq_id);
    fsreq->send.state.dt.offset = 0;

    memtype_ep_config = ucp_ep_config(worker->mem_type_ep[sreq->send.mem_type]);

    for (offset = 0; offset < rndv_size; offset += length) {
        length = ucp_rndv_adjust_zcopy_length(
                     memtype_ep_config->rndv.get_zcopy.min,
                     memtype_ep_config->rndv.get_zcopy.max, 0, rndv_size,
                     offset, rndv_size - offset);
        ucp_rndv_rkey_ptr_get_mem_type(
                fsreq, length,
                (uint64_t)UCS_PTR_BYTE_OFFSET(fsreq->send.buffer, offset),
                UCS_PTR_BYTE_OFFSET(remote_frag_rkey_ptr, offset),
                fsreq->send.mem_type, ucp_rndv_rkey_ptr_frag_completion);
    }

    return UCS_OK;
}

static ucs_status_t ucp_rndv_send_start_put_pipeline(ucp_request_t *sreq,
                                                     ucp_rndv_rtr_hdr_t *rndv_rtr_hdr)
{
    ucp_ep_h ep                     = sreq->send.ep;
    ucp_ep_config_t *config         = ucp_ep_config(ep);
    ucp_worker_h worker             = sreq->send.ep->worker;
    ucp_context_h context           = worker->context;
    size_t rndv_base_offset         = rndv_rtr_hdr->offset;
    ucs_memory_type_t frag_mem_type = context->config.ext.rndv_frag_mem_type;
    size_t rndv_size                = ucs_min(rndv_rtr_hdr->size,
                                              sreq->send.length);
    const uct_md_attr_t *md_attr;
    ucp_request_t *freq;
    ucp_request_t *fsreq;
    size_t max_frag_size, length;
    size_t offset;
    size_t min_zcopy, max_zcopy;
    uct_rkey_t uct_rkey;
    ucs_status_t status;

    sreq->send.rndv.remote_address = rndv_rtr_hdr->address;

    status = ucp_rndv_send_start_peer_seg_pipeline(worker, sreq, rndv_rtr_hdr,
                                                   rndv_size);
    if (status == UCS_OK) {
        goto out;
    }

    ucp_trace_req(sreq, "using put rndv pipeline protocol: va %p size %zu",
                  (void*)rndv_rtr_hdr->address, rndv_rtr_hdr->size);

    status = ucp_ep_rkey_unpack(ep, rndv_rtr_hdr + 1, &sreq->send.rndv.rkey);
    if (status != UCS_OK) {
        ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                  ucp_ep_peer_name(ep), ucs_status_string(status));
    }

    if (rndv_base_offset == 0) {
        ucp_request_send_state_reset(sreq, NULL,
                                     UCP_REQUEST_SEND_PROTO_RNDV_PUT);
        ucp_rndv_req_init_zcopy_lane_map(
                sreq, sreq->send.rndv.rkey->mem_type, rndv_size,
                UCP_REQUEST_SEND_PROTO_RNDV_PUT);

        /* Check if lane could be allocated */
        sreq->send.lane = ucp_rndv_zcopy_get_lane(
                          sreq, &uct_rkey, UCP_REQUEST_SEND_PROTO_RNDV_PUT);
        if (sreq->send.lane == UCP_NULL_LANE) {
            goto err_unsupported;
        }

        /* Check if lane supports bounce buffer memory, to stage sends
         * through it */
        md_attr = ucp_ep_md_attr(sreq->send.ep, sreq->send.lane);
        if (!(md_attr->cap.reg_mem_types & UCS_BIT(frag_mem_type))) {
            goto err_unsupported;
        }

        /* Check if mem type endpoint is exists */
        if (!UCP_MEM_IS_HOST(sreq->send.mem_type) &&
            (worker->mem_type_ep[sreq->send.mem_type] == NULL)) {
            goto err_unsupported;
        }

    }

    /* Internal send request allocated on sender side to handle send fragments
     * for RTR */
    fsreq = ucp_rndv_frag_req_get(worker, sreq, rndv_size, rndv_base_offset, 0,
                                  rndv_rtr_hdr->rreq_id);
    fsreq->send.state.dt.offset = 0;

    min_zcopy     = config->rndv.put_zcopy.min;
    max_zcopy     = config->rndv.put_zcopy.max;
    max_frag_size = ucs_min(
                     context->config.ext.rndv_frag_size[frag_mem_type],
                     max_zcopy);
    offset        = 0;

    while (offset != rndv_size) {
        length = ucp_rndv_adjust_zcopy_length(min_zcopy, max_frag_size, 0,
                                              rndv_size, offset,
                                              rndv_size - offset);

        if (UCP_MEM_IS_HOST(sreq->send.mem_type)) {
            /* sbuf is in host, directly do put */
            freq = ucp_rndv_frag_req_get(worker, fsreq, length, offset,offset,
                                         UCS_PTR_MAP_KEY_INVALID);
            ucp_request_send_state_reset(freq, ucp_rndv_send_frag_put_completion,
                                         UCP_REQUEST_SEND_PROTO_RNDV_PUT);

            freq->send.datatype     = ucp_dt_make_contig(1);
            freq->send.uct.func     = ucp_rndv_progress_rma_put_zcopy;
            freq->send.rndv.mdesc   = NULL;
            freq->send.pending_lane = UCP_NULL_LANE;

            ucp_request_send(freq);
        } else {
            /* Protocol:
             * Step 1: GET fragment from send buffer to HOST fragment buffer
             * Step 2: PUT from fragment HOST buffer to remote HOST fragment buffer
             * Step 3: send ATP for each fragment request
             */
            ucp_rndv_send_frag_get_mem_type(
                    fsreq, length,
                    (uint64_t)UCS_PTR_BYTE_OFFSET(fsreq->send.buffer,
                                                  offset),
                    fsreq->send.mem_type, NULL, NULL, UCS_BIT(0), 1,
                    ucp_rndv_put_pipeline_frag_get_completion);
        }

        offset += length;
    }

    return UCS_OK;

err_unsupported:
    ucp_rkey_destroy(sreq->send.rndv.rkey);
    status = UCS_ERR_UNSUPPORTED;
out:
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_progress_rma_get_zcopy, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    uct_rkey_t uct_rkey;

    req->send.lane = ucp_rndv_zcopy_get_lane(req, &uct_rkey,
                                             UCP_REQUEST_SEND_PROTO_RNDV_GET);
    ucs_assert_always(req->send.lane != UCP_NULL_LANE);

    return ucp_rndv_progress_rma_zcopy_common(req, req->send.lane, uct_rkey,
                                              UCP_REQUEST_SEND_PROTO_RNDV_GET);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_atp_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker      = arg;
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *rtr_sreq, *req;
    ucp_mem_desc_t *mdesc;

    if (worker->context->config.ext.proto_enable) {
        return ucp_proto_rndv_rtr_handle_atp(arg, data, length, flags);
    }

    UCP_SEND_REQUEST_GET_BY_ID(&rtr_sreq, worker, rep_hdr->req_id, 1,
                               return UCS_OK, "RNDV ATP %p", rep_hdr);

    req   = ucp_request_get_super(rtr_sreq);
    mdesc = rtr_sreq->send.rndv.mdesc;
    ucs_assert(req != NULL);
    ucp_request_put(rtr_sreq);

    VALGRIND_MAKE_MEM_DEFINED(req->recv.buffer, req->recv.length);
    if (req->flags & UCP_REQUEST_FLAG_RNDV_FRAG) {
        /* received ATP for frag RTR request */
        UCS_PROFILE_REQUEST_EVENT(req, "rndv_frag_atp_recv", 0);
        ucp_rndv_recv_frag_put_mem_type(ucp_request_get_super(req), req, mdesc,
                                        req->recv.length,
                                        req->recv.frag.offset);
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
    ucp_worker_h worker              = arg;
    ucp_context_h context            = worker->context;
    ucp_rndv_rtr_hdr_t *rndv_rtr_hdr = data;
    ucp_ep_rndv_zcopy_config_t *put_zcopy;
    ucp_request_t *sreq;
    ucp_ep_h ep;
    ucp_ep_config_t *ep_config;
    ucs_status_t status;
    int is_put_pipeline;
    int is_put_supported;
    uct_rkey_t uct_rkey;

    if (context->config.ext.proto_enable) {
        return ucp_proto_rndv_handle_rtr(arg, data, length, flags);
    }

    UCP_SEND_REQUEST_GET_BY_ID(&sreq, arg, rndv_rtr_hdr->sreq_id, 0,
                               return UCS_OK, "RNDV RTR %p", rndv_rtr_hdr);
    ep        = sreq->send.ep;
    ep_config = ucp_ep_config(ep);
    put_zcopy = &ep_config->rndv.put_zcopy;

    ucp_trace_req(sreq, "received rtr address 0x%"PRIx64" remote rreq_id"
                  "0x%"PRIx64, rndv_rtr_hdr->address, rndv_rtr_hdr->rreq_id);
    UCS_PROFILE_REQUEST_EVENT(sreq, "rndv_rtr_recv", 0);

    if (sreq->flags & UCP_REQUEST_FLAG_OFFLOADED) {
        /* Do not deregister memory here, because am zcopy rndv may
         * need it registered (if am and tag is the same lane). */
        ucp_tag_offload_cancel_rndv(sreq);
        ucs_assert(!ucp_ep_use_indirect_id(ep));
    }

    if (UCP_DT_IS_CONTIG(sreq->send.datatype) && rndv_rtr_hdr->address) {
        is_put_supported = ucp_rndv_test_zcopy_scheme_support(sreq->send.length,
                                                              put_zcopy);
        is_put_pipeline = ((!UCP_MEM_IS_HOST(sreq->send.mem_type) ||
                            (sreq->send.length != rndv_rtr_hdr->size)) &&
                           (context->config.ext.rndv_mode != UCP_RNDV_MODE_PUT_ZCOPY)) &&
                          is_put_supported;

        /*
         * Try pipeline protocol for non-host memory, if PUT_ZCOPY protocol is
         * not explicitly required. If pipeline is UNSUPPORTED, fallback to
         * PUT_ZCOPY anyway.
         */
        if (is_put_pipeline) {
            status = ucp_rndv_send_start_put_pipeline(sreq, rndv_rtr_hdr);
            if (status != UCS_ERR_UNSUPPORTED) {
                return status;
            }
            /* If we get here, it means that RNDV pipeline protocol is unsupported
             * and we have to use PUT_ZCOPY RNDV scheme instead */
        }

        status = ucp_ep_rkey_unpack(ep, rndv_rtr_hdr + 1,
                                    &sreq->send.rndv.rkey);
        if (status != UCS_OK) {
            ucs_fatal("failed to unpack rendezvous remote key received from %s: %s",
                      ucp_ep_peer_name(ep), ucs_status_string(status));
        }

        if ((context->config.ext.rndv_mode != UCP_RNDV_MODE_GET_ZCOPY) &&
            is_put_supported) {
            ucp_request_send_state_reset(sreq, ucp_rndv_put_completion,
                                         UCP_REQUEST_SEND_PROTO_RNDV_PUT);
            sreq->send.uct.func            = ucp_rndv_progress_rma_put_zcopy;
            sreq->send.rndv.remote_req_id  = rndv_rtr_hdr->rreq_id;
            sreq->send.rndv.remote_address = rndv_rtr_hdr->address;
            sreq->send.rndv.mdesc          = NULL;
            sreq->send.pending_lane        = UCP_NULL_LANE;

            ucp_rndv_req_init_zcopy_lane_map(sreq, sreq->send.mem_type,
                                             sreq->send.length,
                                             UCP_REQUEST_SEND_PROTO_RNDV_PUT);

            sreq->send.lane =
                ucp_rndv_zcopy_get_lane(sreq, &uct_rkey,
                                        UCP_REQUEST_SEND_PROTO_RNDV_PUT);
            if (sreq->send.lane != UCP_NULL_LANE) {
                goto out_send;
            }
        }

        ucp_rkey_destroy(sreq->send.rndv.rkey);
    }

    ucp_trace_req(sreq, "using rdnv_data protocol");

    /* switch to AM */
    if (UCP_DT_IS_CONTIG(sreq->send.datatype) &&
        (sreq->send.length >=
         ep_config->am.mem_type_zcopy_thresh[sreq->send.mem_type]))
    {
        status = ucp_request_send_buffer_reg_lane(sreq, ucp_ep_get_am_lane(ep), 0);
        ucs_assert_always(status == UCS_OK);

        ucp_request_send_state_reset(sreq, ucp_rndv_am_zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_ZCOPY_AM);

        if ((sreq->send.length + sizeof(ucp_request_data_hdr_t)) <=
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

    sreq->send.rndv_data.remote_req_id = rndv_rtr_hdr->rreq_id;

out_send:
    /* if it is not a PUT pipeline protocol, delete the send request ID */
    ucp_send_request_id_release(sreq);
    ucp_request_send(sreq);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rndv_data_handler,
                 (arg, data, length, flags),
                 void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker                   = arg;
    ucp_request_data_hdr_t *rndv_data_hdr = data;
    ucp_request_t *rreq, *rndv_req;
    size_t recv_len;
    ucs_status_t status;

    if (worker->context->config.ext.proto_enable) {
        return ucp_proto_rndv_handle_data(arg, data, length, flags);
    }

    UCP_SEND_REQUEST_GET_BY_ID(&rndv_req, worker, rndv_data_hdr->req_id, 0,
                               return UCS_OK, "RNDV data %p", rndv_data_hdr);

    rreq = ucp_request_get_super(rndv_req);
    ucs_assert(rreq != NULL);
    ucs_assert(!(rreq->flags & UCP_REQUEST_FLAG_RNDV_FRAG));
    ucs_assert(rreq->flags &
               (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG));

    recv_len = length - sizeof(*rndv_data_hdr);
    UCS_PROFILE_REQUEST_EVENT(rreq, "rndv_data_recv", recv_len);

    status = ucp_request_process_recv_data(rreq, rndv_data_hdr + 1, recv_len,
                                           rndv_data_hdr->offset, 1,
                                           rreq->flags &
                                                   UCP_REQUEST_FLAG_RECV_AM);
    if (status != UCS_INPROGRESS) {
        ucp_send_request_id_release(rndv_req);
        ucp_request_put(rndv_req);
    }

    return UCS_OK;
}

static void ucp_rndv_dump_rkey(const void *rkey_buf, const void *rkey_end,
                               ucs_string_buffer_t *strb)
{
    ucs_string_buffer_appendf(strb, " rkey ");
    ucp_rkey_dump_packed(rkey_buf, UCS_PTR_BYTE_DIFF(rkey_buf, rkey_end), strb);
}

static void ucp_rndv_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                          uint8_t id, const void *data, size_t length,
                          char *buffer, size_t max)
{
    UCS_STRING_BUFFER_FIXED(strb, buffer, max);
    const ucp_rndv_rts_hdr_t *rndv_rts_hdr    = data;
    const ucp_rndv_rtr_hdr_t *rndv_rtr_hdr    = data;
    const ucp_request_data_hdr_t *rndv_data   = data;
    const ucp_rndv_ack_hdr_t *ack_hdr         = data;
    const ucp_reply_hdr_t *rep_hdr            = data;
    const void *data_end                      = UCS_PTR_BYTE_OFFSET(data, length);
    const void *rkey_buf;

    switch (id) {
    case UCP_AM_ID_RNDV_RTS:
        ucs_string_buffer_appendf(&strb, "RNDV_RTS ");
        if (ucp_rndv_rts_is_am(rndv_rts_hdr)) {
            ucs_string_buffer_appendf(&strb, "am_id %u",
                                      ucp_am_hdr_from_rts(rndv_rts_hdr)->am_id);
        } else {
            ucs_assert(ucp_rndv_rts_is_tag(rndv_rts_hdr));
            ucs_string_buffer_appendf(&strb, "tag %" PRIx64,
                                      ucp_tag_hdr_from_rts(rndv_rts_hdr)->tag);
        }

        rkey_buf = rndv_rts_hdr + 1;
        ucs_string_buffer_appendf(&strb,
                                  " ep_id 0x%" PRIx64 " sreq_id 0x%" PRIx64
                                  " address 0x%" PRIx64 " size %zu",
                                  rndv_rts_hdr->sreq.ep_id,
                                  rndv_rts_hdr->sreq.req_id,
                                  rndv_rts_hdr->address, rndv_rts_hdr->size);

        if (rndv_rts_hdr->address != 0) {
            ucp_rndv_dump_rkey(rkey_buf, data_end, &strb);
        }
        break;
    case UCP_AM_ID_RNDV_ATS:
        ucs_string_buffer_appendf(&strb,
                                  "RNDV_ATS sreq_id 0x%" PRIx64 " status '%s'",
                                  rep_hdr->req_id,
                                  ucs_status_string(rep_hdr->status));
        if (length >= sizeof(*ack_hdr)) {
            ucs_string_buffer_appendf(&strb, " size %zu", ack_hdr->size);
        }
        break;
    case UCP_AM_ID_RNDV_RTR:
        ucs_string_buffer_appendf(&strb,
                                  "RNDV_RTR sreq_id 0x%" PRIx64
                                  " rreq_id 0x%" PRIx64 " address 0x%" PRIx64
                                  " size %zu offset %zu",
                                  rndv_rtr_hdr->sreq_id, rndv_rtr_hdr->rreq_id,
                                  rndv_rtr_hdr->address, rndv_rtr_hdr->size,
                                  rndv_rtr_hdr->offset);
        if (rndv_rtr_hdr->address != 0) {
            ucp_rndv_dump_rkey(rndv_rtr_hdr + 1, data_end, &strb);
        }
        break;
    case UCP_AM_ID_RNDV_DATA:
        ucs_string_buffer_appendf(&strb,
                                  "RNDV_DATA rreq_id 0x%" PRIx64 " offset %zu",
                                  rndv_data->req_id, rndv_data->offset);
        break;
    case UCP_AM_ID_RNDV_ATP:
        ucs_string_buffer_appendf(&strb,
                                  "RNDV_ATP sreq_id 0x%" PRIx64 " status '%s'",
                                  rep_hdr->req_id,
                                  ucs_status_string(rep_hdr->status));
        if (length >= sizeof(*ack_hdr)) {
            ucs_string_buffer_appendf(&strb, " size %zu", ack_hdr->size);
        }
        break;
    default:
        return;
    }
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_RTS,
                         ucp_rndv_rts_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_ATS,
                         ucp_rndv_ats_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_ATP,
                         ucp_rndv_atp_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_RTR,
                         ucp_rndv_rtr_handler, ucp_rndv_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG | UCP_FEATURE_AM, UCP_AM_ID_RNDV_DATA,
                         ucp_rndv_data_handler, ucp_rndv_dump, 0);
