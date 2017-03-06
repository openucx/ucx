/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "match.h"
#include "eager.h"
#include "rndv.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/mpool.inl>
#include <string.h>


static ucs_status_t ucp_tag_req_start(ucp_request_t *req, size_t count,
                                      ssize_t max_short, size_t *zcopy_thresh_arr,
                                      size_t rndv_rma_thresh,
                                      size_t rndv_am_thresh,
                                      const ucp_proto_t *proto)
{
    ucp_ep_config_t *config   = ucp_ep_config(req->send.ep);
    ucp_lane_index_t lane     = ucp_ep_get_am_lane(req->send.ep);
    ucp_worker_h worker       = req->send.ep->worker;
    size_t only_hdr_size      = proto->only_hdr_size;
    unsigned flag_iov_single  = 1;
    ucp_rsc_index_t rsc_index;
    unsigned is_iov;
    size_t zcopy_thresh;
    ucs_status_t status;
    size_t length;

    is_iov = UCP_DT_IS_IOV(req->send.datatype);
    if (ucs_unlikely(is_iov)) {
        length = ucp_dt_length(req->send.datatype, count, req->send.buffer,
                               &req->send.state);
        req->send.state.dt.iov.iovcnt_offset = 0;
        req->send.state.dt.iov.iov_offset    = 0;
        req->send.state.dt.iov.iovcnt        = count;
        flag_iov_single                      = (count <= config->am.max_iovcnt);
        if (0 == count) {
            /* disable zcopy */
            zcopy_thresh = SIZE_MAX;
        } else if (!config->am.zcopy_auto_thresh) {
            /* The user defined threshold or no zcopy enabled */
            zcopy_thresh = zcopy_thresh_arr[0];
        } else if (count <= UCP_MAX_IOV) {
            /* Using pre-calculated thresholds */
            zcopy_thresh = zcopy_thresh_arr[count - 1];
        } else {
            /* Calculate threshold */
            rsc_index    = config->key.lanes[lane].rsc_index;
            zcopy_thresh = ucp_ep_config_get_zcopy_auto_thresh(count,
                               &ucp_ep_md_attr(req->send.ep, lane)->reg_cost,
                               worker->context,
                               worker->iface_attrs[rsc_index].bandwidth);
        }
    } else {
        length       = ucp_contig_dt_length(req->send.datatype, count);
        zcopy_thresh = count ? zcopy_thresh_arr[0] : SIZE_MAX;
    }
    req->send.length = length;

    ucs_trace_req("select request(%p) progress algorithm datatype=%lx buffer=%p "
                  " length=%zu max_short=%zd rndv_rma_thresh=%zu ndv_rma_thresh=%zu "
                  "zcopy_thresh=%zu",
                  req, req->send.datatype, req->send.buffer, length, max_short,
                  rndv_rma_thresh, rndv_am_thresh, zcopy_thresh);

    if (((ssize_t)length <= max_short) && !is_iov) {
        /* short */
        req->send.uct.func = proto->contig_short;
        UCS_PROFILE_REQUEST_EVENT(req, "start_contig_short", req->send.length);
    } else if ((((config->key.rndv_lane != UCP_NULL_RESOURCE) &&
               (length >= rndv_rma_thresh)) ||
               (length >= rndv_am_thresh)) && !is_iov) {
        /* RMA/AM rendezvous */
        ucp_tag_send_start_rndv(req);
        UCS_PROFILE_REQUEST_EVENT(req, "start_rndv", req->send.length);
    } else if (length < zcopy_thresh) {
        /* bcopy */
        if (length <= (config->am.max_bcopy - only_hdr_size)) {
            req->send.uct.func   = proto->bcopy_single;
            UCS_PROFILE_REQUEST_EVENT(req, "start_egr_bcopy_single", req->send.length);
        } else {
            req->send.uct.func   = proto->bcopy_multi;
            UCS_PROFILE_REQUEST_EVENT(req, "start_egr_bcopy_multi", req->send.length);
        }
    } else {
        /* eager zcopy */
        status = ucp_request_send_buffer_reg(req, lane);
        if (status != UCS_OK) {
            return status;
        }

        req->send.uct_comp.func  = proto->zcopy_completion;
        req->send.uct_comp.count = 1;

        if ((length <= (config->am.max_zcopy - only_hdr_size)) &&
            flag_iov_single) {
            req->send.uct.func   = proto->zcopy_single;
            UCS_PROFILE_REQUEST_EVENT(req, "start_egr_zcopy_single", req->send.length);
        } else {
            req->send.uct.func   = proto->zcopy_multi;
            UCS_PROFILE_REQUEST_EVENT(req, "start_egr_zcopy_multi", req->send.length);
        }
    }
    return UCS_OK;
}

static void ucp_tag_req_start_generic(ucp_request_t *req, size_t count,
                                      size_t rndv_rma_thresh, size_t rndv_am_thresh,
                                      const ucp_proto_t *proto)
{
    ucp_ep_config_t *config = ucp_ep_config(req->send.ep);
    ucp_dt_generic_t *dt_gen;
    size_t length;
    void *state;

    dt_gen = ucp_dt_generic(req->send.datatype);
    state = dt_gen->ops.start_pack(dt_gen->context, req->send.buffer, count);

    req->send.state.dt.generic.state = state;
    req->send.length = length = dt_gen->ops.packed_size(state);

    if (length >= rndv_am_thresh) {
        /* rendezvous */
        ucp_tag_send_start_rndv(req);
        UCS_PROFILE_REQUEST_EVENT(req, "start_rndv", req->send.length);
    } else if (length <= config->am.max_bcopy - proto->only_hdr_size) {
        /* bcopy single */
        req->send.uct.func = proto->bcopy_single;
        UCS_PROFILE_REQUEST_EVENT(req, "start_gen_bcopy_single", req->send.length);
    } else {
        /* bcopy multi */
        req->send.uct.func = proto->bcopy_multi;
        UCS_PROFILE_REQUEST_EVENT(req, "start_gen_bcopy_multi", req->send.length);
    }
}

static void ucp_send_req_stat(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_RNDV) {
        UCP_EP_STAT_TAG_OP(req->send.ep, RNDV);
    } else if (req->flags & UCP_REQUEST_FLAG_SYNC) {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER_SYNC);
    } else {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER);
    }
}

static inline ucs_status_ptr_t
ucp_tag_send_req(ucp_request_t *req, size_t count, ssize_t max_short,
                 size_t *zcopy_thresh, size_t rndv_rma_thresh, size_t rndv_am_thresh,
                 ucp_send_callback_t cb, const ucp_proto_t *proto)
{
    ucs_status_t status;

    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
    case UCP_DATATYPE_IOV:
        status = ucp_tag_req_start(req, count, max_short, zcopy_thresh,
                                   rndv_rma_thresh, rndv_am_thresh, proto);
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
        break;

    case UCP_DATATYPE_GENERIC:
        ucp_tag_req_start_generic(req, count, rndv_rma_thresh,
                                  rndv_am_thresh, proto);
        break;

    default:
        ucs_error("Invalid data type");
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    ucp_send_req_stat(req);

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_start_send(req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucp_request_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucp_request_set_callback(req, send.cb, cb)
    ucs_trace_req("returning send request %p", req);
    return req + 1;
}

static void ucp_tag_send_req_init(ucp_request_t* req, ucp_ep_h ep,
                                  const void* buffer, uintptr_t datatype,
                                  ucp_tag_t tag, uint16_t flags)
{
    req->flags             = flags;
    req->send.ep           = ep;
    req->send.buffer       = buffer;
    req->send.datatype     = datatype;
    req->send.tag          = tag;
    req->send.state.offset = 0;
#if ENABLE_ASSERT
    req->send.lane         = UCP_NULL_LANE;
#endif
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_nb,
                 (ep, buffer, count, datatype, tag, cb),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucs_status_t status;
    ucp_request_t *req;
    size_t length;
    ucs_status_ptr_t ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    ucs_trace_req("send_nb buffer %p count %zu tag %"PRIx64" to %s cb %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), cb);

    if (ucs_likely(UCP_DT_IS_CONTIG(datatype))) {
        length = ucp_contig_dt_length(datatype, count);
        if (ucs_likely((ssize_t)length <= ucp_ep_config(ep)->am.max_eager_short)) {
            status = UCS_PROFILE_CALL(ucp_tag_send_eager_short, ep, tag, buffer,
                                      length);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                UCP_EP_STAT_TAG_OP(ep, EAGER);
                ret = UCS_STATUS_PTR(status); /* UCS_OK also goes here */
                goto out;
            }
        }
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ucp_tag_send_req_init(req, ep, buffer, datatype, tag, 0);

    ret = ucp_tag_send_req(req, count,
                           ucp_ep_config(ep)->am.max_eager_short,
                           ucp_ep_config(ep)->am.zcopy_thresh,
                           ucp_ep_config(ep)->rndv.rma_thresh,
                           ucp_ep_config(ep)->rndv.am_thresh,
                           cb, &ucp_tag_eager_proto);
out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_sync_nb,
                 (ep, buffer, count, datatype, tag, cb),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucp_request_t *req;
    ucs_status_ptr_t ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    ucs_trace_req("send_sync_nb buffer %p count %zu tag %"PRIx64" to %s cb %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), cb);

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    /* Remote side needs to send reply, so have it connect to us */
    ucp_ep_connect_remote(ep);

    ucp_tag_send_req_init(req, ep, buffer, datatype, tag, UCP_REQUEST_FLAG_SYNC);

    ret = ucp_tag_send_req(req, count,
                           -1, /* disable short method */
                           ucp_ep_config(ep)->am.sync_zcopy_thresh,
                           ucp_ep_config(ep)->rndv.rma_thresh,
                           ucp_ep_config(ep)->rndv.am_thresh,
                           cb, &ucp_tag_eager_sync_proto);
out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return ret;
}

void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, uint64_t sender_uuid,
                                 uintptr_t remote_request)
{
    ucp_request_t *req;

    ucs_trace_req("send_sync_ack sender_uuid %"PRIx64" remote_request 0x%lx",
                  sender_uuid, remote_request);

    req = ucp_worker_allocate_reply(worker, sender_uuid);
    req->send.uct.func             = ucp_proto_progress_am_bcopy_single;
    req->send.proto.am_id          = UCP_AM_ID_EAGER_SYNC_ACK;
    req->send.proto.remote_request = remote_request;
    req->send.proto.status         = UCS_OK;
    ucp_request_start_send(req);
}

/**
 * Common function to copy data with specified offset from particular datatype
 * to the descriptor
 */
size_t ucp_tag_pack_dt_copy(void *dest, const void *src, ucp_frag_state_t *state,
                                size_t length, ucp_datatype_t datatype)
{
    ucp_dt_generic_t *dt;
    size_t result_len = 0;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_CALL(memcpy, dest, src + state->offset, length);
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, dest, src, length,
                              &state->dt.iov.iov_offset,
                              &state->dt.iov.iovcnt_offset);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt = ucp_dt_generic(datatype);
        result_len = UCS_PROFILE_NAMED_CALL("dt_pack", dt->ops.pack,
                                            state->dt.generic.state,
                                            state->offset, dest, length);
        break;

    default:
        ucs_error("Invalid data type");
    }

    state->offset += result_len;
    return result_len;
}

