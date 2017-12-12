/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tag_match.h"
#include "eager.h"
#include "rndv.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/proto/proto_am.inl>
#include <ucs/datastruct/mpool.inl>
#include <string.h>


static UCS_F_ALWAYS_INLINE size_t
ucp_tag_get_rndv_threshold(const ucp_request_t *req, size_t count,
                           size_t max_iov, size_t rndv_rma_thresh,
                           size_t rndv_am_thresh, size_t seg_size)
{
    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_IOV: 
        if ((count > max_iov) &&
            ucp_ep_is_tag_offload_enabled(ucp_ep_config(req->send.ep))) {
            /* Make sure SW RNDV will be used, because tag offload does
             * not support multi-packet eager protocols. */
            return seg_size;
        }
        /* Fall through */
    case UCP_DATATYPE_CONTIG: 
        return ucs_min(rndv_rma_thresh, rndv_am_thresh);
    case UCP_DATATYPE_GENERIC:
        return rndv_am_thresh;
    default:
        ucs_error("Invalid data type %lx", req->send.datatype);
    }
 
    return SIZE_MAX;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_tag_send_req(ucp_request_t *req, size_t count,
                 const ucp_ep_msg_config_t* msg_config,
                 size_t rndv_rma_thresh, size_t rndv_am_thresh,
                 ucp_send_callback_t cb, const ucp_proto_t *proto)
{
    size_t seg_size     = (msg_config->max_bcopy - proto->only_hdr_size);
    size_t rndv_thresh  = ucp_tag_get_rndv_threshold(req, count,
                                                     msg_config->max_iov,
                                                     rndv_rma_thresh,
                                                     rndv_am_thresh, seg_size);
    size_t zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config, count,
                                                        rndv_thresh);
    ssize_t max_short   = ucp_proto_get_short_max(req, msg_config);
    ucs_status_t status;

    ucs_trace_req("select tag request(%p) progress algorithm datatype=%lx "
                  "buffer=%p length=%zu max_short=%zd rndv_thresh=%zu "
                  "zcopy_thresh=%zu",
                  req, req->send.datatype, req->send.buffer, req->send.length,
                  max_short, rndv_thresh, zcopy_thresh);

    status = ucp_request_send_start(req, max_short, zcopy_thresh, seg_size,
                                    rndv_thresh, proto);
    if (ucs_unlikely(status != UCS_OK)) {
        if (status == UCS_ERR_NO_PROGRESS) {
             ucs_assert(req->send.length >= rndv_thresh);
            /* RMA/AM rendezvous */
            status = ucp_tag_send_start_rndv(req);
        }
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
    }

    ucp_request_send_tag_stat(req);

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_send(req);
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

static UCS_F_ALWAYS_INLINE void
ucp_tag_send_req_init(ucp_request_t* req, ucp_ep_h ep,
                                  const void* buffer, uintptr_t datatype,
                                  size_t count, ucp_tag_t tag, uint16_t flags)
{
    req->flags             = flags;
    req->send.ep           = ep;
    req->send.buffer       = buffer;
    req->send.datatype     = datatype;
    req->send.tag          = tag;
    req->send.reg_rsc      = UCP_NULL_RESOURCE;
    ucp_request_send_state_init(req, count);
    req->send.length       = ucp_dt_length(req->send.datatype, count,
                                           req->send.buffer,
                                           &req->send.state.dt);
    req->send.lane         = ucp_ep_config(ep)->tag.lane;
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
        if (ucs_likely((ssize_t)length <= ucp_ep_config(ep)->tag.eager.max_short)) {
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

    ucp_tag_send_req_init(req, ep, buffer, datatype, count, tag, 0);

    ret = ucp_tag_send_req(req, count, &ucp_ep_config(ep)->tag.eager,
                           ucp_ep_config(ep)->tag.rndv.rma_thresh,
                           ucp_ep_config(ep)->tag.rndv.am_thresh,
                           cb, ucp_ep_config(ep)->tag.proto);
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

    if (ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_PEER) {
        ret = UCS_STATUS_PTR(UCS_ERR_UNSUPPORTED);
        goto out;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    /* Remote side needs to send reply, so have it connect to us */
    ucp_ep_connect_remote(ep);

    ucp_tag_send_req_init(req, ep, buffer, datatype, count, tag,
                          UCP_REQUEST_FLAG_SYNC);

    ret = ucp_tag_send_req(req, count, &ucp_ep_config(ep)->tag.eager,
                           ucp_ep_config(ep)->tag.rndv.rma_thresh,
                           ucp_ep_config(ep)->tag.rndv.am_thresh,
                           cb, ucp_ep_config(ep)->tag.sync_proto);
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
    req->send.proto.comp_cb        = ucp_request_put;
    ucp_request_send(req);
}
