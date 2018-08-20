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
                           size_t rndv_am_thresh)
{
    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_IOV:
        if ((count > max_iov) &&
            ucp_ep_is_tag_offload_enabled(ucp_ep_config(req->send.ep))) {
            /* Make sure SW RNDV will be used, because tag offload does
             * not support multi-packet eager protocols. */
            return 1;
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
ucp_tag_send_req(ucp_request_t *req, size_t dt_count,
                 const ucp_ep_msg_config_t* msg_config,
                 size_t rndv_rma_thresh, size_t rndv_am_thresh,
                 ucp_send_callback_t cb, const ucp_proto_t *proto,
                 int enable_zcopy)
{
    size_t rndv_thresh  = ucp_tag_get_rndv_threshold(req, dt_count,
                                                     msg_config->max_iov,
                                                     rndv_rma_thresh,
                                                     rndv_am_thresh);
    ssize_t max_short   = ucp_proto_get_short_max(req, msg_config);
    ucs_status_t status;
    size_t zcopy_thresh;

    if (enable_zcopy || ucs_unlikely(!UCP_MEM_IS_HOST(req->send.mem_type))) {
        zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config, dt_count,
                                                     rndv_thresh);
    } else {
        zcopy_thresh = rndv_thresh;
    }

    ucs_trace_req("select tag request(%p) progress algorithm datatype=%lx "
                  "buffer=%p length=%zu max_short=%zd rndv_thresh=%zu "
                  "zcopy_thresh=%zu zcopy_enabled=%d",
                  req, req->send.datatype, req->send.buffer, req->send.length,
                  max_short, rndv_thresh, zcopy_thresh, enable_zcopy);

    status = ucp_request_send_start(req, max_short, zcopy_thresh, rndv_thresh,
                                    dt_count, msg_config, proto);
    if (ucs_unlikely(status != UCS_OK)) {
        if (status == UCS_ERR_NO_PROGRESS) {
            /* RMA/AM rendezvous */
            ucs_assert(req->send.length >= rndv_thresh);
            status = ucp_tag_send_start_rndv(req);
            if (status != UCS_OK) {
                return UCS_STATUS_PTR(status);
            }

            UCP_EP_STAT_TAG_OP(req->send.ep, RNDV);
        } else {
            return UCS_STATUS_PTR(status);
        }
    }

    if (req->flags & UCP_REQUEST_FLAG_SYNC) {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER_SYNC);
    } else {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER);
    }

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_send(req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        if (enable_zcopy) {
            ucp_request_put(req);
        }
        return UCS_STATUS_PTR(status);
    }

    if (enable_zcopy) {
        ucp_request_set_callback(req, send.cb, cb)
    }

    ucs_trace_req("returning send request %p", req);
    return req + 1;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_send_req_init(ucp_request_t* req, ucp_ep_h ep, const void* buffer,
                      uintptr_t datatype, size_t count, ucp_tag_t tag,
                      uint16_t flags)
{
    req->flags             = flags;
    req->send.ep           = ep;
    req->send.buffer       = (void*)buffer;
    req->send.datatype     = datatype;
    req->send.tag.tag      = tag;
    ucp_request_send_state_init(req, datatype, count);
    req->send.length       = ucp_dt_length(req->send.datatype, count,
                                           req->send.buffer,
                                           &req->send.state.dt);
    ucp_memory_type_detect_mds(ep->worker->context, (void *)buffer,
                               req->send.length, &req->send.mem_type);
    req->send.lane         = ucp_ep_config(ep)->tag.lane;
    req->send.pending_lane = UCP_NULL_LANE;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_send_inline(ucp_ep_h ep, const void *buffer, size_t count,
                    uintptr_t datatype, ucp_tag_t tag)
{
    ucs_status_t status;
    size_t length;

    if (ucs_unlikely(!UCP_DT_IS_CONTIG(datatype))) {
        return UCS_ERR_NO_RESOURCE;
    }

    length = ucp_contig_dt_length(datatype, count);

    if ((ssize_t)length <= ucp_ep_config(ep)->tag.max_eager_short) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(ucp_eager_hdr_t));
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uint64_t));
        status = uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_EAGER_ONLY,
                                 tag, buffer, length);
    } else if ((ssize_t)length <= ucp_ep_config(ep)->tag.offload.max_eager_short) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uct_tag_t));
        status = uct_ep_tag_eager_short(ucp_ep_get_tag_uct_ep(ep), tag, buffer,
                                        length);
    } else {
        return UCS_ERR_NO_RESOURCE;
    }

    if (status != UCS_ERR_NO_RESOURCE) {
        UCP_EP_STAT_TAG_OP(ep, EAGER);
    }

    return status;
}


UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_nb,
                 (ep, buffer, count, datatype, tag, cb),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucs_status_t status;
    ucp_request_t *req;
    ucs_status_ptr_t ret;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("send_nb buffer %p count %zu tag %"PRIx64" to %s cb %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), cb);

    status = UCS_PROFILE_CALL(ucp_tag_send_inline, ep, buffer, count,
                              datatype, tag);
    if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
        ret = UCS_STATUS_PTR(status); /* UCS_OK also goes here */
        goto out;
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
                           cb, ucp_ep_config(ep)->tag.proto, 1);
out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_send_nbr,
                 (ep, buffer, count, datatype, tag, request),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, void *request)
{
    ucp_request_t *req = (ucp_request_t *)request - 1;
    ucs_status_t status;
    ucs_status_ptr_t ret;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("send_nbr buffer %p count %zu tag %"PRIx64" to %s req %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), request);

    status = UCS_PROFILE_CALL(ucp_tag_send_inline, ep, buffer, count,
                              datatype, tag);
    if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
        return status;
    }

    ucp_tag_send_req_init(req, ep, buffer, datatype, count, tag, 0);

    ret = ucp_tag_send_req(req, count, &ucp_ep_config(ep)->tag.eager,
                           ucp_ep_config(ep)->tag.rndv_send_nbr.rma_thresh,
                           ucp_ep_config(ep)->tag.rndv_send_nbr.am_thresh,
                           NULL, ucp_ep_config(ep)->tag.proto, 0);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    if (ucs_unlikely(UCS_PTR_IS_ERR(ret))) {
        return UCS_PTR_STATUS(ret);
    }
    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_sync_nb,
                 (ep, buffer, count, datatype, tag, cb),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucp_request_t *req;
    ucs_status_ptr_t ret;
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("send_sync_nb buffer %p count %zu tag %"PRIx64" to %s cb %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), cb);

    if (ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_PEER) {
        ret = UCS_STATUS_PTR(UCS_ERR_UNSUPPORTED);
        goto out;
    }

    status = ucp_ep_resolve_dest_ep_ptr(ep, ucp_ep_config(ep)->tag.lane);
    if (status != UCS_OK) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ucp_tag_send_req_init(req, ep, buffer, datatype, count, tag,
                          UCP_REQUEST_FLAG_SYNC);

    ret = ucp_tag_send_req(req, count, &ucp_ep_config(ep)->tag.eager,
                           ucp_ep_config(ep)->tag.rndv.rma_thresh,
                           ucp_ep_config(ep)->tag.rndv.am_thresh,
                           cb, ucp_ep_config(ep)->tag.sync_proto, 1);
out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ret;
}
