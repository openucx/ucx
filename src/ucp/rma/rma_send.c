/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "rma.h"
#include "rma.inl"

#include <ucp/core/ucp_mm.h>

#include <ucp/dt/dt_contig.h>
#include <ucs/profile/profile.h>


#define UCP_RMA_CHECK_PARAMS(_buffer, _length) \
    if ((_length) == 0) { \
        return UCS_OK; \
    } \
    if (ENABLE_PARAMS_CHECK && ((_buffer) == NULL)) { \
        return UCS_ERR_INVALID_PARAM; \
    }

#define UCP_RMA_CHECK_PARAMS_PTR(_buffer, _length) \
    if ((_length) == 0) { \
        return UCS_STATUS_PTR(UCS_OK);          \
    } \
    if (ENABLE_PARAMS_CHECK && ((_buffer) == NULL)) { \
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM); \
    }


/* request can be released if
 *  - all fragments were sent (length == 0) (bcopy & zcopy mix)
 *  - all zcopy fragments are done (uct_comp.count == 0)
 *  - and request was allocated from the mpool
 *    (checked in ucp_request_complete_send)
 *
 * Request can be released either immediately or in the completion callback.
 * We must check req length in the completion callback to avoid the following
 * scenario:
 *  partial_send;no_resos;progress;
 *  send_completed;cb called;req free(ooops);
 *  next_partial_send; (oops req already freed)
 */
ucs_status_t ucp_rma_request_advance(ucp_request_t *req, ssize_t frag_length,
                                     ucs_status_t status)
{
    ucs_assert(status != UCS_ERR_NOT_IMPLEMENTED);

    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        if (status != UCS_ERR_NO_RESOURCE) {
            ucp_request_send_buffer_dereg(req);
            ucp_request_complete_send(req, status);
        }
        return status;
    }

    ucs_assert(frag_length >= 0);
    ucs_assert(req->send.length >= frag_length);
    req->send.length -= frag_length;
    if (req->send.length == 0) {
        /* bcopy is the fast path */
        if (ucs_likely(req->send.state.uct_comp.count == 0)) {
            ucp_request_send_buffer_dereg(req);
            ucp_request_complete_send(req, UCS_OK);
        }
        return UCS_OK;
    }
    req->send.buffer          += frag_length;
    req->send.rma.remote_addr += frag_length;
    return UCS_INPROGRESS;
}

static void ucp_rma_request_bcopy_completion(uct_completion_t *self,
                                             ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    if (ucs_likely(req->send.length == req->send.state.dt.offset)) {
        ucp_request_complete_send(req, status);
    }
}

static void ucp_rma_request_zcopy_completion(uct_completion_t *self,
                                             ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    if (ucs_likely(req->send.length == req->send.state.dt.offset)) {
        ucp_request_send_buffer_dereg(req);
        ucp_request_complete_send(req, status);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_rma_request_init(ucp_request_t *req, ucp_ep_h ep, const void *buffer,
                     size_t length, uint64_t remote_addr, ucp_rkey_h rkey,
                     uct_pending_callback_t cb, size_t zcopy_thresh, int flags)
{
    req->flags                = flags; /* Implicit release */
    req->send.ep              = ep;
    req->send.buffer          = (void*)buffer;
    req->send.datatype        = ucp_dt_make_contig(1);
    req->send.mem_type        = UCT_MD_MEM_TYPE_HOST;
    req->send.length          = length;
    req->send.rma.remote_addr = remote_addr;
    req->send.rma.rkey        = rkey;
    req->send.uct.func        = cb;
    req->send.lane            = rkey->cache.rma_lane;
    ucp_request_send_state_init(req, ucp_dt_make_contig(1), length);
    ucp_request_send_state_reset(req,
                                 (length < zcopy_thresh) ?
                                 ucp_rma_request_bcopy_completion :
                                 ucp_rma_request_zcopy_completion,
                                 UCP_REQUEST_SEND_PROTO_RMA);
#if ENABLE_ASSERT
    req->send.cb              = NULL;
#endif
    if (length < zcopy_thresh) {
        return UCS_OK;
    }

    return ucp_request_send_buffer_reg_lane(req, req->send.lane);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_rma_nonblocking(ucp_ep_h ep, const void *buffer, size_t length,
                    uint64_t remote_addr, ucp_rkey_h rkey,
                    uct_pending_callback_t progress_cb, size_t zcopy_thresh)
{
    ucs_status_t status;
    ucp_request_t *req;

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = ucp_rma_request_init(req, ep, buffer, length, remote_addr, rkey,
                                  progress_cb, zcopy_thresh,
                                  UCP_REQUEST_FLAG_RELEASED);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    return ucp_request_send(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_rma_nonblocking_cb(ucp_ep_h ep, const void *buffer, size_t length,
                       uint64_t remote_addr, ucp_rkey_h rkey,
                       uct_pending_callback_t progress_cb, size_t zcopy_thresh,
                       ucp_send_callback_t cb)
{
    ucs_status_t status;
    ucp_request_t *req;

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    status = ucp_rma_request_init(req, ep, buffer, length, remote_addr, rkey,
                                  progress_cb, zcopy_thresh, 0);
    if (ucs_unlikely(status != UCS_OK)) {
        return UCS_STATUS_PTR(status);
    }

    return ucp_rma_send_request_cb(req, cb);
}

ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_ep_rma_config_t *rma_config;
    ucs_status_t status;

    UCP_RMA_CHECK_PARAMS(buffer, length);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("put_nbi buffer %p length %zu remote_addr %"PRIx64" rkey %p to %s",
                   buffer, length, remote_addr, rkey, ucp_ep_peer_name(ep));

    status = UCP_RKEY_RESOLVE(rkey, ep, rma);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    /* Fast path for a single short message */
    if (ucs_likely((ssize_t)length <= (int)rkey->cache.max_put_short)) {
        status = UCS_PROFILE_CALL(uct_ep_put_short, ep->uct_eps[rkey->cache.rma_lane],
                                  buffer, length, remote_addr, rkey->cache.rma_rkey);
        if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
            goto out_unlock;
        }
    }

    rma_config = &ucp_ep_config(ep)->rma[rkey->cache.rma_lane];
    status = ucp_rma_nonblocking(ep, buffer, length, remote_addr, rkey,
                                 rkey->cache.rma_proto->progress_put,
                                 rma_config->put_zcopy_thresh);
out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status;
}

ucs_status_ptr_t ucp_put_nb(ucp_ep_h ep, const void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb)
{
    ucp_ep_rma_config_t *rma_config;
    ucs_status_ptr_t ptr_status;
    ucs_status_t status;

    UCP_RMA_CHECK_PARAMS_PTR(buffer, length);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("put_nb buffer %p length %zu remote_addr %"PRIx64" rkey %p to %s cb %p",
                   buffer, length, remote_addr, rkey, ucp_ep_peer_name(ep), cb);

    status = UCP_RKEY_RESOLVE(rkey, ep, rma);
    if (status != UCS_OK) {
        ptr_status = UCS_STATUS_PTR(status);
        goto out_unlock;
    }

    /* Fast path for a single short message */
    if (ucs_likely((ssize_t)length <= (int)rkey->cache.max_put_short)) {
        status = UCS_PROFILE_CALL(uct_ep_put_short, ep->uct_eps[rkey->cache.rma_lane],
                                  buffer, length, remote_addr, rkey->cache.rma_rkey);
        if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
            ptr_status = UCS_STATUS_PTR(status);
            goto out_unlock;
        }
    }

    rma_config = &ucp_ep_config(ep)->rma[rkey->cache.rma_lane];
    ptr_status = ucp_rma_nonblocking_cb(ep, buffer, length, remote_addr, rkey,
                                        rkey->cache.rma_proto->progress_put,
                                        rma_config->put_zcopy_thresh, cb);
out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ptr_status;
}

ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_ep_rma_config_t *rma_config;
    ucs_status_t status;

    UCP_RMA_CHECK_PARAMS(buffer, length);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("get_nbi buffer %p length %zu remote_addr %"PRIx64" rkey %p from %s",
                   buffer, length, remote_addr, rkey, ucp_ep_peer_name(ep));

    status = UCP_RKEY_RESOLVE(rkey, ep, rma);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    rma_config = &ucp_ep_config(ep)->rma[rkey->cache.rma_lane];
    status = ucp_rma_nonblocking(ep, buffer, length, remote_addr, rkey,
                                 rkey->cache.rma_proto->progress_get,
                                 rma_config->get_zcopy_thresh);
out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status;
}

ucs_status_ptr_t ucp_get_nb(ucp_ep_h ep, void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb)
{
    ucp_ep_rma_config_t *rma_config;
    ucs_status_ptr_t ptr_status;
    ucs_status_t status;

    UCP_RMA_CHECK_PARAMS_PTR(buffer, length);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("get_nb buffer %p length %zu remote_addr %"PRIx64" rkey %p from %s cb %p",
                   buffer, length, remote_addr, rkey, ucp_ep_peer_name(ep), cb);

    status = UCP_RKEY_RESOLVE(rkey, ep, rma);
    if (status != UCS_OK) {
        ptr_status = UCS_STATUS_PTR(status);
        goto out_unlock;
    }

    rma_config = &ucp_ep_config(ep)->rma[rkey->cache.rma_lane];
    ptr_status = ucp_rma_nonblocking_cb(ep, buffer, length, remote_addr, rkey,
                                        rkey->cache.rma_proto->progress_get,
                                        rma_config->get_zcopy_thresh, cb);
out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ptr_status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_put, (ep, buffer, length, remote_addr, rkey),
                 ucp_ep_h ep, const void *buffer, size_t length,
                 uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_rma_wait(ep->worker,
                        ucp_put_nb(ep, buffer, length, remote_addr, rkey,
                                   (void*)ucs_empty_function),
                        "put");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_get, (ep, buffer, length, remote_addr, rkey),
                 ucp_ep_h ep, void *buffer, size_t length,
                 uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_rma_wait(ep->worker,
                        ucp_get_nb(ep, buffer, length, remote_addr, rkey,
                                   (void*)ucs_empty_function),
                        "get");
}
