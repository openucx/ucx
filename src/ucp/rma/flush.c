/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>


static void ucp_ep_flush_error(ucp_request_t *req, ucs_status_t status)
{
    if (ucp_ep_config(req->send.ep)->key.err_mode != UCP_ERR_HANDLING_MODE_PEER) {
        ucs_error("error during flush: %s", ucs_status_string(status));
    }

    req->status = status;
    --req->send.state.uct_comp.count;
}

static void ucp_ep_flush_progress(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;
    ucp_lane_index_t lane;
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_trace("ep %p: progress flush req %p, lanes 0x%x count %d", ep, req,
              req->send.flush.lanes, req->send.state.uct_comp.count);

    while (req->send.flush.lanes) {

        /* Search for next lane to start flush */
        lane   = ucs_ffs64(req->send.flush.lanes);
        uct_ep = ep->uct_eps[lane];
        if (uct_ep == NULL) {
            req->send.flush.lanes &= ~UCS_BIT(lane);
            --req->send.state.uct_comp.count;
            continue;
        }

        /* Start flush operation on UCT endpoint */
        if (req->send.flush.uct_flags & UCT_FLUSH_FLAG_CANCEL) {
            uct_ep_pending_purge(uct_ep, ucp_ep_err_pending_purge,
                                 UCS_STATUS_PTR(UCS_ERR_CANCELED));
        }
        status = uct_ep_flush(uct_ep, req->send.flush.uct_flags,
                              &req->send.state.uct_comp);
        ucs_trace("flushing ep %p lane[%d]: %s", ep, lane,
                  ucs_status_string(status));
        if (status == UCS_OK) {
            req->send.flush.lanes &= ~UCS_BIT(lane);
            --req->send.state.uct_comp.count;
        } else if (status == UCS_INPROGRESS) {
            req->send.flush.lanes &= ~UCS_BIT(lane);
        } else if (status == UCS_ERR_NO_RESOURCE) {
            if (req->send.lane != UCP_NULL_LANE) {
                ucs_trace("ep %p: not adding pending flush %p on lane %d, "
                          "because it's already pending on lane %d",
                          ep, req, lane, req->send.lane);
                break;
            }

            status = uct_ep_pending_add(uct_ep, &req->send.uct);
            ucs_trace("adding pending flush on ep %p lane[%d]: %s", ep, lane,
                      ucs_status_string(status));
            if (status == UCS_OK) {
                req->send.lane        = lane;
                req->send.flush.lanes &= ~UCS_BIT(lane);
            } else if (status != UCS_ERR_BUSY) {
                ucp_ep_flush_error(req, status);
                break;
            }
        } else {
            ucp_ep_flush_error(req, status);
            break;
        }
    }
}

static void ucp_ep_flush_slow_path_remove(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;
    uct_worker_progress_unregister_safe(ep->worker->uct,
                                        &req->send.flush.prog_id);
}

static int ucp_flush_check_completion(ucp_request_t *req)
{
    /* Check if flushed all lanes */
    if (req->send.state.uct_comp.count != 0) {
        return 0;
    }

    ucp_ep_flush_slow_path_remove(req);
    req->send.flush.flushed_cb(req);
    return 1;
}

static unsigned ucp_ep_flush_resume_slow_path_callback(void *arg)
{
    ucp_request_t *req = arg;

    ucp_ep_flush_slow_path_remove(req);
    ucp_ep_flush_progress(req);
    ucp_flush_check_completion(req);
    return 0;
}

static ucs_status_t ucp_ep_flush_progress_pending(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_lane_index_t lane = req->send.lane;
    ucp_ep_h ep = req->send.ep;
    ucs_status_t status;
    int completed;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    status = uct_ep_flush(ep->uct_eps[lane], req->send.flush.uct_flags,
                          &req->send.state.uct_comp);
    ucs_trace("flushing ep %p lane[%d]: %s", ep, lane,
              ucs_status_string(status));
    if (status == UCS_OK) {
        --req->send.state.uct_comp.count; /* UCT endpoint is flushed */
    }

    /* since req->flush.pend.lane is still non-NULL, this function will not
     * put anything on pending.
     */
    ucp_ep_flush_progress(req);
    completed = ucp_flush_check_completion(req);

    /* If the operation has not completed, add slow-path progress to resume */
    if (!completed && req->send.flush.lanes) {
        ucs_trace("ep %p: adding slow-path callback to resume flush", ep);
        uct_worker_progress_register_safe(ep->worker->uct,
                                          ucp_ep_flush_resume_slow_path_callback,
                                          req, 0, &req->send.flush.prog_id);
    }

    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        req->send.lane = UCP_NULL_LANE;
        return UCS_OK;
    } else if (status == UCS_ERR_NO_RESOURCE) {
        return UCS_ERR_NO_RESOURCE;
    } else {
        ucp_ep_flush_error(req, status);
        return UCS_OK;
    }
}

static void ucp_ep_flush_completion(uct_completion_t *self, ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucs_trace("flush completion req=%p status=%d", req, status);

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    req->status = status;

    if (status == UCS_OK) {
        ucp_ep_flush_progress(req);
    } else {
        req->send.state.uct_comp.count = 0;
    }

    ucp_flush_check_completion(req);
}

ucs_status_ptr_t ucp_ep_flush_internal(ucp_ep_h ep, unsigned uct_flags,
                                       ucp_send_callback_t req_cb,
                                       unsigned req_flags,
                                       ucp_request_callback_t flushed_cb)
{
    ucs_status_t status;
    ucp_request_t *req;

    ucs_debug("disconnect ep %p", ep);

    if (ep->flags & UCP_EP_FLAG_FAILED) {
        return NULL;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    /*
     *  Flush operation can be queued on the pending queue of only one of the
     * lanes (indicated by req->send.lane) and scheduled for completion on any
     * number of lanes. req->send.uct_comp.count keeps track of how many lanes
     * are not flushed yet, and when it reaches zero, it means all lanes are
     * flushed. req->send.flush.lanes keeps track of which lanes we still have
     * to start flush on.
      */
    req->flags                  = req_flags;
    req->status                 = UCS_OK;
    req->send.ep                = ep;
    req->send.cb                = req_cb;
    req->send.flush.flushed_cb  = flushed_cb;
    req->send.flush.lanes       = UCS_MASK(ucp_ep_num_lanes(ep));
    req->send.flush.prog_id     = UCS_CALLBACKQ_ID_NULL;
    req->send.flush.uct_flags   = uct_flags;

    req->send.lane              = UCP_NULL_LANE;
    req->send.uct.func          = ucp_ep_flush_progress_pending;
    req->send.state.uct_comp.func   = ucp_ep_flush_completion;
    req->send.state.uct_comp.count  = ucp_ep_num_lanes(ep);

    ucp_ep_flush_progress(req);

    if (req->send.state.uct_comp.count == 0) {
        status = req->status;
        ucs_trace_req("ep %p: releasing flush request %p, returning status %s",
                      ep, req, ucs_status_string(status));
        ucs_mpool_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucs_trace_req("ep %p: return inprogress flush request %p (%p)", ep, req,
                  req + 1);
    return req + 1;
}

static void ucp_ep_flushed_callback(ucp_request_t *req)
{
    ucp_request_complete_send(req, req->status);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_ep_flush_nb, (ep, flags, cb),
                 ucp_ep_h ep, unsigned flags, ucp_send_callback_t cb)
{
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    request = ucp_ep_flush_internal(ep, UCT_FLUSH_FLAG_LOCAL,
                                    cb, UCP_REQUEST_FLAG_CALLBACK,
                                    ucp_ep_flushed_callback);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    return request;
}

static ucs_status_t ucp_worker_flush_check(ucp_worker_h worker)
{
    ucs_status_t status;
    unsigned rsc_index;

    if (worker->wireup_pend_count > 0) {
        return UCS_INPROGRESS;
    }

    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index].iface == NULL) {
            continue;
        }

        status = uct_iface_flush(worker->ifaces[rsc_index].iface, 0, NULL);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static unsigned ucp_worker_flush_progress(void *arg)
{
    ucp_request_t *req  = arg;
    ucp_worker_h worker = req->flush_worker.worker;
    ucs_status_t status;

    status = ucp_worker_flush_check(worker);
    if ((status == UCS_INPROGRESS) || (status == UCS_ERR_NO_RESOURCE)) {
        return 0;
    }

    uct_worker_progress_unregister_safe(worker->uct, &req->flush_worker.prog_id);
    ucp_request_complete(req, flush_worker.cb, status);
    return 0;
}

static ucs_status_ptr_t ucp_worker_flush_nb_internal(ucp_worker_h worker,
                                                     ucp_send_callback_t cb,
                                                     unsigned req_flags)
{
    ucs_status_t status;
    ucp_request_t *req;

    status = ucp_worker_flush_check(worker);
    if ((status != UCS_INPROGRESS) && (status != UCS_ERR_NO_RESOURCE)) {
        return UCS_STATUS_PTR(status);
    }

    req = ucp_request_get(worker);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    req->flags                = req_flags;
    req->status               = UCS_OK;
    req->flush_worker.worker  = worker;
    req->flush_worker.cb      = cb;
    req->flush_worker.prog_id = UCS_CALLBACKQ_ID_NULL;

    uct_worker_progress_register_safe(worker->uct, ucp_worker_flush_progress,
                                      req, 0, &req->flush_worker.prog_id);
    return req + 1;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_worker_flush_nb, (worker, flags, cb),
                 ucp_worker_h worker, unsigned flags, ucp_send_callback_t cb)
{
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    request = ucp_worker_flush_nb_internal(worker, cb,
                                           UCP_REQUEST_FLAG_CALLBACK);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return request;
}

static ucs_status_t ucp_flush_wait(ucp_worker_h worker, void *request)
{
    ucs_status_t status;

    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        ucs_warn("flush failed: %s", ucs_status_string(UCS_PTR_STATUS(request)));
        return UCS_PTR_STATUS(request);
    } else {
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(request);
        return status;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_flush, (worker), ucp_worker_h worker)
{
    ucs_status_t status;
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    request = ucp_worker_flush_nb_internal(worker, NULL, 0);
    status = ucp_flush_wait(worker, request);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_flush, (ep), ucp_ep_h ep)
{
    ucs_status_t status;
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    request = ucp_ep_flush_internal(ep, UCT_FLUSH_FLAG_LOCAL, NULL, 0,
                                    ucp_ep_flushed_callback);
    status = ucp_flush_wait(ep->worker, request);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_fence, (worker), ucp_worker_h worker)
{
    unsigned rsc_index;
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index].iface == NULL) {
            continue;
        }

        status = uct_iface_fence(worker->ifaces[rsc_index].iface, 0);
        if (status != UCS_OK) {
            goto out;
        }
    }
    status = UCS_OK;

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;
}
