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
    --req->send.uct_comp.count;
}

static void ucp_ep_flush_progress(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;
    ucp_lane_index_t lane;
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_trace("ep %p: progress flush req %p, lanes 0x%x count %d", ep, req,
              req->send.flush.lanes, req->send.uct_comp.count);

    while (req->send.flush.lanes) {

        /* Search for next lane to start flush */
        lane   = ucs_ffs64(req->send.flush.lanes);
        uct_ep = ep->uct_eps[lane];
        if (uct_ep == NULL) {
            req->send.flush.lanes &= ~UCS_BIT(lane);
            --req->send.uct_comp.count;
            continue;
        }

        /* Start flush operation on UCT endpoint */
        if (req->send.flush.uct_flags & UCT_FLUSH_FLAG_CANCEL) {
            uct_ep_pending_purge(uct_ep, ucp_ep_err_pending_purge,
                                 UCS_STATUS_PTR(UCS_ERR_CANCELED));
        }
        status = uct_ep_flush(uct_ep, req->send.flush.uct_flags,
                              &req->send.uct_comp);
        ucs_trace("flushing ep %p lane[%d]: %s", ep, lane,
                  ucs_status_string(status));
        if (status == UCS_OK) {
            req->send.flush.lanes &= ~UCS_BIT(lane);
            --req->send.uct_comp.count;
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
                                        &req->send.flush.slow_cb_id);
}

static unsigned ucp_ep_flushed_slow_path_callback(void *arg)
{
    ucp_request_t *req = arg;
    ucp_ep_h ep = req->send.ep;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    ucs_trace("flush req %p ep %p remove from uct_worker %p", req, ep,
              ep->worker->uct);
    ucp_ep_flush_slow_path_remove(req);
    req->send.flush.flushed_cb(req);

    /* Complete send request from here, to avoid releasing the request while
     * slow-path element is still pending */
    ucp_request_complete_send(req, req->status);

    return 0;
}

static int ucp_flush_check_completion(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;

    /* Check if flushed all lanes */
    if (req->send.uct_comp.count != 0) {
        return 0;
    }

    ucs_trace("adding slow-path callback to destroy ep %p", ep);
    ucp_ep_flush_slow_path_remove(req);
    uct_worker_progress_register_safe(ep->worker->uct,
                                      ucp_ep_flushed_slow_path_callback, req, 0,
                                      &req->send.flush.slow_cb_id);
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
                          &req->send.uct_comp);
    ucs_trace("flushing ep %p lane[%d]: %s", ep, lane,
              ucs_status_string(status));
    if (status == UCS_OK) {
        --req->send.uct_comp.count; /* UCT endpoint is flushed */
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
                                          req, 0, &req->send.flush.slow_cb_id);
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
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct_comp);

    ucs_trace("flush completion req=%p status=%d", req, status);

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    if (status == UCS_OK) {
        req->status = status;
    }

    ucp_ep_flush_progress(req);
    ucp_flush_check_completion(req);
}

static void ucp_ep_flushed_callback(ucp_request_t *req)
{
    ucp_ep_disconnected(req->send.ep);
}

ucs_status_ptr_t ucp_disconnect_nb_internal(ucp_ep_h ep, unsigned mode)
{
    ucs_status_t status;
    ucp_request_t *req;

    ucs_debug("disconnect ep %p", ep);

    if (ep->flags & UCP_EP_FLAG_FAILED) {
        ucp_ep_disconnected(ep);
        return NULL;
    }

    req = ucs_mpool_get(&ep->worker->req_mp);
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
     *  If a flush is completed from a pending/completion callback, we need to
     * schedule slow-path callback to release the endpoint later, since a UCT
     * endpoint cannot be released from pending/completion callback context.
     */
    req->flags                  = 0;
    req->status                 = UCS_OK;
    req->send.ep                = ep;
    req->send.flush.flushed_cb  = ucp_ep_flushed_callback;
    req->send.flush.lanes       = UCS_MASK(ucp_ep_num_lanes(ep));
    req->send.flush.slow_cb_id  = UCS_CALLBACKQ_ID_NULL;
    req->send.flush.uct_flags   = (mode == UCP_EP_CLOSE_MODE_FLUSH) ?
                                  UCT_FLUSH_FLAG_LOCAL : UCT_FLUSH_FLAG_CANCEL;

    req->send.lane              = UCP_NULL_LANE;
    req->send.uct.func          = ucp_ep_flush_progress_pending;
    req->send.uct_comp.func     = ucp_ep_flush_completion;
    req->send.uct_comp.count    = ucp_ep_num_lanes(ep);

    ucp_ep_flush_progress(req);

    if (req->send.uct_comp.count == 0) {
        status = req->status;
        ucs_trace_req("ep %p: releasing flush request %p, returning status %s",
                      ep, req, ucs_status_string(status));
        ucs_mpool_put(req);
        ucp_ep_disconnected(ep);
        return UCS_STATUS_PTR(status);
    }

    ucs_trace_req("ep %p: return inprogress flush request %p (%p)", ep, req,
                  req + 1);
    return req + 1;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_flush, (worker), ucp_worker_h worker)
{
    unsigned rsc_index;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    while (worker->wireup_pend_count > 0) {
        ucp_worker_progress(worker);
    }

    /* TODO flush in parallel */
    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index].iface == NULL) {
            continue;
        }

        while (uct_iface_flush(worker->ifaces[rsc_index].iface, 0, NULL) != UCS_OK) {
            ucp_worker_progress(worker);
        }
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_flush, (ep), ucp_ep_h ep)
{
    ucp_lane_index_t lane;
    ucs_status_t     status;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        for (;;) {
            status = uct_ep_flush(ep->uct_eps[lane], 0, NULL);
            if (status == UCS_OK) {
                break;
            } else if ((status != UCS_INPROGRESS) && (status != UCS_ERR_NO_RESOURCE)) {
                goto out;
            }
            ucp_worker_progress(ep->worker);
        }
    }

    status = UCS_OK;
out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return status;
}
