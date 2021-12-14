/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>

#include "rma.inl"


static void ucp_ep_flush_error(ucp_request_t *req, ucs_status_t status)
{
    ucs_log_level_t level = (ucp_ep_config(req->send.ep)->key.err_mode ==
                             UCP_ERR_HANDLING_MODE_PEER) ?
                             UCS_LOG_LEVEL_TRACE_REQ : UCS_LOG_LEVEL_ERROR;

    req->status = status;
    --req->send.state.uct_comp.count;
    ucs_log(level, "req %p: error during flush: %s, flush comp %p count reduced to %d",
            req, ucs_status_string(status), &req->send.state.uct_comp,
            req->send.state.uct_comp.count);
}

static int ucp_ep_flush_is_completed(ucp_request_t *req)
{
    return (req->send.state.uct_comp.count == 0) && req->send.flush.sw_done;
}

static void ucp_ep_flush_progress(ucp_request_t *req)
{
    ucp_ep_h ep              = req->send.ep;
    unsigned num_lanes       = ucp_ep_num_lanes(ep);
    ucp_lane_map_t all_lanes = UCS_MASK(num_lanes);
    ucp_ep_flush_state_t *flush_state;
    ucp_lane_index_t lane;
    ucs_status_t status;
    uct_ep_h uct_ep;
    int diff;
    ucp_lane_map_t destroyed_lanes;

    /* If the number of lanes changed since flush operation was submitted, adjust
     * the number of expected completions */
    diff = num_lanes - req->send.flush.num_lanes;
    if (ucs_unlikely(diff != 0)) {
        if (diff > 0) {
            ucs_debug("ep %p: flush req %p lanes changed from %d to %d, "
                      "adding %d to completion count",
                      ep, req, req->send.flush.num_lanes, num_lanes, diff);
            req->send.state.uct_comp.count += diff;
        } else {
            /* Some lanes that we wanted to flush were destroyed. If we already
               started to flush them, they would be completed by discard flow,
               so reduce completion count only by the lanes we have not started
               to flush yet. */
            destroyed_lanes = UCS_MASK(req->send.flush.num_lanes) & ~all_lanes &
                              ~req->send.flush.started_lanes;
            ucs_debug("ep %p: flush req %p lanes changed from %d to %d, "
                      "destroyed_lanes 0x%x, reducing completion count by %d",
                      ep, req, req->send.flush.num_lanes, num_lanes,
                      destroyed_lanes, ucs_popcount(destroyed_lanes));
            ucs_assert(!(req->send.flush.started_lanes & destroyed_lanes));
            req->send.state.uct_comp.count -= ucs_popcount(destroyed_lanes);
        }
        req->send.flush.num_lanes = num_lanes;
    }

    ucs_trace("ep %p flags 0x%x: progress flush req %p, started_lanes 0x%x "
              "count %d",
              ep, ep->flags, req, req->send.flush.started_lanes,
              req->send.state.uct_comp.count);

    while (req->send.flush.started_lanes < all_lanes) {

        /* Search for next lane to start flush */
        lane   = ucs_ffs64(all_lanes & ~req->send.flush.started_lanes);
        uct_ep = ep->uct_eps[lane];
        if (uct_ep == NULL) {
            req->send.flush.started_lanes |= UCS_BIT(lane);
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
        ucp_trace_req(req, "ep %p flush lane[%d]=%p flags 0x%x: %s",
                      ep, lane, uct_ep, req->send.flush.uct_flags,
                      ucs_status_string(status));
        if (status == UCS_OK) {
            req->send.flush.started_lanes |= UCS_BIT(lane);
            --req->send.state.uct_comp.count;
            ucs_trace("ep %p: flush comp %p count reduced to %d", ep,
                      &req->send.state.uct_comp, req->send.state.uct_comp.count);
        } else if (status == UCS_INPROGRESS) {
            req->send.flush.started_lanes |= UCS_BIT(lane);
        } else if (status == UCS_ERR_NO_RESOURCE) {
            if (req->send.lane != UCP_NULL_LANE) {
                ucs_trace("ep %p: not adding pending flush %p on lane %d, "
                          "because it's already pending on lane %d",
                          ep, req, lane, req->send.lane);
                break;
            }

            status = uct_ep_pending_add(uct_ep, &req->send.uct, 0);
            ucs_trace("adding pending flush on ep %p lane[%d]: %s", ep, lane,
                      ucs_status_string(status));
            if (status == UCS_OK) {
                req->send.lane                 = lane;
                req->send.flush.started_lanes |= UCS_BIT(lane);
            } else if (status != UCS_ERR_BUSY) {
                ucp_ep_flush_error(req, status);
                break;
            }
        } else {
            ucp_ep_flush_error(req, status);
            req->send.flush.started_lanes |= UCS_BIT(lane);
        }
    }

    if (!req->send.flush.sw_started && (req->send.state.uct_comp.count == 0)) {
        /* Start waiting for remote completions only after all lanes are flushed
         * on the transport level, so we are sure all pending requests were sent.
         * We don't need to wait for remote completions in these cases:
         * - The flush operation is in 'cancel' mode
         * - The endpoint is either not used or did not resolve the peer endpoint,
         *   which means we didn't have any user operations which require remote
         *   completion. In this case, the flush state may not even be initialized.
         */
        if ((req->send.flush.uct_flags & UCT_FLUSH_FLAG_CANCEL) ||
            !ucs_test_all_flags(ep->flags, UCP_EP_FLAG_USED |
                                           UCP_EP_FLAG_REMOTE_ID)) {
            ucs_trace_req("flush request %p not waiting for remote completions",
                          req);
            req->send.flush.sw_done = 1;
        } else {
            /* All pending requests were sent, so 'send_sn' value is up-to-date */
            flush_state = ucp_ep_flush_state(ep);
            if (flush_state->send_sn == flush_state->cmpl_sn) {
                req->send.flush.sw_done = 1;
                ucs_trace_req("flush request %p remote completions done", req);
            } else {
                req->send.flush.cmpl_sn = flush_state->send_sn;
                ucs_hlist_add_tail(&flush_state->reqs, &req->send.list);
                ucs_trace_req("added flush request %p to ep remote completion"
                              " queue with sn %d",
                              req, req->send.flush.cmpl_sn);
            }
        }
        req->send.flush.sw_started = 1;
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
    if (!ucp_ep_flush_is_completed(req)) {
        return 0;
    }

    ucs_trace_req("flush req %p completed", req);
    ucp_ep_flush_slow_path_remove(req);
    req->send.flushed_cb(req);
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

ucs_status_t ucp_ep_flush_progress_pending(uct_pending_req_t *self)
{
    ucp_request_t *req    = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_lane_index_t lane = req->send.lane;
    ucp_ep_h ep           = req->send.ep;
    ucs_status_t status;
    int completed;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    status = uct_ep_flush(ep->uct_eps[lane], req->send.flush.uct_flags,
                          &req->send.state.uct_comp);
    ucs_trace("flushing ep %p lane[%d]=%p: %s", ep, lane, ep->uct_eps[lane],
              ucs_status_string(status));
    if (status == UCS_OK) {
        --req->send.state.uct_comp.count; /* UCT endpoint is flushed */
    } else if (UCS_STATUS_IS_ERR(status) && (status != UCS_ERR_NO_RESOURCE)) {
        ucp_ep_flush_error(req, status);
    }

    /* since req->flush.pend.lane is still non-NULL, this function will not
     * put anything on pending.
     */
    ucp_ep_flush_progress(req);
    completed = ucp_flush_check_completion(req);

    /* If the operation has not completed, and not started on all lanes, add
     * slow-path progress to resume */
    if (!completed &&
        (req->send.flush.started_lanes != UCS_MASK(ucp_ep_num_lanes(ep)))) {
        ucs_trace("ep %p: adding slow-path callback to resume flush", ep);
        uct_worker_progress_register_safe(ep->worker->uct,
                                          ucp_ep_flush_resume_slow_path_callback,
                                          req, 0, &req->send.flush.prog_id);
    }

    if (status == UCS_ERR_NO_RESOURCE) {
        return UCS_ERR_NO_RESOURCE;
    } else if (!UCS_STATUS_IS_ERR(status)) {
        /* flushed callback might release the request */
        if (!completed) {
            req->send.lane = UCP_NULL_LANE;
        }
    }

    return UCS_OK;
}

void ucp_ep_flush_completion(uct_completion_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucs_status_t status = self->status;

    ucp_trace_req(req, "flush completion status=%d", status);

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));
    ucs_assert(status != UCS_INPROGRESS);

    req->status = status;

    if (status == UCS_OK) {
        ucp_ep_flush_progress(req);
    } else {
        /* force flush completion in case of error */
        req->send.flush.sw_done        = 1;
        req->send.state.uct_comp.count = 0;
    }


    ucp_trace_req(req, "flush completion comp_count %d status %s",
                  req->send.state.uct_comp.count, ucs_status_string(status));
    ucp_flush_check_completion(req);
}

void ucp_ep_flush_request_ff(ucp_request_t *req, ucs_status_t status)
{
    /* Calculate how many completions to emulate: 1 for every lane we did not
     * start to flush yet, plus one for the lane from which we just removed
     * this request from its pending queue
     */
    int num_comps = req->send.flush.num_lanes -
                    ucs_popcount(req->send.flush.started_lanes)
                    + 1;

    ucp_trace_req(req, "fast-forward flush, comp-=%d num_lanes %d started 0x%x",
                  num_comps, req->send.flush.num_lanes,
                  req->send.flush.started_lanes);

    req->send.flush.started_lanes = UCS_MASK(req->send.flush.num_lanes);

    ucs_assert(req->send.state.uct_comp.count >= num_comps);
    req->send.state.uct_comp.count -= num_comps;
    uct_completion_update_status(&req->send.state.uct_comp, status);
    ucp_send_request_invoke_uct_completion(req);
}

void ucp_ep_flush_remote_completed(ucp_request_t *req)
{
    ucs_trace_req("flush remote ops completed req=%p", req);

    if (!req->send.flush.sw_done) {
        req->send.flush.sw_done = 1;
        ucp_flush_check_completion(req);
    }
}

ucs_status_ptr_t ucp_ep_flush_internal(ucp_ep_h ep, unsigned req_flags,
                                       const ucp_request_param_t *param,
                                       ucp_request_t *worker_req,
                                       ucp_request_callback_t flushed_cb,
                                       const char *debug_name)
{
    ucs_status_t status;
    ucp_request_t *req;

    ucs_debug("%s ep %p", debug_name, ep);

    req = ucp_request_get_param(ep->worker, param,
                                {return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);});

    /*
     * Flush operation can be queued on the pending queue of only one of the
     * lanes (indicated by req->send.lane) and scheduled for completion on any
     * number of lanes. req->send.uct_comp.count keeps track of how many lanes
     * are not flushed yet, and when it reaches zero, it means all lanes are
     * flushed. req->send.flush.lanes keeps track of which lanes we still have
     * to start flush on.
     */
    req->flags                      = req_flags;
    req->status                     = UCS_OK;
    req->send.ep                    = ep;
    req->send.flushed_cb            = flushed_cb;
    req->send.flush.prog_id         = UCS_CALLBACKQ_ID_NULL;
    req->send.flush.uct_flags       = UCT_FLUSH_FLAG_LOCAL;
    req->send.flush.sw_started      = 0;
    req->send.flush.sw_done         = 0;
    req->send.flush.num_lanes       = ucp_ep_num_lanes(ep);
    req->send.flush.started_lanes   = 0;
    req->send.lane                  = UCP_NULL_LANE;
    req->send.uct.func              = ucp_ep_flush_progress_pending;
    req->send.state.uct_comp.func   = ucp_ep_flush_completion;
    req->send.state.uct_comp.count  = ucp_ep_num_lanes(ep);
    req->send.state.uct_comp.status = UCS_OK;

    ucp_request_set_super(req, worker_req);
    ucp_request_set_send_callback_param(param, req, send);
    ucp_ep_flush_progress(req);

    if (ucp_ep_flush_is_completed(req)) {
        status = req->status;
        ucs_trace_req("ep %p: releasing flush request %p, returning status %s",
                      ep, req, ucs_status_string(status));
        ucp_request_put_param(param, req)
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
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb
    };

    return ucp_ep_flush_nbx(ep, &param);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_ep_flush_nbx, (ep, param),
                 ucp_ep_h ep, const ucp_request_param_t *param)
{
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    request = ucp_ep_flush_internal(ep, 0, param, NULL,
                                    ucp_ep_flushed_callback, "flush_nbx");

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    return request;
}

static ucs_status_t ucp_worker_flush_check(ucp_worker_h worker)
{
    ucp_rsc_index_t iface_id;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;

    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        wiface = worker->ifaces[iface_id];
        if (wiface->iface == NULL) {
            continue;
        }

        status = uct_iface_flush(wiface->iface, 0, NULL);
        if (status != UCS_OK) {
            if (UCS_STATUS_IS_ERR(status)) {
                ucs_error("iface[%d] "UCT_TL_RESOURCE_DESC_FMT" flush failed: %s",
                          iface_id,
                          UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[wiface->rsc_index].tl_rsc),
                          ucs_status_string(status));
            }
            return status;
        }
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucp_ep_h
ucp_worker_flush_req_set_next_ep(ucp_request_t *req, int is_current_ep_valid,
                                 ucs_list_link_t *next_ep_iter)
{
    ucp_worker_h worker              = req->flush_worker.worker;
    ucp_ep_ext_gen_t *next_ep_ext    = ucs_container_of(next_ep_iter,
                                                        ucp_ep_ext_gen_t,
                                                        ep_list);
    ucp_ep_h next_ep                 = ucp_ep_from_ext_gen(next_ep_ext);
    ucp_ep_ext_gen_t *current_ep_ext = req->flush_worker.next_ep;
    ucp_ep_h current_ep;

    req->flush_worker.next_ep = next_ep_ext;

    if (next_ep_iter != &worker->all_eps) {
        /* Increment UCP EP reference counter to avoid destroying UCP EP while
         * it is being scheduled to be flushed */
        ucp_ep_refcount_add(next_ep, flush);
    }

    if (!is_current_ep_valid) {
        return NULL;
    }

    ucs_assert(&current_ep_ext->ep_list != &worker->all_eps);

    current_ep = ucp_ep_from_ext_gen(current_ep_ext);
    return ucp_ep_refcount_remove(current_ep, flush) ? NULL : current_ep;
}

static void ucp_worker_flush_complete_one(ucp_request_t *req, ucs_status_t status,
                                          int force_progress_unreg)
{
    ucp_worker_h worker = req->flush_worker.worker;
    int complete;

    --req->flush_worker.comp_count;
    complete = (req->flush_worker.comp_count == 0) || (status != UCS_OK);

    if (complete || force_progress_unreg) {
        uct_worker_progress_unregister_safe(worker->uct,
                                            &req->flush_worker.prog_id);
    }

    if (complete) {
        ucs_assert(status != UCS_INPROGRESS);

        if (&req->flush_worker.next_ep->ep_list != &worker->all_eps) {
            /* Cleanup EP iterator */
            ucp_worker_flush_req_set_next_ep(req, 1, &worker->all_eps);
        }

        /* Coverity wrongly resolves completion callback function to
         * 'ucp_cm_server_conn_request_progress' */
        /* coverity[offset_free] */
        ucp_request_complete(req, flush_worker.cb, status, req->user_data);
    }
}

static void ucp_worker_flush_ep_flushed_cb(ucp_request_t *req)
{
    ucp_worker_flush_complete_one(ucp_request_get_super(req), UCS_OK, 0);
    ucp_request_put(req);
}

static unsigned ucp_worker_flush_progress(void *arg)
{
    ucp_request_t *req        = arg;
    ucp_worker_h worker       = req->flush_worker.worker;
    ucp_ep_ext_gen_t *next_ep = req->flush_worker.next_ep;
    void *ep_flush_request;
    ucs_status_t status;
    ucp_ep_h ep;

    if (worker->flush_ops_count == 0) {
        /* all scheduled progress operations on worker were completed */
        status = ucp_worker_flush_check(worker);
        if ((status == UCS_OK) || (&next_ep->ep_list == &worker->all_eps)) {
            /* If all ifaces are flushed, or we finished going over all
             * endpoints, no need to progress this request actively anymore
             * and we complete the flush operation with UCS_OK status. */
            ucp_worker_flush_complete_one(req, UCS_OK, 1);
            goto out;
        } else if (status != UCS_INPROGRESS) {
            /* Error returned from uct iface flush, no need to progress
             * this request actively anymore and we complete the flush
             * operation with an error status. */
            ucp_worker_flush_complete_one(req, status, 1);
            goto out;
        }
    }

    if (worker->context->config.ext.flush_worker_eps &&
        (&next_ep->ep_list != &worker->all_eps)) {
        /* Some endpoints are not flushed yet. Take the endpoint from the list
         * and start flush operation on it. */
        ep = ucp_worker_flush_req_set_next_ep(req, 1, next_ep->ep_list.next);
        if (ep == NULL) {
            goto out;
        }

        ep_flush_request = ucp_ep_flush_internal(ep, UCP_REQUEST_FLAG_RELEASED,
                                                 &ucp_request_null_param, req,
                                                 ucp_worker_flush_ep_flushed_cb,
                                                 "flush_worker");
        if (UCS_PTR_IS_ERR(ep_flush_request)) {
            /* endpoint flush resulted in an error */
            status = UCS_PTR_STATUS(ep_flush_request);
            ucs_diag("ucp_ep_flush_internal() failed: %s",
                     ucs_status_string(status));
        } else if (ep_flush_request != NULL) {
            /* endpoint flush started, increment refcount */
            ++req->flush_worker.comp_count;
        }
    }

out:
    return 0;
}

static ucs_status_ptr_t
ucp_worker_flush_nbx_internal(ucp_worker_h worker,
                              const ucp_request_param_t *param)
{
    ucs_status_t status;
    ucp_request_t *req;

    if (!worker->flush_ops_count) {
        status = ucp_worker_flush_check(worker);
        if ((status != UCS_INPROGRESS) && (status != UCS_ERR_NO_RESOURCE)) {
            /* UCS_OK is returned here as well */
            return UCS_STATUS_PTR(status);
        }
    }

    req = ucp_request_get_param(worker, param,
                                {return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);});

    req->flags                   = 0;
    req->status                  = UCS_OK;
    req->flush_worker.worker     = worker;
    req->flush_worker.comp_count = 1; /* counting starts from 1, and decremented
                                         when finished going over all endpoints */
    req->flush_worker.prog_id    = UCS_CALLBACKQ_ID_NULL;

    ucp_worker_flush_req_set_next_ep(req, 0, worker->all_eps.next);
    ucp_request_set_send_callback_param(param, req, flush_worker);
    uct_worker_progress_register_safe(worker->uct, ucp_worker_flush_progress,
                                      req, 0, &req->flush_worker.prog_id);
    return req + 1;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_worker_flush_nb, (worker, flags, cb),
                 ucp_worker_h worker, unsigned flags, ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb
    };

    return ucp_worker_flush_nbx(worker, &param);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_worker_flush_nbx, (worker, param),
                 ucp_worker_h worker, const ucp_request_param_t *param)
{
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    request = ucp_worker_flush_nbx_internal(worker, param);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return request;
}

static ucs_status_t ucp_flush_wait(ucp_worker_h worker, void *request)
{
    return ucp_rma_wait(worker, request, "flush");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_flush, (worker), ucp_worker_h worker)
{
    ucs_status_t status;
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    request = ucp_worker_flush_nbx_internal(worker, &ucp_request_null_param);
    status  = ucp_flush_wait(worker, request);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_flush, (ep), ucp_ep_h ep)
{
    ucs_status_t status;
    void *request;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    request = ucp_ep_flush_internal(ep, 0, &ucp_request_null_param, NULL,
                                    ucp_ep_flushed_callback, "flush");
    status = ucp_flush_wait(ep->worker, request);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_fence, (worker), ucp_worker_h worker)
{
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    UCS_BITMAP_FOR_EACH_BIT(worker->context->tl_bitmap, rsc_index) {
        wiface = ucp_worker_iface(worker, rsc_index);
        if (wiface->iface == NULL) {
            continue;
        }

        status = uct_iface_fence(wiface->iface, 0);
        if (status != UCS_OK) {
            goto out;
        }
    }
    status = UCS_OK;

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;
}
