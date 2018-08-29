/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_worker.h"
#include "ucp_mm.h"
#include "ucp_request.inl"

#include <ucp/wireup/address.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/tag/eager.h>
#include <ucp/tag/offload.h>
#include <ucp/stream/stream.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/type/cpu_set.h>
#include <ucs/sys/string.h>
#include <sys/poll.h>
#include <sys/eventfd.h>


#define UCP_WORKER_HEADROOM_SIZE \
    (sizeof(ucp_recv_desc_t) + UCP_WORKER_HEADROOM_PRIV_SIZE)


#if ENABLE_STATS
static ucs_stats_class_t ucp_worker_stats_class = {
    .name           = "ucp_worker",
    .num_counters   = UCP_WORKER_STAT_LAST,
    .counter_names  = {
        [UCP_WORKER_STAT_TAG_RX_EAGER_MSG]         = "rx_eager_msg",
        [UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG]    = "rx_sync_msg",
        [UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP]   = "rx_eager_chunk_exp",
        [UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP] = "rx_eager_chunk_unexp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_EXP]          = "rx_rndv_rts_exp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_UNEXP]        = "rx_rndv_rts_unexp"
    }
};

static ucs_stats_class_t ucp_worker_tm_offload_stats_class = {
    .name           = "tag_offload",
    .num_counters   = UCP_WORKER_STAT_TAG_OFFLOAD_LAST,
    .counter_names  = {
        [UCP_WORKER_STAT_TAG_OFFLOAD_POSTED]           = "posted",
        [UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED]          = "matched",
        [UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED_SW_RNDV]  = "matched_sw_rndv",
        [UCP_WORKER_STAT_TAG_OFFLOAD_CANCELED]         = "canceled",
        [UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_TAG_EXCEED] = "block_tag_exceed",
        [UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NON_CONTIG] = "block_non_contig",
        [UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_WILDCARD]   = "block_wildcard",
        [UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_SW_PEND]    = "block_sw_pend",
        [UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NO_IFACE]   = "block_no_iface",
        [UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_EGR]     = "rx_unexp_egr",
        [UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_RNDV]    = "rx_unexp_rndv",
        [UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV] = "rx_unexp_sw_rndv",
    }
};
#endif


ucs_mpool_ops_t ucp_am_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucs_empty_function,
    .obj_cleanup   = ucs_empty_function
};


ucs_mpool_ops_t ucp_reg_mpool_ops = {
    .chunk_alloc   = ucp_reg_mpool_malloc,
    .chunk_release = ucp_reg_mpool_free,
    .obj_init      = ucp_mpool_obj_init,
    .obj_cleanup   = ucs_empty_function
};

ucs_mpool_ops_t ucp_frag_mpool_ops = {
    .chunk_alloc   = ucp_frag_mpool_malloc,
    .chunk_release = ucp_frag_mpool_free,
    .obj_init      = ucp_mpool_obj_init,
    .obj_cleanup   = ucs_empty_function
};

void ucp_worker_iface_check_events(ucp_worker_iface_t *wiface, int force);


static ucs_status_t ucp_worker_wakeup_ctl_fd(ucp_worker_h worker, int op,
                                             int event_fd)
{
    struct epoll_event event = {0};
    int ret;

    if (!(worker->context->config.features & UCP_FEATURE_WAKEUP)) {
        return UCS_OK;
    }

    memset(&event.data, 0, sizeof(event.data));
    event.data.ptr = worker->user_data;
    event.events   = EPOLLIN;
    if (worker->flags & UCP_WORKER_FLAG_EDGE_TRIGGERED) {
        event.events |= EPOLLET;
    }

    ret = epoll_ctl(worker->epfd, op, event_fd, &event);
    if (ret == -1) {
        ucs_error("epoll_ctl(epfd=%d, op=%d, fd=%d) failed: %m", worker->epfd,
                  op, event_fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static void ucp_worker_set_am_handlers(ucp_worker_iface_t *wiface, int is_proxy)
{
    ucp_worker_h worker   = wiface->worker;
    ucp_context_h context = worker->context;
    ucs_status_t status;
    unsigned am_id;

    ucs_trace_func("iface=%p is_proxy=%d", wiface->iface, is_proxy);

    for (am_id = 0; am_id < UCP_AM_ID_LAST; ++am_id) {
        if (!(wiface->attr.cap.flags & (UCT_IFACE_FLAG_AM_SHORT |
                                        UCT_IFACE_FLAG_AM_BCOPY |
                                        UCT_IFACE_FLAG_AM_ZCOPY))) {
            continue;
        }

        if (!(context->config.features & ucp_am_handlers[am_id].features)) {
            continue;
        }

        if ((ucp_am_handlers[am_id].flags & UCT_CB_FLAG_SYNC) &&
            !(wiface->attr.cap.flags & UCT_IFACE_FLAG_CB_SYNC))
        {
            /* Do not register a sync callback on interface which does not
             * support it. The transport selection logic should not use async
             * transports for protocols with sync active message handlers.
             */
            continue;
        }

        if (is_proxy && (ucp_am_handlers[am_id].proxy_cb != NULL)) {
            /* we care only about sync active messages, and this also makes sure
             * the counter is not accessed from another thread.
             */
            ucs_assert(ucp_am_handlers[am_id].flags & UCT_CB_FLAG_SYNC);
            status = uct_iface_set_am_handler(wiface->iface, am_id,
                                              ucp_am_handlers[am_id].proxy_cb,
                                              wiface,
                                              ucp_am_handlers[am_id].flags);
        } else {
            status = uct_iface_set_am_handler(wiface->iface, am_id,
                                              ucp_am_handlers[am_id].cb,
                                              worker,
                                              ucp_am_handlers[am_id].flags);
        }
        if (status != UCS_OK) {
            ucs_fatal("failed to set active message handler id %d: %s", am_id,
                      ucs_status_string(status));
        }
    }
}

static ucs_status_t ucp_stub_am_handler(void *arg, void *data, size_t length,
                                        unsigned flags)
{
    ucp_worker_h worker = arg;
    ucs_trace("worker %p: drop message", worker);
    return UCS_OK;
}

static void ucp_worker_remove_am_handlers(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t tl_id;
    unsigned am_id;

    ucs_debug("worker %p: remove active message handlers", worker);

    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        if (!(worker->ifaces[tl_id].attr.cap.flags & (UCT_IFACE_FLAG_AM_SHORT |
                                                      UCT_IFACE_FLAG_AM_BCOPY |
                                                      UCT_IFACE_FLAG_AM_ZCOPY))) {
            continue;
        }
        for (am_id = 0; am_id < UCP_AM_ID_LAST; ++am_id) {
            if (context->config.features & ucp_am_handlers[am_id].features) {
                (void)uct_iface_set_am_handler(worker->ifaces[tl_id].iface,
                                               am_id, ucp_stub_am_handler,
                                               worker, UCT_CB_FLAG_ASYNC);
            }
        }
    }
}

static void ucp_worker_am_tracer(void *arg, uct_am_trace_type_t type,
                                 uint8_t id, const void *data, size_t length,
                                 char *buffer, size_t max)
{
    ucp_worker_h worker = arg;
    ucp_am_tracer_t tracer;

    if (id < UCP_AM_ID_LAST) {
        tracer = ucp_am_handlers[id].tracer;
        if (tracer != NULL) {
            tracer(worker, type, id, data, length, buffer, max);
        }
    }
}

static ucs_status_t ucp_worker_wakeup_init(ucp_worker_h worker,
                                           const ucp_worker_params_t *params)
{
    ucp_context_h context = worker->context;
    ucp_wakeup_event_t events;
    ucs_status_t status;

    if (!(context->config.features & UCP_FEATURE_WAKEUP)) {
        worker->epfd       = -1;
        worker->eventfd    = -1;
        worker->uct_events = 0;
        status = UCS_OK;
        goto out;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_EVENTS) {
        events = params->events;
    } else {
        events = UCP_WAKEUP_RMA | UCP_WAKEUP_AMO | UCP_WAKEUP_TAG_SEND |
                 UCP_WAKEUP_TAG_RECV | UCP_WAKEUP_TX | UCP_WAKEUP_RX;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_EVENT_FD) {
        worker->epfd            = params->event_fd;
        worker->flags          |= UCP_WORKER_FLAG_EXTERNAL_EVENT_FD;
    } else {
        worker->epfd = epoll_create(context->num_tls);
        if (worker->epfd == -1) {
            ucs_error("Failed to create epoll file descriptor: %m");
            status = UCS_ERR_IO_ERROR;
            goto out;
        }
    }

    if (events & UCP_WAKEUP_EDGE) {
        worker->flags |= UCP_WORKER_FLAG_EDGE_TRIGGERED;
    }

    worker->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (worker->eventfd == -1) {
        ucs_error("Failed to create event fd: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_close_epfd;
    }

    ucp_worker_wakeup_ctl_fd(worker, EPOLL_CTL_ADD, worker->eventfd);

    worker->uct_events = 0;

    /* FIXME: any TAG flag initializes all types of completion because of
     *        possible issues in RNDV protocol. The optimization may be
     *        implemented with using of separated UCP descriptors or manual
     *        signaling in RNDV and similar cases, see conversation in PR #1277
     */
    if ((events & UCP_WAKEUP_TAG_SEND) ||
        ((events & UCP_WAKEUP_TAG_RECV) &&
         (context->config.ext.rndv_thresh != UCS_CONFIG_MEMUNITS_INF)))
    {
        worker->uct_events |= UCT_EVENT_SEND_COMP;
    }

    if (events & (UCP_WAKEUP_TAG_RECV | UCP_WAKEUP_RX)) {
        worker->uct_events |= UCT_EVENT_RECV;
    }

    if (events & (UCP_WAKEUP_RMA | UCP_WAKEUP_AMO | UCP_WAKEUP_TX)) {
        worker->uct_events |= UCT_EVENT_SEND_COMP;
    }

    return UCS_OK;

err_close_epfd:
    close(worker->epfd);
out:
    return status;
}

static void ucp_worker_wakeup_cleanup(ucp_worker_h worker)
{
    if ((worker->epfd != -1) &&
        !(worker->flags & UCP_WORKER_FLAG_EXTERNAL_EVENT_FD)) {
        close(worker->epfd);
    }
    if (worker->eventfd != -1) {
        close(worker->eventfd);
    }
}

static void ucp_worker_iface_disarm(ucp_worker_iface_t *wiface)
{
    ucs_status_t status;

    if (wiface->flags & UCP_WORKER_IFACE_FLAG_ON_ARM_LIST) {
        status = ucp_worker_wakeup_ctl_fd(wiface->worker, EPOLL_CTL_DEL,
                                          wiface->event_fd);
        ucs_assert_always(status == UCS_OK);
        ucs_list_del(&wiface->arm_list);
        wiface->flags &= ~UCP_WORKER_IFACE_FLAG_ON_ARM_LIST;
    }
}

static ucs_status_t ucp_worker_wakeup_signal_fd(ucp_worker_h worker)
{
    uint64_t dummy = 1;
    int ret;

    ucs_trace_func("worker=%p fd=%d", worker, worker->eventfd);

    do {
        ret = write(worker->eventfd, &dummy, sizeof(dummy));
        if (ret == sizeof(dummy)) {
            return UCS_OK;
        } else if (ret == -1) {
            if (errno == EAGAIN) {
                return UCS_OK;
            } else if (errno != EINTR) {
                ucs_error("Signaling wakeup failed: %m");
                return UCS_ERR_IO_ERROR;
            }
        } else {
            ucs_assert(ret == 0);
        }
    } while (ret == 0);

    return UCS_OK;
}

void ucp_worker_signal_internal(ucp_worker_h worker)
{
    if (worker->context->config.features & UCP_FEATURE_WAKEUP) {
        ucp_worker_wakeup_signal_fd(worker);
    }
}

static unsigned ucp_worker_iface_err_handle_progress(void *arg)
{
    ucp_worker_err_handle_arg_t *err_handle_arg = arg;
    ucp_worker_h worker                         = err_handle_arg->worker;
    ucp_ep_h ucp_ep                             = err_handle_arg->ucp_ep;
    uct_ep_h uct_ep                             = err_handle_arg->uct_ep;
    ucs_status_t status                         = err_handle_arg->status;
    ucp_lane_index_t failed_lane                = err_handle_arg->failed_lane;
    ucp_lane_index_t lane;
    ucp_ep_config_key_t key;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_debug("ep %p: handle error on lane[%d]=%p: %s",
              ucp_ep, failed_lane, uct_ep, ucs_status_string(status));

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_FAILED);

    /* Destroy all lanes except failed one since ucp_ep becomes unusable as well */
    for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
        if (ucp_ep->uct_eps[lane] == NULL) {
            continue;
        }

        /* Purge pending queue */
        ucs_trace("ep %p: purge pending on uct_ep[%d]=%p", ucp_ep, lane,
                  ucp_ep->uct_eps[lane]);
        uct_ep_pending_purge(ucp_ep->uct_eps[lane], ucp_ep_err_pending_purge,
                             UCS_STATUS_PTR(status));

        if (lane != failed_lane) {
            ucs_trace("ep %p: destroy uct_ep[%d]=%p", ucp_ep, lane,
                      ucp_ep->uct_eps[lane]);
            uct_ep_destroy(ucp_ep->uct_eps[lane]);
            ucp_ep->uct_eps[lane] = NULL;
        }
    }

    /* Move failed lane to index 0 */
    if (failed_lane != 0) {
        ucp_ep->uct_eps[0] = ucp_ep->uct_eps[failed_lane];
        ucp_ep->uct_eps[failed_lane] = NULL;
    }

    /* NOTE: if failed ep is wireup auxiliary/sockaddr then we need to replace
     *       the lane with failed ep and destroy wireup ep
     */
    if (ucp_ep->uct_eps[0] != uct_ep) {
        ucs_assert(ucp_wireup_ep_is_owner(ucp_ep->uct_eps[0], uct_ep));
        ucp_wireup_ep_disown(ucp_ep->uct_eps[0], uct_ep);
        ucs_trace("ep %p: destroy failed wireup ep %p", ucp_ep, ucp_ep->uct_eps[0]);
        uct_ep_destroy(ucp_ep->uct_eps[0]);
        ucp_ep->uct_eps[0] = uct_ep;
    }

    /* Redirect all lanes to failed one */
    key                    = ucp_ep_config(ucp_ep)->key;
    key.am_lane            = 0;
    key.wireup_lane        = 0;
    key.tag_lane           = 0;
    key.rma_lanes[0]       = 0;
    key.rma_bw_lanes[0]    = 0;
    key.amo_lanes[0]       = 0;
    key.lanes[0].rsc_index = UCP_NULL_RESOURCE;
    key.num_lanes          = 1;
    key.status             = status;

    ucp_ep->cfg_index = ucp_worker_get_ep_config(worker, &key);
    ucp_ep->am_lane   = 0;

    if (ucp_ep_ext_gen(ucp_ep)->err_cb != NULL) {
        ucs_assert(ucp_ep->flags & UCP_EP_FLAG_USED);
        ucs_debug("ep %p: calling user error callback %p with arg %p", ucp_ep,
                  ucp_ep_ext_gen(ucp_ep)->err_cb,  ucp_ep_ext_gen(ucp_ep)->user_data);
        ucp_ep_ext_gen(ucp_ep)->err_cb(ucp_ep_ext_gen(ucp_ep)->user_data, ucp_ep,
                                       status);
    } else if (!(ucp_ep->flags & UCP_EP_FLAG_USED)) {
        ucs_debug("ep %p: destroy internal endpoint due to peer failure", ucp_ep);
        ucp_ep_disconnected(ucp_ep, 1);
    }

    ucs_free(err_handle_arg);
    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

int ucp_worker_err_handle_remove_filter(const ucs_callbackq_elem_t *elem,
                                        void *arg)
{
    ucp_worker_err_handle_arg_t *err_handle_arg = elem->arg;

    return (elem->cb == ucp_worker_iface_err_handle_progress) &&
           (err_handle_arg->ucp_ep == arg);
}

static ucs_status_t
ucp_worker_iface_error_handler(void *arg, uct_ep_h uct_ep, ucs_status_t status)
{
    ucp_worker_h worker         = (ucp_worker_h)arg;
    uct_worker_cb_id_t prog_id  = UCS_CALLBACKQ_ID_NULL;
    uct_tl_resource_desc_t* tl_rsc;
    ucp_lane_index_t lane, failed_lane;
    ucp_worker_err_handle_arg_t *err_handle_arg;
    ucs_status_t ret_status;
    ucp_rsc_index_t rsc_index;
    ucp_ep_ext_gen_t *ep_ext;
    ucp_ep_h ucp_ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_debug("worker %p: error handler called for uct_ep %p: %s",
              worker, uct_ep, ucs_status_string(status));

    /* TODO: need to optimize uct_ep -> ucp_ep lookup */
    ucs_list_for_each(ep_ext, &worker->all_eps, ep_list) {
        ucp_ep = ucp_ep_from_ext_gen(ep_ext);
        for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
            if ((uct_ep == ucp_ep->uct_eps[lane]) ||
                ucp_wireup_ep_is_owner(ucp_ep->uct_eps[lane], uct_ep)) {
                failed_lane = lane;
                goto found_ucp_ep;
            }
        }
    }

    ucs_fatal("no uct_ep_h %p associated with ucp_ep_h on ucp_worker_h %p",
              uct_ep, worker);

found_ucp_ep:
    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        goto out_ok;
    }

    /* set endpoint to failed to prevent wireup_ep switch */
    ucp_ep->flags |= UCP_EP_FLAG_FAILED;

    if (ucp_ep_config(ucp_ep)->key.err_mode == UCP_ERR_HANDLING_MODE_NONE) {
        /* NOTE: if user has not requested error handling on the endpoint,
         *       the failure is considered unhandled */
        ret_status = status;
        goto out;
    }

    err_handle_arg = ucs_malloc(sizeof(*err_handle_arg), "ucp_worker_err_handle_arg");
    if (err_handle_arg == NULL) {
        ucs_error("failed to allocate ucp_worker_err_handle_arg");
        ret_status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    err_handle_arg->worker      = worker;
    err_handle_arg->ucp_ep      = ucp_ep;
    err_handle_arg->uct_ep      = uct_ep;
    err_handle_arg->status      = status;
    err_handle_arg->failed_lane = failed_lane;

    /* invoke the rest of the error handling flow from the main thread */
    uct_worker_progress_register_safe(worker->uct,
                                      ucp_worker_iface_err_handle_progress,
                                      err_handle_arg, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);

    if ((ucp_ep_ext_gen(ucp_ep)->err_cb == NULL) &&
        (ucp_ep->flags & UCP_EP_FLAG_USED)) {
        rsc_index = ucp_ep_get_rsc_index(ucp_ep, lane);
        tl_rsc    = &worker->context->tl_rscs[rsc_index].tl_rsc;
        ucs_error("error '%s' will not be handled for ep %p - "
                  UCT_TL_RESOURCE_DESC_FMT, ucs_status_string(status), ucp_ep,
                  UCT_TL_RESOURCE_DESC_ARG(tl_rsc));
        ret_status = status;
        goto out;
    }

out_ok:
    ret_status = UCS_OK;

out:
    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(worker);

    UCS_ASYNC_UNBLOCK(&worker->async);

    return ret_status;
}

void ucp_worker_iface_activate(ucp_worker_iface_t *wiface, unsigned uct_flags)
{
    ucp_worker_h worker = wiface->worker;
    ucs_status_t status;

    ucs_trace("activate iface %p acount=%u", wiface->iface,
              wiface->activate_count);

    if (wiface->activate_count++ > 0) {
        return; /* was already activated */
    }

    /* Stop ongoing activation process, if such exists */
    uct_worker_progress_unregister_safe(worker->uct, &wiface->check_events_id);

    /* Set default active message handlers */
    ucp_worker_set_am_handlers(wiface, 0);

    /* Add to user wakeup */
    if (wiface->attr.cap.flags & UCP_WORKER_UCT_ALL_EVENT_CAP_FLAGS) {
        status = ucp_worker_wakeup_ctl_fd(worker, EPOLL_CTL_ADD, wiface->event_fd);
        ucs_assert_always(status == UCS_OK);
        wiface->flags |= UCP_WORKER_IFACE_FLAG_ON_ARM_LIST;
        ucs_list_add_tail(&worker->arm_ifaces, &wiface->arm_list);
    }

    uct_iface_progress_enable(wiface->iface,
                              UCT_PROGRESS_SEND | UCT_PROGRESS_RECV | uct_flags);
}

static void ucp_worker_iface_deactivate(ucp_worker_iface_t *wiface, int force)
{
    ucs_trace("deactivate iface %p force=%d acount=%u", wiface->iface, force,
              wiface->activate_count);

    if (!force) {
        ucs_assert(wiface->activate_count > 0);
        if (--wiface->activate_count > 0) {
            return; /* not completely deactivated yet */
        }
    }

    /* Avoid progress on the interface to reduce overhead */
    uct_iface_progress_disable(wiface->iface,
                               UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    /* Remove from user wakeup */
    ucp_worker_iface_disarm(wiface);

    /* Set proxy active message handlers to count receives */
    ucp_worker_set_am_handlers(wiface, 1);

    /* Prepare for next receive event */
    ucp_worker_iface_check_events(wiface, force);
}

void ucp_worker_iface_progress_ep(ucp_worker_iface_t *wiface)
{
    ucs_trace_func("iface=%p", wiface->iface);

    UCS_ASYNC_BLOCK(&wiface->worker->async);

    /* This function may be called from progress thread (such as when processing
     * wireup messages), so ask UCT to be thread-safe.
     */
    ucp_worker_iface_activate(wiface, UCT_PROGRESS_THREAD_SAFE);

    UCS_ASYNC_UNBLOCK(&wiface->worker->async);
}

void ucp_worker_iface_unprogress_ep(ucp_worker_iface_t *wiface)
{
    ucs_trace_func("iface=%p", wiface->iface);

    UCS_ASYNC_BLOCK(&wiface->worker->async);
    ucp_worker_iface_deactivate(wiface, 0);
    UCS_ASYNC_UNBLOCK(&wiface->worker->async);
}

/*
 * If active messages were received by am proxy handler, activate the interface.
 * Otherwise, arm the interface event and make sure that when an active message
 * is received in the future, the interface would be activated.
 */
static ucs_status_t ucp_worker_iface_check_events_do(ucp_worker_iface_t *wiface,
                                                     unsigned *progress_count)
{
    unsigned prev_recv_count;
    ucs_status_t status;

    ucs_trace_func("iface=%p", wiface->iface);

    if (wiface->activate_count > 0) {
        ucs_trace("iface %p already activated", wiface->iface);
        *progress_count = 0;
        return UCS_OK;
    }

    prev_recv_count = wiface->proxy_recv_count;

    *progress_count = uct_iface_progress(wiface->iface);
    if (prev_recv_count != wiface->proxy_recv_count) {
        /* Received relevant active messages, activate the interface */
        ucp_worker_iface_activate(wiface, 0);
        return UCS_OK;
    } else if (*progress_count == 0) {
        /* Arm the interface to wait for next event */
        ucs_assert(wiface->attr.cap.flags & UCP_WORKER_UCT_RECV_EVENT_CAP_FLAGS);
        status = uct_iface_event_arm(wiface->iface,
                                     UCP_WORKER_UCT_RECV_EVENT_ARM_FLAGS);
        if (status == UCS_OK) {
            ucs_trace("armed iface %p", wiface->iface);

            /* re-enable events, which were disabled by ucp_suspended_iface_event() */
            status = ucs_async_modify_handler(wiface->event_fd, POLLIN);
            if (status != UCS_OK) {
                ucs_fatal("failed to modify %d event handler to POLLIN: %s",
                          wiface->event_fd, ucs_status_string(status));
            }

            return UCS_OK;
        } else if (status != UCS_ERR_BUSY) {
            ucs_fatal("failed to arm iface %p: %s", wiface->iface,
                      ucs_status_string(status));
        } else {
            ucs_trace("arm iface %p returned BUSY", wiface->iface);
            return UCS_ERR_BUSY;
        }
    } else {
        ucs_trace("iface %p progress returned %u, but no active messages were received",
                  wiface->iface, *progress_count);
        return UCS_ERR_BUSY;
    }
}

static unsigned ucp_worker_iface_check_events_progress(void *arg)
{
    ucp_worker_iface_t *wiface = arg;
    ucp_worker_h worker = wiface->worker;
    unsigned progress_count;
    ucs_status_t status;

    ucs_trace_func("iface=%p, worker=%p", wiface->iface, worker);

    /* Check if we either had active messages or were able to arm the interface.
     * In these cases, the work is done and this progress callback can be removed.
     */
    UCS_ASYNC_BLOCK(&worker->async);
    status = ucp_worker_iface_check_events_do(wiface, &progress_count);
    if (status == UCS_OK) {
        uct_worker_progress_unregister_safe(worker->uct, &wiface->check_events_id);
    }
    UCS_ASYNC_UNBLOCK(&worker->async);

    return progress_count;
}

void ucp_worker_iface_check_events(ucp_worker_iface_t *wiface, int force)
{
    unsigned progress_count;
    ucs_status_t status;

    ucs_trace_func("iface=%p, force=%d", wiface->iface, force);

    if (force) {
        do {
            status = ucp_worker_iface_check_events_do(wiface, &progress_count);
            ucs_assert(progress_count == 0);
        } while (status == UCS_ERR_BUSY);
        ucs_assert(status == UCS_OK);
    } else {
        /* Check events on the main progress loop, to make this function safe to
         * call from async context, and avoid starvation of other progress callbacks.
         */
        uct_worker_progress_register_safe(wiface->worker->uct,
                                          ucp_worker_iface_check_events_progress,
                                          wiface, 0, &wiface->check_events_id);
    }
}

void ucp_worker_iface_event(int fd, void *arg)
{
    ucp_worker_iface_t *wiface = arg;
    ucp_worker_h worker = wiface->worker;
    ucs_status_t status;

    ucs_trace_func("fd=%d iface=%p", fd, wiface->iface);

    status = ucs_async_modify_handler(wiface->event_fd, 0);
    if (status != UCS_OK) {
        ucs_fatal("failed to modify %d event handler to <empty>: %s",
                  wiface->event_fd, ucs_status_string(status));
    }

    /* Do more work on the main thread */
    ucp_worker_iface_check_events(wiface, 0);

    /* Signal user wakeup, to report the first message on the interface */
    ucp_worker_signal_internal(worker);
}

static ucs_status_t ucp_worker_add_resource_ifaces(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_tl_resource_desc_t *resource;
    uct_iface_params_t iface_params;
    ucp_rsc_index_t tl_id;
    ucs_status_t status;

    /* Open all resources as interfaces on this worker */
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        memset(&iface_params, 0, sizeof(iface_params));
        resource            = &context->tl_rscs[tl_id];

        if (resource->flags & UCP_TL_RSC_FLAG_SOCKADDR) {
            iface_params.open_mode            = UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT;
        } else {
            iface_params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
            iface_params.mode.device.tl_name  = resource->tl_rsc.tl_name;
            iface_params.mode.device.dev_name = resource->tl_rsc.dev_name;
        }

        status = ucp_worker_iface_init(worker, tl_id, &iface_params,
                                       &worker->ifaces[tl_id]);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucp_worker_close_ifaces(ucp_worker_h worker)
{
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *wiface;

    UCS_ASYNC_BLOCK(&worker->async);
    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        wiface = &worker->ifaces[rsc_index];
        if (wiface->iface != NULL) {
            ucp_worker_iface_cleanup(wiface);
        }
    }
    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_worker_iface_init(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   uct_iface_params_t *iface_params,
                                   ucp_worker_iface_t *wiface)
{
    ucp_context_h context            = worker->context;
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[tl_id];
    uct_md_h md                      = context->tl_mds[resource->md_index].md;
    uct_iface_config_t *iface_config;
    const char *cfg_tl_name;
    ucs_status_t status;

    wiface->rsc_index        = tl_id;
    wiface->worker           = worker;
    wiface->event_fd         = -1;
    wiface->activate_count   = 0;
    wiface->check_events_id  = UCS_CALLBACKQ_ID_NULL;
    wiface->proxy_recv_count = 0;
    wiface->post_count       = 0;
    wiface->flags            = 0;

    /* Read interface or md configuration */
    if (resource->flags & UCP_TL_RSC_FLAG_SOCKADDR) {
        cfg_tl_name = NULL;
    } else {
        cfg_tl_name = resource->tl_rsc.tl_name;
    }
    status = uct_md_iface_config_read(md, cfg_tl_name, NULL, NULL, &iface_config);
    if (status != UCS_OK) {
        return status;
    }

    UCS_STATIC_ASSERT(UCP_WORKER_HEADROOM_PRIV_SIZE >= sizeof(ucp_eager_sync_hdr_t));

    /* Fill rest of uct_iface params (caller should fill specific mode fields) */
    iface_params->stats_root        = UCS_STATS_RVAL(worker->stats);
    iface_params->rx_headroom       = UCP_WORKER_HEADROOM_SIZE;
    iface_params->err_handler_arg   = worker;
    iface_params->err_handler       = ucp_worker_iface_error_handler;
    iface_params->err_handler_flags = UCT_CB_FLAG_ASYNC;
    iface_params->eager_arg         = iface_params->rndv_arg = wiface;
    iface_params->eager_cb          = ucp_tag_offload_unexp_eager;
    iface_params->rndv_cb           = ucp_tag_offload_unexp_rndv;
    iface_params->cpu_mask          = worker->cpu_mask;

    /* Open UCT interface */
    status = uct_iface_open(md, worker->uct, iface_params, iface_config,
                            &wiface->iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
        goto out;
    }

    ucs_debug("created interface[%d]=%p using "UCT_TL_RESOURCE_DESC_FMT" on worker %p",
               tl_id, wiface->iface, UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc),
               worker);

    VALGRIND_MAKE_MEM_UNDEFINED(&wiface->attr, sizeof(wiface->attr));
    status = uct_iface_query(wiface->iface, &wiface->attr);
    if (status != UCS_OK) {
        goto out;
    }

    /* Set wake-up handlers */
    if (wiface->attr.cap.flags & UCP_WORKER_UCT_ALL_EVENT_CAP_FLAGS) {
        status = uct_iface_event_fd_get(wiface->iface, &wiface->event_fd);
        if (status != UCS_OK) {
            goto out_close_iface;
        }

        /* Register event handler without actual events so we could modify it later. */
        status = ucs_async_set_event_handler(worker->async.mode, wiface->event_fd,
                                             0, ucp_worker_iface_event, wiface,
                                             &worker->async);
        if (status != UCS_OK) {
            ucs_fatal("failed to register event handler: %s",
                      ucs_status_string(status));
        }
    }

    /* Set active message handlers */
    if ((wiface->attr.cap.flags & (UCT_IFACE_FLAG_AM_SHORT|
                                   UCT_IFACE_FLAG_AM_BCOPY|
                                   UCT_IFACE_FLAG_AM_ZCOPY)))
    {
        status = uct_iface_set_am_tracer(wiface->iface, ucp_worker_am_tracer,
                                         worker);
        if (status != UCS_OK) {
            goto out_close_iface;
        }

        if (context->config.ext.adaptive_progress &&
            (wiface->attr.cap.flags & UCP_WORKER_UCT_RECV_EVENT_CAP_FLAGS))
        {
            ucp_worker_iface_deactivate(wiface, 1);
        } else {
            ucp_worker_iface_activate(wiface, 0);
        }
    }

    context->mem_type_tls[context->tl_mds[resource->md_index].
                          attr.cap.mem_type] |= UCS_BIT(tl_id);

    return UCS_OK;

out_close_iface:
    uct_iface_close(wiface->iface);
out:
    /* coverity[leaked_storage] */
    return status;
}

void ucp_worker_iface_cleanup(ucp_worker_iface_t *wiface)
{
    ucs_status_t status;

    uct_worker_progress_unregister_safe(wiface->worker->uct,
                                        &wiface->check_events_id);

    ucp_worker_iface_disarm(wiface);

    if (wiface->attr.cap.flags & UCP_WORKER_UCT_ALL_EVENT_CAP_FLAGS) {
        status = ucs_async_remove_handler(wiface->event_fd, 1);
        if (status != UCS_OK) {
            ucs_warn("failed to remove event handler for fd %d: %s",
                     wiface->event_fd, ucs_status_string(status));
        }
    }

    uct_iface_close(wiface->iface);
}

static void ucp_worker_enable_atomic_tl(ucp_worker_h worker, const char *mode,
                                        ucp_rsc_index_t rsc_index)
{
    ucs_assert(rsc_index != UCP_NULL_RESOURCE);
    ucs_trace("worker %p: using %s atomics on iface[%d]=" UCT_TL_RESOURCE_DESC_FMT,
              worker, mode, rsc_index,
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc));
    worker->atomic_tls |= UCS_BIT(rsc_index);
}

static void ucp_worker_init_cpu_atomics(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t rsc_index;

    /* Enable all interfaces which have host-based atomics */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index].attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU) {
            ucp_worker_enable_atomic_tl(worker, "cpu", rsc_index);
        }
    }
}

static void ucp_worker_init_device_atomics(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_address_iface_attr_t dummy_iface_attr;
    ucp_tl_resource_desc_t *rsc, *best_rsc;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t rsc_index;
    uint64_t iface_cap_flags;
    double score, best_score;
    ucp_rsc_index_t md_index;
    uct_md_attr_t *md_attr;
    uint64_t supp_tls;
    uint8_t priority, best_priority;
    ucp_tl_iface_atomic_flags_t atomic;

    ucp_context_uct_atomic_iface_flags(context, &atomic);

    iface_cap_flags             = UCT_IFACE_FLAG_ATOMIC_DEVICE;

    dummy_iface_attr.bandwidth  = 1e12;
    dummy_iface_attr.cap_flags  = -1;
    dummy_iface_attr.overhead   = 0;
    dummy_iface_attr.priority   = 0;
    dummy_iface_attr.lat_ovh    = 0;

    supp_tls                    = 0;
    best_score                  = -1;
    best_rsc                    = NULL;
    best_priority               = 0;

    /* Select best interface for atomics device */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        rsc        = &context->tl_rscs[rsc_index];
        md_index   = rsc->md_index;
        md_attr    = &context->tl_mds[md_index].attr;
        iface_attr = &worker->ifaces[rsc_index].attr;

        if (!(md_attr->cap.flags & UCT_MD_FLAG_REG) ||
            !ucs_test_all_flags(iface_attr->cap.flags, iface_cap_flags)                        ||
            !ucs_test_all_flags(iface_attr->cap.atomic32.op_flags, atomic.atomic32.op_flags)   ||
            !ucs_test_all_flags(iface_attr->cap.atomic32.fop_flags, atomic.atomic32.fop_flags) ||
            !ucs_test_all_flags(iface_attr->cap.atomic64.op_flags, atomic.atomic64.op_flags)   ||
            !ucs_test_all_flags(iface_attr->cap.atomic64.fop_flags, atomic.atomic64.fop_flags))
        {
            continue;
        }

        supp_tls |= UCS_BIT(rsc_index);
        priority  = iface_attr->priority;

        score = ucp_wireup_amo_score_func(context, md_attr, iface_attr,
                                          &dummy_iface_attr);
        if ((score > best_score) ||
            ((score == best_score) && (priority > best_priority)))
        {
            best_rsc      = rsc;
            best_score    = score;
            best_priority = priority;
        }
    }

    if (best_rsc == NULL) {
        ucs_debug("worker %p: no support for atomics", worker);
        return;
    }

    /* Enable atomics on all resources using same device as the "best" resource */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        rsc = &context->tl_rscs[rsc_index];
        if ((supp_tls & UCS_BIT(rsc_index)) &&
            (rsc->md_index == best_rsc->md_index) &&
            !strncmp(rsc->tl_rsc.dev_name, best_rsc->tl_rsc.dev_name,
                     UCT_DEVICE_NAME_MAX))
        {
            ucp_worker_enable_atomic_tl(worker, "device", rsc_index);
        }
    }
}

static void ucp_worker_init_guess_atomics(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t rsc_index;
    uint64_t accumulated_flags = 0;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        accumulated_flags |= worker->ifaces[rsc_index].attr.cap.flags;
    }

    if (accumulated_flags & UCT_IFACE_FLAG_ATOMIC_DEVICE) {
        ucp_worker_init_device_atomics(worker);
    } else {
        ucp_worker_init_cpu_atomics(worker);
    }
}

static void ucp_worker_init_atomic_tls(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;

    worker->atomic_tls = 0;

    if (context->config.features & UCP_FEATURE_AMO) {
        switch(context->config.ext.atomic_mode) {
        case UCP_ATOMIC_MODE_CPU:
            ucp_worker_init_cpu_atomics(worker);
            break;
        case UCP_ATOMIC_MODE_DEVICE:
            ucp_worker_init_device_atomics(worker);
            break;
        case UCP_ATOMIC_MODE_GUESS:
            ucp_worker_init_guess_atomics(worker);
            break;
        default:
            ucs_fatal("unsupported atomic mode: %d",
                      context->config.ext.atomic_mode);
        }
    }
}

static ucs_status_t ucp_worker_init_mpools(ucp_worker_h worker)
{
    size_t           max_mp_entry_size = 0;
    ucp_context_t    *context          = worker->context;
    uct_iface_attr_t *if_attr;
    size_t           tl_id;
    ucs_status_t     status;

    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        if_attr = &worker->ifaces[tl_id].attr;
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_short);
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_bcopy);
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_zcopy);
    }

    status = ucs_mpool_init(&worker->am_mp, 0,
                            max_mp_entry_size + UCP_WORKER_HEADROOM_SIZE,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_am_mpool_ops, "ucp_am_bufs");
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_mpool_init(&worker->reg_mp, 0,
                            context->config.ext.seg_size + sizeof(ucp_mem_desc_t),
                            sizeof(ucp_mem_desc_t), UCS_SYS_CACHE_LINE_SIZE,
                            128, UINT_MAX, &ucp_reg_mpool_ops, "ucp_reg_bufs");
    if (status != UCS_OK) {
        goto err_release_am_mpool;
    }

    status = ucs_mpool_init(&worker->rndv_frag_mp, 0,
                            context->config.ext.rndv_frag_size + sizeof(ucp_mem_desc_t),
                            sizeof(ucp_mem_desc_t), UCS_SYS_CACHE_LINE_SIZE, 128,
                            UINT_MAX, &ucp_frag_mpool_ops, "ucp_rndv_frags");
    if (status != UCS_OK) {
        goto err_release_reg_mpool;
    }

    return UCS_OK;

err_release_reg_mpool:
    ucs_mpool_cleanup(&worker->reg_mp, 0);
err_release_am_mpool:
    ucs_mpool_cleanup(&worker->am_mp, 0);
out:
    return status;
}

/* All the ucp endpoints will share the configurations. No need for every ep to
 * have it's own configuration (to save memory footprint). Same config can be used
 * by different eps.
 * A 'key' identifies an entry in the ep_config array. An entry holds the key and
 * additional configuration parameters and thresholds.
 */
unsigned ucp_worker_get_ep_config(ucp_worker_h worker,
                                  const ucp_ep_config_key_t *key)
{
    ucp_ep_config_t *config;
    unsigned config_idx;

    /* Search for the given key in the ep_config array */
    for (config_idx = 0; config_idx < worker->ep_config_count; ++config_idx) {
        if (ucp_ep_config_is_equal(&worker->ep_config[config_idx].key, key)) {
            goto out;
        }
    }

    if (worker->ep_config_count >= worker->ep_config_max) {
        /* TODO support larger number of configurations */
        ucs_fatal("too many ep configurations: %d", worker->ep_config_count);
    }

    /* Create new configuration */
    config_idx = worker->ep_config_count++;
    config     = &worker->ep_config[config_idx];

    memset(config, 0, sizeof(*config));
    config->key = *key;
    ucp_ep_config_init(worker, config);

out:
    return config_idx;
}

ucs_status_t ucp_worker_create(ucp_context_h context,
                               const ucp_worker_params_t *params,
                               ucp_worker_h *worker_p)
{
    ucp_worker_h worker;
    ucs_status_t status;
    unsigned config_count;
    unsigned name_length;
    ucs_thread_mode_t thread_mode;

    config_count = ucs_min((context->num_tls + 1) * (context->num_tls + 1) * context->num_tls,
                           UINT8_MAX);

    worker = ucs_calloc(1, sizeof(*worker) +
                           sizeof(*worker->ep_config) * config_count,
                        "ucp worker");
    if (worker == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_THREAD_MODE) {
        thread_mode = params->thread_mode;
    } else {
        thread_mode = UCS_THREAD_MODE_SINGLE;
    }

    if (thread_mode != UCS_THREAD_MODE_MULTI) {
        worker->mt_lock.mt_type = UCP_MT_TYPE_NONE;
    } else if (context->config.ext.use_mt_mutex) {
        worker->mt_lock.mt_type = UCP_MT_TYPE_MUTEX;
    } else {
        worker->mt_lock.mt_type = UCP_MT_TYPE_SPINLOCK;
    }

    UCP_THREAD_LOCK_INIT(&worker->mt_lock);

    worker->context           = context;
    worker->uuid              = ucs_generate_uuid((uintptr_t)worker);
    worker->flush_ops_count   = 0;
    worker->flags             = 0;
    worker->inprogress        = 0;
    worker->ep_config_max     = config_count;
    worker->ep_config_count   = 0;
    ucs_list_head_init(&worker->arm_ifaces);
    ucs_list_head_init(&worker->stream_ready_eps);
    ucs_list_head_init(&worker->all_eps);
    ucp_ep_match_init(&worker->ep_match_ctx);

    UCS_STATIC_ASSERT(sizeof(ucp_ep_ext_gen_t) <= sizeof(ucp_ep_t));
    if (context->config.features & UCP_FEATURE_STREAM) {
        UCS_STATIC_ASSERT(sizeof(ucp_ep_ext_proto_t) <= sizeof(ucp_ep_t));
        ucs_strided_alloc_init(&worker->ep_alloc, sizeof(ucp_ep_t), 3);
    } else {
        ucs_strided_alloc_init(&worker->ep_alloc, sizeof(ucp_ep_t), 2);
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_USER_DATA) {
        worker->user_data = params->user_data;
    } else {
        worker->user_data = NULL;
    }

    name_length = ucs_min(UCP_WORKER_NAME_MAX,
                          context->config.ext.max_worker_name + 1);
    ucs_snprintf_zero(worker->name, name_length, "%s:%d", ucs_get_host_name(),
                      getpid());

    worker->ifaces = ucs_calloc(context->num_tls, sizeof(ucp_worker_iface_t),
                                "ucp iface");
    if (worker->ifaces == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&worker->stats, &ucp_worker_stats_class,
                                  ucs_stats_get_root(), "-%p", worker);
    if (status != UCS_OK) {
        goto err_free_ifaces;
    }

    status = UCS_STATS_NODE_ALLOC(&worker->tm_offload_stats,
                                  &ucp_worker_tm_offload_stats_class,
                                  worker->stats);
    if (status != UCS_OK) {
        goto err_free_stats;
    }

    status = ucs_async_context_init(&worker->async, UCS_ASYNC_MODE_THREAD);
    if (status != UCS_OK) {
        goto err_free_tm_offload_stats;
    }

    /* Create the underlying UCT worker */
    status = uct_worker_create(&worker->async, thread_mode, &worker->uct);
    if (status != UCS_OK) {
        goto err_destroy_async;
    }

    /* Create memory pool for requests */
    status = ucs_mpool_init(&worker->req_mp, 0,
                            sizeof(ucp_request_t) + context->config.request.size,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_request_mpool_ops, "ucp_requests");
    if (status != UCS_OK) {
        goto err_destroy_uct_worker;
    }

    /* Create epoll set which combines events from all transports */
    status = ucp_worker_wakeup_init(worker, params);
    if (status != UCS_OK) {
        goto err_req_mp_cleanup;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_CPU_MASK) {
        worker->cpu_mask = params->cpu_mask;
    } else {
        UCS_CPU_ZERO(&worker->cpu_mask);
    }

    /* Initialize tag matching */
    status = ucp_tag_match_init(&worker->tm);
    if (status != UCS_OK) {
        goto err_wakeup_cleanup;
    }

    /* Open all resources as interfaces on this worker */
    status = ucp_worker_add_resource_ifaces(worker);
    if (status != UCS_OK) {
        goto err_close_ifaces;
    }

    /* create mem type endponts */
    status = ucp_worker_create_mem_type_endpoints(worker);;
    if (status != UCS_OK) {
        goto err_close_ifaces;
    }

    /* Init AM and registered memory pools */
    status = ucp_worker_init_mpools(worker);
    if (status != UCS_OK) {
        goto err_close_ifaces;
    }

    /* Select atomic resources */
    ucp_worker_init_atomic_tls(worker);

    *worker_p = worker;
    return UCS_OK;

err_close_ifaces:
    ucp_worker_close_ifaces(worker);
    ucp_tag_match_cleanup(&worker->tm);
err_wakeup_cleanup:
    ucp_worker_wakeup_cleanup(worker);
err_req_mp_cleanup:
    ucs_mpool_cleanup(&worker->req_mp, 1);
err_destroy_uct_worker:
    uct_worker_destroy(worker->uct);
err_destroy_async:
    ucs_async_context_cleanup(&worker->async);
err_free_tm_offload_stats:
    UCS_STATS_NODE_FREE(worker->tm_offload_stats);
err_free_stats:
    UCS_STATS_NODE_FREE(worker->stats);
err_free_ifaces:
    ucs_free(worker->ifaces);
err_free:
    ucs_strided_alloc_cleanup(&worker->ep_alloc);
    UCP_THREAD_LOCK_FINALIZE(&worker->mt_lock);
    ucs_free(worker);
    return status;
}

static void ucp_worker_destroy_eps(ucp_worker_h worker)
{
    ucp_ep_ext_gen_t *ep_ext, *tmp;

    ucs_debug("worker %p: destroy all endpoints", worker);
    ucs_list_for_each_safe(ep_ext, tmp, &worker->all_eps, ep_list) {
        ucp_ep_disconnected(ucp_ep_from_ext_gen(ep_ext), 1);
    }
}

void ucp_worker_destroy(ucp_worker_h worker)
{
    ucs_trace_func("worker=%p", worker);

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_worker_destroy_eps(worker);
    ucp_worker_remove_am_handlers(worker);
    UCS_ASYNC_UNBLOCK(&worker->async);

    ucs_mpool_cleanup(&worker->am_mp, 1);
    ucs_mpool_cleanup(&worker->reg_mp, 1);
    ucs_mpool_cleanup(&worker->rndv_frag_mp, 1);
    ucp_worker_close_ifaces(worker);
    ucp_tag_match_cleanup(&worker->tm);
    ucp_worker_wakeup_cleanup(worker);
    ucs_mpool_cleanup(&worker->req_mp, 1);
    uct_worker_destroy(worker->uct);
    ucs_async_context_cleanup(&worker->async);
    ucs_free(worker->ifaces);
    ucp_ep_match_cleanup(&worker->ep_match_ctx);
    ucs_strided_alloc_cleanup(&worker->ep_alloc);
    UCP_THREAD_LOCK_FINALIZE(&worker->mt_lock);
    UCS_STATS_NODE_FREE(worker->tm_offload_stats);
    UCS_STATS_NODE_FREE(worker->stats);
    ucs_free(worker);
}

ucs_status_t ucp_worker_query(ucp_worker_h worker,
                              ucp_worker_attr_t *attr)
{
    if (attr->field_mask & UCP_WORKER_ATTR_FIELD_THREAD_MODE) {
        if (UCP_THREAD_IS_REQUIRED(&worker->mt_lock)) {
            attr->thread_mode = UCS_THREAD_MODE_MULTI;
        } else {
            attr->thread_mode = UCS_THREAD_MODE_SINGLE;
        }
    }

    return UCS_OK;
}

unsigned ucp_worker_progress(ucp_worker_h worker)
{
    unsigned count;

    /* worker->inprogress is used only for assertion check.
     * coverity[assert_side_effect]
     */
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    /* check that ucp_worker_progress is not called from within ucp_worker_progress */
    ucs_assert(worker->inprogress++ == 0);
    count = uct_worker_progress(worker->uct);
    ucs_async_check_miss(&worker->async);

    /* coverity[assert_side_effect] */
    ucs_assert(--worker->inprogress == 0);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return count;
}

ssize_t ucp_stream_worker_poll(ucp_worker_h worker,
                               ucp_stream_poll_ep_t *poll_eps,
                               size_t max_eps, unsigned flags)
{
    ucp_ep_ext_proto_t *ep_ext;
    ssize_t count = 0;
    ucp_ep_h ep;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    while ((count < max_eps) && !ucs_list_is_empty(&worker->stream_ready_eps)) {
        ep_ext                    = ucp_stream_worker_dequeue_ep_head(worker);
        ep                        = ucp_ep_from_ext_proto(ep_ext);
        poll_eps[count].ep        = ep;
        poll_eps[count].user_data = ucp_ep_ext_gen(ep)->user_data;
        ++count;
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return count;
}

ucs_status_t ucp_worker_get_efd(ucp_worker_h worker, int *fd)
{
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
    if (worker->flags & UCP_WORKER_FLAG_EXTERNAL_EVENT_FD) {
        status = UCS_ERR_UNSUPPORTED;
    } else {
        *fd = worker->epfd;
        status = UCS_OK;
    }
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;
}

ucs_status_t ucp_worker_arm(ucp_worker_h worker)
{
    ucp_worker_iface_t *wiface;
    ucs_status_t status;
    uint64_t dummy;
    int ret;

    ucs_trace_func("worker=%p", worker);

    /* Read from event pipe. If some events are found, return BUSY,
     * Otherwise, continue to arm the transport interfaces.
     */
    do {
        ret = read(worker->eventfd, &dummy, sizeof(dummy));
        if (ret == sizeof(dummy)) {
            status = UCS_ERR_BUSY;
            goto out;
        } else if (ret == -1) {
            if (errno == EAGAIN) {
                break; /* No more events */
            } else if (errno != EINTR) {
                ucs_error("Read from internal event fd failed: %m");
                status = UCS_ERR_IO_ERROR;
                goto out;
            }
        } else {
            ucs_assert(ret == 0);
        }
    } while (ret != 0);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    /* Go over arm_list of active interfaces which support events and arm them */
    ucs_list_for_each(wiface, &worker->arm_ifaces, arm_list) {
        ucs_assert(wiface->activate_count > 0);
        status = uct_iface_event_arm(wiface->iface, worker->uct_events);
        ucs_trace("arm iface %p returned %s", wiface->iface,
                  ucs_status_string(status));
        if (status != UCS_OK) {
            goto out_unlock;
        }
    }

    status = UCS_OK;

out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
out:
    ucs_trace("ucp_worker_arm returning %s", ucs_status_string(status));
    return status;
}

void ucp_worker_wait_mem(ucp_worker_h worker, void *address)
{
   ucs_arch_wait_mem(address);
}

ucs_status_t ucp_worker_wait(ucp_worker_h worker)
{
    ucp_worker_iface_t *wiface;
    struct pollfd *pfd;
    ucs_status_t status;
    nfds_t nfds;
    int ret;

    ucs_trace_func("worker %p", worker);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    status = ucp_worker_arm(worker);
    if (status == UCS_ERR_BUSY) { /* if UCS_ERR_BUSY returned - no poll() must called */
        status = UCS_OK;
        goto out;
    } else if (status != UCS_OK) {
        goto out;
    }

    if (worker->flags & UCP_WORKER_FLAG_EXTERNAL_EVENT_FD) {
        pfd = ucs_alloca(sizeof(*pfd) * worker->context->num_tls);
        nfds = 0;
        ucs_list_for_each(wiface, &worker->arm_ifaces, arm_list) {
            pfd[nfds].fd     = wiface->event_fd;
            pfd[nfds].events = POLLIN;
            ++nfds;
        }
    } else {
        pfd = ucs_alloca(sizeof(*pfd));
        pfd->fd      = worker->epfd;
        pfd->events  = POLLIN;
        nfds         = 1;
    }

    for (;;) {
        ret = poll(pfd, nfds, -1);
        if (ret >= 0) {
            ucs_assertv(ret == 1, "ret=%d", ret);
            status = UCS_OK;
            goto out;
        } else {
            if (errno != EINTR) {
                ucs_error("poll(nfds=%d) returned %d: %m", (int)nfds, ret);
                status = UCS_ERR_IO_ERROR;
                goto out;
            }
        }
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;
}

ucs_status_t ucp_worker_signal(ucp_worker_h worker)
{
    ucs_trace_func("worker %p", worker);
    return ucp_worker_wakeup_signal_fd(worker);
}

ucs_status_t ucp_worker_get_address(ucp_worker_h worker, ucp_address_t **address_p,
                                    size_t *address_length_p)
{
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    status = ucp_address_pack(worker, NULL, -1, NULL, address_length_p,
                              (void**)address_p);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return status;
}

void ucp_worker_release_address(ucp_worker_h worker, ucp_address_t *address)
{
    ucs_free(address);
}


void ucp_worker_print_info(ucp_worker_h worker, FILE *stream)
{
    ucp_context_h context = worker->context;
    ucp_address_t *address;
    size_t address_length;
    ucs_status_t status;
    ucp_rsc_index_t rsc_index;
    int first;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP worker '%s'\n", ucp_worker_get_name(worker));
    fprintf(stream, "#\n");

    status = ucp_worker_get_address(worker, &address, &address_length);
    if (status == UCS_OK) {
        ucp_worker_release_address(worker, address);
        fprintf(stream, "#                 address: %zu bytes\n", address_length);
    } else {
        fprintf(stream, "# <failed to get address>\n");
    }

    if (context->config.features & UCP_FEATURE_AMO) {
        fprintf(stream, "#                 atomics: ");
        first = 1;
        for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
            if (worker->atomic_tls & UCS_BIT(rsc_index)) {
                if (!first) {
                    fprintf(stream, ", ");
                }
                fprintf(stream, "%d:"UCT_TL_RESOURCE_DESC_FMT, rsc_index,
                        UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc));
                first = 0;
            }
        }
        fprintf(stream, "\n");
    }

    fprintf(stream, "#\n");

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
}
