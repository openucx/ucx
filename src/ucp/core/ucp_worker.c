/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_am.h"
#include "ucp_worker.h"
#include "ucp_rkey.h"
#include "ucp_request.inl"

#include <ucp/wireup/address.h>
#include <ucp/wireup/wireup_cm.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/tag/eager.h>
#include <ucp/tag/offload.h>
#include <ucp/stream/stream.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/ptr_map.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/type/cpu_set.h>
#include <ucs/sys/string.h>
#include <ucs/arch/atomic.h>
#include <sys/poll.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>


#define UCP_WORKER_HEADROOM_SIZE \
    (sizeof(ucp_recv_desc_t) + UCP_WORKER_HEADROOM_PRIV_SIZE)

typedef enum ucp_worker_event_fd_op {
    UCP_WORKER_EPFD_OP_ADD,
    UCP_WORKER_EPFD_OP_DEL
} ucp_worker_event_fd_op_t;

#ifdef ENABLE_STATS
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
        [UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_MEM_REG]    = "block_mem_reg",
        [UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_EGR]     = "rx_unexp_egr",
        [UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_RNDV]    = "rx_unexp_rndv",
        [UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV] = "rx_unexp_sw_rndv"
    }
};

static ucs_stats_class_t ucp_worker_stats_class = {
    .name           = "ucp_worker",
    .num_counters   = UCP_WORKER_STAT_LAST,
    .counter_names  = {
        [UCP_WORKER_STAT_TAG_RX_EAGER_MSG]         = "rx_eager_msg",
        [UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG]    = "rx_sync_msg",
        [UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP]   = "rx_eager_chunk_exp",
        [UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP] = "rx_eager_chunk_unexp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_EXP]          = "rx_rndv_rts_exp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_UNEXP]        = "rx_rndv_rts_unexp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_GET_ZCOPY]    = "rx_rndv_get_zcopy",
        [UCP_WORKER_STAT_TAG_RX_RNDV_SEND_RTR]     = "rx_rndv_send_rtr",
        [UCP_WORKER_STAT_TAG_RX_RNDV_RKEY_PTR]     = "rx_rndv_rkey_ptr"
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

static ucs_mpool_ops_t ucp_rkey_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

#define ucp_worker_discard_uct_ep_hash_key(_uct_ep) \
    kh_int64_hash_func((uintptr_t)(_uct_ep))


KHASH_IMPL(ucp_worker_discard_uct_ep_hash, uct_ep_h, ucp_request_t*, 1,
           ucp_worker_discard_uct_ep_hash_key, kh_int64_hash_equal);


static ucs_status_t ucp_worker_wakeup_ctl_fd(ucp_worker_h worker,
                                             ucp_worker_event_fd_op_t op,
                                             int event_fd)
{
    ucs_event_set_types_t events = UCS_EVENT_SET_EVREAD;
    ucs_status_t status;

    if (!(worker->context->config.features & UCP_FEATURE_WAKEUP)) {
        return UCS_OK;
    }

    if (worker->flags & UCP_WORKER_FLAG_EDGE_TRIGGERED) {
        events |= UCS_EVENT_SET_EDGE_TRIGGERED;
    }

    switch (op) {
    case UCP_WORKER_EPFD_OP_ADD:
        status = ucs_event_set_add(worker->event_set, event_fd,
                                   events, worker->user_data);
        break;
    case UCP_WORKER_EPFD_OP_DEL:
        status = ucs_event_set_del(worker->event_set, event_fd);
        break;
    default:
        ucs_bug("Unknown operation (%d) was passed", op);
        status = UCS_ERR_INVALID_PARAM;
        break;
    }

    return status;
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

        if (!(ucp_am_handlers[am_id].flags & UCT_CB_FLAG_ASYNC) &&
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
            ucs_assert(!(ucp_am_handlers[am_id].flags & UCT_CB_FLAG_ASYNC));
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
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t iface_id;
    unsigned am_id;

    ucs_debug("worker %p: remove active message handlers", worker);

    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        wiface = worker->ifaces[iface_id];
        if (!(wiface->attr.cap.flags & (UCT_IFACE_FLAG_AM_SHORT |
                                        UCT_IFACE_FLAG_AM_BCOPY |
                                        UCT_IFACE_FLAG_AM_ZCOPY))) {
            continue;
        }
        for (am_id = 0; am_id < UCP_AM_ID_LAST; ++am_id) {
            if (context->config.features & ucp_am_handlers[am_id].features) {
                (void)uct_iface_set_am_handler(wiface->iface,
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
    unsigned events;
    ucs_status_t status;

    if (!(context->config.features & UCP_FEATURE_WAKEUP)) {
        worker->event_fd   = -1;
        worker->event_set  = NULL;
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
        worker->flags |= UCP_WORKER_FLAG_EXTERNAL_EVENT_FD;
        status = ucs_event_set_create_from_fd(&worker->event_set,
                                              params->event_fd);
    } else {
        status = ucs_event_set_create(&worker->event_set);
    }
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_event_set_fd_get(worker->event_set, &worker->event_fd);
    if (status != UCS_OK) {
        goto err_cleanup_event_set;
    }

    if (events & UCP_WAKEUP_EDGE) {
        worker->flags |= UCP_WORKER_FLAG_EDGE_TRIGGERED;
    }

    worker->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (worker->eventfd == -1) {
        ucs_error("Failed to create event fd: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_cleanup_event_set;
    }

    ucp_worker_wakeup_ctl_fd(worker, UCP_WORKER_EPFD_OP_ADD, worker->eventfd);

    worker->uct_events = 0;

    /* FIXME: any TAG flag initializes all types of completion because of
     *        possible issues in RNDV protocol. The optimization may be
     *        implemented with using of separated UCP descriptors or manual
     *        signaling in RNDV and similar cases, see conversation in PR #1277
     */
    if ((events & UCP_WAKEUP_TAG_SEND) ||
        ((events & UCP_WAKEUP_TAG_RECV) &&
         (context->config.ext.rndv_thresh != UCS_MEMUNITS_INF)))
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

err_cleanup_event_set:
    ucs_event_set_cleanup(worker->event_set);
    worker->event_set = NULL;
    worker->event_fd  = -1;
out:
    return status;
}

static void ucp_worker_wakeup_cleanup(ucp_worker_h worker)
{
    if (worker->event_set != NULL) {
        ucs_assert(worker->event_fd != -1);
        ucs_event_set_cleanup(worker->event_set);
        worker->event_set = NULL;
        worker->event_fd  = -1;
    }
    if (worker->eventfd != -1) {
        close(worker->eventfd);
    }
}

static UCS_F_ALWAYS_INLINE
int ucp_worker_iface_has_event_notify(const ucp_worker_iface_t *wiface)
{
    return (wiface->attr.cap.event_flags & (UCT_IFACE_FLAG_EVENT_FD |
                                            UCT_IFACE_FLAG_EVENT_ASYNC_CB));
}

static UCS_F_ALWAYS_INLINE
int ucp_worker_iface_use_event_fd(const ucp_worker_iface_t *wiface)
{
    /* use iface's fd if it is supported by UCT iface and asynchronous
     * callback mechanism isn't supported (this is preferred mechanism,
     * since it will be called anyway) */
    return (wiface->attr.cap.event_flags & UCT_IFACE_FLAG_EVENT_FD) &&
           !(wiface->attr.cap.event_flags & UCT_IFACE_FLAG_EVENT_ASYNC_CB);
}

static UCS_F_ALWAYS_INLINE
int ucp_worker_iface_get_event_fd(const ucp_worker_iface_t *wiface)
{
    ucs_assert(ucp_worker_iface_use_event_fd(wiface));
    return wiface->event_fd;
}

static UCS_F_ALWAYS_INLINE
void ucp_worker_iface_event_fd_ctl(ucp_worker_iface_t *wiface,
                                   ucp_worker_event_fd_op_t op)
{
    ucs_status_t status;

    status = ucp_worker_wakeup_ctl_fd(wiface->worker, op,
                                      ucp_worker_iface_get_event_fd(wiface));
    ucs_assert_always(status == UCS_OK);
}

static void ucp_worker_iface_disarm(ucp_worker_iface_t *wiface)
{
    if (wiface->flags & UCP_WORKER_IFACE_FLAG_ON_ARM_LIST) {
        if (ucp_worker_iface_use_event_fd(wiface)) {
            ucp_worker_iface_event_fd_ctl(wiface, UCP_WORKER_EPFD_OP_DEL);
        }
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
    ucp_ep_h ucp_ep                             = err_handle_arg->ucp_ep;
    ucs_status_t status                         = err_handle_arg->status;
    ucp_worker_h worker                         = ucp_ep->worker;
    ucp_request_t *close_req;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_debug("ep %p: handle error: %s", ucp_ep, ucs_status_string(status));

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_FAILED);

    ucp_ep_discard_lanes(ucp_ep, status);
    ucp_stream_ep_cleanup(ucp_ep);
    if (ucp_ep->flags & UCP_EP_FLAG_USED) {
        if (ucp_ep->flags & UCP_EP_FLAG_CLOSE_REQ_VALID) {
            ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CLOSED);
            /* Promote close operation to CANCEL in case of transport error,
             * since the disconnect event may never arrive. */
            close_req                        = ucp_ep_ext_control(ucp_ep)->
                                               close_req.req;
            close_req->send.flush.uct_flags |= UCT_FLUSH_FLAG_CANCEL;
            ucp_ep_local_disconnect_progress(close_req);
        } else {
            ucp_ep_invoke_err_cb(ucp_ep, status);
        }
    } else {
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

    if ((elem->cb == ucp_worker_iface_err_handle_progress) &&
        (err_handle_arg->ucp_ep == arg)) {
        /* release err handling argument to avoid memory leak */
        ucs_free(err_handle_arg);
        return 1;
    }

    return 0;
}

/*
 * Caller must acquire lock
 */
ucs_status_t ucp_worker_set_ep_failed(ucp_worker_h worker, ucp_ep_h ucp_ep,
                                      uct_ep_h uct_ep, ucp_lane_index_t lane,
                                      ucs_status_t status)
{
    uct_worker_cb_id_t          prog_id    = UCS_CALLBACKQ_ID_NULL;
    ucs_status_t                ret_status = UCS_OK;
    ucp_rsc_index_t             rsc_index;
    uct_tl_resource_desc_t      *tl_rsc;
    ucp_worker_err_handle_arg_t *err_handle_arg;
    ucs_log_level_t             log_level;

    /* In case if this is a local failure we need to notify remote side */
    if (ucp_ep_is_cm_local_connected(ucp_ep)) {
        ucp_ep_cm_disconnect_cm_lane(ucp_ep);
    }

    /* set endpoint to failed to prevent wireup_ep switch */
    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        goto out_ok;
    }

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

    err_handle_arg->ucp_ep      = ucp_ep;
    err_handle_arg->status      = status;

    /* invoke the rest of the error handling flow from the main thread */
    uct_worker_progress_register_safe(worker->uct,
                                      ucp_worker_iface_err_handle_progress,
                                      err_handle_arg, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);

    if ((ucp_ep_ext_control(ucp_ep)->err_cb == NULL) &&
        (ucp_ep->flags & UCP_EP_FLAG_USED)) {
        /* do not print error if connection reset by remote peer since it can
         * be part of user level close protocol  */
        log_level = (status == UCS_ERR_CONNECTION_RESET) ? UCS_LOG_LEVEL_DIAG :
                    UCS_LOG_LEVEL_ERROR;

        if (lane != UCP_NULL_LANE) {
            rsc_index = ucp_ep_get_rsc_index(ucp_ep, lane);
            tl_rsc    = &worker->context->tl_rscs[rsc_index].tl_rsc;
            ucs_log(log_level, "error '%s' will not be handled for ep %p - "
                    UCT_TL_RESOURCE_DESC_FMT " since no error callback is installed",
                    ucs_status_string(status), ucp_ep,
                    UCT_TL_RESOURCE_DESC_ARG(tl_rsc));
        } else {
            ucs_assert(uct_ep == NULL);
            ucs_log(log_level, "error '%s' occurred on wireup will not be "
                    "handled for ep %p since no error callback is installed",
                    ucs_status_string(status), ucp_ep);
        }
        ret_status = status;
        goto out;
    }

out_ok:
    ret_status = UCS_OK;

out:
    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(worker);

    return ret_status;
}

static ucs_status_t
ucp_worker_iface_handle_uct_ep_failure(ucp_ep_h ucp_ep, ucp_lane_index_t lane,
                                       uct_ep_h uct_ep, ucs_status_t status)
{
    ucp_worker_h worker = ucp_ep->worker;
    ucp_wireup_ep_t *wireup_ep;
    uct_ep_h aux_uct_ep;

    /* If the failure happened on AUX EP of CM lane on a server EP,
     * it means that client closed its CM_WIREUP_EP/AUX_EP and it
     * was detected before receiving WIREUP_MSG/ACK from a client or
     * marking a server's EP as REMOTE_CONNECTED was scheduled on a
     * progress, but not completed yet (CM_WIREUP_EP/AUX_EP is
     * closed when moving an EP to REMOTE_CONNECTED state) */
    wireup_ep = ucp_wireup_ep(ucp_ep->uct_eps[lane]);
    if ((lane == ucp_ep_get_cm_lane(ucp_ep))         &&
        (lane == ucp_ep_get_wireup_msg_lane(ucp_ep)) &&
        (wireup_ep != NULL)                          &&
        ucp_wireup_aux_ep_is_owner(wireup_ep, uct_ep)) {
        ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED);

        /* No need to invoke the error handling flow, just flush and
         * destroy CM_WIREUP/AUX_EP */
        aux_uct_ep = wireup_ep->aux_ep;

        ucp_wireup_ep_disown(ucp_ep->uct_eps[lane], aux_uct_ep);
        ucp_worker_discard_uct_ep(ucp_ep, aux_uct_ep, UCT_FLUSH_FLAG_CANCEL,
                                  (uct_pending_purge_callback_t)
                                  ucs_empty_function_do_assert, NULL);
        return UCS_OK;
    }

    return ucp_worker_set_ep_failed(worker, ucp_ep, uct_ep, lane, status);
}

static ucs_status_t
ucp_worker_iface_error_handler(void *arg, uct_ep_h uct_ep, ucs_status_t status)
{
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucp_lane_index_t lane;
    ucs_status_t ret_status;
    ucp_ep_ext_gen_t *ep_ext;
    ucp_ep_h ucp_ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_debug("worker %p: error handler called for UCT EP %p: %s",
              worker, uct_ep, ucs_status_string(status));

    if (ucp_worker_is_uct_ep_discarding(worker, uct_ep)) {
        ucs_debug("UCT EP %p is being discarded on UCP Worker %p",
                  uct_ep, worker);
        ret_status = UCS_OK;
        goto out;
    }

    /* TODO: need to optimize uct_ep -> ucp_ep lookup */
    ucs_list_for_each(ep_ext, &worker->all_eps, ep_list) {
        ucp_ep = ucp_ep_from_ext_gen(ep_ext);
        for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
            if ((uct_ep == ucp_ep->uct_eps[lane]) ||
                ucp_wireup_ep_is_owner(ucp_ep->uct_eps[lane], uct_ep)) {
                ret_status = ucp_worker_iface_handle_uct_ep_failure(ucp_ep,
                                                                    lane,
                                                                    uct_ep,
                                                                    status);
                goto out;
            }
        }
    }

    ucs_error("UCT EP %p isn't associated with UCP EP and was not scheduled "
              "to be discarded on UCP Worker %p",
              uct_ep, worker);
    ret_status = UCS_ERR_NO_ELEM;

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return ret_status;
}

void ucp_worker_iface_activate(ucp_worker_iface_t *wiface, unsigned uct_flags)
{
    ucp_worker_h worker = wiface->worker;

    ucs_trace("activate iface %p acount=%u aifaces=%u", wiface->iface,
              wiface->activate_count, worker->num_active_ifaces);

    if (wiface->activate_count++ > 0) {
        return; /* was already activated */
    }

    /* Stop ongoing activation process, if such exists */
    uct_worker_progress_unregister_safe(worker->uct, &wiface->check_events_id);

    /* Set default active message handlers */
    ucp_worker_set_am_handlers(wiface, 0);

    if (ucp_worker_iface_has_event_notify(wiface)) {
        if (ucp_worker_iface_use_event_fd(wiface)) {
            /* Add to user wakeup */
            ucp_worker_iface_event_fd_ctl(wiface, UCP_WORKER_EPFD_OP_ADD);
        }

        /* Add to the list of UCT ifaces that should be armed */
        wiface->flags |= UCP_WORKER_IFACE_FLAG_ON_ARM_LIST;
        ucs_list_add_tail(&worker->arm_ifaces, &wiface->arm_list);
    }

    ++worker->num_active_ifaces;

    uct_iface_progress_enable(wiface->iface,
                              UCT_PROGRESS_SEND | UCT_PROGRESS_RECV | uct_flags);
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

    ucs_trace_func("wiface=%p iface=%p", wiface, wiface->iface);

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
        ucs_assert(wiface->attr.cap.event_flags & UCT_IFACE_FLAG_EVENT_RECV);
        status = uct_iface_event_arm(wiface->iface, UCT_EVENT_RECV);
        if (status == UCS_OK) {
            ucs_trace("armed iface %p", wiface->iface);

            if (ucp_worker_iface_use_event_fd(wiface)) {
                /* re-enable events, which were disabled by
                 * ucp_worker_iface_async_fd_event() */
                status = ucs_async_modify_handler(wiface->event_fd,
                                                  UCS_EVENT_SET_EVREAD);
                if (status != UCS_OK) {
                    ucs_fatal("failed to modify %d event handler to UCS_EVENT_SET_EVREAD: %s",
                              wiface->event_fd, ucs_status_string(status));
                }
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
        ucs_trace("wiface %p progress returned %u, but no active messages were received",
                  wiface, *progress_count);
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

static void ucp_worker_iface_check_events(ucp_worker_iface_t *wiface, int force)
{
    unsigned progress_count;
    ucs_status_t status;

    ucs_trace_func("iface=%p, force=%d", wiface->iface, force);

    if (force) {
        do {
            /* coverity wrongly resolves rc's progress to ucp_listener_conn_request_progress
             * which in turn releases wiface->iface. this leads coverity to assume
             * that ucp_worker_iface_check_events_do() dereferences a freed pointer
             * in the subsequent call in the following loop */
            /* coverity[freed_arg] */
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

static void ucp_worker_iface_deactivate(ucp_worker_iface_t *wiface, int force)
{
    ucs_trace("deactivate iface %p force=%d acount=%u aifaces=%u",
              wiface->iface, force, wiface->activate_count,
              wiface->worker->num_active_ifaces);

    if (!force) {
        ucs_assert(wiface->activate_count > 0);
        if (--wiface->activate_count > 0) {
            return; /* not completely deactivated yet */
        }
        --wiface->worker->num_active_ifaces;
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

static UCS_F_ALWAYS_INLINE void
ucp_worker_iface_event_common(ucp_worker_iface_t *wiface)
{
    ucp_worker_h worker = wiface->worker;

    /* Do more work on the main thread */
    ucp_worker_iface_check_events(wiface, 0);

    /* Signal user wakeup to report the first message on the interface */
    ucp_worker_signal_internal(worker);
}

static void ucp_worker_iface_async_cb_event(void *arg, unsigned flags)
{
    ucp_worker_iface_t *wiface = arg;

    ucs_assert(wiface->attr.cap.event_flags & UCT_IFACE_FLAG_EVENT_ASYNC_CB);
    ucs_trace_func("async_cb for iface=%p", wiface->iface);

    ucp_worker_iface_event_common(wiface);
}

static void
ucp_worker_iface_async_fd_event(int fd, ucs_event_set_types_t events, void *arg)
{
    ucp_worker_iface_t *wiface = arg;
    int event_fd               = ucp_worker_iface_get_event_fd(wiface);
    ucs_status_t status;

    ucs_assertv(fd == event_fd, "fd=%d vs wiface::event_fd=%d", fd, event_fd);
    ucs_trace_func("fd=%d iface=%p", event_fd, wiface->iface);

    status = ucs_async_modify_handler(event_fd, 0);
    if (status != UCS_OK) {
        ucs_fatal("failed to modify %d event handler to <empty>: %s",
                  event_fd, ucs_status_string(status));
    }

    ucp_worker_iface_event_common(wiface);
}

static void ucp_worker_uct_iface_close(ucp_worker_iface_t *wiface)
{
    if (wiface->iface != NULL) {
        uct_iface_close(wiface->iface);
        wiface->iface = NULL;
    }
}

static int ucp_worker_iface_find_better(ucp_worker_h worker,
                                        ucp_worker_iface_t *wiface,
                                        ucp_rsc_index_t *better_index)
{
    ucp_context_h ctx = worker->context;
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *if_iter;
    uint64_t test_flags;
    double latency_iter, latency_cur, bw_cur;

    ucs_assert(wiface != NULL);

    latency_cur = ucp_tl_iface_latency(ctx, &wiface->attr.latency);
    bw_cur      = ucp_tl_iface_bandwidth(ctx, &wiface->attr.bandwidth);

    test_flags = wiface->attr.cap.flags & ~(UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                            UCT_IFACE_FLAG_CONNECT_TO_EP);

    for (rsc_index = 0; rsc_index < ctx->num_tls; ++rsc_index) {
        if_iter = worker->ifaces[rsc_index];

        /* Need to check resources which belong to the same device only */
        if ((ctx->tl_rscs[rsc_index].dev_index != ctx->tl_rscs[wiface->rsc_index].dev_index) ||
            (if_iter->flags & UCP_WORKER_IFACE_FLAG_UNUSED) ||
            (rsc_index == wiface->rsc_index)) {
            continue;
        }

        latency_iter = ucp_tl_iface_latency(ctx, &if_iter->attr.latency);

        /* Check that another iface: */
        if (/* 1. Supports all capabilities of the target iface (at least),
             *    except ...CONNECT_TO... caps. */
            ucs_test_all_flags(if_iter->attr.cap.flags, test_flags) &&
            /* 2. Has the same or better performance characteristics */
            (if_iter->attr.overhead <= wiface->attr.overhead) &&
            (ucp_tl_iface_bandwidth(ctx, &if_iter->attr.bandwidth) >= bw_cur) &&
            /* swap latencies in args list since less is better */
            (ucp_score_prio_cmp(latency_cur,  if_iter->attr.priority,
                                latency_iter, wiface->attr.priority) >= 0) &&
            /* 3. The found transport is scalable enough or both
             *    transport are unscalable */
            (ucp_is_scalable_transport(ctx, if_iter->attr.max_num_eps) ||
             !ucp_is_scalable_transport(ctx, wiface->attr.max_num_eps)))
        {
            *better_index = rsc_index;
            /* Do not check this iface anymore, because better one exists.
             * It helps to avoid the case when two interfaces with the same
             * caps and performance exclude each other. */
            wiface->flags |= UCP_WORKER_IFACE_FLAG_UNUSED;
            return 1;
        }
    }

    /* Better resource wasn't found */
    *better_index = 0;
    return 0;
}

/**
 * @brief Find the minimal possible set of tl interfaces for each device
 *
 * @param [in]  worker     UCP worker.
 * @param [out] tl_bitmap  Map of the relevant tl resources.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
static void ucp_worker_select_best_ifaces(ucp_worker_h worker,
                                          uint64_t *tl_bitmap_p)
{
    ucp_context_h context = worker->context;
    uint64_t tl_bitmap    = 0;
    ucp_rsc_index_t repl_ifaces[UCP_MAX_RESOURCES];
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t tl_id, iface_id;

    /* For each iface check whether there is another iface, which:
     * 1. Supports at least the same capabilities
     * 2. Provides equivalent or better performance
     */
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        wiface = worker->ifaces[tl_id];
        if (!ucp_worker_iface_find_better(worker, wiface, &repl_ifaces[tl_id])) {
            tl_bitmap |= UCS_BIT(tl_id);
        }
    }

    *tl_bitmap_p       = tl_bitmap;
    worker->num_ifaces = ucs_popcount(tl_bitmap);
    ucs_assert(worker->num_ifaces <= context->num_tls);

    if (worker->num_ifaces == context->num_tls) {
        return;
    }

    ucs_assert(worker->num_ifaces < context->num_tls);

    /* Some ifaces need to be closed */
    for (tl_id = 0, iface_id = 0; tl_id < context->num_tls; ++tl_id) {
        wiface = worker->ifaces[tl_id];
        if (tl_bitmap & UCS_BIT(tl_id)) {
            if (iface_id != tl_id) {
                worker->ifaces[iface_id] = wiface;
            }
            ++iface_id;
        } else {
            ucs_debug("closing resource[%d] "UCT_TL_RESOURCE_DESC_FMT
                      ", since resource[%d] "UCT_TL_RESOURCE_DESC_FMT
                      " is better, worker %p",
                      tl_id, UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[tl_id].tl_rsc),
                      repl_ifaces[tl_id],
                      UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[repl_ifaces[tl_id]].tl_rsc),
                      worker);
            /* Ifaces should not be initialized yet, just close it
             * (no need for cleanup) */
            ucp_worker_uct_iface_close(wiface);
            ucs_free(wiface);
        }
    }
}

/**
 * @brief  Open all resources as interfaces on this worker
 *
 * This routine opens interfaces on the tl resources according to the
 * bitmap in the context. If bitmap is not set, the routine opens interfaces
 * on all available resources and select the best ones. Then it caches obtained
 * bitmap on the context, so the next workers could use it instead of
 * constructing it themselves.
 *
 * @param [in]  worker     UCP worker.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
static ucs_status_t ucp_worker_add_resource_ifaces(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_tl_resource_desc_t *resource;
    uct_iface_params_t iface_params;
    ucp_rsc_index_t tl_id, iface_id;
    ucp_worker_iface_t *wiface;
    uint64_t ctx_tl_bitmap, tl_bitmap;
    unsigned num_ifaces;
    ucs_status_t status;

    /* If tl_bitmap is already set, just use it. Otherwise open ifaces on all
     * available resources and then select the best ones. */
    ctx_tl_bitmap  = context->tl_bitmap;
    if (ctx_tl_bitmap) {
        num_ifaces = ucs_popcount(ctx_tl_bitmap);
        tl_bitmap  = ctx_tl_bitmap;
    } else {
        num_ifaces = context->num_tls;
        tl_bitmap  = UCS_MASK(context->num_tls);
    }

    worker->ifaces = ucs_calloc(num_ifaces, sizeof(*worker->ifaces),
                                "ucp ifaces array");
    if (worker->ifaces == NULL) {
        ucs_error("failed to allocate worker ifaces");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    worker->num_ifaces = num_ifaces;
    iface_id           = 0;

    ucs_for_each_bit(tl_id, tl_bitmap) {
        iface_params.field_mask = UCT_IFACE_PARAM_FIELD_OPEN_MODE;
        resource = &context->tl_rscs[tl_id];

        if (resource->flags & UCP_TL_RSC_FLAG_SOCKADDR) {
            iface_params.open_mode            = UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT;
        } else {
            iface_params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
            iface_params.field_mask          |= UCT_IFACE_PARAM_FIELD_DEVICE;
            iface_params.mode.device.tl_name  = resource->tl_rsc.tl_name;
            iface_params.mode.device.dev_name = resource->tl_rsc.dev_name;
        }

        status = ucp_worker_iface_open(worker, tl_id, &iface_params,
                                       &worker->ifaces[iface_id++]);
        if (status != UCS_OK) {
            goto err_close_ifaces;
        }
    }

    if (!ctx_tl_bitmap) {
        /* Context bitmap is not set, need to select the best tl resources */
        tl_bitmap = 0;
        ucp_worker_select_best_ifaces(worker, &tl_bitmap);
        ucs_assert(tl_bitmap);

        /* Cache tl_bitmap on the context, so the next workers would not need
         * to select best ifaces. */
        context->tl_bitmap = tl_bitmap;
        ucs_debug("selected tl bitmap: 0x%"PRIx64" (%d tls)",
                  tl_bitmap, ucs_popcount(tl_bitmap));
    }

    worker->scalable_tl_bitmap = 0;
    ucs_for_each_bit(tl_id, context->tl_bitmap) {
        ucs_assert(ucp_worker_is_tl_p2p(worker, tl_id) ||
                   ucp_worker_is_tl_2iface(worker, tl_id) ||
                   ucp_worker_is_tl_2sockaddr(worker, tl_id));
        wiface = ucp_worker_iface(worker, tl_id);
        if (ucp_is_scalable_transport(context, wiface->attr.max_num_eps)) {
            worker->scalable_tl_bitmap |= UCS_BIT(tl_id);
        }
    }

    ucs_debug("selected scalable tl bitmap: 0x%"PRIx64" (%d tls)",
              worker->scalable_tl_bitmap,
              ucs_popcount(worker->scalable_tl_bitmap));

    iface_id = 0;
    ucs_for_each_bit(tl_id, tl_bitmap) {
        status = ucp_worker_iface_init(worker, tl_id,
                                       worker->ifaces[iface_id++]);
        if (status != UCS_OK) {
            goto err_cleanup_ifaces;
        }
    }

    return UCS_OK;

err_cleanup_ifaces:
    /* cleanup ucp_worker_iface_t structure and close UCT ifaces */
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        if (worker->ifaces[iface_id] != NULL) {
            ucp_worker_iface_cleanup(worker->ifaces[iface_id]);
            worker->ifaces[iface_id] = NULL;
        }
    }
err_close_ifaces:
    /* only close UCT ifaces, if they weren't closed already */
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        if (worker->ifaces[iface_id] != NULL) {
            ucp_worker_uct_iface_close(worker->ifaces[iface_id]);
        }
    }
    ucs_free(worker->ifaces);
err:
    return status;
}

static void ucp_worker_close_ifaces(ucp_worker_h worker)
{
    ucp_rsc_index_t iface_id;
    ucp_worker_iface_t *wiface;

    UCS_ASYNC_BLOCK(&worker->async);
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        wiface = worker->ifaces[iface_id];
        if (wiface != NULL) {
            ucp_worker_iface_cleanup(wiface);
        }
    }
    ucs_free(worker->ifaces);
    UCS_ASYNC_UNBLOCK(&worker->async);
}

static ucs_status_t
ucp_worker_get_sys_device_distance(ucp_context_h context,
                                   ucp_rsc_index_t rsc_index,
                                   ucs_sys_dev_distance_t *distance)
{
    ucs_sys_device_t device     = UCS_SYS_DEVICE_ID_UNKNOWN;
    ucs_sys_device_t cmp_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    ucp_rsc_index_t md_index, i;

    for (i = 0; i < context->num_tls; i++) {
        md_index = context->tl_rscs[i].md_index;
        if (strcmp(context->tl_mds[md_index].rsc.md_name,
                   context->config.selection_cmp)) {
            continue;
        }

        device     = context->tl_rscs[rsc_index].tl_rsc.sys_device;
        cmp_device = context->tl_rscs[i].tl_rsc.sys_device;

        return ucs_topo_get_distance(device, cmp_device, distance);
    }

    return UCS_ERR_NO_RESOURCE;
}

ucs_status_t ucp_worker_iface_open(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   uct_iface_params_t *iface_params,
                                   ucp_worker_iface_t **wiface_p)
{
    ucp_context_h context            = worker->context;
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[tl_id];
    uct_md_h md                      = context->tl_mds[resource->md_index].md;
    ucs_sys_dev_distance_t distance  = {.latency = 0, .bandwidth = 0};
    uct_iface_config_t *iface_config;
    const char *cfg_tl_name;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;

    wiface = ucs_calloc(1, sizeof(*wiface), "ucp_iface");
    if (wiface == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

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
        goto err_free_iface;
    }

    UCS_STATIC_ASSERT(UCP_WORKER_HEADROOM_PRIV_SIZE >= sizeof(ucp_eager_sync_hdr_t));

    /* Fill rest of uct_iface params (caller should fill specific mode fields) */
    iface_params->field_mask       |= UCT_IFACE_PARAM_FIELD_STATS_ROOT        |
                                      UCT_IFACE_PARAM_FIELD_RX_HEADROOM       |
                                      UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG   |
                                      UCT_IFACE_PARAM_FIELD_ERR_HANDLER       |
                                      UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS |
                                      UCT_IFACE_PARAM_FIELD_CPU_MASK;
    iface_params->stats_root        = UCS_STATS_RVAL(worker->stats);
    iface_params->rx_headroom       = UCP_WORKER_HEADROOM_SIZE;
    iface_params->err_handler_arg   = worker;
    iface_params->err_handler       = ucp_worker_iface_error_handler;
    iface_params->err_handler_flags = UCT_CB_FLAG_ASYNC;
    iface_params->cpu_mask          = worker->cpu_mask;

    if (context->config.features & UCP_FEATURE_TAG) {
        iface_params->eager_arg     = iface_params->rndv_arg = wiface;
        iface_params->eager_cb      = ucp_tag_offload_unexp_eager;
        iface_params->rndv_cb       = ucp_tag_offload_unexp_rndv;
        iface_params->field_mask   |= UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_ARG |
                                      UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_ARG  |
                                      UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_CB   |
                                      UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_CB;
    }

    iface_params->async_event_arg   = wiface;
    iface_params->async_event_cb    = ucp_worker_iface_async_cb_event;
    iface_params->field_mask       |= UCT_IFACE_PARAM_FIELD_ASYNC_EVENT_ARG |
                                      UCT_IFACE_PARAM_FIELD_ASYNC_EVENT_CB;

    if (ucp_worker_keepalive_is_enabled(worker)) {
        iface_params->field_mask        |= UCT_IFACE_PARAM_FIELD_KEEPALIVE_INTERVAL;
        iface_params->keepalive_interval = context->config.keepalive_interval;
    }

    /* Open UCT interface */
    status = uct_iface_open(md, worker->uct, iface_params, iface_config,
                            &wiface->iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
       goto err_free_iface;
    }

    VALGRIND_MAKE_MEM_UNDEFINED(&wiface->attr, sizeof(wiface->attr));

    status = uct_iface_query(wiface->iface, &wiface->attr);
    if (status != UCS_OK) {
        goto err_close_iface;
    }

    status = ucp_worker_get_sys_device_distance(context, wiface->rsc_index,
                                                &distance);
    if (status == UCS_OK) {
        wiface->attr.latency.c          += distance.latency;
        wiface->attr.bandwidth.shared    =
            ucs_min(wiface->attr.bandwidth.shared, distance.bandwidth);
        wiface->attr.bandwidth.dedicated =
            ucs_min(wiface->attr.bandwidth.dedicated, distance.bandwidth);
    }

    ucs_debug("created interface[%d]=%p using "UCT_TL_RESOURCE_DESC_FMT" on worker %p",
              tl_id, wiface->iface, UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc),
              worker);

    *wiface_p = wiface;

    return UCS_OK;

err_close_iface:
    uct_iface_close(wiface->iface);
err_free_iface:
    ucs_free(wiface);
    return status;
}

static void ucp_worker_iface_remove_event_handler(ucp_worker_iface_t *wiface)
{
    ucs_status_t status;

    if (wiface->event_fd == -1) {
        return;
    }

    ucs_assertv(ucp_worker_iface_use_event_fd(wiface),
                "%p: has event fd %d, but it has to not use this mechanism",
                wiface, wiface->event_fd);

    status = ucs_async_remove_handler(wiface->event_fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove event handler for fd %d: %s",
                 wiface->event_fd, ucs_status_string(status));
    }
}

ucs_status_t ucp_worker_iface_init(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   ucp_worker_iface_t *wiface)
{
    ucp_context_h context            = worker->context;
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[tl_id];
    uint8_t mem_type_index;
    ucs_status_t status;

    ucs_assert(wiface != NULL);

    /* Set wake-up handlers */
    if (ucp_worker_iface_use_event_fd(wiface)) {
        status = uct_iface_event_fd_get(wiface->iface, &wiface->event_fd);
        if (status != UCS_OK) {
            goto err;
        }

        /* Register event handler without actual events so we could modify it later. */
        status = ucs_async_set_event_handler(worker->async.mode, wiface->event_fd,
                                             0, ucp_worker_iface_async_fd_event,
                                             wiface, &worker->async);
        if (status != UCS_OK) {
            ucs_error("failed to set event handler on "
                      UCT_TL_RESOURCE_DESC_FMT " fd %d: %s",
                      UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc),
                      wiface->event_fd, ucs_status_string(status));
            goto err;
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
            goto err_unset_handler;
        }

        if (context->config.ext.adaptive_progress &&
            (wiface->attr.cap.event_flags & UCT_IFACE_FLAG_EVENT_RECV))
        {
            ucp_worker_iface_deactivate(wiface, 1);
        } else {
            ucp_worker_iface_activate(wiface, 0);
        }
    }

    ucs_for_each_bit(mem_type_index,
        context->tl_mds[resource->md_index].attr.cap.access_mem_types) {
        ucs_assert(mem_type_index < UCS_MEMORY_TYPE_LAST);
        context->mem_type_access_tls[mem_type_index] |= UCS_BIT(tl_id);
    }

    return UCS_OK;

err_unset_handler:
    ucp_worker_iface_remove_event_handler(wiface);
err:
    return status;
}

void ucp_worker_iface_cleanup(ucp_worker_iface_t *wiface)
{
    uct_worker_progress_unregister_safe(wiface->worker->uct,
                                        &wiface->check_events_id);
    ucp_worker_iface_disarm(wiface);
    ucp_worker_iface_remove_event_handler(wiface);
    ucp_worker_uct_iface_close(wiface);
    ucs_free(wiface);
}

static void ucp_worker_close_cms(ucp_worker_h worker)
{
    const ucp_rsc_index_t num_cms = ucp_worker_num_cm_cmpts(worker);
    ucp_rsc_index_t i;

    for (i = 0; (i < num_cms) && (worker->cms[i].cm != NULL); ++i) {
        uct_cm_close(worker->cms[i].cm);
    }

    ucs_free(worker->cms);
    worker->cms = NULL;
}

static ucs_status_t ucp_worker_add_resource_cms(ucp_worker_h worker)
{
    ucp_context_h   context = worker->context;
    uct_cm_config_t *cm_config;
    uct_component_h cmpt;
    ucp_rsc_index_t cmpt_index, cm_cmpt_index, i;
    ucs_status_t    status;

    if (!ucp_worker_sockaddr_is_cm_proto(worker)) {
        worker->cms = NULL;
        return UCS_OK;
    }

    UCS_ASYNC_BLOCK(&worker->async);

    worker->cms = ucs_calloc(ucp_worker_num_cm_cmpts(worker),
                             sizeof(*worker->cms), "ucp cms");
    if (worker->cms == NULL) {
        ucs_error("can't allocate CMs array");
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    for (i = 0, cm_cmpt_index = 0; cm_cmpt_index < context->config.num_cm_cmpts;
         ++cm_cmpt_index) {
        cmpt_index = context->config.cm_cmpt_idxs[cm_cmpt_index];
        cmpt       = context->tl_cmpts[cmpt_index].cmpt;

        status = uct_cm_config_read(cmpt, NULL, NULL, &cm_config);
        if (status != UCS_OK) {
            ucs_error("failed to read cm configuration on component %s",
                      context->tl_cmpts[cmpt_index].attr.name);
            goto err_free_cms;
        }

        status = uct_cm_open(cmpt, worker->uct, cm_config, &worker->cms[i].cm);
        if (status != UCS_OK) {
            ucs_error("failed to open CM on component %s with status %s",
                      context->tl_cmpts[cmpt_index].attr.name,
                      ucs_status_string(status));
            goto err_free_cms;
        }

        worker->cms[i].attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
        status                         = uct_cm_query(worker->cms[i].cm,
                                                      &worker->cms[i].attr);
        if (status != UCS_OK) {
            ucs_error("failed to query CM on component %s with status %s",
                      context->tl_cmpts[cmpt_index].attr.name,
                      ucs_status_string(status));
            goto err_free_cms;
        }

        uct_config_release(cm_config);
        worker->cms[i++].cmpt_idx = cmpt_index;
    }

    status = UCS_OK;
    goto out;

err_free_cms:
    ucp_worker_close_cms(worker);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
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
    ucp_rsc_index_t iface_id;
    ucp_worker_iface_t *wiface;

    ucs_debug("worker %p: using cpu atomics", worker);

    /* Enable all interfaces which have host-based atomics */
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        wiface = worker->ifaces[iface_id];
        if (wiface->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU) {
            ucp_worker_enable_atomic_tl(worker, "cpu", wiface->rsc_index);
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
    ucp_rsc_index_t iface_id;
    uint64_t iface_cap_flags;
    double score, best_score;
    ucp_md_index_t md_index;
    ucp_worker_iface_t *wiface;
    uct_md_attr_t *md_attr;
    uint64_t supp_tls;
    uint8_t priority, best_priority;
    ucp_tl_iface_atomic_flags_t atomic;

    ucp_context_uct_atomic_iface_flags(context, &atomic);

    iface_cap_flags                      = UCT_IFACE_FLAG_ATOMIC_DEVICE;

    dummy_iface_attr.bandwidth.dedicated = 1e12;
    dummy_iface_attr.bandwidth.shared    = 0;
    dummy_iface_attr.cap_flags           = UINT64_MAX;
    dummy_iface_attr.overhead            = 0;
    dummy_iface_attr.priority            = 0;
    dummy_iface_attr.lat_ovh             = 0;

    supp_tls                             = 0;
    best_score                           = -1;
    best_rsc                             = NULL;
    best_priority                        = 0;

    /* Select best interface for atomics device */
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        wiface     = worker->ifaces[iface_id];
        rsc_index  = wiface->rsc_index;
        rsc        = &context->tl_rscs[rsc_index];
        md_index   = rsc->md_index;
        md_attr    = &context->tl_mds[md_index].attr;
        iface_attr = &wiface->attr;

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
        if (ucp_is_scalable_transport(worker->context,
                                      iface_attr->max_num_eps) &&
            ((score > best_score) ||
             ((score == best_score) && (priority > best_priority))))
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

    ucs_debug("worker %p: using device atomics", worker);

    /* Enable atomics on all resources using same device as the "best" resource */
    ucs_for_each_bit(rsc_index, context->tl_bitmap) {
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
    uint64_t accumulated_flags = 0;
    ucp_rsc_index_t iface_id;

    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        if (ucp_is_scalable_transport(worker->context,
                                      worker->ifaces[iface_id]->attr.max_num_eps)) {
            accumulated_flags |= worker->ifaces[iface_id]->attr.cap.flags;
        }
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

static char* ucp_worker_add_feature_rsc(ucp_context_h context,
                                        const ucp_ep_config_key_t *key,
                                        ucp_lane_map_t lanes_bitmap,
                                        const char *feature_str,
                                        char *buf, size_t max)
{
    char *p    = buf;
    char *endp = buf + max;
    int   sep  = 0;
    ucp_rsc_index_t rsc_idx;
    ucp_lane_index_t lane;

    if (!lanes_bitmap) {
        return p;
    }

    snprintf(p, endp - p, "%s(", feature_str);
    p += strlen(p);

    ucs_for_each_bit(lane, lanes_bitmap) {
        ucs_assert(lane < UCP_MAX_LANES); /* make coverity happy */
        rsc_idx = key->lanes[lane].rsc_index;
        snprintf(p, endp - p, "%*s"UCT_TL_RESOURCE_DESC_FMT, sep, "",
                 UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_idx].tl_rsc));
        p  += strlen(p);
        sep = 1; /* add space between tl names */
    }

    snprintf(p, endp - p, "); ");
    p += strlen(p);

    return p;
}

static void ucp_worker_print_used_tls(const ucp_ep_config_key_t *key,
                                      ucp_context_h context,
                                      ucp_worker_cfg_index_t config_idx)
{
    char info[256]                  = {0};
    ucp_lane_map_t tag_lanes_map    = 0;
    ucp_lane_map_t rma_lanes_map    = 0;
    ucp_lane_map_t amo_lanes_map    = 0;
    ucp_lane_map_t stream_lanes_map = 0;
    ucp_lane_map_t am_lanes_map     = 0;
    ucp_lane_index_t lane;
    char *p, *endp;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_INFO)) {
        return;
    }

    p    = info;
    endp = p + sizeof(info);

    snprintf(p, endp - p,  "ep_cfg[%d]: ", config_idx);
    p += strlen(p);

    for (lane = 0; lane < key->num_lanes; ++lane) {
        if ((key->am_lane == lane) ||
            (ucp_ep_config_get_multi_lane_prio(key->am_bw_lanes, lane) >= 0)  ||
            (ucp_ep_config_get_multi_lane_prio(key->rma_bw_lanes, lane) >= 0)) {
            if (context->config.features & UCP_FEATURE_TAG) {
                tag_lanes_map |= UCS_BIT(lane);
            }

            if (context->config.features & UCP_FEATURE_AM) {
                am_lanes_map |= UCS_BIT(lane);
            }
        }

        if (key->tag_lane == lane) {
            /* tag_lane is initialized if TAG feature is requested */
            ucs_assert(context->config.features & UCP_FEATURE_TAG);
            tag_lanes_map |= UCS_BIT(lane);
        }

        if ((key->am_lane == lane) &&
            (context->config.features & UCP_FEATURE_STREAM)) {
            stream_lanes_map |= UCS_BIT(lane);
        }

        if ((ucp_ep_config_get_multi_lane_prio(key->rma_lanes, lane) >= 0)) {
            rma_lanes_map |= UCS_BIT(lane);
        }

        if ((ucp_ep_config_get_multi_lane_prio(key->amo_lanes, lane) >= 0)) {
            amo_lanes_map |= UCS_BIT(lane);
        }
    }

    p = ucp_worker_add_feature_rsc(context, key, tag_lanes_map, "tag",
                                   p, endp - p);
    p = ucp_worker_add_feature_rsc(context, key, rma_lanes_map, "rma",
                                   p, endp - p);
    p = ucp_worker_add_feature_rsc(context, key, amo_lanes_map, "amo",
                                   p, endp - p);
    p = ucp_worker_add_feature_rsc(context, key, am_lanes_map, "am",
                                   p, endp - p);
    ucp_worker_add_feature_rsc(context, key, stream_lanes_map, "stream",
                               p, endp - p);
    ucs_info("%s", info);
}

static ucs_status_t ucp_worker_init_mpools(ucp_worker_h worker)
{
    size_t           max_mp_entry_size = 0;
    ucp_context_t    *context          = worker->context;
    uct_iface_attr_t *if_attr;
    ucp_rsc_index_t  iface_id;
    ucs_status_t     status;

    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        if_attr           = &worker->ifaces[iface_id]->attr;
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_short);
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_bcopy);
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_zcopy);
    }

    /* Create memory pool for requests */
    status = ucs_mpool_init(&worker->req_mp, 0,
                            sizeof(ucp_request_t) + context->config.request.size,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_request_mpool_ops, "ucp_requests");
    if (status != UCS_OK) {
        goto err;
    }

    /* Create memory pool for small rkeys */
    status = ucs_mpool_init(&worker->rkey_mp, 0,
                            sizeof(ucp_rkey_t) +
                            sizeof(ucp_tl_rkey_t) * UCP_RKEY_MPOOL_MAX_MD,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_rkey_mpool_ops, "ucp_rkeys");
    if (status != UCS_OK) {
        goto err_req_mp_cleanup;
    }

    /* Create memory pool for incoming UCT messages without a UCT descriptor */
    status = ucs_mpool_init(&worker->am_mp, 0,
                            max_mp_entry_size + UCP_WORKER_HEADROOM_SIZE,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_am_mpool_ops, "ucp_am_bufs");
    if (status != UCS_OK) {
        goto err_rkey_mp_cleanup;
    }

    /* Create memory pool of bounce buffers */
    status = ucs_mpool_init(&worker->reg_mp, 0,
                            context->config.ext.seg_size + sizeof(ucp_mem_desc_t),
                            sizeof(ucp_mem_desc_t), UCS_SYS_CACHE_LINE_SIZE,
                            128, UINT_MAX, &ucp_reg_mpool_ops, "ucp_reg_bufs");
    if (status != UCS_OK) {
        goto err_am_mp_cleanup;
    }

    /* Create memory pool for pipelined rndv fragments */
    status = ucs_mpool_init(&worker->rndv_frag_mp, 0,
                            context->config.ext.rndv_frag_size + sizeof(ucp_mem_desc_t),
                            sizeof(ucp_mem_desc_t), UCS_SYS_PCI_MAX_PAYLOAD, 128,
                            UINT_MAX, &ucp_frag_mpool_ops, "ucp_rndv_frags");
    if (status != UCS_OK) {
        goto err_reg_mp_cleanup;
    }

    return UCS_OK;

err_reg_mp_cleanup:
    ucs_mpool_cleanup(&worker->reg_mp, 0);
err_am_mp_cleanup:
    ucs_mpool_cleanup(&worker->am_mp, 0);
err_rkey_mp_cleanup:
    ucs_mpool_cleanup(&worker->rkey_mp, 0);
err_req_mp_cleanup:
    ucs_mpool_cleanup(&worker->req_mp, 0);
err:
    return status;
}

static void ucp_worker_destroy_mpools(ucp_worker_h worker)
{
    ucs_mpool_cleanup(&worker->rndv_frag_mp, 1);
    ucs_mpool_cleanup(&worker->reg_mp, 1);
    ucs_mpool_cleanup(&worker->am_mp, 1);
    ucs_mpool_cleanup(&worker->rkey_mp, 1);
    ucs_mpool_cleanup(&worker->req_mp, 1);
}

/* All the ucp endpoints will share the configurations. No need for every ep to
 * have it's own configuration (to save memory footprint). Same config can be used
 * by different eps.
 * A 'key' identifies an entry in the ep_config array. An entry holds the key and
 * additional configuration parameters and thresholds.
 */
ucs_status_t
ucp_worker_get_ep_config(ucp_worker_h worker, const ucp_ep_config_key_t *key,
                         int print_cfg, ucp_worker_cfg_index_t *cfg_index_p)
{
    ucp_context_h context = worker->context;
    ucp_worker_cfg_index_t ep_cfg_index;
    ucp_proto_select_short_t tag_short;
    ucp_ep_config_t *ep_config;
    ucs_status_t status;

    /* Search for the given key in the ep_config array */
    for (ep_cfg_index = 0; ep_cfg_index < worker->ep_config_count;
         ++ep_cfg_index) {
        if (ucp_ep_config_is_equal(&worker->ep_config[ep_cfg_index].key, key)) {
            goto out;
        }
    }

    if (worker->ep_config_count >= UCP_WORKER_MAX_EP_CONFIG) {
        ucs_error("too many ep configurations: %d (max: %d)",
                  worker->ep_config_count, UCP_WORKER_MAX_EP_CONFIG);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    /* Create new configuration */
    ep_cfg_index = worker->ep_config_count;
    ep_config    = &worker->ep_config[ep_cfg_index];
    status       = ucp_ep_config_init(worker, ep_config, key);
    if (status != UCS_OK) {
        return status;
    }

    if (context->config.ext.proto_enable) {
        if (context->config.features & UCP_FEATURE_TAG) {
            /* Set threshold for short send */
            ucp_proto_select_short_init(worker, &ep_config->proto_select,
                                        ep_cfg_index, UCP_WORKER_CFG_INDEX_NULL,
                                        UCP_OP_ID_TAG_SEND, 0,
                                        UCP_PROTO_FLAG_AM_SHORT, &tag_short);
            /* short protocol should be either disabled, or use key->am_lane */
            ucs_assert((tag_short.max_length_host_mem < 0) ||
                       (tag_short.lane == key->am_lane));
        } else {
            ucp_proto_select_short_disable(&tag_short);
        }

        /* TODO replace ep_config->tag.max_eager_short by this struct */
        ep_config->tag.max_eager_short.memtype_off = tag_short.max_length_unknown_mem;
        ep_config->tag.max_eager_short.memtype_on  = tag_short.max_length_host_mem;
    }

    if (print_cfg) {
        ucp_worker_print_used_tls(key, context, ep_cfg_index);
    }

    ++worker->ep_config_count;

out:
    *cfg_index_p = ep_cfg_index;
    return UCS_OK;
}

ucs_status_t
ucp_worker_add_rkey_config(ucp_worker_h worker, const ucp_rkey_config_key_t *key,
                           ucp_worker_cfg_index_t *cfg_index_p)
{
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_rkey_config_t *rkey_config;
    ucs_status_t status;
    khiter_t khiter;
    int khret;

    ucs_assert(worker->context->config.ext.proto_enable);

    if (worker->rkey_config_count >= UCP_WORKER_MAX_RKEY_CONFIG) {
        ucs_error("too many rkey configurations: %d (max: %d)",
                  worker->rkey_config_count, UCP_WORKER_MAX_RKEY_CONFIG);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err;
    }

    /* initialize rkey configuration */
    rkey_cfg_index   = worker->rkey_config_count;
    rkey_config      = &worker->rkey_config[rkey_cfg_index];
    rkey_config->key = *key;
    status           = ucp_proto_select_init(&rkey_config->proto_select);
    if (status != UCS_OK) {
        goto err;
    }

    if (worker->context->config.features & UCP_FEATURE_RMA) {
       /* Set threshold for short put */
        ucp_proto_select_short_init(worker, &rkey_config->proto_select,
                                    key->ep_cfg_index, rkey_cfg_index,
                                    UCP_OP_ID_PUT, UCP_OP_ATTR_FLAG_FAST_CMPL,
                                    UCP_PROTO_FLAG_PUT_SHORT,
                                    &rkey_config->put_short);
    } else {
        ucp_proto_select_short_disable(&rkey_config->put_short);
    }

    khiter = kh_put(ucp_worker_rkey_config, &worker->rkey_config_hash, *key,
                    &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        status = UCS_ERR_NO_MEMORY;
        goto err_proto_cleanup;
    }

    /* we should not get into this function if key already exists */
    ucs_assert_always(khret != UCS_KH_PUT_KEY_PRESENT);

    kh_value(&worker->rkey_config_hash, khiter) = rkey_cfg_index;

    ++worker->rkey_config_count;
    *cfg_index_p = rkey_cfg_index;
    return UCS_OK;

err_proto_cleanup:
    ucp_proto_select_cleanup(&rkey_config->proto_select);
err:
    return status;
}

static UCS_F_ALWAYS_INLINE void ucp_worker_keepalive_reset(ucp_worker_h worker)
{
    worker->keepalive.iter = &worker->all_eps;
}

static void ucp_worker_destroy_configs(ucp_worker_h worker)
{
    unsigned i;

    for (i = 0; i < worker->ep_config_count; ++i) {
        ucp_ep_config_cleanup(worker, &worker->ep_config[i]);
    }
    worker->ep_config_count = 0;

    for (i = 0; i < worker->rkey_config_count; ++i) {
        ucp_proto_select_cleanup(&worker->rkey_config[i].proto_select);
    }
    worker->rkey_config_count = 0;
}

ucs_status_t ucp_worker_create(ucp_context_h context,
                               const ucp_worker_params_t *params,
                               ucp_worker_h *worker_p)
{
    ucs_thread_mode_t uct_thread_mode;
    unsigned name_length;
    ucp_worker_h worker;
    ucs_status_t status;

    worker = ucs_calloc(1, sizeof(*worker), "ucp worker");
    if (worker == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    uct_thread_mode = UCS_THREAD_MODE_SINGLE;
    worker->flags   = 0;

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_THREAD_MODE) {
#if ENABLE_MT
        if (params->thread_mode != UCS_THREAD_MODE_SINGLE) {
            /* UCT is serialized by UCP lock or by UCP user */
            uct_thread_mode = UCS_THREAD_MODE_SERIALIZED;
        }

        if (params->thread_mode == UCS_THREAD_MODE_MULTI) {
            worker->flags |= UCP_WORKER_FLAG_MT;
        }
#else
        if (params->thread_mode != UCS_THREAD_MODE_SINGLE) {
            ucs_debug("forced single thread mode on worker create");
        }
#endif
    }

    worker->context              = context;
    worker->uuid                 = ucs_generate_uuid((uintptr_t)worker);
    worker->flush_ops_count      = 0;
    worker->inprogress           = 0;
    worker->rkey_config_count    = 0;
    worker->ep_config_count      = 0;
    worker->num_active_ifaces    = 0;
    worker->num_ifaces           = 0;
    worker->am_message_id        = ucs_generate_uuid(0);
    worker->rkey_ptr_cb_id       = UCS_CALLBACKQ_ID_NULL;
    worker->keepalive.cb_id      = UCS_CALLBACKQ_ID_NULL;
    worker->keepalive.last_round = 0;
    worker->keepalive.lane_map   = 0;
    worker->keepalive.ep_count   = 0;
    ucp_worker_keepalive_reset(worker);
    ucs_queue_head_init(&worker->rkey_ptr_reqs);
    ucs_list_head_init(&worker->arm_ifaces);
    ucs_list_head_init(&worker->stream_ready_eps);
    ucs_list_head_init(&worker->all_eps);
    kh_init_inplace(ucp_worker_rkey_config, &worker->rkey_config_hash);
    kh_init_inplace(ucp_worker_discard_uct_ep_hash, &worker->discard_uct_ep_hash);

    UCS_STATIC_ASSERT(sizeof(ucp_ep_ext_gen_t) <= sizeof(ucp_ep_t));
    if (context->config.features & (UCP_FEATURE_STREAM | UCP_FEATURE_AM)) {
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

    status = ucs_ptr_map_init(&worker->ptr_map);
    if (status != UCS_OK) {
        goto err_free;
    }

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&worker->stats, &ucp_worker_stats_class,
                                  ucs_stats_get_root(), "-%p", worker);
    if (status != UCS_OK) {
        goto err_destroy_ptr_map;
    }

    status = UCS_STATS_NODE_ALLOC(&worker->tm_offload_stats,
                                  &ucp_worker_tm_offload_stats_class,
                                  worker->stats);
    if (status != UCS_OK) {
        goto err_free_stats;
    }

    status = ucs_async_context_init(&worker->async,
                                    context->config.ext.use_mt_mutex ?
                                    UCS_ASYNC_MODE_THREAD_MUTEX :
                                    UCS_ASYNC_THREAD_LOCK_TYPE);
    if (status != UCS_OK) {
        goto err_free_tm_offload_stats;
    }

    /* Create the underlying UCT worker */
    status = uct_worker_create(&worker->async, uct_thread_mode, &worker->uct);
    if (status != UCS_OK) {
        goto err_destroy_async;
    }

    /* Create UCS event set which combines events from all transports */
    status = ucp_worker_wakeup_init(worker, params);
    if (status != UCS_OK) {
        goto err_destroy_uct_worker;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_CPU_MASK) {
        worker->cpu_mask = params->cpu_mask;
    } else {
        UCS_CPU_ZERO(&worker->cpu_mask);
    }

    /* Initialize connection matching structure */
    ucs_conn_match_init(&worker->conn_match_ctx, sizeof(uint64_t),
                        &ucp_ep_match_ops);

    /* Open all resources as interfaces on this worker */
    status = ucp_worker_add_resource_ifaces(worker);
    if (status != UCS_OK) {
        goto err_conn_match_cleanup;
    }

    /* Open all resources as connection managers on this worker */
    status = ucp_worker_add_resource_cms(worker);
    if (status != UCS_OK) {
        goto err_close_ifaces;
    }

    /* Create loopback endpoints to copy across memory types */
    status = ucp_worker_create_mem_type_endpoints(worker);
    if (status != UCS_OK) {
        goto err_close_cms;
    }

    /* Initialize memory pools, should be done after resources are added */
    status = ucp_worker_init_mpools(worker);
    if (status != UCS_OK) {
        goto err_destroy_memtype_eps;
    }

    /* Initialize tag matching */
    status = ucp_tag_match_init(&worker->tm);
    if (status != UCS_OK) {
        goto err_destroy_mpools;
    }

    /* Initialize UCP AMs */
    status = ucp_am_init(worker);
    if (status != UCS_OK) {
        goto err_tag_match_cleanup;
    }

    /* Select atomic resources */
    ucp_worker_init_atomic_tls(worker);

    /* At this point all UCT memory domains and interfaces are already created
     * so warn about unused environment variables.
     */
    ucs_config_parser_warn_unused_env_vars_once(context->config.env_prefix);

    *worker_p = worker;
    return UCS_OK;

err_tag_match_cleanup:
    ucp_tag_match_cleanup(&worker->tm);
err_destroy_mpools:
    ucp_worker_destroy_mpools(worker);
err_destroy_memtype_eps:
    ucp_worker_destroy_mem_type_endpoints(worker);
err_close_cms:
    ucp_worker_close_cms(worker);
err_close_ifaces:
    ucp_worker_close_ifaces(worker);
err_conn_match_cleanup:
    ucs_conn_match_cleanup(&worker->conn_match_ctx);
    ucp_worker_wakeup_cleanup(worker);
err_destroy_uct_worker:
    uct_worker_destroy(worker->uct);
err_destroy_async:
    ucs_async_context_cleanup(&worker->async);
err_free_tm_offload_stats:
    UCS_STATS_NODE_FREE(worker->tm_offload_stats);
err_free_stats:
    UCS_STATS_NODE_FREE(worker->stats);
err_destroy_ptr_map:
    ucs_ptr_map_destroy(&worker->ptr_map);
err_free:
    ucs_strided_alloc_cleanup(&worker->ep_alloc);
    kh_destroy_inplace(ucp_worker_discard_uct_ep_hash,
                       &worker->discard_uct_ep_hash);
    kh_destroy_inplace(ucp_worker_rkey_config, &worker->rkey_config_hash);
    ucp_worker_destroy_configs(worker);
    ucs_free(worker);
    return status;
}

static void ucp_worker_put_flush_req(ucp_request_t *req)
{
    ucp_ep_h ucp_ep = req->send.ep;

    ucp_worker_flush_ops_count_dec(ucp_ep->worker);
    ucp_ep_destroy_base(ucp_ep);
    ucp_request_put(req);
}

static unsigned ucp_worker_discard_uct_ep_destroy_progress(void *arg)
{
    ucp_request_t *req  = (ucp_request_t*)arg;
    uct_ep_h uct_ep     = req->send.discard_uct_ep.uct_ep;
    ucp_ep_h ucp_ep     = req->send.ep;
    ucp_worker_h worker = ucp_ep->worker;
    khiter_t iter;

    ucp_trace_req(req, "destroy uct_ep=%p", uct_ep);

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_worker_put_flush_req(req);
    iter = kh_get(ucp_worker_discard_uct_ep_hash,
                  &worker->discard_uct_ep_hash, uct_ep);
    if (iter == kh_end(&worker->discard_uct_ep_hash)) {
        ucs_fatal("no %p UCT EP in the %p worker hash of discarded UCT EPs",
                  uct_ep, worker);
    }
    kh_del(ucp_worker_discard_uct_ep_hash,
           &worker->discard_uct_ep_hash, iter);
    UCS_ASYNC_UNBLOCK(&worker->async);

    uct_ep_destroy(uct_ep);

    return 1;
}

static void
ucp_worker_discard_uct_ep_flush_comp(uct_completion_t *self)
{
    uct_worker_cb_id_t cb_id = UCS_CALLBACKQ_ID_NULL;
    ucp_request_t *req       = ucs_container_of(self, ucp_request_t,
                                                send.state.uct_comp);
    ucp_worker_h worker      = req->send.ep->worker;

    ucp_trace_req(req, "discard_uct_ep flush completion status %s",
                  ucs_status_string(self->status));

    if (self->status == UCS_ERR_CANCELED) {
        /* we run from EP cleanup - just release request */
        ucp_worker_put_flush_req(req);
        return;
    }

    /* don't destroy UCT EP from the flush completion callback, schedule
     * a progress callback on the main thread to destroy UCT EP */
    uct_worker_progress_register_safe(worker->uct,
                                      ucp_worker_discard_uct_ep_destroy_progress,
                                      req, UCS_CALLBACKQ_FLAG_ONESHOT, &cb_id);
}

static ucs_status_t
ucp_worker_discard_uct_ep_pending_cb(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    uct_ep_h uct_ep    = req->send.discard_uct_ep.uct_ep;
    ucs_status_t status;

    status = uct_ep_flush(uct_ep, req->send.discard_uct_ep.ep_flush_flags,
                          &req->send.state.uct_comp);
    if (status == UCS_OK) {
        ucp_worker_discard_uct_ep_destroy_progress(req);
        return UCS_OK;
    } else if (status == UCS_INPROGRESS) {
        return UCS_OK;
    } else if (status == UCS_ERR_NO_RESOURCE) {
        return UCS_ERR_NO_RESOURCE;
    }

    uct_completion_update_status(&req->send.state.uct_comp, status);
    ucp_worker_discard_uct_ep_flush_comp(&req->send.state.uct_comp);
    return UCS_OK;
}

static unsigned ucp_worker_discard_uct_ep_progress(void *arg)
{
    uct_worker_cb_id_t cb_id = UCS_CALLBACKQ_ID_NULL;
    ucp_request_t *req       = (ucp_request_t*)arg;
    uct_ep_h uct_ep          = req->send.discard_uct_ep.uct_ep;
    ucp_worker_h worker      = req->send.ep->worker;
    ucs_status_t status;

    status = ucp_worker_discard_uct_ep_pending_cb(&req->send.uct);
    if (status == UCS_ERR_NO_RESOURCE) {
        status = uct_ep_pending_add(uct_ep, &req->send.uct, 0);
        ucs_assert((status == UCS_ERR_BUSY) || (status == UCS_OK));
        if (status == UCS_ERR_BUSY) {
            /* adding to the pending queue failed, schedule the UCT EP discard
             * operation on UCT worker progress again */
            uct_worker_progress_register_safe(worker->uct,
                                              ucp_worker_discard_uct_ep_progress,
                                              req, UCS_CALLBACKQ_FLAG_ONESHOT,
                                              &cb_id);
        }

        return 0;
    }

    return 1;
}

static int ucp_worker_discard_remove_filter(const ucs_callbackq_elem_t *elem,
                                            void *arg)
{
    if ((elem->cb != ucp_worker_discard_uct_ep_destroy_progress) &&
        (elem->cb != ucp_worker_discard_uct_ep_progress)) {
        return 0;
    }

    ucp_worker_put_flush_req((ucp_request_t*)elem->arg);
    return 1;
}

static void ucp_worker_discarded_uct_eps_cleanup(ucp_worker_h worker)
{
    uct_ep_h uct_ep;

    /* if ep owns the discard operation ep_destroy will cancel it.
     * we are after uct_worker_progress_unregister_safe and
     * ucp_worker_discard_remove_filter, so either we canceled req
     * or it was finished and removed from kh before */
    kh_foreach_key(&worker->discard_uct_ep_hash, uct_ep, {
        uct_ep_destroy(uct_ep);
    })
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
    uct_worker_progress_unregister_safe(worker->uct, &worker->keepalive.cb_id);
    ucs_callbackq_remove_if(&worker->uct->progress_q,
                            ucp_worker_discard_remove_filter, NULL);
    ucp_worker_destroy_eps(worker);
    ucp_worker_remove_am_handlers(worker);
    ucp_am_cleanup(worker);
    ucp_worker_discarded_uct_eps_cleanup(worker);

    if (worker->flush_ops_count != 0) {
        ucs_warn("not all pending operations (%u) were flushed on worker %p "
                 "that is being destroyed",
                 worker->flush_ops_count, worker);
    }
    UCS_ASYNC_UNBLOCK(&worker->async);

    ucp_tag_match_cleanup(&worker->tm);
    ucp_worker_destroy_mpools(worker);
    ucp_worker_destroy_mem_type_endpoints(worker);
    ucp_worker_close_cms(worker);
    ucp_worker_close_ifaces(worker);
    ucs_conn_match_cleanup(&worker->conn_match_ctx);
    ucp_worker_wakeup_cleanup(worker);
    uct_worker_destroy(worker->uct);
    ucs_async_context_cleanup(&worker->async);
    UCS_STATS_NODE_FREE(worker->tm_offload_stats);
    UCS_STATS_NODE_FREE(worker->stats);
    ucs_ptr_map_destroy(&worker->ptr_map);
    ucs_strided_alloc_cleanup(&worker->ep_alloc);
    kh_destroy_inplace(ucp_worker_discard_uct_ep_hash,
                       &worker->discard_uct_ep_hash);
    kh_destroy_inplace(ucp_worker_rkey_config, &worker->rkey_config_hash);
    ucp_worker_destroy_configs(worker);
    ucs_free(worker);
}

ucs_status_t ucp_worker_query(ucp_worker_h worker,
                              ucp_worker_attr_t *attr)
{
    ucp_context_h context = worker->context;
    ucs_status_t status   = UCS_OK;
    uint64_t tl_bitmap;
    ucp_rsc_index_t tl_id;

    if (attr->field_mask & UCP_WORKER_ATTR_FIELD_THREAD_MODE) {
        if (worker->flags & UCP_WORKER_FLAG_MT) {
            attr->thread_mode = UCS_THREAD_MODE_MULTI;
        } else {
            attr->thread_mode = UCS_THREAD_MODE_SINGLE;
        }
    }

    if (attr->field_mask & UCP_WORKER_ATTR_FIELD_ADDRESS) {
        /* If UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS is not set,
         * pack all tl addresses */
        tl_bitmap = UINT64_MAX;

        if (attr->field_mask & UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS) {
            if (attr->address_flags & UCP_WORKER_ADDRESS_FLAG_NET_ONLY) {
                tl_bitmap = 0;
                ucs_for_each_bit(tl_id, context->tl_bitmap) {
                    if (context->tl_rscs[tl_id].tl_rsc.dev_type == UCT_DEVICE_TYPE_NET) {
                        tl_bitmap |= UCS_BIT(tl_id);
                    }
                }
            }
        }

        status = ucp_address_pack(worker, NULL, tl_bitmap,
                                  UCP_ADDRESS_PACK_FLAGS_WORKER_DEFAULT,
                                  NULL, &attr->address_length,
                                  (void**)&attr->address);
    }

    if (attr->field_mask & UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER) {
        attr->max_am_header = ucp_am_max_header_size(worker);
    }

    return status;
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
    ssize_t            count = 0;
    ucp_ep_ext_proto_t *ep_ext;
    ucp_ep_h           ep;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_STREAM,
                                    return UCS_ERR_INVALID_PARAM);

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

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_WAKEUP,
                                    return UCS_ERR_INVALID_PARAM);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
    if (worker->flags & UCP_WORKER_FLAG_EXTERNAL_EVENT_FD) {
        status = UCS_ERR_UNSUPPORTED;
    } else {
        *fd    = worker->event_fd;
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

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_WAKEUP,
                                    return UCS_ERR_INVALID_PARAM);

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

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_WAKEUP,
                                    return UCS_ERR_INVALID_PARAM);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    status = ucp_worker_arm(worker);
    if (status == UCS_ERR_BUSY) { /* if UCS_ERR_BUSY returned - no poll() must called */
        status = UCS_OK;
        goto out_unlock;
    } else if (status != UCS_OK) {
        goto out_unlock;
    }

    if (worker->flags & UCP_WORKER_FLAG_EXTERNAL_EVENT_FD) {
        pfd = ucs_alloca(sizeof(*pfd) * worker->context->num_tls);
        nfds = 0;
        ucs_list_for_each(wiface, &worker->arm_ifaces, arm_list) {
            if (!ucp_worker_iface_use_event_fd(wiface)) {
                /* if UCT iface supports asynchronous event callback, we
                 * prefer this method, since it will be called anyway. So,
                 * no need to get event fd. */
                continue;
            }

            pfd[nfds].fd     = ucp_worker_iface_get_event_fd(wiface);
            pfd[nfds].events = POLLIN;
            ++nfds;
        }
    } else {
        pfd = ucs_alloca(sizeof(*pfd));
        pfd->fd     = worker->event_fd;
        pfd->events = POLLIN;
        nfds        = 1;
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    /* poll is thread safe system call, though can have unpredictable results
     * because of using the same descriptor in multiple threads.
     */
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

out_unlock:
     UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
out:
    return status;
}

ucs_status_t ucp_worker_signal(ucp_worker_h worker)
{
    ucs_trace_func("worker %p", worker);
    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_WAKEUP,
                                    return UCS_ERR_INVALID_PARAM);
    return ucp_worker_wakeup_signal_fd(worker);
}

ucs_status_t ucp_worker_get_address(ucp_worker_h worker, ucp_address_t **address_p,
                                    size_t *address_length_p)
{
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    status = ucp_address_pack(worker, NULL, UINT64_MAX,
                              UCP_ADDRESS_PACK_FLAGS_WORKER_DEFAULT, NULL,
                              address_length_p, (void**)address_p);

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

static UCS_F_ALWAYS_INLINE ucp_ep_h
ucp_worker_keepalive_current_ep(ucp_worker_h worker)
{
    ucp_ep_ext_gen_t *ep_ext = ucs_container_of(worker->keepalive.iter,
                                                ucp_ep_ext_gen_t, ep_list);
    return ucp_ep_from_ext_gen(ep_ext);
}

static UCS_F_ALWAYS_INLINE void
ucp_worker_keepalive_next_ep(ucp_worker_h worker)
{
    ucp_ep_h ep;

    worker->keepalive.iter = worker->keepalive.iter->next;
    if (worker->keepalive.iter == &worker->all_eps) {
        /* if next list item points to all_eps then step one more time */
        worker->keepalive.iter = worker->keepalive.iter->next;
    }

    ucs_assert(worker->keepalive.iter != &worker->all_eps);
    ep                         = ucp_worker_keepalive_current_ep(worker);
    worker->keepalive.lane_map = ((ep->cfg_index != UCP_WORKER_CFG_INDEX_NULL) &&
                                  !(ep->flags & UCP_EP_FLAG_FAILED)) ?
                                 ucp_ep_config(ep)->key.ep_check_map : 0;
}

static unsigned ucp_worker_keepalive_progress(void *arg)
{
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucs_time_t now      = ucs_get_time();
    ucs_list_link_t *iter_begin;
    ucp_ep_h ep;

    ucs_assert(worker->context->config.ext.keepalive_num_eps != 0);

    if (ucs_likely((now - worker->keepalive.last_round) <
                   worker->context->config.keepalive_interval)) {
        return 0;
    }

    if (ucs_unlikely(ucs_list_is_empty(&worker->all_eps))) {
        ucs_assert(worker->keepalive.iter == &worker->all_eps);
        uct_worker_progress_unregister_safe(worker->uct,
                                            &worker->keepalive.cb_id);
        return 0;
    }

    if (ucs_unlikely(worker->keepalive.iter == &worker->all_eps)) {
        ucp_worker_keepalive_next_ep(worker);
    }

    iter_begin = worker->keepalive.iter;
    /* use own loop for elements because standard for_each skips
     * head element */
    /* TODO: use more optimal algo to enumerate EPs to keepalive
     * (linked list) */
    do {
        ep = ucp_worker_keepalive_current_ep(worker);
        ucp_ep_do_keepalive(ep, &worker->keepalive.lane_map);
        if (worker->keepalive.lane_map != 0) {
            /* in case if EP has no resources to send keepalive message
             * then just return without update of last_round timestamp,
             * on next progress iteration we will continue from this point */
            goto out_no_resources;
        }

        worker->keepalive.ep_count++;
        ucp_worker_keepalive_next_ep(worker);
    } while ((iter_begin != worker->keepalive.iter) &&
             (worker->keepalive.ep_count < worker->context->config.ext.keepalive_num_eps));

    worker->keepalive.last_round = now;
    worker->keepalive.ep_count   = 0;

out_no_resources:
    return worker->keepalive.ep_count;
}

void ucp_worker_keepalive_add_ep(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;

    if (ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_NONE) {
        return;
    }

    if (!ucp_worker_keepalive_is_enabled(worker)) {
        return;
    }

    uct_worker_progress_register_safe(worker->uct,
                                      ucp_worker_keepalive_progress, worker, 0,
                                      &worker->keepalive.cb_id);
}

/* EP is removed from worker */
void ucp_worker_keepalive_remove_ep(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;

    if (!ucp_worker_keepalive_is_enabled(worker)) {
        ucs_assert(worker->keepalive.iter == &worker->all_eps);
        return;
    }

    ucs_assert(!ucs_list_is_empty(&worker->all_eps));

    if (ucs_list_is_only(&worker->all_eps, &ucp_ep_ext_gen(ep)->ep_list)) {
        /* this is the last EP in worker */
        worker->keepalive.iter = &worker->all_eps;
    } else if (worker->keepalive.iter == &ucp_ep_ext_gen(ep)->ep_list) {
        /* if iterator points into EP to be removed - then
         * step to next EP */
        ucp_worker_keepalive_next_ep(worker);
    }
}

static void
ucp_worker_discard_tl_uct_ep(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                             unsigned ep_flush_flags)
{
    ucp_worker_h worker = ucp_ep->worker;
    ucp_request_t *req;
    int ret;
    khiter_t iter;

    req = ucp_request_get(worker);
    if (ucs_unlikely(req == NULL)) {
        ucs_error("unable to allocate request for discarding UCT EP %p "
                  "on UCP worker %p", uct_ep, worker);
        return;
    }

    ucs_assert(ucp_ep->ref_cnt < UINT8_MAX);
    ++ucp_ep->ref_cnt;
    ucp_worker_flush_ops_count_inc(worker);
    iter = kh_put(ucp_worker_discard_uct_ep_hash, &worker->discard_uct_ep_hash,
                  uct_ep, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_fatal("failed to put %p UCT EP into the %p worker hash",
                  uct_ep, worker);
    } else if (ret == UCS_KH_PUT_KEY_PRESENT) {
        ucs_fatal("%p UCT EP is already present in the %p worker hash",
                  uct_ep, worker);
    }
    kh_value(&worker->discard_uct_ep_hash, iter) = req;

    ucs_assert(!ucp_wireup_ep_test(uct_ep));
    req->flags                              = 0;
    req->send.ep                            = ucp_ep;
    req->send.uct.func                      = ucp_worker_discard_uct_ep_pending_cb;
    req->send.state.uct_comp.func           = ucp_worker_discard_uct_ep_flush_comp;
    req->send.state.uct_comp.count          = 1;
    req->send.state.uct_comp.status         = UCS_OK;
    req->send.discard_uct_ep.uct_ep         = uct_ep;
    req->send.discard_uct_ep.ep_flush_flags = ep_flush_flags;

    ucp_worker_discard_uct_ep_progress(req);
}

static void
ucp_worker_discard_wireup_uct_ep(ucp_ep_h ucp_ep, ucp_wireup_ep_t *wireup_ep,
                                 unsigned ep_flush_flags, uct_ep_h uct_ep)
{
    if (uct_ep == NULL) {
        return;
    }

    ucp_wireup_ep_disown(&wireup_ep->super.super, uct_ep);
    /* discard the WIREUP EP's UCT EP */
    ucp_worker_discard_uct_ep(ucp_ep, uct_ep, ep_flush_flags,
                              /* make sure that there are no WIREUP MSGs
                               * anymore that are scheduled on the UCT EP, i.e.
                               * the purge callback hasn't be invoked here */
                              (uct_pending_purge_callback_t)
                              ucs_empty_function_do_assert, NULL);
}

static uct_ep_h
ucp_worker_discard_wireup_ep(ucp_ep_h ucp_ep, ucp_wireup_ep_t *wireup_ep,
                             unsigned ep_flush_flags,
                             uct_pending_purge_callback_t purge_cb,
                             void *purge_arg)
{
    uct_ep_h uct_ep;
    int is_owner;

    ucs_assert(wireup_ep != NULL);

    ucp_worker_discard_wireup_uct_ep(ucp_ep, wireup_ep, ep_flush_flags,
                                     wireup_ep->aux_ep);
    ucp_worker_discard_wireup_uct_ep(ucp_ep, wireup_ep, ep_flush_flags,
                                     wireup_ep->sockaddr_ep);

    is_owner = wireup_ep->super.is_owner;
    uct_ep   = ucp_wireup_ep_extract_next_ep(&wireup_ep->super.super);

    /* destroy WIREUP EP allocated for this UCT EP, since discard operation
     * most likely won't have an access to UCP EP as it could be destroyed
     * by the caller */
    uct_ep_destroy(&wireup_ep->super.super);

    /* do nothing, if this wireup EP is not an owner for UCT EP */
    return is_owner ? uct_ep : NULL;
}

/* must be called with async lock held */
int ucp_worker_is_uct_ep_discarding(ucp_worker_h worker, uct_ep_h uct_ep)
{
    return kh_get(ucp_worker_discard_uct_ep_hash,
                  &worker->discard_uct_ep_hash, uct_ep) !=
           kh_end(&worker->discard_uct_ep_hash);
}

/* must be called with async lock held */
void ucp_worker_discard_uct_ep(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                               unsigned ep_flush_flags,
                               uct_pending_purge_callback_t purge_cb,
                               void *purge_arg)
{
    ucs_assert(uct_ep != NULL);
    ucs_assert(purge_cb != NULL);

    uct_ep_pending_purge(uct_ep, purge_cb, purge_arg);

    if (ucp_wireup_ep_test(uct_ep)) {
        uct_ep = ucp_worker_discard_wireup_ep(ucp_ep, ucp_wireup_ep(uct_ep),
                                              ep_flush_flags,
                                              purge_cb, purge_arg);
        if (uct_ep == NULL) {
            return;
        }
    }

    ucp_worker_discard_tl_uct_ep(ucp_ep, uct_ep, ep_flush_flags);
}
