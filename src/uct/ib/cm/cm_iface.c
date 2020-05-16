/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cm.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/arch/atomic.h>
#include <ucs/async/async.h>
#include <ucs/debug/log.h>
#include <poll.h>


static ucs_config_field_t uct_cm_iface_config_table[] = {
  {UCT_IB_CONFIG_PREFIX, "RX_INLINE=0", NULL,
   ucs_offsetof(uct_cm_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {"TIMEOUT", "300ms", "Timeout for MAD layer",
   ucs_offsetof(uct_cm_iface_config_t, timeout), UCS_CONFIG_TYPE_TIME},

  {"RETRY_COUNT", "100", "Number of retries for MAD layer",
   ucs_offsetof(uct_cm_iface_config_t, retry_count), UCS_CONFIG_TYPE_UINT},

  {"MAX_OP", "1024", "Maximal number of outstanding SIDR operations",
   ucs_offsetof(uct_cm_iface_config_t, max_outstanding), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static uct_ib_iface_ops_t uct_cm_iface_ops;


static unsigned uct_cm_iface_progress(void *arg)
{
    uct_cm_iface_t *iface = arg;
    uct_cm_pending_req_priv_t *priv;
    uct_cm_iface_op_t *op;
    unsigned count;

    uct_cm_enter(iface);

    /* Invoke flush completions at the head of the queue - the sends which
     * started before them were already completed.
     */
    count = 0;
    ucs_queue_for_each_extract(op, &iface->outstanding_q, queue, !op->is_id) {
        uct_invoke_completion(op->comp, UCS_OK);
        ucs_free(op);
        ++count;
    }

    /* we are in the progress() context. Now it is safe to release resources. */
    iface->num_outstanding -= iface->num_completions;
    iface->num_completions  = 0;

    /* Dispatch pending operations */
    uct_pending_queue_dispatch(priv, &iface->notify_q,
                               iface->num_outstanding < iface->config.max_outstanding);

    /* Remove the progress callback only if there is no user completion at the
     * head of the queue. It could be added by the progress callback.
     */
    if (ucs_queue_is_empty(&iface->outstanding_q) ||
        ucs_queue_head_elem_non_empty(&iface->outstanding_q, uct_cm_iface_op_t, queue)->is_id)
    {
        uct_worker_progress_unregister_safe(&uct_cm_iface_worker(iface)->super,
                                            &iface->slow_prog_id);
    }

    uct_cm_leave(iface);

    return count;
}

ucs_status_t uct_cm_iface_flush_do(uct_cm_iface_t *iface, uct_completion_t *comp)
{
    uct_cm_iface_op_t *op;

    if (iface->num_outstanding == 0) {
        return UCS_OK;
    }

    /* If user request a completion callback, allocate a new operation and put
     * it in the tail of the queue. It will be called when all operations which
     * were sent before are completed.
     */
    if (comp != NULL) {
        op = ucs_malloc(sizeof *op, "cm_op");
        if (op == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        op->is_id = 0;
        op->comp  = comp;
        ucs_queue_push(&iface->outstanding_q, &op->queue);
    }

    sched_yield();
    return UCS_INPROGRESS;
}

ucs_status_t uct_cm_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);
    ucs_status_t status;

    uct_cm_enter(iface);
    status = uct_cm_iface_flush_do(iface, comp);
    if (status == UCS_OK) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    } else if (status == UCS_INPROGRESS){
        UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    }
    uct_cm_leave(iface);

    return status;
}

static void uct_cm_iface_handle_sidr_req(uct_cm_iface_t *iface,
                                         struct ib_cm_event *event)
{
    uct_cm_hdr_t *hdr = event->private_data;
    struct ib_cm_sidr_rep_param rep;
    int ret;

    VALGRIND_MAKE_MEM_DEFINED(hdr, sizeof(hdr));
    VALGRIND_MAKE_MEM_DEFINED(hdr + 1, hdr->length);

    uct_cm_iface_trace_data(iface, UCT_AM_TRACE_TYPE_RECV, hdr, "RX: SIDR_REQ");

    /* Send reply */
    ucs_trace_data("TX: SIDR_REP [id %p{%u}]", event->cm_id,
                   event->cm_id->handle);
    memset(&rep, 0, sizeof rep);
    rep.status = IB_SIDR_SUCCESS;
    ret = ib_cm_send_sidr_rep(event->cm_id, &rep);
    if (ret) {
        ucs_error("ib_cm_send_sidr_rep() failed: %m");
    }

    uct_iface_invoke_am(&iface->super.super, hdr->am_id, hdr + 1, hdr->length, 0);
}

static void uct_cm_iface_outstanding_remove(uct_cm_iface_t* iface,
                                            struct ib_cm_id* id)
{
    uct_cm_iface_op_t *op;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(op, iter, &iface->outstanding_q, queue) {
        if (op->is_id && (op->id == id)) {
            ucs_queue_del_iter(&iface->outstanding_q, iter);
            /* Must not release resources from the async context
             * because it will break pending op ordering.
             * For example bcopy() may succeed while there are queued
             * pending ops:
             * bcopy() -> no resources
             * pending_add() -> ok
             * <-- async event: resources available
             * bcopy() --> ok. oops this is out of order send
             *
             * save the number and do actual release in the
             * progress() context.
             */
            ++iface->num_completions;
            ucs_free(op);
            return;
        }
    }

    ucs_fatal("outstanding cm id %p not found", id);
}

static void uct_cm_iface_outstanding_purge(uct_cm_iface_t *iface)
{
    uct_cm_iface_op_t *op;

    ucs_queue_for_each_extract(op, &iface->outstanding_q, queue, 1) {
        if (op->is_id) {
            ib_cm_destroy_id(op->id);
        } else {
            uct_invoke_completion(op->comp, UCS_ERR_CANCELED);
        }
        ucs_free(op);
    }
    iface->num_outstanding = 0;
}

static void uct_cm_iface_event_handler(int fd, int events, void *arg)
{
    uct_cm_iface_t *iface = arg;
    struct ib_cm_event *event;
    struct ib_cm_id *id;
    int destroy_id;
    int ret;

    ucs_trace_func("");

    for (;;) {
        /* Fetch all events */
        ret = ib_cm_get_event(iface->cmdev, &event);
        if (ret) {
            if (errno != EAGAIN) {
                ucs_warn("ib_cm_get_event() failed: %m");
            }
            return;
        }

        id  = event->cm_id;

        /* Handle the event */
        switch (event->event) {
        case IB_CM_SIDR_REQ_ERROR:
            ucs_error("SIDR request error, status: %s",
                      ibv_wc_status_str(event->param.send_status));
            destroy_id = 1;
            break;
        case IB_CM_SIDR_REQ_RECEIVED:
            uct_cm_iface_handle_sidr_req(iface, event);
            destroy_id = 1; /* Destroy the ID created by the driver */
            break;
        case IB_CM_SIDR_REP_RECEIVED:
            ucs_trace_data("RX: SIDR_REP [id %p{%u}]", id, id->handle);
            uct_cm_iface_outstanding_remove(iface, id);
            destroy_id = 1; /* Destroy the ID which was used for sending */
            break;
        default:
            ucs_warn("Unexpected CM event: %d", event->event);
            destroy_id = 0;
            break;
        }

        /* Acknowledge CM event, remember the id, in case we would destroy it */
        ret = ib_cm_ack_event(event);
        if (ret) {
            ucs_warn("ib_cm_ack_event() failed: %m");
        }

        /* If there is an id which should be destroyed, do it now, after
         * acknowledging all events.
         */
        if (destroy_id) {
            ret = ib_cm_destroy_id(id);
            if (ret) {
                ucs_error("ib_cm_destroy_id() failed: %m");
            }
        }

        uct_worker_progress_register_safe(&uct_cm_iface_worker(iface)->super,
                                          uct_cm_iface_progress, iface, 0,
                                          &iface->slow_prog_id);
    }
}

static void uct_cm_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_ib_iface_t *iface = ucs_container_of(self, uct_ib_iface_t, release_desc);
    /* Don't use UCS_PTR_BYTE_OFFSET here due to coverity false positive report */
    ucs_free((char*)desc - iface->config.rx_headroom_offset);
}

static UCS_CLASS_INIT_FUNC(uct_cm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cm_iface_config_t *config = ucs_derived_of(tl_config, uct_cm_iface_config_t);
    uct_ib_iface_init_attr_t init_attr = {};
    ucs_status_t status;
    int ret;

    ucs_trace_func("");

    init_attr.cq_len[UCT_IB_DIR_TX] = 1;
    init_attr.cq_len[UCT_IB_DIR_RX] = config->super.rx.queue_len;
    init_attr.seg_size              = ucs_min(IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE,
                                              config->super.seg_size);

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &uct_cm_iface_ops, md, worker,
                              params, &config->super, &init_attr);

    if (self->super.super.worker->async == NULL) {
        ucs_error("cm must have async!=NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    self->num_outstanding     = 0;
    self->num_completions     = 0;
    self->service_id          = 0;
    self->config.timeout_ms   = (int)(config->timeout * 1e3 + 0.5);
    self->config.max_outstanding = config->max_outstanding;
    self->config.retry_count  = ucs_min(config->retry_count, UINT8_MAX);
    self->notify_q.head       = NULL;
    self->slow_prog_id        = UCS_CALLBACKQ_ID_NULL;
    ucs_queue_head_init(&self->notify_q);
    ucs_queue_head_init(&self->outstanding_q);

    /* Redefine receive desc release callback */
    self->super.release_desc.cb = uct_cm_iface_release_desc;

    self->cmdev = ib_cm_open_device(uct_ib_iface_device(&self->super)->ibv_context);
    if (self->cmdev == NULL) {
        ucs_error("ib_cm_open_device() failed: %m. Check if ib_ucm.ko module is loaded.");
        status = UCS_ERR_NO_DEVICE;
        goto err;
    }

    status = ucs_sys_fcntl_modfl(self->cmdev->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close_device;
    }

    ret = ib_cm_create_id(self->cmdev, &self->listen_id, self);
    if (ret) {
        ucs_error("ib_cm_create_id() failed: %m");
        status = UCS_ERR_NO_DEVICE;
        goto err_close_device;
    }

    do {
        self->service_id = (uint32_t)(ucs_generate_uuid((uintptr_t)self) &
                                      (~IB_CM_ASSIGN_SERVICE_ID_MASK));
        ret = ib_cm_listen(self->listen_id, self->service_id, 0);
        if (ret) {
            if (errno == EBUSY) {
                /* The generated service id is already in use - try to
                 * generate another one.
                 */
                ucs_debug("ib_cm service id 0x%x already in use, "
                          "trying another one", self->service_id);
                continue;
            } else {
                ucs_error("ib_cm_listen(service_id=0x%x) failed: %m",
                          self->service_id);
                status = UCS_ERR_INVALID_ADDR;
                goto err_destroy_id;
            }
        }
    } while (ret);

    if (self->super.super.worker->async->mode == UCS_ASYNC_MODE_SIGNAL) {
        ucs_warn("ib_cm fd does not support SIGIO");
    }

    status = ucs_async_set_event_handler(self->super.super.worker->async->mode,
                                         self->cmdev->fd, UCS_EVENT_SET_EVREAD,
                                         uct_cm_iface_event_handler, self,
                                         self->super.super.worker->async);
    if (status != UCS_OK) {
        ucs_error("failed to set event handler");
        goto err_destroy_id;
    }

    ucs_debug("listening for SIDR service_id 0x%x on fd %d", self->service_id,
              self->cmdev->fd);
    return UCS_OK;

err_destroy_id:
    ib_cm_destroy_id(self->listen_id);
err_close_device:
    ib_cm_close_device(self->cmdev);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cm_iface_t)
{

    ucs_trace_func("");

    ucs_async_remove_handler(self->cmdev->fd, 1);

    uct_cm_enter(self);
    uct_cm_iface_outstanding_purge(self);
    ib_cm_destroy_id(self->listen_id);
    ib_cm_close_device(self->cmdev);
    uct_worker_progress_unregister_safe(&uct_cm_iface_worker(self)->super,
                                        &self->slow_prog_id);
    uct_cm_leave(self);

    /* At this point all outstanding have been removed, and no further events
     * can be added.
     */
}

UCS_CLASS_DEFINE(uct_cm_iface_t, uct_ib_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                                 const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_iface_t, uct_iface_t);

static ucs_status_t uct_cm_iface_query(uct_iface_h tl_iface,
                                       uct_iface_attr_t *iface_attr)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);
    ucs_status_t status;
    size_t mtu;

    status = uct_ib_iface_query(&iface->super, 32 /* TODO */, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->overhead = 1200e-9;

    mtu = ucs_min(IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE - sizeof(uct_cm_hdr_t),
                  UINT8_MAX);

    iface_attr->cap.am.max_bcopy      = mtu;
    iface_attr->iface_addr_len        = sizeof(uint32_t);
    iface_attr->ep_addr_len           = 0;
    iface_attr->max_conn_priv         = 0;
    iface_attr->cap.flags             = UCT_IFACE_FLAG_AM_BCOPY |
                                        UCT_IFACE_FLAG_AM_DUP   |
                                        UCT_IFACE_FLAG_PENDING  |
                                        UCT_IFACE_FLAG_CB_ASYNC |
                                        UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    return UCS_OK;
}

static ucs_status_t uct_cm_iface_get_address(uct_iface_h tl_iface,
                                             uct_iface_addr_t *iface_addr)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);
    *(uint32_t*)iface_addr = iface->service_id;
    return UCS_OK;
}


static uct_ib_iface_ops_t uct_cm_iface_ops = {
    {
    .ep_am_bcopy              = uct_cm_ep_am_bcopy,
    .ep_pending_add           = uct_cm_ep_pending_add,
    .ep_pending_purge         = uct_cm_ep_pending_purge,
    .ep_flush                 = uct_cm_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_cm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cm_ep_t),
    .iface_flush              = uct_cm_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cm_iface_t),
    .iface_query              = uct_cm_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_get_address        = uct_cm_iface_get_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable
    },
    .create_cq                = uct_ib_verbs_create_cq,
    .arm_cq                   = ucs_empty_function_return_success,
};

static int uct_cm_is_module_loaded(uct_ib_md_t *ib_md)
{
    struct ib_cm_device *cmdev = NULL;

    cmdev = ib_cm_open_device(ib_md->dev.ibv_context);
    if (cmdev == NULL) {
        ucs_debug("ib_cm_open_device() for %s failed: %m. "
                  "Check if ib_ucm.ko module is loaded.",
                  uct_ib_device_name(&ib_md->dev));
        return 0;
    }

    ib_cm_close_device(cmdev);
    return 1;
}

static ucs_status_t
uct_cm_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                        unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    if (!uct_cm_is_module_loaded(ib_md)) {
        *num_tl_devices_p = 0;
        *tl_devices_p     = NULL;
        return UCS_OK;
    }

   return uct_ib_device_query_ports(&ib_md->dev, UCT_IB_DEVICE_FLAG_LINK_IB,
                                    tl_devices_p, num_tl_devices_p);
}

UCT_TL_DEFINE(&uct_ib_component, cm, uct_cm_query_tl_devices, uct_cm_iface_t,
              "CM_", uct_cm_iface_config_table, uct_cm_iface_config_t);
