/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "cm.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/tl/context.h>
#include <ucs/async/async.h>
#include <ucs/debug/log.h>
#include <poll.h>
#include <infiniband/arch.h>


static ucs_config_field_t uct_cm_iface_config_table[] = {
  {"IB_", "RX_INLINE=0", NULL,
   ucs_offsetof(uct_cm_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {"ASYNC_MODE", "thread", "Async mode to use",
   ucs_offsetof(uct_cm_iface_config_t, async_mode), UCS_CONFIG_TYPE_ENUM(ucs_async_mode_names)},

  {"TIMEOUT", "100ms", "Timeout for MAD layer",
   ucs_offsetof(uct_cm_iface_config_t, timeout), UCS_CONFIG_TYPE_TIME},

  {"RETRY_COUNT", "20", "Number of retries for MAD layer",
   ucs_offsetof(uct_cm_iface_config_t, retry_count), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

static uct_iface_ops_t uct_cm_iface_ops;


ucs_status_t uct_cm_iface_flush(uct_iface_h tl_iface)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);

    if (iface->inflight == 0) {
        return UCS_OK;
    }

    sched_yield();
    return UCS_INPROGRESS;
}

static void uct_cm_iface_handle_sidr_req(uct_cm_iface_t *iface,
                                         struct ib_cm_event *event)
{
    uct_cm_hdr_t *hdr = event->private_data;
    struct ib_cm_sidr_rep_param rep;
    ucs_status_t status;
    void *desc;
    int ret;

    VALGRIND_MAKE_MEM_DEFINED(hdr, sizeof(hdr));
    VALGRIND_MAKE_MEM_DEFINED(hdr + 1, hdr->length);

    ucs_trace_data("RECV SIDR_REQ am_id %d length %d", hdr->am_id,
                   hdr->length);

    /* Allocate temporary buffer to serve as receive descriptor */
    desc = ucs_malloc(iface->super.config.rx_payload_offset + hdr->length,
                      "cm_recv_desc");
    if (desc == NULL) {
        ucs_error("failed to allocate cm receive descriptor");
        return;
    }

    /* Call active message handler */
    status = uct_iface_invoke_am(&iface->super.super, hdr->am_id, hdr + 1, hdr->length,
                                 desc + iface->super.config.rx_headroom_offset);
    if (status == UCS_OK) {
        ucs_free(desc);
    }

    /* Send reply */
    ucs_trace_data("SEND SIDR_REP (dummy)");
    memset(&rep, 0, sizeof rep);
    rep.status = IB_SIDR_SUCCESS;
    ret = ib_cm_send_sidr_rep(event->cm_id, &rep);
    if (ret) {
        ucs_error("ib_cm_send_sidr_rep() failed: %m");
    }
}

static void uct_cm_iface_event_handler(void *arg)
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
            ucs_trace_data("RECV SIDR_REP (dummy)");
            ucs_assert(iface->inflight > 0);
            ucs_atomic_add32(&iface->inflight, -1);
            destroy_id      = 1; /* Destroy the ID which was used for sending */
            break;
        default:
            ucs_warn("Unexpected CM event: %d", event->event);
            destroy_id = 0;
            break;
        }

        /* Acknowledge CM event, remember the id, in case we would destroy it */
        id  = event->cm_id;
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
    }
}

static void uct_cm_iface_release_desc(uct_iface_t *tl_iface, void *desc)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);
    ucs_free(desc - iface->super.config.rx_payload_offset);
}

static UCS_CLASS_INIT_FUNC(uct_cm_iface_t, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_cm_iface_config_t *config = ucs_derived_of(tl_config, uct_cm_iface_config_t);
    ucs_status_t status;
    int ret;

    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &uct_cm_iface_ops, worker,
                              dev_name, rx_headroom, 0 /* rx_priv_len */,
                              0 /* rx_hdr_len */, 1 /* tx_cq_len */,
                              &config->super);

    self->service_id         = (uint32_t)(ucs_generate_uuid((uintptr_t)self) &
                                            (~IB_CM_ASSIGN_SERVICE_ID_MASK));
    self->inflight           = 0;
    self->config.timeout_ms  = (int)(config->timeout * 1e3 + 0.5);
    self->config.retry_count = ucs_min(config->retry_count, UINT8_MAX);

    self->cmdev = ib_cm_open_device(uct_ib_iface_device(&self->super)->ibv_context);
    if (self->cmdev == NULL) {
        ucs_error("ib_cm_open_device() failed: %m");
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

    ret = ib_cm_listen(self->listen_id, self->service_id, 0);
    if (ret) {
        ucs_error("ib_cm_listen() failed: %m");
        status = UCS_ERR_INVALID_ADDR;
        goto err_destroy_id;
    }

    if (config->async_mode == UCS_ASYNC_MODE_SIGNAL) {
        ucs_warn("ib_cm fd does not support SIGIO");
    }

    status = ucs_async_set_event_handler(config->async_mode, self->cmdev->fd,
                                         POLLIN, uct_cm_iface_event_handler, self,
                                         NULL);
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

    if (self->inflight) {
        ucs_debug("waiting for %d in-flight requests to complete", self->inflight);
        while (self->inflight) {
            sched_yield();
        }
    }
    ucs_async_unset_event_handler(self->cmdev->fd);
    ib_cm_destroy_id(self->listen_id);
    ib_cm_close_device(self->cmdev);
}

UCS_CLASS_DEFINE(uct_cm_iface_t, uct_ib_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_iface_t, uct_iface_t, uct_worker_h,
                                 const char*, size_t, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_iface_t, uct_iface_t);

static ucs_status_t uct_cm_iface_query(uct_iface_h tl_iface,
                                       uct_iface_attr_t *iface_attr)
{
    size_t mtu;

    mtu = ucs_min(IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE - sizeof(uct_cm_hdr_t),
                  UINT8_MAX);

    memset(iface_attr, 0, sizeof(*iface_attr));
    iface_attr->cap.am.max_short      = mtu;
    iface_attr->cap.am.max_bcopy      = mtu;
    iface_attr->cap.am.max_zcopy      = 0;
    iface_attr->iface_addr_len        = sizeof(struct ibv_sa_path_rec);
    iface_attr->ep_addr_len           = 0;
    iface_attr->cap.flags             = UCT_IFACE_FLAG_AM_SHORT |
                                        UCT_IFACE_FLAG_AM_BCOPY |
                                        UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->completion_priv_len   = 0;
    return UCS_OK;
}

ucs_status_t uct_cm_iface_get_addr(uct_cm_iface_t *iface, uct_cm_iface_addr_t *addr)
{
    int ret;

    ret = ibv_query_gid(uct_ib_iface_device(&iface->super)->ibv_context,
                        iface->super.port_num, 0 /* TODO */, &addr->gid);
    if (ret) {
        return UCS_ERR_INVALID_ADDR;
    }

    addr->lid        = iface->super.addr.lid;
    addr->service_id = iface->service_id;
    return UCS_OK;
}

static ucs_status_t uct_cm_iface_get_address(uct_iface_h tl_iface,
                                             uct_iface_addr_t *iface_addr)
{
    return uct_cm_iface_get_addr(ucs_derived_of(tl_iface, uct_cm_iface_t),
                                 ucs_derived_of(iface_addr, uct_cm_iface_addr_t));
}

static uct_iface_ops_t uct_cm_iface_ops = {
    .iface_query           = uct_cm_iface_query,
    .iface_get_address     = uct_cm_iface_get_address,
    .iface_flush           = uct_cm_iface_flush,
    .iface_close           = UCS_CLASS_DELETE_FUNC_NAME(uct_cm_iface_t),
    .iface_release_am_desc = uct_cm_iface_release_desc,
    .ep_connect_to_iface   = uct_cm_ep_connect_to_iface,
    .ep_am_short           = uct_cm_ep_am_short,
    .ep_create             = UCS_CLASS_NEW_FUNC_NAME(uct_cm_ep_t),
    .ep_destroy            = UCS_CLASS_DELETE_FUNC_NAME(uct_cm_ep_t),
    .ep_am_bcopy           = uct_cm_ep_am_bcopy,
    .ep_flush              = uct_cm_ep_flush,
};

static ucs_status_t uct_cm_query_resources(uct_context_h context,
                                                 uct_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p)
{
    return uct_ib_query_resources(context, 0, /* TODO require IB link layer? */
                                  512, /* TODO */
                                  800,
                                  resources_p, num_resources_p);
}

static uct_tl_ops_t uct_cm_tl_ops = {
    .query_resources     = uct_cm_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_cm_iface_t),
};

static void uct_cm_register(uct_context_t *context)
{
    uct_register_tl(context, "cm", uct_cm_iface_config_table,
                    sizeof(uct_cm_iface_config_t), "CM_", &uct_cm_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, cm, uct_cm_register, ucs_empty_function, 0)
