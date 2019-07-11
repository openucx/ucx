/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h" /* Defines HAVE_RDMACM_QP_LESS */
#endif

#include "rdmacm_cm.h"
#include "rdmacm_listener.h"
#include <ucs/async/async.h>

#include <poll.h>
#include <rdma/rdma_cma.h>


static ucs_status_t uct_rdmacm_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    if (cm_attr->field_mask & UCT_CM_ATTR_FIELD_MAX_CONN_PRIV) {
        cm_attr->max_conn_priv = uct_rdmacm_cm_get_max_conn_priv();
    }
    return UCS_OK;
}


static void uct_rdmacm_cm_event_handler(int fd, void *arg){}


static uct_cm_ops_t uct_rdmacm_cm_ops = {
    .close            = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cm_t),
    .cm_query         = uct_rdmacm_cm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_listener_t),
    .listener_reject  = (void*)ucs_empty_function,
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t),
    .ep_create        = (void*)ucs_empty_function
};

UCS_CLASS_INIT_FUNC(uct_rdmacm_cm_t, uct_component_h component,
                    uct_worker_h worker)
{
    uct_priv_worker_t *worker_priv;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_t, &uct_rdmacm_cm_ops, component);

    self->worker          = worker;
    self->ev_ch           = rdma_create_event_channel();
    if (self->ev_ch == NULL) {
        ucs_error("rdma_create_event_channel failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    /* Set the event_channel fd to non-blocking mode
     * (so that rdma_get_cm_event won't be blocking) */
    status = ucs_sys_fcntl_modfl(self->ev_ch->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_ev_ch;
    }

    worker_priv = ucs_derived_of(worker, uct_priv_worker_t);
    status = ucs_async_set_event_handler(worker_priv->async->mode,
                                         self->ev_ch->fd, POLLIN,
                                         uct_rdmacm_cm_event_handler, self,
                                         worker_priv->async);
    if (status != UCS_OK) {
        goto err_destroy_ev_ch;
    }

    return UCS_OK;

err_destroy_ev_ch:
    rdma_destroy_event_channel(self->ev_ch);
err:
    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_cm_t)
{
    ucs_status_t status;

    status = ucs_async_remove_handler(self->ev_ch->fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove event handler for fd %d: %s",
                 self->ev_ch->fd, ucs_status_string(status));
    }

    rdma_destroy_event_channel(self->ev_ch);
}

UCS_CLASS_DEFINE(uct_rdmacm_cm_t, uct_cm_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cm_t, uct_cm_t, uct_component_h, uct_worker_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cm_t, uct_cm_t);
