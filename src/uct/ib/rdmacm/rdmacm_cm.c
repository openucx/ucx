/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h" /* Defines HAVE_RDMACM_QP_LESS */
#endif

#if HAVE_RDMACM_QP_LESS

#include "rdmacm_cm.h"
#include <ucs/async/async.h>

#include <poll.h>
#include <rdma/rdma_cma.h>

UCS_CLASS_INIT_FUNC(uct_rdmacm_listener_t, uct_cm_h cm,
                    const struct sockaddr *saddr, socklen_t socklen,
                    const uct_listener_params_t *params)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_listener_t){}

UCS_CLASS_DEFINE(uct_rdmacm_listener_t, uct_listener_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_listener_t, uct_listener_t,
                          uct_cm_h , const struct sockaddr *, socklen_t ,
                          const uct_listener_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_listener_t, uct_listener_t);

static void uct_rdmacm_cm_cleanup(uct_cm_h cm)
{
    uct_rdmacm_cm_t *rdmacm_cm = ucs_derived_of(cm, uct_rdmacm_cm_t);
    ucs_status_t status;

    status = ucs_async_remove_handler(rdmacm_cm->ev_ch->fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove event handler for fd %d: %s",
                 rdmacm_cm->ev_ch->fd, ucs_status_string(status));
    }
    rdma_destroy_event_channel(rdmacm_cm->ev_ch);
    ucs_free(cm);
}

static ucs_status_t uct_rdmacm_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    if (cm_attr->field_mask & UCT_CM_ATTR_FIELD_MAX_CONN_PRIV) {
        cm_attr->max_conn_priv = UCT_RDMACM_CM_MAX_CONN_PRIV;
    }
    return UCS_OK;
}

uct_cm_ops_t uct_rdmacm_cm_ops = {
    .close            = uct_rdmacm_cm_cleanup,
    .cm_query         = uct_rdmacm_cm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_listener_t),
    .listener_reject  = (void*)ucs_empty_function,
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t),
    .ep_create        = (void*)ucs_empty_function
};

static void uct_rdmacm_cm_event_handler(int fd, void *arg){}

ucs_status_t uct_rdmacm_cm_open(uct_component_h component, uct_worker_h worker,
                                uct_cm_h *uct_cm_p)
{
    uct_priv_worker_t *worker_priv;
    uct_rdmacm_cm_t *rdmacm_cm;
    ucs_status_t status;

    rdmacm_cm = ucs_calloc(1, sizeof(uct_rdmacm_cm_t), "uct_rdmacm_cm_open");
    if (rdmacm_cm == NULL) {
        ucs_error("failed to allocate an rdmacm_cm %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rdmacm_cm->super.ops       = &uct_rdmacm_cm_ops;
    rdmacm_cm->super.component = component;
    rdmacm_cm->worker          = worker;
    rdmacm_cm->ev_ch           = rdma_create_event_channel();
    if (rdmacm_cm->ev_ch == NULL) {
        ucs_error("rdma_create_event_channel failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_cm;
    }

    /* Set the event_channel fd to non-blocking mode
     * (so that rdma_get_cm_event won't be blocking) */
    status = ucs_sys_fcntl_modfl(rdmacm_cm->ev_ch->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_ev_ch;
    }

    worker_priv = ucs_derived_of(worker, uct_priv_worker_t);
    status = ucs_async_set_event_handler(worker_priv->async->mode,
                                         rdmacm_cm->ev_ch->fd, POLLIN,
                                         uct_rdmacm_cm_event_handler, rdmacm_cm,
                                         worker_priv->async);
    if (status != UCS_OK) {
        goto err_destroy_ev_ch;
    }

    *uct_cm_p = &rdmacm_cm->super;
    return UCS_OK;

err_destroy_ev_ch:
    rdma_destroy_event_channel(rdmacm_cm->ev_ch);
err_free_cm:
    ucs_free(rdmacm_cm);
err:
    return status;
}

#endif /* HAVE_RDMACM_QP_LESS */
