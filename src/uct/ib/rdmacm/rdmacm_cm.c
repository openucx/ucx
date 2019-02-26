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
#include "rdmacm_iface.h"

#include <ucs/async/async.h>

#include <poll.h>
#include <rdma/rdma_cma.h>

typedef struct uct_rdmacm_cm {
    uct_cm_t                  super;
    struct rdma_event_channel *ev_ch;
    uct_worker_h              worker;
} uct_rdmacm_cm_t;


typedef struct uct_rdmacm_listener {
    uct_listener_t                       super;
    struct rdma_cm_id                    *id;
    uct_listener_conn_request_callback_t conn_request_cb;
    void                                 *user_data;
} uct_rdmacm_listener_t;


UCS_CLASS_INIT_FUNC(uct_rdmacm_listener_t, const uct_listener_params_t *params)
{
    uct_rdmacm_cm_t *rdmacm_cm  = ucs_derived_of(params->cm, uct_rdmacm_cm_t);
    ucs_status_t status = UCS_OK;
    int backlog;

    UCS_CLASS_CALL_SUPER_INIT(uct_listener_t, params->cm);
    self->id              = NULL;
    self->conn_request_cb = params->conn_request_cb;
    self->user_data       = (params->field_mask &
                             UCT_LISTENER_PARAM_FIELD_USER_DATA) ?
                            params->user_data : NULL;

    if (rdma_create_id(rdmacm_cm->ev_ch, &self->id, self, RDMA_PS_TCP)) {
        ucs_error("rdma_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    if (rdma_bind_addr(self->id, (struct sockaddr *)params->sockaddr.addr)) {
        ucs_error("rdma_bind_addr() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    backlog = (params->field_mask & UCT_LISTENER_PARAM_FIELD_BACKLOG) ?
              params->backlog : SOMAXCONN;
    if (rdma_listen(self->id, backlog)) {
        ucs_error("rdma_listen() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

out:
    if ((status != UCS_OK) && (self->id != NULL)) {
        rdma_destroy_id(self->id);
    }
    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_listener_t)
{
    rdma_destroy_id(self->id);
}

UCS_CLASS_DEFINE(uct_rdmacm_listener_t, uct_listener_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_listener_t, uct_listener_t,
                          const uct_listener_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_listener_t, uct_listener_t);

static void uct_rdmacm_cm_cleanup(uct_cm_h cm)
{
    uct_rdmacm_cm_t *rdmacm_cm = ucs_derived_of(cm, uct_rdmacm_cm_t);

    ucs_async_remove_handler(rdmacm_cm->ev_ch->fd, 1);
    rdma_destroy_event_channel(rdmacm_cm->ev_ch);
    ucs_free(cm);
}

static ucs_status_t uct_rdmacm_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    memset(cm_attr, 0, sizeof(*cm_attr));
    cm_attr->max_conn_priv = UCT_RDMACM_CM_MAX_CONN_PRIV;
    return UCS_OK;
}

uct_cm_ops_t uct_rdmacm_cm_ops = {
    .close            = uct_rdmacm_cm_cleanup,
    .cm_query         = uct_rdmacm_cm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_listener_t),
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t)
};


static void uct_rdmacm_cm_event_handler(int fd, void *arg)
{
}

ucs_status_t uct_rdmacm_cm_open(const uct_cm_params_t *params,
                                uct_cm_h *uct_cm_p)
{
    uct_priv_worker_t *worker_priv;
    uct_rdmacm_cm_t *rdmacm_cm;
    uct_md_component_t *mdc;
    ucs_status_t status;

    rdmacm_cm = ucs_calloc(1, sizeof(uct_rdmacm_cm_t), "uct_rdmacm_cm_open");
    if (rdmacm_cm == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_find_md_component(params->md_name, &mdc);
    if (status != UCS_OK) {
        goto err;
    }

    rdmacm_cm->super.ops       = &uct_rdmacm_cm_ops;
    rdmacm_cm->super.component = mdc;
    rdmacm_cm->worker          = params->worker;
    rdmacm_cm->ev_ch           = rdma_create_event_channel();
    if (rdmacm_cm->ev_ch == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err;
    }
    /* Set the event_channel fd to non-blocking mode
     * (so that rdma_get_cm_event won't be blocking) */
    status = ucs_sys_fcntl_modfl(rdmacm_cm->ev_ch->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_ev_ch;
    }

    worker_priv = ucs_derived_of(params->worker, uct_priv_worker_t);
    status = ucs_async_set_event_handler(worker_priv->async->mode,
                                         rdmacm_cm->ev_ch->fd, POLLIN,
                                         uct_rdmacm_cm_event_handler, rdmacm_cm,
                                         worker_priv->async);
    if (status == UCS_OK) {
        *uct_cm_p = &rdmacm_cm->super;
        return UCS_OK;
    }

err_destroy_ev_ch:
    rdma_destroy_event_channel(rdmacm_cm->ev_ch);
err:
    ucs_free(rdmacm_cm);
    return status;
}

#endif /* HAVE_RDMACM_QP_LESS */
