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


typedef struct uct_rdmacm_cep {
    uct_base_ep_t                     super;
    uct_rdmacm_cm_t                   *cm;
    struct rdma_cm_id                 *id;
    void                              *user_data;
    uct_ep_sockaddr_disconnected_cb_t disconnected_cb;

    /* TODO: Allocate separately to reduce memory consumption. These fields are
     *       relevant only for connection establishment. */
    struct {
        char                              priv_data[UCT_RDMACM_CM_MAX_CONN_PRIV];
        uct_sockaddr_priv_pack_callback_t priv_pack_cb;
        union {
            struct {
                uct_ep_client_connected_cb_t connected_cb;
            } client;
            struct {
                uct_ep_server_connected_cb_t connected_cb;
            } server;
        };
    } wireup;
} uct_rdmacm_cep_t;


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

UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cep_t, uct_ep_t);

ucs_status_t uct_rdmacm_cep_disconnect(uct_ep_h ep)
{
    uct_rdmacm_cep_t *cep = ucs_derived_of(ep, uct_rdmacm_cep_t);

    if (rdma_disconnect(cep->id)) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

uct_base_iface_t dummy_iface = {
    .super = {
        .ops = {
            .ep_disconnect = uct_rdmacm_cep_disconnect,
            .ep_destroy    = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cep_t)
        }
    }
};

static void
uct_rdamcm_cm_fill_conn_param(struct rdma_conn_param *conn_param,
                              const uct_rdmacm_priv_data_hdr_t *hdr) {
    static uint32_t qp_num = 0xffffff;

    memset(conn_param, 0, sizeof(*conn_param));
    conn_param->private_data        = hdr;
    conn_param->private_data_len    = sizeof(*hdr) + hdr->length;
    conn_param->responder_resources = 1;
    conn_param->initiator_depth     = 1;
    conn_param->retry_count         = 7;
    conn_param->rnr_retry_count     = 7;
    conn_param->qp_num              = qp_num--;
    if (qp_num == 0) {
        qp_num = 0xffffff;
    }
}

UCS_CLASS_INIT_FUNC(uct_rdmacm_cep_t, const uct_ep_params_t *params)
{
    ucs_status_t status = UCS_OK;
    struct rdma_cm_event *event;
    struct rdma_conn_param conn_param;
    uct_rdmacm_priv_data_hdr_t *hdr;

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_CM)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS) ||
        !(params->sockaddr_cb_flags & UCT_CB_FLAG_ASYNC)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(params->field_mask & (UCT_EP_PARAM_FIELD_SOCKADDR |
                                UCT_EP_PARAM_FIELD_CONN_REQUEST))) {
        return UCS_ERR_INVALID_PARAM;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &dummy_iface);

    self->cm                  = ucs_derived_of(params->cm, uct_rdmacm_cm_t);
    self->wireup.priv_pack_cb = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB) ?
                                params->sockaddr_pack_cb : NULL;
    self->disconnected_cb     = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECTED_CB) ?
                                params->disconnected_cb : NULL;
    self->user_data           = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_USER_DATA) ?
                                params->user_data : NULL;

    /* TODO: to separate server & cliet funcs */
    if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR) {
        self->wireup.client.connected_cb = params->sockaddr_connected_cb.client;

        if (rdma_create_id(self->cm->ev_ch, &self->id, self, RDMA_PS_TCP)) {
            ucs_error("rdma_create_id() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto out;
        }
        if (rdma_resolve_addr(self->id, NULL,
                              (struct sockaddr *)params->sockaddr->addr,
                              1000/* TODO */)) {
            ucs_error("rdma_resolve_addr() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto out;
        }
    } else {
        ucs_assert(params->field_mask & UCT_EP_PARAM_FIELD_CONN_REQUEST);

        self->wireup.server.connected_cb = params->sockaddr_connected_cb.server;

        event = (struct rdma_cm_event *)params->conn_request;

        /* TODO: migrate id if cm is different */
        ucs_assert(event->listen_id->channel == self->cm->ev_ch);

        self->id          = event->id;
        self->id->context = self;
        hdr = (uct_rdmacm_priv_data_hdr_t *)self->wireup.priv_data;
        hdr->length = self->wireup.priv_pack_cb(self->user_data,
                                                self->id->verbs->device->name,
                                                hdr + 1);
        if ((hdr->length < 0) || (hdr->length > UCT_RDMACM_CM_MAX_CONN_PRIV)) {
            status = UCS_ERR_INVALID_PARAM;
            goto out;
        }
        hdr->status = UCS_OK;
        uct_rdamcm_cm_fill_conn_param(&conn_param, hdr);
        if (rdma_accept(event->id, &conn_param)) {
            status = UCS_ERR_IO_ERROR;
        }
        if (rdma_ack_cm_event(event)) {
            status = UCS_ERR_IO_ERROR;
        }
    }
out:
    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_cep_t)
{
    rdma_destroy_id(self->id);
}

UCS_CLASS_DEFINE(uct_rdmacm_cep_t, uct_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cep_t, uct_ep_t);

uct_cm_ops_t uct_rdmacm_cm_ops = {
    .close            = uct_rdmacm_cm_cleanup,
    .cm_query         = uct_rdmacm_cm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_listener_t),
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t),
    .ep_create        = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_cep_t)
};


static int
uct_rdmacm_cm_process_event(uct_rdmacm_cm_t *cm, struct rdma_cm_event *event)
{
    struct rdma_conn_param      conn_param;
    uct_rdmacm_priv_data_hdr_t  *hdr;
    uct_rdmacm_cep_t            *cep;
    uct_rdmacm_listener_t       *listener;

    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        rdma_resolve_route(event->id, 1000);
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        cep = (uct_rdmacm_cep_t *)event->id->context;
        ucs_assert(event->id == cep->id);
        rdma_ack_cm_event(event);
        hdr = (uct_rdmacm_priv_data_hdr_t *)cep->wireup.priv_data;
        hdr->length = cep->wireup.priv_pack_cb(cep->user_data,
                                               cep->id->verbs->device->name,
                                               hdr + 1);
        hdr->status = (hdr->length < 0) ? hdr->length : UCS_OK;

        if (hdr->status != UCS_OK) {
            return hdr->status;
        }
        uct_rdamcm_cm_fill_conn_param(&conn_param, hdr);
        return rdma_connect(cep->id, &conn_param);
    case RDMA_CM_EVENT_CONNECT_REQUEST:
        listener = event->listen_id->context;
        hdr      = (uct_rdmacm_priv_data_hdr_t *)event->param.conn.private_data;
        if (hdr->status == UCS_OK) {
            listener->conn_request_cb(&listener->super, listener->user_data,
                                      event->id->verbs->device->name,
                                      (uct_conn_request_h)event, hdr + 1,
                                      hdr->length);
            return 0;
            /* Do not ack event here, ep create does this */
        }

        hdr->status = UCS_ERR_REJECTED;
        hdr->length = 0;
        if (rdma_reject(event->id, hdr, sizeof(*hdr))) {
            return 1;
        }
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
        if (rdma_establish(event->id)) {
            return 1;
        }
        cep = event->id->context;
        hdr = (uct_rdmacm_priv_data_hdr_t *)event->param.conn.private_data;
        cep->wireup.client.connected_cb(&cep->super.super, cep->user_data,
                                        hdr + 1, hdr->length, hdr->status);
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_ESTABLISHED:
        cep = event->id->context;
        cep->wireup.server.connected_cb(&cep->super.super, cep->user_data,
                                        UCS_OK);
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_DISCONNECTED:
        cep = event->id->context;
        cep->disconnected_cb(&cep->super.super, cep->user_data);
        return rdma_ack_cm_event(event);
    default:
        assert(0);
    }
    return 1;
}

static void uct_rdmacm_cm_event_handler(int fd, void *arg)
{
    uct_rdmacm_cm_t      *cm = (uct_rdmacm_cm_t *)arg;
    struct rdma_cm_event *event;
    int                  ret;

    while (1) {
        /* Fetch an event */
        ret = rdma_get_cm_event(cm->ev_ch, &event);
        if (ret) {
            /* EAGAIN (in a non-blocking rdma_get_cm_event) means that
             * there are no more events */
            if (errno != EAGAIN) {
                ucs_warn("rdma_get_cm_event() failed: %m");
            }
            return;
        }
        uct_rdmacm_cm_process_event(cm, event);
    }
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
