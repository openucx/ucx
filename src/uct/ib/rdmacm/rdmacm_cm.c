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

#include <uct/ib/base/ib_iface.h>
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


enum {
    UCT_RDMACM_CEP_FLAG_LOCAL_DISCONNECTED  = UCS_BIT(0),
    UCT_RDMACM_CEP_FLAG_REMOTE_DISCONNECTED = UCS_BIT(1)
};


typedef struct uct_rdmacm_cep {
    uct_base_ep_t                     super;
    unsigned                          flags;
    uct_rdmacm_cm_t                   *cm;
    struct rdma_cm_id                 *id;
    void                              *user_data;
    uct_ep_sockaddr_disconnected_cb_t disconnected_cb;

    /* TODO: Allocate separately to reduce memory consumption. These fields are
     *       relevant only for connection establishment. */
    struct {
        char                              priv_data[UCT_RDMACM_TCP_PRIV_DATA_LEN];
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

ucs_status_t uct_rdmacm_cep_disconnect(uct_ep_h ep, unsigned flags)
{
    uct_rdmacm_cep_t *cep = ucs_derived_of(ep, uct_rdmacm_cep_t);

    ucs_trace("uct_rdmacm_cep_disconnect on ep %p", cep);
//    ucs_warn("uct_rdmacm_cep_disconnect on ep %p", cep);

    ucs_assert(!(cep->flags & UCT_RDMACM_CEP_FLAG_LOCAL_DISCONNECTED));
    cep->flags |= UCT_RDMACM_CEP_FLAG_LOCAL_DISCONNECTED;
    if (rdma_disconnect(cep->id)) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

uct_base_iface_t dummy_iface = {
    .super = {
        .ops = {
            .ep_pending_purge = (void *)ucs_empty_function_return_success,
            .ep_disconnect    = uct_rdmacm_cep_disconnect,
            .ep_destroy       = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cep_t)
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
    char dev_name[UCT_DEVICE_NAME_MAX];
    struct rdma_conn_param conn_param;
    uct_rdmacm_priv_data_hdr_t *hdr;
    struct rdma_cm_event *event;

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

    self->flags               = 0;
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
        uct_rdmacm_cm_id_to_dev_name(self->id, dev_name);
        hdr = (uct_rdmacm_priv_data_hdr_t *)self->wireup.priv_data;
        hdr->length = self->wireup.priv_pack_cb(self->user_data, dev_name,
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

static size_t uct_rdmacm_cm_fill_addr_flags(const struct rdma_cm_id *id,
                                            const union ibv_gid *gid,
                                            int link_layer,
                                            uct_ib_address_t *addr)
{
    if (link_layer == IBV_LINK_LAYER_ETHERNET) {
        addr->flags = (UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH |
                       UCT_IB_ADDRESS_FLAG_GID);
        return sizeof(uct_ib_address_t) + sizeof(union ibv_gid);  /* raw gid */
    } else {
        addr->flags = (UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB |
                       UCT_IB_ADDRESS_FLAG_LID);
    }

    if (gid->global.subnet_prefix != UCT_IB_LINK_LOCAL_PREFIX) {
        addr->flags |= UCT_IB_ADDRESS_FLAG_IF_ID;
        if (((gid->global.subnet_prefix & UCT_IB_SITE_LOCAL_MASK) ==
             UCT_IB_SITE_LOCAL_PREFIX)) {
            addr->flags |= UCT_IB_ADDRESS_FLAG_SUBNET16;
            return sizeof(uct_ib_address_t) +
                   sizeof(uint16_t) + /* lid */
                   sizeof(uint64_t) + /* if_id */
                   sizeof(uint16_t);  /* subnet16 */
        }
        addr->flags |= UCT_IB_ADDRESS_FLAG_SUBNET64;
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t) + /* lid */
               sizeof(uint64_t) + /* if_id */
               sizeof(uint64_t);  /* subnet64 */
    }
    return sizeof(uct_ib_address_t) + sizeof(uint16_t); /* lid */
}

static ucs_status_t uct_rdmacm_cm_id_to_dev_addr(struct rdma_cm_id *cm_id,
                                          uct_device_addr_t **dev_addr_p,
                                          size_t *dev_addr_len_p)
{
    uct_ib_address_t addr_dummy;
    uct_ib_address_t *dev_addr;
    struct ibv_qp_attr qp_attr;
    int qp_attr_mask;

    qp_attr.qp_state = IBV_QPS_RTR;
    if (rdma_init_qp_attr(cm_id, &qp_attr, &qp_attr_mask)) {
        return UCS_ERR_IO_ERROR;
    }

//    dev_addr->flags = UCT_IB_ADDRESS_FLAG_AH_ATTRS;

/*
    union ibv_gid gid;
    if (ibv_query_gid(cm_id->verbs, cm_id->port_num,
                      qp_attr.ah_attr.grh.sgid_index, &gid)) {
        return UCS_ERR_IO_ERROR;
    }
*/
    struct ibv_port_attr port_arrt;
    if (ibv_query_port(cm_id->verbs, cm_id->port_num, &port_arrt)) {
        return UCS_ERR_IO_ERROR;
    }

    size_t addr_length = uct_rdmacm_cm_fill_addr_flags(cm_id,
                                                       &qp_attr.ah_attr.grh.dgid,
                                                       port_arrt.link_layer,
                                                       &addr_dummy);
    dev_addr = ucs_malloc(addr_length, "IB device address");
    if (dev_addr == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *dev_addr = addr_dummy;

    uct_ib_address_pack(&qp_attr.ah_attr.grh.dgid, qp_attr.ah_attr.dlid, dev_addr);

    *dev_addr_p     = (uct_device_addr_t *)dev_addr;
    *dev_addr_len_p = addr_length;

    return UCS_OK;
}

static int
uct_rdmacm_cm_process_event(uct_rdmacm_cm_t *cm, struct rdma_cm_event *event)
{
    struct rdma_conn_param     conn_param;
    uct_rdmacm_priv_data_hdr_t *hdr;
    uct_rdmacm_cep_t           *cep;
    uct_rdmacm_listener_t      *listener;
    char                       dev_name[UCT_DEVICE_NAME_MAX];
    uct_device_addr_t          *dev_addr;
    ucs_status_t               status;
    size_t                     addr_length;

    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        rdma_resolve_route(event->id, 1000);
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        cep = (uct_rdmacm_cep_t *)event->id->context;
        ucs_assert(event->id == cep->id);
        rdma_ack_cm_event(event);
        hdr = (uct_rdmacm_priv_data_hdr_t *)cep->wireup.priv_data;
        uct_rdmacm_cm_id_to_dev_name(cep->id, dev_name);
        hdr->length = cep->wireup.priv_pack_cb(cep->user_data, dev_name,
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
            uct_rdmacm_cm_id_to_dev_name(event->id, dev_name);
            status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr,
                                                  &addr_length);
            ucs_assert_always(status == UCS_OK);

            listener->conn_request_cb(&listener->super, listener->user_data,
                                      dev_name, dev_addr, addr_length, event,
                                      hdr + 1, hdr->length);
            ucs_free(dev_addr);
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
        status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr,
                                              &addr_length);
        if (status == UCS_OK) {
            cep->wireup.client.connected_cb(&cep->super.super, cep->user_data,
                                            dev_addr, addr_length,  hdr + 1,
                                            hdr->length, hdr->status);
            ucs_free(dev_addr);
            return rdma_ack_cm_event(event);
        } else {
            rdma_ack_cm_event(event);
            return status;
        }
    case RDMA_CM_EVENT_ESTABLISHED:
        cep = event->id->context;
        cep->wireup.server.connected_cb(&cep->super.super, cep->user_data,
                                        UCS_OK);
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_DISCONNECTED:
        ucs_trace("RDMA_CM_EVENT_DISCONNECTED event");
//        ucs_warn("RDMA_CM_EVENT_DISCONNECTED event");
        cep = event->id->context;
        ucs_assert(!(cep->flags & UCT_RDMACM_CEP_FLAG_REMOTE_DISCONNECTED));
        cep->flags |= UCT_RDMACM_CEP_FLAG_REMOTE_DISCONNECTED;
        if (cep->disconnected_cb != NULL) {
            cep->disconnected_cb(&cep->super.super, cep->user_data);
        } else {
            ucs_warn("RDMA_CM_EVENT_DISCONNECTED event is missed");
        }
        return rdma_ack_cm_event(event);
    case RDMA_CM_EVENT_TIMEWAIT_EXIT:
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
