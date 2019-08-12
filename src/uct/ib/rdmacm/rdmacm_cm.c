/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h" /* Defines HAVE_RDMACM_QP_LESS */
#endif

#include "rdmacm_cm_ep.h"
#include <uct/ib/base/ib_iface.h>
#include <ucs/async/async.h>

#include <poll.h>
#include <rdma/rdma_cma.h>


ucs_status_t uct_rdmacm_cm_destroy_id(struct rdma_cm_id *id)
{
    ucs_trace("destroying cm_id %p", id);

    if (rdma_destroy_id(id)) {
        ucs_warn("rdma_destroy_id() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_rdmacm_cm_ack_event(struct rdma_cm_event *event)
{
    ucs_trace("ack event %p, cm_id %p", event, event->id);

    if (rdma_ack_cm_event(event)) {
        ucs_warn("rdma_ack_cm_event failed on event %s: %m",
                 rdma_event_str(event->event));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_rdmacm_cm_reject(struct rdma_cm_id *id, ucs_status_t hdr_status)
{
    uct_rdmacm_priv_data_hdr_t hdr;

    ucs_trace("reject on cm_id %p", id);

    hdr.status = hdr_status;
    hdr.length = 0;
    if (rdma_reject(id, &hdr, sizeof(hdr))) {
        ucs_error("rdma_reject (id=%p) failed with error: %m", id);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

size_t uct_rdmacm_cm_get_max_conn_priv()
{
    return UCT_RDMACM_TCP_PRIV_DATA_LEN - sizeof(uct_rdmacm_priv_data_hdr_t);
}

static ucs_status_t uct_rdmacm_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    if (cm_attr->field_mask & UCT_CM_ATTR_FIELD_MAX_CONN_PRIV) {
        cm_attr->max_conn_priv = uct_rdmacm_cm_get_max_conn_priv();
    }
    return UCS_OK;
}

static void uct_rdmacm_cm_handle_event_addr_resolved(struct rdma_cm_event *event)
{
    struct sockaddr    *remote_addr = rdma_get_peer_addr(event->id);
    uct_rdmacm_cm_ep_t *cep         = (uct_rdmacm_cm_ep_t *)event->id->context;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    uct_cm_remote_data_t remote_data;

    ucs_assert(event->id == cep->id);

    ucs_trace("rdma_resolve_route on cm_id %p", event->id);

    if (rdma_resolve_route(event->id, 1000 /* TODO */)) {
        ucs_error("rdma_resolve_route(to addr=%s) failed: %m",
                  ucs_sockaddr_str(remote_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data, UCS_ERR_IO_ERROR);
    }
}

static void uct_rdmacm_cm_handle_event_route_resolved(struct rdma_cm_event *event)
{
    struct sockaddr        *remote_addr = rdma_get_peer_addr(event->id);
    uct_rdmacm_cm_ep_t     *cep         = (uct_rdmacm_cm_ep_t *)event->id->context;
    uct_cm_remote_data_t   remote_data;
    ucs_status_t           status;
    struct rdma_conn_param conn_param;
    char                   ip_port_str[UCS_SOCKADDR_STRING_LEN];

    ucs_assert(event->id == cep->id);

    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.private_data = ucs_alloca(uct_rdmacm_cm_get_max_conn_priv() +
                                         sizeof(uct_rdmacm_priv_data_hdr_t));

    status = uct_rdmacm_cm_ep_conn_param_init(cep, &conn_param);
    if (status != UCS_OK) {
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data, status);
        return;
    }

    ucs_trace("rdma_connect on ep %p, cm_id %p", cep, cep->id);

    if (rdma_connect(cep->id, &conn_param)) {
        ucs_error("rdma_connect(to addr=%s) failed: %m",
                  ucs_sockaddr_str(remote_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data, UCS_ERR_IO_ERROR);
    }
}

static ucs_status_t uct_rdmacm_cm_id_to_dev_addr(struct rdma_cm_id *cm_id,
                                                 uct_device_addr_t **dev_addr_p,
                                                 size_t *dev_addr_len_p)
{
    struct ibv_port_attr port_attr;
    uct_ib_address_t *dev_addr;
    struct ibv_qp_attr qp_attr;
    size_t addr_length;
    int qp_attr_mask;
    char dev_name[UCT_DEVICE_NAME_MAX];

    /* get the qp attributes in order to modify the qp state.
     * the ah_attr fields from them are required to extract the device address
     * of the remote peer.
     */
    qp_attr.qp_state = IBV_QPS_RTR;
    if (rdma_init_qp_attr(cm_id, &qp_attr, &qp_attr_mask)) {
        ucs_error("rdma_init_qp_attr (id=%p, qp_state=%d) failed: %m",
                  cm_id, qp_attr.qp_state);
        return UCS_ERR_IO_ERROR;
     }

    if (ibv_query_port(cm_id->verbs, cm_id->port_num, &port_attr)) {
        uct_rdmacm_cm_id_to_dev_name(cm_id, dev_name);
        ucs_error("ibv_query_port (%s) failed: %m", dev_name);
        return UCS_ERR_IO_ERROR;
    }

    addr_length = uct_ib_address_size(&qp_attr.ah_attr.grh.dgid,
                                      qp_attr.ah_attr.is_global,
                                      IBV_PORT_IS_LINK_LAYER_ETHERNET(&port_attr));

    dev_addr = ucs_malloc(addr_length, "IB device address");
    if (dev_addr == NULL) {
        ucs_error("failed to allocate IB device address");
        return UCS_ERR_NO_MEMORY;
    }

    uct_ib_address_pack(&qp_attr.ah_attr.grh.dgid, qp_attr.ah_attr.dlid,
                        IBV_PORT_IS_LINK_LAYER_ETHERNET(&port_attr),
                        qp_attr.ah_attr.is_global,
                        dev_addr);

    *dev_addr_p     = (uct_device_addr_t *)dev_addr;
    *dev_addr_len_p = addr_length;
    return UCS_OK;
}

static void uct_rdmacm_cm_handle_event_connect_request(struct rdma_cm_event *event)
{
    uct_rdmacm_priv_data_hdr_t *hdr      = (uct_rdmacm_priv_data_hdr_t *)
                                           event->param.conn.private_data;
    uct_rdmacm_listener_t      *listener = event->listen_id->context;
    char                       dev_name[UCT_DEVICE_NAME_MAX];
    uct_device_addr_t          *dev_addr;
    size_t                     addr_length;
    uct_cm_remote_data_t       remote_data;
    ucs_status_t               status;

    ucs_assert(hdr->status == UCS_OK);

    uct_rdmacm_cm_id_to_dev_name(event->id, dev_name);

    status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr, &addr_length);
    if (status != UCS_OK) {
        uct_rdmacm_cm_reject(event->id, status);
        uct_rdmacm_cm_destroy_id(event->id);
        return;
    }

    remote_data.field_mask            = UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                                        UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
    remote_data.dev_addr              = dev_addr;
    remote_data.dev_addr_length       = addr_length;
    remote_data.conn_priv_data        = hdr + 1;
    remote_data.conn_priv_data_length = hdr->length;

    listener->conn_request_cb(&listener->super, listener->user_data,
                              dev_name, event, &remote_data);
    ucs_free(dev_addr);
}

static void uct_rdmacm_cm_handle_event_connect_response(struct rdma_cm_event *event)
{
    struct sockaddr            *remote_addr = rdma_get_peer_addr(event->id);
    uct_rdmacm_priv_data_hdr_t *hdr         = (uct_rdmacm_priv_data_hdr_t *)
                                              event->param.conn.private_data;
    uct_rdmacm_cm_ep_t         *cep         = event->id->context;
    char                       ip_port_str[UCS_SOCKADDR_STRING_LEN];
    uct_device_addr_t          *dev_addr;
    size_t                     addr_length;
    uct_cm_remote_data_t       remote_data;
    ucs_status_t               status;

    ucs_assert(event->id == cep->id);

    remote_data.field_mask            = UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
    remote_data.conn_priv_data        = hdr + 1;
    remote_data.conn_priv_data_length = hdr->length;

    status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr, &addr_length);
    if (status != UCS_OK) {
        ucs_error("client (ep=%p id=%p) failed to process a connect response "
                  "from server %s.", cep, event->id,
                  ucs_sockaddr_str(remote_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data, status);
        return;
    }

    remote_data.field_mask       |= UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR |
                                    UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH;
    remote_data.dev_addr          = dev_addr;
    remote_data.dev_addr_length   = addr_length;

    uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data, hdr->status);

    ucs_free(dev_addr);

    if (rdma_establish(event->id)) {
        ucs_error("rdma_establish on ep %p (to server addr=%s) failed: %m",
                  cep, ucs_sockaddr_str(remote_addr, ip_port_str,
                                        UCS_SOCKADDR_STRING_LEN));
        /* in case of an error here, call disconnect because the client already
         * called the connect_cb with status UCS_OK so the client is considered
         * as connected to the server at this point. */
        cep->disconnect_cb(&cep->super.super, cep->user_data);
    }
}

static void uct_rdmacm_cm_handle_event_rejected(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t         *cep = event->id->context;
    uct_rdmacm_priv_data_hdr_t *hdr;
    uct_cm_remote_data_t       remote_data;
    ucs_status_t               status;

    ucs_assert(event->id == cep->id);

    /* Network reject or called by rdma_reject on the server */
    remote_data.field_mask = 0;
    if (event->param.conn.private_data != NULL) {
        /* The server always passes private data when rejecting a connection request */
        hdr    = (uct_rdmacm_priv_data_hdr_t *)event->param.conn.private_data;
        status = hdr->status;
        ucs_assert(hdr->length == 0);
    } else {
        /* Network reject */
        status = UCS_ERR_REJECTED;
    }

    ucs_debug("rdmacm event rejected on ep %p (cm_id %p) with status %s.",
              cep, event->id, ucs_status_string(status));

    uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data, status);
}

static void uct_rdmacm_cm_handle_event_established(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t *cep = event->id->context;

    ucs_assert(event->id == cep->id);
    cep->wireup.server.connect_cb(&cep->super.super, cep->user_data, UCS_OK);
}

static void uct_rdmacm_cm_handle_event_disconnected(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t *cep = event->id->context;

    cep->disconnect_cb(&cep->super.super, cep->user_data);
}

static void
uct_rdmacm_cm_process_event(uct_rdmacm_cm_t *cm, struct rdma_cm_event *event)
{
    struct sockaddr *remote_addr = rdma_get_peer_addr(event->id);
    uint8_t         ack_event    = 1;
    char            ip_port_str[UCS_SOCKADDR_STRING_LEN];

    ucs_trace("rdmacm event (fd=%d cm_id %p): %s. Peer: %s.",
              cm->ev_ch->fd, event->id, rdma_event_str(event->event),
              ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    /* The following applies for rdma_cm_id of type RDMA_PS_TCP only */
    ucs_assert(event->id->ps == RDMA_PS_TCP);

    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        /* Client side event */
        uct_rdmacm_cm_handle_event_addr_resolved(event);
        break;
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        /* Client side event */
        uct_rdmacm_cm_handle_event_route_resolved(event);
        break;
    case RDMA_CM_EVENT_CONNECT_REQUEST:
        /* Server side event */
        uct_rdmacm_cm_handle_event_connect_request(event);
        /* The server will ack the event after accepting/rejecting the request
         * (in ep_create). */
        ack_event = 0;
        break;
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
        /* Client side event */
        uct_rdmacm_cm_handle_event_connect_response(event);
        break;
    case RDMA_CM_EVENT_REJECTED:
        /* Client side event */
        uct_rdmacm_cm_handle_event_rejected(event);
        break;
    case RDMA_CM_EVENT_ESTABLISHED:
        /* Server side event */
        uct_rdmacm_cm_handle_event_established(event);
        break;
    case RDMA_CM_EVENT_DISCONNECTED:
        /* Client and Server side event */
        uct_rdmacm_cm_handle_event_disconnected(event);
        break;
    case RDMA_CM_EVENT_TIMEWAIT_EXIT:
        /* This event is generated when the QP associated with the connection
         * has exited its timewait state and is now ready to be re-used.
         * After a QP has been disconnected, it is maintained in a timewait
         * state to allow any in flight packets to exit the network.
         * After the timewait state has completed, the rdma_cm will report this event.*/
        break;
        /* client error events */
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
        /* client and server error events */
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_DEVICE_REMOVAL:
    case RDMA_CM_EVENT_ADDR_CHANGE:
        ucs_error("received an error event %s. status = %d. Peer: %s.",
                  rdma_event_str(event->event), event->status,
                  ucs_sockaddr_str(remote_addr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        break;
    default:
        ucs_warn("unexpected RDMACM event: %s", rdma_event_str(event->event));
        break;
    }

    if (ack_event) {
        uct_rdmacm_cm_ack_event(event);
    }
}

static void uct_rdmacm_cm_event_handler(int fd, void *arg)
{
    uct_rdmacm_cm_t      *cm = (uct_rdmacm_cm_t *)arg;
    struct rdma_cm_event *event;
    int                  ret;

    for (;;) {
        /* Fetch an event */
        ret = rdma_get_cm_event(cm->ev_ch, &event);
        if (ret) {
            /* EAGAIN (in a non-blocking rdma_get_cm_event) means that
             * there are no more events */
            if ((errno != EAGAIN) && (errno != EINTR)) {
                ucs_warn("rdma_get_cm_event() failed: %m");
            }
            return;
        }
        uct_rdmacm_cm_process_event(cm, event);
    }
}

static uct_cm_ops_t uct_rdmacm_cm_ops = {
    .close            = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cm_t),
    .cm_query         = uct_rdmacm_cm_query,
    .listener_create  = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_listener_t),
    .listener_reject  = uct_rdmacm_listener_reject,
    .listener_query   = uct_rdmacm_listener_query,
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t),
    .ep_create        = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_cm_ep_t)
};

static uct_iface_ops_t uct_rdmacm_cm_iface_ops = {
    .ep_pending_purge         = ucs_empty_function,
    .ep_disconnect            = uct_rdmacm_cm_ep_disconnect,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cm_ep_t),
    .ep_put_short             = (void *)ucs_empty_function_return_unsupported,
    .ep_put_bcopy             = (void *)ucs_empty_function_return_zero_int64,
    .ep_get_bcopy             = (void *)ucs_empty_function_return_unsupported,
    .ep_am_short              = (void *)ucs_empty_function_return_unsupported,
    .ep_am_bcopy              = (void *)ucs_empty_function_return_zero_int64,
    .ep_atomic_cswap64        = (void *)ucs_empty_function_return_unsupported,
    .ep_atomic64_post         = (void *)ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch        = (void *)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32        = (void *)ucs_empty_function_return_unsupported,
    .ep_atomic32_post         = (void *)ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch        = (void *)ucs_empty_function_return_unsupported,
    .ep_pending_add           = (void *)ucs_empty_function_return_unsupported,
    .ep_flush                 = (void *)ucs_empty_function_return_unsupported,
    .ep_fence                 = (void *)ucs_empty_function_return_unsupported,
    .ep_check                 = (void *)ucs_empty_function_return_unsupported,
    .ep_create                = (void *)ucs_empty_function_return_unsupported,
    .iface_flush              = (void *)ucs_empty_function_return_unsupported,
    .iface_fence              = (void *)ucs_empty_function_return_unsupported,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (void *)ucs_empty_function_return_zero,
    .iface_event_fd_get       = (void *)ucs_empty_function_return_unsupported,
    .iface_event_arm          = (void *)ucs_empty_function_return_unsupported,
    .iface_close              = ucs_empty_function,
    .iface_query              = (void *)ucs_empty_function_return_unsupported,
    .iface_get_device_address = (void *)ucs_empty_function_return_unsupported,
    .iface_get_address        = (void *)ucs_empty_function_return_unsupported,
    .iface_is_reachable       = (void *)ucs_empty_function_return_zero
};

UCS_CLASS_INIT_FUNC(uct_rdmacm_cm_t, uct_component_h component,
                    uct_worker_h worker)
{
    uct_priv_worker_t *worker_priv;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_t, &uct_rdmacm_cm_ops,
                              &uct_rdmacm_cm_iface_ops, worker, component);

    self->ev_ch  = rdma_create_event_channel();
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
                                         self->ev_ch->fd, UCS_EVENT_SET_EVREAD,
                                         uct_rdmacm_cm_event_handler, self,
                                         worker_priv->async);
    if (status != UCS_OK) {
        goto err_destroy_ev_ch;
    }

    ucs_debug("created rdmacm_cm %p with event_channel %p (fd=%d)",
              self, self->ev_ch, self->ev_ch->fd);

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

    ucs_trace("destroying event_channel %p", self->ev_ch);
    rdma_destroy_event_channel(self->ev_ch);
}

UCS_CLASS_DEFINE(uct_rdmacm_cm_t, uct_cm_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cm_t, uct_cm_t, uct_component_h, uct_worker_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cm_t, uct_cm_t);
