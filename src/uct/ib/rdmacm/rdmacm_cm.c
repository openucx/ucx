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


static void
uct_rdmacm_cm_process_event(uct_rdmacm_cm_t *cm, struct rdma_cm_event *event)
{
    struct sockaddr            *remote_addr = rdma_get_peer_addr(event->id);
    uint8_t                    ack_event = 1;
    struct rdma_conn_param     conn_param;
    uct_rdmacm_priv_data_hdr_t hdr;
    uct_rdmacm_cm_ep_t         *cep;
    char                       dev_name[UCT_DEVICE_NAME_MAX];
    char                       ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t               status;
    ssize_t                    priv_data_ret;

    ucs_trace("rdmacm event (fd=%d cm_id %p): %s. Peer: %s.",
              cm->ev_ch->fd, event->id, rdma_event_str(event->event),
              ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    /* The following applies for rdma_cm_id of type RDMA_PS_TCP only */
    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        /* Client side event */
        cep = (uct_rdmacm_cm_ep_t *)event->id->context;
        ucs_assert(event->id == cep->id);

        if (rdma_resolve_route(event->id, 1000 /* TODO */)){
            ucs_error("rdma_resolve_route(to addr=%s) failed: %m",
                      ucs_sockaddr_str(remote_addr, ip_port_str,
                                       UCS_SOCKADDR_STRING_LEN));
        }
        break;
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        /* Client side event */
        cep = (uct_rdmacm_cm_ep_t *)event->id->context;
        ucs_assert(event->id == cep->id);

        uct_rdmacm_cm_id_to_dev_name(cep->id, dev_name);

        memset(&conn_param, 0, sizeof(conn_param));
        conn_param.private_data = ucs_alloca(uct_rdmacm_cm_get_max_conn_priv() +
                                             sizeof(uct_rdmacm_priv_data_hdr_t));
        /* Pack data to send inside the connection request to the server */
        priv_data_ret = cep->wireup.priv_pack_cb(cep->user_data, dev_name,
                                                 (void*)(conn_param.private_data +
                                                  sizeof(uct_rdmacm_priv_data_hdr_t)));

        if ((priv_data_ret < 0) || (priv_data_ret > uct_rdmacm_cm_get_max_conn_priv())) {
            ucs_error("failed to pack data on the client (ep=%p). packed data size: %zu.",
                      cep, priv_data_ret);
            break;
        }

        hdr.length = (uint8_t)priv_data_ret;
        hdr.status = UCS_OK;

        memcpy((void*)conn_param.private_data, &hdr, sizeof(uct_rdmacm_priv_data_hdr_t));
        status = uct_rdamcm_cm_ep_set_remaining_conn_param(&conn_param, &hdr, cep);
        if (status != UCS_OK) {
            break;
        }

        if (rdma_connect(cep->id, &conn_param)) {
            ucs_error("rdma_connect(to addr=%s) failed: %m",
                      ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

        }
        break;
    case RDMA_CM_EVENT_CONNECT_REQUEST:
        /* Server side event */
        break;
    case RDMA_CM_EVENT_CONNECT_RESPONSE:
        /* Client side event */
        ucs_fatal("UCS_ERR_NOT_IMPLEMENTED");
        break;
    case RDMA_CM_EVENT_REJECTED:
        /* Client side event */
        ucs_fatal("UCS_ERR_NOT_IMPLEMENTED");
        break;
    case RDMA_CM_EVENT_ESTABLISHED:
        /* Server side event */
        ucs_fatal("UCS_ERR_NOT_IMPLEMENTED");
        break;
    case RDMA_CM_EVENT_DISCONNECTED:
        /* Client and Server side event */
        ucs_fatal("UCS_ERR_NOT_IMPLEMENTED");
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

    if (ack_event && rdma_ack_cm_event(event)) {
        ucs_warn("rdma_ack_cm_event() failed on event %s: %m",
                 rdma_event_str(event->event));
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
            if (errno != EAGAIN) {
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
    .listener_destroy = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_listener_t),
    .ep_create        = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_cm_ep_t)
};

UCS_CLASS_INIT_FUNC(uct_rdmacm_cm_t, uct_component_h component,
                    uct_worker_h worker)
{
    uct_priv_worker_t *worker_priv;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_t, &uct_rdmacm_cm_ops, component);

    self->worker = worker;
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
