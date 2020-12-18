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

ucs_status_t uct_rdmacm_cm_reject(struct rdma_cm_id *id)
{
    uct_rdmacm_priv_data_hdr_t hdr;
    char remote_ip_port_str[UCS_SOCKADDR_STRING_LEN];
    char local_ip_port_str[UCS_SOCKADDR_STRING_LEN];

    hdr.length = 0;
    hdr.status = (uint8_t)UCS_ERR_REJECTED;

    ucs_trace("reject on cm_id %p", id);

    if (rdma_reject(id, &hdr, sizeof(hdr))) {
        ucs_error("rdma_reject (id=%p local addr=%s remote addr=%s) failed "
                  "with error: %m", id,
                  ucs_sockaddr_str(rdma_get_local_addr(id), local_ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN),
                  ucs_sockaddr_str(rdma_get_peer_addr(id), remote_ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_rdmacm_cm_get_cq(uct_rdmacm_cm_t *cm, struct ibv_context *verbs,
                                  uint32_t pd_key, struct ibv_cq **cq_p)
{
    struct ibv_cq *cq;
    khiter_t iter;
    int ret;

    iter = kh_put(uct_rdmacm_cm_cqs, &cm->cqs, pd_key, &ret);
    if (ret == -1) {
        ucs_error("cm %p: cannot allocate hash entry for CQ", cm);
        return UCS_ERR_NO_MEMORY;
    }

    if (ret == 0) {
        /* already exists so use it */
        cq = kh_value(&cm->cqs, iter);
    } else {
        /* Create a dummy completion queue */
        cq = ibv_create_cq(verbs, 1, NULL, NULL, 0);
        if (cq == NULL) {
            kh_del(uct_rdmacm_cm_cqs, &cm->cqs, iter);
            ucs_error("ibv_create_cq() failed: %m");
            return UCS_ERR_IO_ERROR;
        }

        kh_value(&cm->cqs, iter) = cq;
    }

    *cq_p = cq;
    return UCS_OK;
}

void uct_rdmacm_cm_cqs_cleanup(uct_rdmacm_cm_t *cm)
{
    struct ibv_cq *cq;
    int ret;

    kh_foreach_value(&cm->cqs, cq, {
        ret = ibv_destroy_cq(cq);
        if (ret != 0) {
            ucs_warn("ibv_destroy_cq() returned %d: %m", ret);
        }
    });

    kh_destroy_inplace(uct_rdmacm_cm_cqs, &cm->cqs);
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
    uct_rdmacm_cm_ep_t *cep = (uct_rdmacm_cm_ep_t*)event->id->context;
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    uct_cm_remote_data_t remote_data;

    ucs_assert(event->id == cep->id);

    ucs_trace("%s rdma_resolve_route on cm_id %p",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              event->id);

    if (rdma_resolve_route(event->id, 1000 /* TODO */)) {
        ucs_error("%s: rdma_resolve_route failed: %m",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN));
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_set_failed(cep, &remote_data, UCS_ERR_IO_ERROR);
    }
}

static void uct_rdmacm_cm_handle_event_route_resolved(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t     *cep = (uct_rdmacm_cm_ep_t*)event->id->context;
    uct_cm_remote_data_t   remote_data;
    ucs_status_t           status;
    struct rdma_conn_param conn_param;
    char                   ep_str[UCT_RDMACM_EP_STRING_LEN];

    ucs_assert(event->id == cep->id);

    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.private_data = ucs_alloca(uct_rdmacm_cm_get_max_conn_priv() +
                                         sizeof(uct_rdmacm_priv_data_hdr_t));

    status = uct_rdmacm_cm_ep_pack_cb(cep, &conn_param);
    if (status != UCS_OK) {
        cep->status = status;
        cep->flags |= UCT_RDMACM_CM_EP_FAILED;
        return;
    }

    status = uct_rdamcm_cm_ep_set_qp_num(&conn_param, cep);
    if (status != UCS_OK) {
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_set_failed(cep, &remote_data, status);
        return;
    }

    ucs_trace("%s rdma_connect, cm_id %p",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN), cep->id);

    if (rdma_connect(cep->id, &conn_param)) {
        ucs_error("%s rdma_connect failed: %m",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN));
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_set_failed(cep, &remote_data, UCS_ERR_IO_ERROR);
    }
}

static ucs_status_t uct_rdmacm_cm_id_to_dev_addr(struct rdma_cm_id *cm_id,
                                                 uct_device_addr_t **dev_addr_p,
                                                 size_t *dev_addr_len_p)
{
    uct_ib_address_pack_params_t params;
    struct ibv_port_attr port_attr;
    uct_ib_address_t *dev_addr;
    struct ibv_qp_attr qp_attr;
    size_t addr_length;
    int qp_attr_mask;
    char dev_name[UCT_DEVICE_NAME_MAX];
    char ah_attr_str[128];
    uct_ib_roce_version_info_t roce_info;
    int ret;

    params.flags = 0;

    /* get the qp attributes in order to modify the qp state.
     * the ah_attr fields from them are required to extract the device address
     * of the remote peer.
     */
    qp_attr.qp_state = IBV_QPS_RTR;
    ret              = rdma_init_qp_attr(cm_id, &qp_attr, &qp_attr_mask);
    if (ret) {
        ucs_error("rdma_init_qp_attr (id=%p, qp_state=%d) failed: %m",
                  cm_id, qp_attr.qp_state);
        return UCS_ERR_IO_ERROR;
    }

    ret = ibv_query_port(cm_id->pd->context, cm_id->port_num, &port_attr);
    if (ret) {
        uct_rdmacm_cm_id_to_dev_name(cm_id, dev_name);
        ucs_error("ibv_query_port (%s) failed: %m", dev_name);
        return UCS_ERR_IO_ERROR;
    }

    if (qp_attr.ah_attr.is_global) {
        params.flags    |= UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX;
        params.gid_index = qp_attr.ah_attr.grh.sgid_index;
        params.gid       = qp_attr.ah_attr.grh.dgid;
    }

    ucs_debug("cm_id %p: ah_attr %s", cm_id,
              uct_ib_ah_attr_str(ah_attr_str, sizeof(ah_attr_str),
                                 &qp_attr.ah_attr));
    ucs_assert_always(qp_attr.path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
    params.flags   |= UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU;
    params.path_mtu = qp_attr.path_mtu;

    if (IBV_PORT_IS_LINK_LAYER_ETHERNET(&port_attr)) {
        /* Ethernet address */
        ucs_assert(qp_attr.ah_attr.is_global);

        /* pack the remote RoCE version as ANY assuming that rdmacm guarantees
         * that the remote peer is reachable to the local one */
        roce_info.ver         = UCT_IB_DEVICE_ROCE_ANY;
        roce_info.addr_family = 0;
        params.roce_info      = roce_info;
        params.flags         |= UCT_IB_ADDRESS_PACK_FLAG_ETH;
    } else if (qp_attr.ah_attr.is_global) {
        params.flags         |= UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX |
                                UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID;
    } else {
        /* For local IB address, assume the remote subnet prefix is the same
         * and pack it to make reachability check pass */
        ret = ibv_query_gid(cm_id->verbs, cm_id->port_num,
                            UCT_IB_MD_DEFAULT_GID_INDEX, &params.gid);
        if (ret) {
            ucs_error("ibv_query_gid(dev=%s port=%d index=%d) failed: %m",
                      ibv_get_device_name(cm_id->verbs->device),
                      cm_id->port_num, UCT_IB_MD_DEFAULT_GID_INDEX);
            return UCS_ERR_IO_ERROR;
        }

        params.gid_index = UCT_IB_MD_DEFAULT_GID_INDEX;
        params.flags    |= UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX |
                           UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX;
    }

    params.lid  = qp_attr.ah_attr.dlid;
    addr_length = uct_ib_address_size(&params);
    dev_addr    = ucs_malloc(addr_length, "IB device address");
    if (dev_addr == NULL) {
        ucs_error("failed to allocate IB device address");
        return UCS_ERR_NO_MEMORY;
    }

    uct_ib_address_pack(&params, dev_addr);

    *dev_addr_p     = (uct_device_addr_t*)dev_addr;
    *dev_addr_len_p = addr_length;
    return UCS_OK;
}

static void uct_rdmacm_cm_handle_event_connect_request(struct rdma_cm_event *event)
{
    uct_rdmacm_priv_data_hdr_t          *hdr      = (uct_rdmacm_priv_data_hdr_t*)
                                                    event->param.conn.private_data;
    uct_rdmacm_listener_t               *listener = event->listen_id->context;
    char                                dev_name[UCT_DEVICE_NAME_MAX];
    uct_device_addr_t                   *dev_addr;
    size_t                              addr_length;
    uct_cm_remote_data_t                remote_data;
    ucs_status_t                        status;
    uct_cm_listener_conn_request_args_t conn_req_args;
    ucs_sock_addr_t                     client_saddr;
    size_t                              size;

    ucs_assert(hdr->status == UCS_OK);

    uct_rdmacm_cm_id_to_dev_name(event->id, dev_name);

    status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr, &addr_length);
    if (status != UCS_OK) {
        goto err;
    }

    remote_data.field_mask            = UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                                        UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
    remote_data.dev_addr              = dev_addr;
    remote_data.dev_addr_length       = addr_length;
    remote_data.conn_priv_data        = hdr + 1;
    remote_data.conn_priv_data_length = hdr->length;

    client_saddr.addr = rdma_get_peer_addr(event->id);

    status = ucs_sockaddr_sizeof(client_saddr.addr, &size);
    if (status != UCS_OK) {
        goto err_free_dev_addr;
    }

    client_saddr.addrlen = size;

    conn_req_args.field_mask     = UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_DEV_NAME     |
                                   UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST |
                                   UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_REMOTE_DATA  |
                                   UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CLIENT_ADDR;
    conn_req_args.conn_request   = event;
    conn_req_args.remote_data    = &remote_data;
    conn_req_args.client_address = client_saddr;
    ucs_strncpy_safe(conn_req_args.dev_name, dev_name, UCT_DEVICE_NAME_MAX);

    listener->conn_request_cb(&listener->super, listener->user_data,
                              &conn_req_args);
    ucs_free(dev_addr);

    return;

err_free_dev_addr:
    ucs_free(dev_addr);
err:
    uct_rdmacm_cm_reject(event->id);
    uct_rdmacm_cm_destroy_id(event->id);
    uct_rdmacm_cm_ack_event(event);
}

static void uct_rdmacm_cm_handle_event_connect_response(struct rdma_cm_event *event)
{
    uct_rdmacm_priv_data_hdr_t *hdr = (uct_rdmacm_priv_data_hdr_t*)
                                       event->param.conn.private_data;
    uct_rdmacm_cm_ep_t         *cep = event->id->context;
    char                       ep_str[UCT_RDMACM_EP_STRING_LEN];
    uct_device_addr_t          *dev_addr;
    size_t                     addr_length;
    uct_cm_remote_data_t       remote_data;
    ucs_status_t               status;

    ucs_assert(event->id == cep->id);
    ucs_trace("%s client received connect_response",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN));

    /* Do not notify user on disconnected EP, RDMACM out of order case */
    if (cep->flags & UCT_RDMACM_CM_EP_GOT_DISCONNECT) {
        return;
    }

    remote_data.field_mask            = UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
    remote_data.conn_priv_data        = hdr + 1;
    remote_data.conn_priv_data_length = hdr->length;

    status = uct_rdmacm_cm_id_to_dev_addr(event->id, &dev_addr, &addr_length);
    if (status != UCS_OK) {
        ucs_error("%s client (ep=%p id=%p) failed to process a connect response ",
                  uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
                  cep, event->id);
        uct_rdmacm_cm_ep_set_failed(cep, &remote_data, status);
        return;
    }

    remote_data.field_mask       |= UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR |
                                    UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH;
    remote_data.dev_addr          = dev_addr;
    remote_data.dev_addr_length   = addr_length;

    uct_rdmacm_cm_ep_client_connect_cb(cep, &remote_data,
                                       (ucs_status_t)hdr->status);
    ucs_free(dev_addr);
}

static void uct_rdmacm_cm_handle_event_established(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t *cep = event->id->context;

    ucs_assert(event->id == cep->id);
    /* do not call connect callback again, RDMACM out of order case */
    if (cep->flags & UCT_RDMACM_CM_EP_GOT_DISCONNECT) {
        return;
    }

    uct_rdmacm_cm_ep_server_conn_notify_cb(cep, UCS_OK);
}

static void uct_rdmacm_cm_handle_event_disconnected(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t   *cep = event->id->context;
    char                 ep_str[UCT_RDMACM_EP_STRING_LEN];
    uct_cm_remote_data_t remote_data;

    ucs_debug("%s got disconnect event, status %d",
              uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
              event->status);

    cep->flags |= UCT_RDMACM_CM_EP_GOT_DISCONNECT;
    /* calling error_cb instead of disconnect CB directly handles out-of-order
     * disconnect event prior connect_response/connect_established event */
    remote_data.field_mask = 0;
    uct_rdmacm_cm_ep_error_cb(cep, &remote_data, UCS_ERR_CONNECTION_RESET);
}

static void uct_rdmacm_cm_handle_error_event(struct rdma_cm_event *event)
{
    uct_rdmacm_cm_ep_t *cep      = event->id->context;
    char ep_str[UCT_RDMACM_EP_STRING_LEN];
    uct_cm_remote_data_t remote_data;
    const uct_rdmacm_priv_data_hdr_t *hdr;
    ucs_log_level_t log_level;
    ucs_status_t status;

    switch (event->event) {
    case RDMA_CM_EVENT_REJECTED:
        if (cep->flags & UCT_RDMACM_CM_EP_ON_SERVER) {
            /* response was rejected by the client in the middle of
             * connection establishment, so report connection reset */
            status = UCS_ERR_CONNECTION_RESET;
        } else {
            ucs_assert(cep->flags & UCT_RDMACM_CM_EP_ON_CLIENT);
            hdr = (const uct_rdmacm_priv_data_hdr_t*)event->param.conn.private_data;

            if ((hdr != NULL) && (event->param.conn.private_data_len > 0) &&
                ((ucs_status_t)hdr->status == UCS_ERR_REJECTED)) {
                ucs_assert(hdr->length == 0);
                /* the actual amount of data transferred to the remote side is
                 * transport dependent and may be larger than that requested.*/
                ucs_assert(event->param.conn.private_data_len >= sizeof(*hdr));
                status = UCS_ERR_REJECTED;
            } else {
                status = UCS_ERR_UNREACHABLE;
            }
        }

        log_level = UCS_LOG_LEVEL_DEBUG;
        break;
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
    case RDMA_CM_EVENT_CONNECT_ERROR:
        status    = UCS_ERR_UNREACHABLE;
        log_level = uct_rdmacm_cm_ep_get_cm(cep)->super.config.failure_level;
        break;
    default:
        status    = UCS_ERR_IO_ERROR;
        log_level = UCS_LOG_LEVEL_ERROR;
    }

    ucs_log(log_level, "%s got error event %s, event status %d ",
            uct_rdmacm_cm_ep_str(cep, ep_str, UCT_RDMACM_EP_STRING_LEN),
            rdma_event_str(event->event), event->status);

    if (uct_rdmacm_ep_is_connected(cep) &&
        !(cep->flags & UCT_RDMACM_CM_EP_FAILED)) {
        /* first failure on connected EP has to be reported as disconnect event
         * to allow user to call disconnect due to UCT API limitation -
         * disconnect callback does not have status arg */
        uct_rdmacm_cm_handle_event_disconnected(event);
    } else {
        remote_data.field_mask = 0;
        uct_rdmacm_cm_ep_set_failed(cep, &remote_data, status);
    }
}

static void
uct_rdmacm_cm_process_event(uct_rdmacm_cm_t *cm, struct rdma_cm_event *event)
{
    struct sockaddr UCS_V_UNUSED *remote_addr = rdma_get_peer_addr(event->id);
    uint8_t         ack_event                 = 1;
    char            ip_port_str[UCS_SOCKADDR_STRING_LEN];

    ucs_trace("rdmacm event (fd=%d cm_id %p cm %p event_channel %p status %s): %s. Peer: %s.",
              cm->ev_ch->fd, event->id, cm, cm->ev_ch, strerror(event->status),
              rdma_event_str(event->event),
              ucs_sockaddr_str(remote_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    /* The following applies for rdma_cm_id of type RDMA_PS_TCP only */
    ucs_assert(event->id->ps == RDMA_PS_TCP);

    /* Using https://linux.die.net/man/3/rdma_get_cm_event to distinguish
     * between client and server events */
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
    case RDMA_CM_EVENT_DEVICE_REMOVAL:
    case RDMA_CM_EVENT_ADDR_CHANGE:
        /* client and server error events */
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_CONNECT_ERROR:
        uct_rdmacm_cm_handle_error_event(event);
        break;
    default:
        ucs_warn("unexpected RDMACM event: %s", rdma_event_str(event->event));
        break;
    }

    if (ack_event) {
        uct_rdmacm_cm_ack_event(event);
    }
}

static void uct_rdmacm_cm_event_handler(int fd, ucs_event_set_types_t events,
                                        void *arg)
{
    uct_rdmacm_cm_t      *cm = (uct_rdmacm_cm_t*)arg;
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

        UCS_ASYNC_BLOCK(uct_rdmacm_cm_get_async(cm));
        uct_rdmacm_cm_process_event(cm, event);
        UCS_ASYNC_UNBLOCK(uct_rdmacm_cm_get_async(cm));
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
    .cm_ep_conn_notify        = uct_rdmacm_cm_ep_conn_notify,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rdmacm_cm_ep_t),
    .ep_put_short             = (uct_ep_put_short_func_t)ucs_empty_function_return_unsupported,
    .ep_put_bcopy             = (uct_ep_put_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_get_bcopy             = (uct_ep_get_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short              = (uct_ep_am_short_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short_iov          = (uct_ep_am_short_iov_func_t)ucs_empty_function_return_unsupported,
    .ep_am_bcopy              = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap64        = (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic64_post         = (uct_ep_atomic64_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch        = (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32        = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_post         = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch        = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_pending_add           = (uct_ep_pending_add_func_t)ucs_empty_function_return_unsupported,
    .ep_flush                 = (uct_ep_flush_func_t)ucs_empty_function_return_success,
    .ep_fence                 = (uct_ep_fence_func_t)ucs_empty_function_return_unsupported,
    .ep_check                 = (uct_ep_check_func_t)ucs_empty_function_return_unsupported,
    .ep_create                = (uct_ep_create_func_t)ucs_empty_function_return_unsupported,
    .iface_flush              = (uct_iface_flush_func_t)ucs_empty_function_return_unsupported,
    .iface_fence              = (uct_iface_fence_func_t)ucs_empty_function_return_unsupported,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (uct_iface_progress_func_t)ucs_empty_function_return_zero,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)ucs_empty_function_return_unsupported,
    .iface_close              = ucs_empty_function,
    .iface_query              = (uct_iface_query_func_t)ucs_empty_function_return_unsupported,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_unsupported,
    .iface_get_address        = (uct_iface_get_address_func_t)ucs_empty_function_return_unsupported,
    .iface_is_reachable       = (uct_iface_is_reachable_func_t)ucs_empty_function_return_zero
};

static ucs_status_t
uct_rdmacm_cm_ipstr_to_sockaddr(const char *ip_str, struct sockaddr **saddr_p,
                                const char *debug_name)
{
    struct sockaddr_storage *sa_storage;
    ucs_status_t status;

    /* NULL-pointer for empty parameter */
    if (ip_str[0] == '\0') {
        sa_storage = NULL;
        goto out;
    }

    sa_storage = ucs_calloc(1, sizeof(struct sockaddr_storage), debug_name);
    if (sa_storage == NULL) {
        status = UCS_ERR_NO_MEMORY;
        ucs_error("cannot allocate memory for rdmacm source address");
        goto err;
    }

    status = ucs_sock_ipstr_to_sockaddr(ip_str, sa_storage);
    if (status != UCS_OK) {
        goto err_free;
    }

out:
    *saddr_p = (struct sockaddr*)sa_storage;
    return UCS_OK;

err_free:
    ucs_free(sa_storage);
err:
    return status;
}

UCS_CLASS_INIT_FUNC(uct_rdmacm_cm_t, uct_component_h component,
                    uct_worker_h worker, const uct_cm_config_t *config)
{
    const uct_rdmacm_cm_config_t *rdmacm_config = ucs_derived_of(config,
                                                                 uct_rdmacm_cm_config_t);
    uct_priv_worker_t *worker_priv;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_t, &uct_rdmacm_cm_ops,
                              &uct_rdmacm_cm_iface_ops, worker, component,
                              config);

    kh_init_inplace(uct_rdmacm_cm_cqs, &self->cqs);

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

    status = uct_rdmacm_cm_ipstr_to_sockaddr(rdmacm_config->src_addr,
                                             &self->config.src_addr,
                                             "rdmacm_src_addr");
    if (status != UCS_OK) {
        goto ucs_async_remove_handler;
    }

    ucs_debug("created rdmacm_cm %p with event_channel %p (fd=%d)",
              self, self->ev_ch, self->ev_ch->fd);

    return UCS_OK;

ucs_async_remove_handler:
    ucs_async_remove_handler(self->ev_ch->fd, 1);
err_destroy_ev_ch:
    rdma_destroy_event_channel(self->ev_ch);
err:
    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_cm_t)
{
    ucs_status_t status;

    ucs_free(self->config.src_addr);

    status = ucs_async_remove_handler(self->ev_ch->fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove event handler for fd %d: %s",
                 self->ev_ch->fd, ucs_status_string(status));
    }

    ucs_trace("destroying event_channel %p on cm %p", self->ev_ch, self);
    rdma_destroy_event_channel(self->ev_ch);
    uct_rdmacm_cm_cqs_cleanup(self);
}

UCS_CLASS_DEFINE(uct_rdmacm_cm_t, uct_cm_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cm_t, uct_cm_t, uct_component_h,
                          uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cm_t, uct_cm_t);
