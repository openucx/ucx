/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>


void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep,
                                  uct_tcp_ep_conn_state_t new_conn_state)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    uct_tcp_ep_conn_state_t old_conn_state;

    old_conn_state = ep->conn_state;
    ep->conn_state = new_conn_state;

    switch(ep->conn_state) {
    case UCT_TCP_EP_CONN_STATE_CONNECTING:
    case UCT_TCP_EP_CONN_STATE_WAITING_ACK:
        if (old_conn_state == UCT_TCP_EP_CONN_STATE_CLOSED) {
            uct_tcp_iface_outstanding_inc(iface);
        } else {
            ucs_assert((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
                       (old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING));
        }
        break;
    case UCT_TCP_EP_CONN_STATE_CONNECTED:
        ucs_assert((old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING));
        if (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) {
            uct_tcp_iface_outstanding_dec(iface);
        }
        break;
    case UCT_TCP_EP_CONN_STATE_CLOSED:
        ucs_assert(old_conn_state != UCT_TCP_EP_CONN_STATE_CLOSED);
        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) {
            uct_tcp_iface_outstanding_dec(iface);
        }
        break;
    default:
        ucs_assert(ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING);
        /* Since ep::peer_addr is 0'ed and client's <address:port>
         * has already been logged, no need to log here */
        return;
    }

    ucs_debug("tcp_ep %p: %s -> %s for the [%s]<->[%s] connection",
              ep, uct_tcp_ep_cm_state[old_conn_state].name,
              uct_tcp_ep_cm_state[ep->conn_state].name,
              ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                               str_local_addr, UCS_SOCKADDR_STRING_LEN),
              ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                               str_remote_addr, UCS_SOCKADDR_STRING_LEN));
}

static void uct_tcp_cm_io_err_handler_cb(void *arg, int errno)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;

    /* check whether this is possible somaxconn exceeded reason or not */    
    if (((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
         (ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) &&
        ((errno == ECONNRESET) || (errno == ECONNREFUSED))) {
        ucs_error("try to increase \"net.core.somaxconn\" on the remote node");
    }
}

static void uct_tcp_cm_trace_conn_pkt(const uct_tcp_ep_t *ep, const char *msg,
                                      const struct sockaddr_in *peer_addr)
{
    char str_addr[UCS_SOCKADDR_STRING_LEN];

    ucs_debug("tcp_ep %p: %s %s", ep, msg,
              ucs_sockaddr_str((const struct sockaddr*)peer_addr,
                               str_addr, UCS_SOCKADDR_STRING_LEN));
}

static ucs_status_t
uct_tcp_cm_conn_pkt_check_event(const uct_tcp_ep_t *ep,
                                uct_tcp_cm_conn_event_t expected_event,
                                uct_tcp_cm_conn_event_t actual_event)
{
    char str_addr[UCS_SOCKADDR_STRING_LEN];

    if (expected_event != actual_event) {
        ucs_error("tcp_ep %p: received wrong CM event (expected: %u, "
                  "actual: %u) from the peer with iface listener "
                  "address: %s", ep, expected_event, actual_event,
                  ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                   str_addr, UCS_SOCKADDR_STRING_LEN));
        return UCS_ERR_INVALID_PARAM;
    }
    return UCS_OK;
}

static void uct_tcp_cm_handle_maxconn_exceed(ucs_status_t status)
{
    /* check whether this is possible somaxconn exceeded reason or not */
    if (status == UCS_ERR_IO_ERROR) {
        ucs_error("try to increase \"net.core.somaxconn\" on the remote node");
    }
}

static ucs_status_t uct_tcp_cm_send_conn_req(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface                  = ucs_derived_of(ep->super.super.iface,
                                                             uct_tcp_iface_t);
    uct_tcp_cm_conn_req_pkt_t conn_pkt      = {
        .event                              = UCT_TCP_CM_CONN_REQ,
        .iface_addr                         = iface->config.ifaddr
    };
    
    ucs_socket_io_err_handler_t err_handler = {
        .cb                                 = uct_tcp_cm_io_err_handler_cb,
        .arg                                = ep
    };
    ucs_status_t status;

    status = ucs_socket_send(ep->fd, &conn_pkt, sizeof(conn_pkt), &err_handler);
    if (status != UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, "unable to send connection request to",
                                  &ep->peer_addr);
        uct_tcp_cm_handle_maxconn_exceed(status);
        return status;
    }

    uct_tcp_cm_trace_conn_pkt(ep, "connection request sent to",
                              &ep->peer_addr);
    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_recv_conn_req(uct_tcp_ep_t *ep,
                                             struct sockaddr_in *peer_addr)
{
    uct_tcp_cm_conn_req_pkt_t conn_pkt;
    ucs_status_t status;

    status = ucs_socket_recv(ep->fd, &conn_pkt, sizeof(conn_pkt), NULL);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_tcp_cm_conn_pkt_check_event(ep, UCT_TCP_CM_CONN_REQ,
                                             conn_pkt.event);
    if (status != UCS_OK) {
        return status;
    }

    *peer_addr = conn_pkt.iface_addr;

    uct_tcp_cm_trace_conn_pkt(ep, "received the connection request from",
                              peer_addr);
    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_send_conn_ack(uct_tcp_ep_t *ep)
{
    uct_tcp_cm_conn_event_t event = UCT_TCP_CM_CONN_ACK;
    ucs_status_t status;

    status = ucs_socket_send(ep->fd, &event, sizeof(event), NULL);
    if (status != UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, "unable to send connection ack to",
                                  &ep->peer_addr);
        return status;
    }

    uct_tcp_cm_trace_conn_pkt(ep, "connection ack sent to",
                              &ep->peer_addr);
    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_recv_conn_ack(uct_tcp_ep_t *ep)
{
    ucs_socket_io_err_handler_t err_handler = {
        .cb                                 = uct_tcp_cm_io_err_handler_cb,
        .arg                                = ep
    };
    uct_tcp_cm_conn_event_t event;
    ucs_status_t status; 

    status = ucs_socket_recv(ep->fd, &event, sizeof(event), &err_handler);
    if (status != UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, "unable to receive connection ack from",
                                  &ep->peer_addr);
        uct_tcp_cm_handle_maxconn_exceed(status);
        return status;
    }

    status = uct_tcp_cm_conn_pkt_check_event(ep, UCT_TCP_CM_CONN_ACK, event);
    if (status != UCS_OK) {
        return status;
    }

    uct_tcp_cm_trace_conn_pkt(ep, "connection ack received from",
                              &ep->peer_addr);
    return UCS_OK;
}

unsigned uct_tcp_cm_conn_progress(uct_tcp_ep_t *ep)
{
    ucs_status_t status;

    status = ucs_socket_connect_nb_get_status(ep->fd);
    if (status != UCS_OK) {
        if (status == UCS_INPROGRESS) {
            return 0;
        }
        goto err;
    }

    status = uct_tcp_cm_send_conn_req(ep);
    if (status != UCS_OK) {
        return 0;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_WAITING_ACK);
    uct_tcp_ep_mod_events(ep, EPOLLIN, EPOLLOUT);

    ucs_assertv((ep->tx.length == 0) && (ep->tx.offset == 0) &&
                (ep->tx.buf == NULL), "ep=%p", ep);
    return 1;

err:
    uct_tcp_ep_set_failed(ep);
    return 0;
}

unsigned uct_tcp_cm_conn_ack_rx_progress(uct_tcp_ep_t *ep)
{
    ucs_status_t status;

    status = uct_tcp_cm_recv_conn_ack(ep);
    if (status != UCS_OK) {
        uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
        return 0;
    }

    ucs_assertv(ep->tx.buf == NULL, "ep=%p", ep);
    ucs_assert(!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX)));

    ep->ctx_caps |= UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX);
    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);
    uct_tcp_ep_mod_events(ep, EPOLLOUT, EPOLLIN);

    /* Progress possibly pending TX operations */
    return 1 + uct_tcp_ep_progress(ep, UCT_TCP_EP_CTX_TYPE_TX);
}

unsigned uct_tcp_cm_conn_req_rx_progress(uct_tcp_ep_t *ep)
{
    struct sockaddr_in peer_addr;
    ucs_status_t status;

    status = uct_tcp_cm_recv_conn_req(ep, &peer_addr);
    if (status != UCS_OK) {
        return 0;
    }

    ep->peer_addr = peer_addr;

    status = uct_tcp_cm_send_conn_ack(ep);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_assertv(ep->rx.buf == NULL, "ep=%p", ep);
    ucs_assert(!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_RX)));

    ep->ctx_caps |= UCS_BIT(UCT_TCP_EP_CTX_TYPE_RX);
    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);

    return 2;

 err:
    uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
    uct_tcp_ep_destroy(&ep->super.super);
    return 1;
}

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep)
{
    uct_tcp_ep_conn_state_t new_conn_state;
    uint32_t req_events;
    ucs_status_t status;

    status = ucs_socket_connect(ep->fd, (const struct sockaddr*)&ep->peer_addr);
    if (status == UCS_INPROGRESS) {
        new_conn_state  = UCT_TCP_EP_CONN_STATE_CONNECTING;
        req_events      = EPOLLOUT;
        status          = UCS_OK;
    } else if (status == UCS_OK) {
        status = uct_tcp_cm_send_conn_req(ep);
        if (status != UCS_OK) {
            return status;
        }

        new_conn_state  = UCT_TCP_EP_CONN_STATE_WAITING_ACK;
        req_events      = EPOLLIN;
    } else {
        new_conn_state  = UCT_TCP_EP_CONN_STATE_CLOSED;
        req_events      = 0;
    }

    uct_tcp_cm_change_conn_state(ep, new_conn_state);
    uct_tcp_ep_mod_events(ep, req_events, 0);
    return status;
}

/* This function is called from async thread */
ucs_status_t uct_tcp_cm_handle_incoming_conn(uct_tcp_iface_t *iface,
                                             const struct sockaddr_in *peer_addr,
                                             int fd)
{
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    uct_tcp_ep_t *ep;

    status = uct_tcp_ep_init(iface, fd, NULL, &ep);
    if (status != UCS_OK) {
        return status;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_ACCEPTING);
    uct_tcp_ep_mod_events(ep, EPOLLIN, 0);

    ucs_debug("tcp_iface %p: accepted connection from "
              "%s on %s to tcp_ep %p (fd %d)", iface,
              ucs_sockaddr_str((const struct sockaddr*)peer_addr,
                               str_remote_addr, UCS_SOCKADDR_STRING_LEN),
              ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                               str_local_addr, UCS_SOCKADDR_STRING_LEN),
              ep, fd);
    return UCS_OK;
}
