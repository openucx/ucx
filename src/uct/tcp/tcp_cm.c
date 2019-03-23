/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>


const static char *uct_tcp_cm_conn_state_str[][UCT_TCP_EP_CONN_STATE_STR_MAX] = {
    UCT_TCP_EP_CONN_STATES(UCT_TCP_EP_CONN_STATE_STR)
};


unsigned uct_tcp_cm_conn_progress(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;

    status = ucs_socket_connect_nb_get_status(ep->fd);
    if (status == UCS_INPROGRESS) {
        return 0;
    } else if (status != UCS_OK) {
        goto err;
    }

    iface->outstanding--;

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CTX_TYPE_TX,
                                 UCT_TCP_EP_CONN_CONNECTED);

    ucs_assertv((ep->tx.length == 0) && (ep->tx.offset == 0) &&
                (ep->tx.buf == NULL), "ep=%p", ep);

    return uct_tcp_ep_progress(ep, UCT_TCP_EP_CTX_TYPE_TX);

err:
    iface->outstanding--;
    ucs_error("non-blocking connect(%s) failed",
              ucs_sockaddr_str(ep->peer_addr.addr, str_remote_addr,
                               UCS_SOCKADDR_STRING_LEN));
    uct_tcp_ep_set_failed(ep, UCT_TCP_EP_CTX_TYPE_TX);
    return 0;
}

void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep, uct_tcp_ep_ctx_type_t ctx_type,
                                  uct_tcp_ep_conn_state_t new_conn_state)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_tcp_iface_t);
    char str_local_addr[UCS_SOCKADDR_STRING_LEN], str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    uct_tcp_ep_conn_state_t old_conn_state;

    old_conn_state = ep->conn_state;
    ep->conn_state = new_conn_state;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        return;
    }

    ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                     str_local_addr, UCS_SOCKADDR_STRING_LEN);
    ucs_sockaddr_str(ep->peer_addr.addr, str_remote_addr, UCS_SOCKADDR_STRING_LEN);

    ucs_debug("[%s -> %s] tcp_ep %p: %s ([%s]<->[%s])",
              uct_tcp_cm_conn_state_str[old_conn_state][UCT_TCP_EP_CONN_STATE_STR_NAME],
              uct_tcp_cm_conn_state_str[ep->conn_state][UCT_TCP_EP_CONN_STATE_STR_NAME],
              ep, uct_tcp_cm_conn_state_str[ep->conn_state][UCT_TCP_EP_CONN_STATE_STR_DESCR],
              str_local_addr, str_remote_addr);
}

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_conn_state_t new_conn_state;
    ucs_status_t status;

    status = ucs_socket_connect(ep->fd, ep->peer_addr.addr);
    if (status == UCS_INPROGRESS) {
        iface->outstanding++;

        new_conn_state  = UCT_TCP_EP_CONN_CONNECTING;
        status          = UCS_OK;

        uct_tcp_ep_mod_events(ep, EPOLLOUT, 0);
    } else if (status == UCS_OK) {
        new_conn_state  = UCT_TCP_EP_CONN_CONNECTED;
    } else {
        new_conn_state  = UCT_TCP_EP_CONN_CLOSED;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CTX_TYPE_TX,
                                 new_conn_state);

    return status;
}

ucs_status_t uct_tcp_cm_handle_incoming_conn(uct_tcp_iface_t *iface,
                                             const struct sockaddr *peer_addr, int fd)
{
    ucs_status_t status;
    uct_tcp_ep_t *ep;

    status = uct_tcp_ep_init(iface, fd, peer_addr, &ep);
    if (status != UCS_OK) {
        return status;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CTX_TYPE_RX,
                                 UCT_TCP_EP_CONN_CONNECTED);
    uct_tcp_ep_mod_events(ep, EPOLLIN, 0);

    return UCS_OK;
}
