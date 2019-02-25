/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>

static ucs_status_t uct_tcp_cm_send_conn_ack(uct_tcp_ep_t *ep)
{    
    uct_tcp_ep_conn_pkt_t conn_pkt = {
        .event                     = UCT_TCP_EP_CONN_ACK,
    };
    ucs_status_t status;

    status = uct_tcp_send_blocking(ep->fd, &conn_pkt, sizeof(conn_pkt));
    if (status != UCS_OK) {
        ucs_error("Blocking send failed on fd %d: %m", ep->fd);
        return status;
    } else {
        ucs_debug("tcp_ep %p: conection ack sent to %s:%d",
                  ep, inet_ntoa(ep->peer_addr->sin_addr),
                  ntohs(ep->peer_addr->sin_port));
    }

    return UCS_OK;
}

ucs_status_t uct_tcp_cm_send_conn_req(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface         = ucs_derived_of(ep->super.super.iface,
                                                    uct_tcp_iface_t);
    uct_tcp_ep_conn_pkt_t conn_pkt = {
        .event                     = UCT_TCP_EP_CONN_REQ,
        .data.req.iface_addr       = iface->config.ifaddr,
    };
    ucs_status_t status;

    status = uct_tcp_send_blocking(ep->fd, &conn_pkt, sizeof(conn_pkt));
    if (status != UCS_OK) {
        ucs_debug("Blocking send failed on fd %d: %m", ep->fd);
        return status;
    } else {
        ucs_debug("tcp_ep %p: conection request sent to %s:%d",
                  ep, inet_ntoa(ep->peer_addr->sin_addr),
                  ntohs(ep->peer_addr->sin_port));
    }

    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_recv_conn_req(uct_tcp_ep_t *ep,
                                             struct sockaddr_in *peer_addr)
{
    uct_tcp_ep_conn_pkt_t conn_pkt;
    ucs_status_t status;

    status = uct_tcp_recv_blocking(ep->fd, &conn_pkt, sizeof(conn_pkt));
    if (status == UCS_OK) {
        ucs_assertv(conn_pkt.event == UCT_TCP_EP_CONN_REQ, "ep=%p", ep);
    } else {
        ucs_debug("Blocking recv failed on fd %d: %m", ep->fd);
        return status;
    }

    if (peer_addr) {
        *peer_addr = conn_pkt.data.req.iface_addr;
        ucs_debug("tcp_ep %p: Received the connection request from %s:%d",
                  ep, inet_ntoa(peer_addr->sin_addr),
                  ntohs(peer_addr->sin_port));
    }

    return UCS_OK;
}

static unsigned uct_tcp_cm_conn_req_rx_connected_progress(uct_tcp_ep_t *ep)
{
    ucs_status_t status;
    struct sockaddr_in peer_addr;

    status = uct_tcp_cm_recv_conn_req(ep, &peer_addr);
    if (status != UCS_OK) {
        return 0;
    }

    ucs_assert(uct_tcp_sockaddr_cmp(ep->peer_addr, &peer_addr) == 0);

    status = uct_tcp_ep_assign_rx(ep);
    if (status != UCS_OK) {
        return 0;
    }

    return 0;
}

static unsigned uct_tcp_cm_conn_ack_rx_progress(uct_tcp_ep_t *ep)
{
    uct_tcp_ep_conn_pkt_t conn_pkt;

    if (uct_tcp_recv_blocking(ep->fd, &conn_pkt, sizeof(conn_pkt)) != UCS_OK) {
        ucs_debug("Blocking recv failed on fd %d: %m. Perhaps the peer "
                  "closed the connection. Nothing to do with %p, just "
                  "wait when it will be re-used", ep->fd, ep);
        uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
        return 0;
    }

    switch (conn_pkt.event) {
    case UCT_TCP_EP_CONN_ACK:
        break;
    case UCT_TCP_EP_CONN_REQ:
        /* The peer want that we set RX to receive data from it,
         * check whether we recieved such request from this peer
         * or not - it may occur when we rejecting the connection
         * request to us and we are unable to receive the first
         * connection request, because the socket `fd` is closed. */
        if (ep->rx == NULL) {
            if (uct_tcp_ep_assign_rx(ep) != UCS_OK) {
                return 0;
            }
        }

        /* This blocking receive must always be successful */
        if (uct_tcp_recv_blocking(ep->fd, &conn_pkt, sizeof(conn_pkt)) != UCS_OK) {
            ucs_error("Blocking recv failed on fd %d (%p): %m", ep->fd, ep);
            uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
            return 0;
        }
 
        ucs_assertv(conn_pkt.event == UCT_TCP_EP_CONN_ACK, "ep=%p", ep);
        break;
    }

    uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_CONNECTED);

    ep->progress_tx = uct_tcp_ep_progress_tx;

    if (ep->rx != NULL) {
        ep->progress_rx = uct_tcp_ep_progress_rx;
    } else {
        /* Set this callback to receive a connection request from
         * the peer when it will want to send us data */
        ep->progress_rx = uct_tcp_cm_conn_req_rx_connected_progress;
    }

    uct_tcp_ep_mod_events(ep, EPOLLOUT, 0);

    ucs_assertv(ep->events & EPOLLIN,
                "EPOLLIN must be set for ep=%p", ep);

    /* Progress possible pending TX operations */
    return ep->progress_tx(ep);
}

static unsigned uct_tcp_cm_simult_conn_accept_remote_conn(uct_tcp_ep_t *accept_ep,
                                                          uct_tcp_ep_t *connect_ep)
{
    ucs_status_t status;

    /* 1. Close the allocated socket `fd` to avoid reading any
     *    events for this socket and assign the socket `fd` returned
     *    from `accept()` to the found EP */
    uct_tcp_ep_mod_events(connect_ep, 0, EPOLLOUT | EPOLLIN);
    ucs_assertv(connect_ep->events == 0,
                "Requsted epoll events must be 0-ed for ep=%p", connect_ep);

    close(connect_ep->fd);
    connect_ep->fd = accept_ep->fd;

    /* 2. Migrate RX from the EP allocated during accepting connection to
     *    the found EP */
    uct_tcp_ep_migrate_rx(connect_ep, accept_ep);

    /* 3. Destroy the EP allocated during accepting connection
     *    (set its socket `fd` to -1 prior to avoid closing this socket) */
    uct_tcp_ep_mod_events(accept_ep, 0, EPOLLIN);
    accept_ep->fd = -1;
    uct_tcp_ep_destroy(&accept_ep->super.super);
    accept_ep = NULL;

    connect_ep->progress_tx = uct_tcp_ep_progress_tx;
    connect_ep->progress_rx = uct_tcp_ep_progress_rx;

    /* 4. Send ACK to the peer */    
    status = uct_tcp_cm_send_conn_ack(connect_ep);
    if (status != UCS_OK) {
        return status;
    }

    /* 5. Ok, now we fully connected to the peer */
    uct_tcp_ep_change_conn_state(connect_ep, UCT_TCP_EP_CONN_CONNECTED);

    uct_tcp_ep_mod_events(connect_ep, EPOLLIN | EPOLLOUT, 0);

    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_handle_simult_conn(uct_tcp_iface_t *iface,
                                                  uct_tcp_ep_t *accept_ep,
                                                  uct_tcp_ep_t *connect_ep,
                                                  unsigned *progress_count)
{
    ucs_status_t status = UCS_OK;

    if (connect_ep->conn_state == UCT_TCP_EP_CONN_CONNECTED ||
        uct_tcp_sockaddr_cmp(connect_ep->peer_addr, &iface->config.ifaddr) > 0) {
        /* We must live with our connection and reject this one */

        /* 1. Migrate RX from the EP allocated during accepting connection to
         *    the found E. Don't set anything if != CONNECTED. because we need to handle
	 *    a connection data */

        if (connect_ep->rx == NULL) {
            uct_tcp_ep_migrate_rx(connect_ep, accept_ep);
        }

        if (connect_ep->conn_state == UCT_TCP_EP_CONN_CONNECTED) {
            connect_ep->progress_rx = uct_tcp_ep_progress_rx;

            uct_tcp_ep_mod_events(connect_ep, EPOLLIN, 0);
        }

        /* 2. Destroy the EP allocated during accepting connection */
        uct_tcp_ep_mod_events(accept_ep, 0, EPOLLIN);
        uct_tcp_ep_destroy(&accept_ep->super.super);
    } else /* our ifacce address less than remote && we are not connected */ {
        /* We must accept this connection and close our one */

        /* 1. If we're still connecting, send connection request to the peer
         *    using new socket `fd` to ensure opening RX on the peer's EP
         *    (i.e. it must be done before any data sent to the peer).
         *    NOTE: the peer must be able to handle the receiving of:
         *    - 1 connection request: the 1st connection request failed to
         *      be received (old socket `fd` is closed), the connection
         *      request recevied only using the new socket `fd`.
         *    - 2 connection requests: the 1st and 2nd connection request are
         *      successful, no action should be done for the 2nd one. */
        if (connect_ep->conn_state == UCT_TCP_EP_CONN_CONNECTING ||
            connect_ep->conn_state == UCT_TCP_EP_CONN_CONNECT_ACK) {
            status = uct_tcp_cm_send_conn_req(accept_ep);
            if (status != UCS_OK) {
                ucs_error("tcp_ep %p: unable to send connection request to %s:%d",
                          accept_ep, inet_ntoa(accept_ep->peer_addr->sin_addr),
                          ntohs(accept_ep->peer_addr->sin_port));
                return status;
            }
        }

        /* 2. Accept remote connection and close our one */
        status = uct_tcp_cm_simult_conn_accept_remote_conn(accept_ep, connect_ep);
        if (status != UCS_OK) {
            return status;
        }

        /* 3. Progress TX, because we may have some pending operations queued */
        *progress_count += connect_ep->progress_tx(connect_ep);
    }

    return status;
}

static unsigned uct_tcp_cm_conn_req_rx_progress(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_t *pair_ep  = NULL, *iter_ep;
    unsigned count = 0;
    struct sockaddr_in peer_addr;
    ucs_status_t status;

    status = uct_tcp_cm_recv_conn_req(ep, &peer_addr);
    if (status != UCS_OK) {
        ucs_debug("Receiving of connection request for %d (%p) failed. "
                  "Someone is unable to connect to us (%s:%d)",
                  ep->fd, ep, inet_ntoa(iface->config.ifaddr.sin_addr),
                  ntohs(iface->config.ifaddr.sin_port));
        uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
        uct_tcp_ep_destroy(&ep->super.super);
        return 0;
    }

    *ep->peer_addr = peer_addr;

    uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_CONNECTING);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_for_each(iter_ep, &iface->ep_list, list) {
        if (uct_tcp_sockaddr_cmp(iter_ep->peer_addr, ep->peer_addr) == 0 &&
            uct_tcp_sockaddr_cmp(iter_ep->peer_addr, &iface->config.ifaddr) != 0 &&
            iter_ep->rx == NULL) {
            pair_ep = iter_ep;
            break;
        }
    }
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_assert(!pair_ep || uct_tcp_sockaddr_cmp(pair_ep->peer_addr,
                                                &iface->config.ifaddr) != 0);

    if (pair_ep) {
        status = uct_tcp_cm_handle_simult_conn(iface, ep, pair_ep, &count);
        if (status != UCS_OK) {
            goto err;
        }
    } else {
        /* Just accept this connection and make it operational for RX events only */
        status = uct_tcp_cm_send_conn_ack(ep);
        if (status != UCS_OK) {
            goto err;
        }

        uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_CONNECTED);

        ep->progress_rx = uct_tcp_ep_progress_rx;

        UCS_ASYNC_BLOCK(iface->super.worker->async);
        ucs_list_add_tail(&iface->ep_list, &ep->list);
        UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    }

    return count;

err:
    uct_tcp_ep_set_failed(ep);
    return 0;
}

static unsigned uct_tcp_cm_conn_progress(uct_tcp_ep_t *ep)
{
    ucs_status_t status;

    status = uct_tcp_socket_connect_nb_get_status(ep->fd);
    if (status == UCS_INPROGRESS) {
        return 0;
    } else if (status != UCS_OK) {
        goto err;
    }

    if (uct_tcp_cm_send_conn_req(ep) != UCS_OK) {
        ucs_debug("EP (%p) is unable to send connection request to %s:%d",
                  ep, inet_ntoa(ep->peer_addr->sin_addr),
                  ntohs(ep->peer_addr->sin_port));
        return 0;
    }

    ep->progress_tx = uct_tcp_ep_empty_progress;
    ep->progress_rx = uct_tcp_cm_conn_ack_rx_progress;

    uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_CONNECT_ACK);

    uct_tcp_ep_mod_events(ep, EPOLLIN, EPOLLOUT);

    return 0;

 err:
    ucs_error("Non-blocking connect(%s:%d) failed",
              inet_ntoa(ep->peer_addr->sin_addr),
              ntohs(ep->peer_addr->sin_port));
    uct_tcp_ep_set_failed(ep);
    return 0;
}

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep)
{
    ucs_status_t status;
    uint32_t req_events;
    uct_tcp_ep_conn_state_t new_conn_state;

    ucs_assertv(ep->progress_tx == uct_tcp_ep_progress_tx, "ep=%p", ep);

    status = uct_tcp_socket_connect(ep->fd, ep->peer_addr);
    if (status == UCS_INPROGRESS) {
        ep->progress_tx = uct_tcp_cm_conn_progress;

        new_conn_state = UCT_TCP_EP_CONN_CONNECTING;
        req_events = EPOLLOUT;

        status = UCS_OK;
    } else if (status != UCS_OK) {
        return status;
    } else {
        status = uct_tcp_cm_send_conn_req(ep);
        if (status != UCS_OK) {
            ucs_error("tcp_ep %p: Failed to initiate the connection with the peer (%s:%d)",
                      ep, inet_ntoa(ep->peer_addr->sin_addr),
                      ntohs(ep->peer_addr->sin_port));
            return status;
        }

        ep->progress_tx = uct_tcp_ep_empty_progress;
        ep->progress_rx = uct_tcp_cm_conn_ack_rx_progress;

        new_conn_state = UCT_TCP_EP_CONN_CONNECT_ACK;
        req_events = EPOLLIN;
    }

    uct_tcp_ep_change_conn_state(ep, new_conn_state);

    uct_tcp_ep_mod_events(ep, req_events, 0);

    return status;
}

ucs_status_t uct_tcp_cm_handle_incoming_conn(uct_tcp_iface_t *iface, int fd,
                                             const struct sockaddr_in *peer_addr)
{
    ucs_status_t status;
    uct_tcp_ep_t *ep;

    ucs_debug("tcp_iface %p: accepted connection from %s:%d to fd %d", iface,
              inet_ntoa(peer_addr->sin_addr), ntohs(peer_addr->sin_port), fd);

    status = uct_tcp_ep_create(iface, fd, NULL, &ep);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_tcp_ep_assign_rx(ep);
    if (status != UCS_OK) {
        goto err_ep_destroy; 
    }

    ep->progress_rx = uct_tcp_cm_conn_req_rx_progress;

    uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_ACCEPTING);

    uct_tcp_ep_mod_events(ep, EPOLLIN, 0);

    return UCS_OK;

err_ep_destroy:
    uct_tcp_ep_destroy(&ep->super.super);
    return status;
}
