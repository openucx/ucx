/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>

uct_tcp_ep_conn_state_t
uct_tcp_cm_set_conn_state(uct_tcp_ep_t *ep,
                          uct_tcp_ep_conn_state_t new_conn_state)
{
    uct_tcp_ep_conn_state_t old_conn_state = ep->conn_state;

    ep->conn_state = new_conn_state;
    ucs_debug("tcp_ep %p: set new state %s",
              ep, uct_tcp_ep_cm_state[ep->conn_state].name);
    return old_conn_state;
}

void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep,
                                  uct_tcp_ep_conn_state_t new_conn_state)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    char str_ctx_caps[UCT_TCP_EP_CTX_CAPS_STR_MAX];
    uct_tcp_ep_conn_state_t old_conn_state;

    old_conn_state = uct_tcp_cm_set_conn_state(ep, new_conn_state);

    switch(ep->conn_state) {
    case UCT_TCP_EP_CONN_STATE_CONNECTING:
    case UCT_TCP_EP_CONN_STATE_WAITING_ACK:
        if (old_conn_state == UCT_TCP_EP_CONN_STATE_CLOSED) {
            uct_tcp_iface_outstanding_inc(iface);
            ucs_debug("Increased for EP %p (%zu)", ep, iface->outstanding);
        } else {
            ucs_assert((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
                       (old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING));
        }
        break;
    case UCT_TCP_EP_CONN_STATE_CONNECTED:
        ucs_assert((old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING));
        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
            /* it may happen when a peer is going to use this EP with socket
             * from accepted connection in case of handling simultaneous
             * connection establishment */
            (old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING)) {
            uct_tcp_iface_outstanding_dec(iface);
            ucs_debug("Decreased for EP %p (%zu)", ep, iface->outstanding);
        }
        break;
    case UCT_TCP_EP_CONN_STATE_CLOSED:
        ucs_assert(old_conn_state != UCT_TCP_EP_CONN_STATE_CLOSED);
        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) {
            uct_tcp_iface_outstanding_dec(iface);
            ucs_debug("Decreased for EP %p (%zu)", ep, iface->outstanding);
        }
        break;
    default:
        ucs_assert(ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING);
        /* Since ep::peer_addr is 0'ed and client's <address:port>
         * has already been logged, no need to log here */
        return;
    }

    ucs_debug("tcp_ep %p: %s -> %s for the [%s]<->[%s] connection %s",
              ep, uct_tcp_ep_cm_state[old_conn_state].name,
              uct_tcp_ep_cm_state[ep->conn_state].name,
              ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                               str_local_addr, UCS_SOCKADDR_STRING_LEN),
              ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                               str_remote_addr, UCS_SOCKADDR_STRING_LEN),
              uct_tcp_ep_ctx_caps_str(ep->ctx_caps, str_ctx_caps));
}

static void uct_tcp_cm_io_err_handler_cb(void *arg,
                                         const ucs_socket_io_err_info_t *err_info)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;

    /* check whether this is possible somaxconn exceeded reason or not */
    if (((err_info->err_no == ECONNRESET) || (err_info->err_no == ECONNREFUSED)) &&
        ((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
         (ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK))) {
        err_info->print_err(err_info);
        ucs_error("try to increase \"net.core.somaxconn\" on the remote node");
    } else {
        err_info->print_err(err_info);
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
                                uct_tcp_cm_conn_event_t actual_event)
{
    char expected_event_str[64] = { 0 };
    char str_addr[UCS_SOCKADDR_STRING_LEN];
    int not_equal = 0;

    switch (ep->conn_state) {
    case UCT_TCP_EP_CONN_STATE_ACCEPTING:
        snprintf(expected_event_str, sizeof(expected_event_str),
                 "%u", UCT_TCP_CM_CONN_REQ);
        not_equal = (actual_event != UCT_TCP_CM_CONN_REQ);
        break;
    case UCT_TCP_EP_CONN_STATE_WAITING_ACK:
        snprintf(expected_event_str, sizeof(expected_event_str),
                 "%u or %u", UCT_TCP_CM_CONN_ACK, UCT_TCP_CM_CONN_REQ);
        not_equal = ((actual_event != UCT_TCP_CM_CONN_REQ) &&
                     (actual_event != UCT_TCP_CM_CONN_ACK));
        break;
    case UCT_TCP_EP_CONN_STATE_CONNECTED:
        ucs_assert((ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_RX)) == 0);
        snprintf(expected_event_str, sizeof(expected_event_str),
                 "%u", UCT_TCP_CM_CONN_REQ);
        not_equal = (actual_event != UCT_TCP_CM_CONN_REQ);
        break;
    default:
        ucs_assertv_always(0, "this mustn't happen ep=%p", ep);
        break;
    }

    if (not_equal) {
        ucs_error("tcp_ep %p: received wrong CM event (expected: %s, "
                  "actual: %u) from the peer with iface listener "
                  "address: %s",
                  ep, expected_event_str, actual_event,
                  ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                   str_addr, UCS_SOCKADDR_STRING_LEN));
        return UCS_ERR_INVALID_PARAM;
    }
    return UCS_OK;
}

ucs_status_t uct_tcp_cm_send_conn_req(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface             = ucs_derived_of(ep->super.super.iface,
                                                        uct_tcp_iface_t);
    uct_tcp_cm_conn_req_pkt_t conn_pkt = {
        .event                         = UCT_TCP_CM_CONN_REQ,
        .iface_addr                    = iface->config.ifaddr
    };
    ucs_status_t status;

    status = ucs_socket_send(ep->fd, &conn_pkt, sizeof(conn_pkt),
                             uct_tcp_cm_io_err_handler_cb, ep);
    if (status != UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, "unable to send connection req to",
                                  &ep->peer_addr);
        return status;
    }

    uct_tcp_cm_trace_conn_pkt(ep, "connection req sent to",
                              &ep->peer_addr);
    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_recv_conn_req(uct_tcp_ep_t *ep,
                                             struct sockaddr_in *peer_addr)
{
    uct_tcp_cm_conn_req_pkt_t conn_pkt;
    ucs_status_t status;

    status = ucs_socket_recv(ep->fd, &conn_pkt, sizeof(conn_pkt), NULL, NULL);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_tcp_cm_conn_pkt_check_event(ep, conn_pkt.event);
    if (status != UCS_OK) {
        return status;
    }

    *peer_addr = conn_pkt.iface_addr;

    uct_tcp_cm_trace_conn_pkt(ep, "connection req received from",
                              peer_addr);
    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_send_conn_ack(uct_tcp_ep_t *ep)
{
    uct_tcp_cm_conn_event_t event = UCT_TCP_CM_CONN_ACK;
    ucs_status_t status;

    status = ucs_socket_send(ep->fd, &event, sizeof(event), NULL, NULL);
    if (status != UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, "unable to send connection ack to",
                                  &ep->peer_addr);
        return status;
    }

    uct_tcp_cm_trace_conn_pkt(ep, "connection ack sent to",
                              &ep->peer_addr);
    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_recv_conn_ack_or_req(uct_tcp_ep_t *ep,
                                                    unsigned *progress_count)
{
    struct sockaddr_in peer_addr = { 0 };
    char str_expected_addr[UCS_SOCKADDR_STRING_LEN];
    char str_actual_addr[UCS_SOCKADDR_STRING_LEN];
    uct_tcp_cm_conn_event_t event;
    ucs_status_t status;
    int cmp;

    do {
        status = ucs_socket_recv(ep->fd, &event, sizeof(event),
                                 uct_tcp_cm_io_err_handler_cb, ep);
        if (status != UCS_OK) {
            uct_tcp_cm_trace_conn_pkt(ep,
                                      "unable to receive connection ack or req from",
                                      &ep->peer_addr);
            return status;
        }

        status = uct_tcp_cm_conn_pkt_check_event(ep, event);
        if (status != UCS_OK) {
            return status;
        }

        if (event == UCT_TCP_CM_CONN_ACK) {
            uct_tcp_cm_trace_conn_pkt(ep, "connection ack received from",
                                      &ep->peer_addr);
        } else {
            /* The peer want that the reciever sets RX to receive data from
             * it, receive the remaining part of the request (iface address)
             * check whether we recieved such request from this peer or not.
             * It may occur when we rejecting the connection request to us
             * and we are unable to receive the first connection request,
             * because the socket `fd` is closed. */
            status = ucs_socket_recv(ep->fd, &peer_addr,
                                     ucs_field_sizeof(uct_tcp_cm_conn_req_pkt_t,
                                                      iface_addr),
                                     uct_tcp_cm_io_err_handler_cb, ep);
            if (status != UCS_OK) {
                uct_tcp_cm_trace_conn_pkt(ep,
                                          "unable to receive connection ack or req from",
                                          &ep->peer_addr);
                return status;
            }

            (*progress_count)++;

            cmp = ucs_sockaddr_is_equal((const struct sockaddr*)&peer_addr,
                                        (const struct sockaddr*)&ep->peer_addr,
                                        &status);
            if (status != UCS_OK) {
                return status;
            } else  if (!cmp) {
                ucs_error("tcp_ep %p: received request with wrong peer addr "
                          "(expected: %s, actual: %s)", ep,
                          ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                           str_expected_addr, UCS_SOCKADDR_STRING_LEN),
                          ucs_sockaddr_str((const struct sockaddr*)&peer_addr,
                                           str_actual_addr, UCS_SOCKADDR_STRING_LEN));
                return UCS_ERR_INVALID_ADDR;
            }

            uct_tcp_cm_trace_conn_pkt(ep, "connection req received from",
                                      &ep->peer_addr);
            if (!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_RX))) {
                status = uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_CTX_TYPE_RX);
                if (status != UCS_OK) {
                    return status;
                }
            }
        }

        (*progress_count)++;
    } while (event != UCT_TCP_CM_CONN_ACK);

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
    unsigned progress_count = 0;
    ucs_status_t status;

    status = uct_tcp_cm_recv_conn_ack_or_req(ep, &progress_count);
    if (status != UCS_OK) {
        uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
        return 0;
    }

    ucs_assertv(ep->tx.buf == NULL, "ep=%p", ep);

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);

    /* Progress possibly pending TX operations */
    return progress_count + uct_tcp_ep_progress(ep, UCT_TCP_EP_CTX_TYPE_TX);
}

ucs_status_t uct_tcp_cm_add_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep)
{
    ucs_list_link_t *ep_list;
    khiter_t iter;
    int ret;

    iter = kh_get(uct_tcp_cm_eps, &iface->ep_cm_map, ep->peer_addr);
    if (iter == kh_end(&iface->ep_cm_map)) {
        ep_list = ucs_calloc(1, sizeof(*ep_list), "tcp_ep_cm_map_entry");
        if (ep_list == NULL) {
            return UCS_ERR_NO_MEMORY;
        }
        ucs_assertv_always(ep_list != NULL, "iface=%p", iface);
        ucs_list_head_init(ep_list);

        iter = kh_put(uct_tcp_cm_eps, &iface->ep_cm_map, ep->peer_addr, &ret);
        kh_value(&iface->ep_cm_map, iter) = ep_list;

        ucs_debug("tcp_iface %p: %p list added to map",
                  iface, ep_list);
    } else {
        ep_list = kh_value(&iface->ep_cm_map, iter);
        ucs_assertv(!ucs_list_is_empty(ep_list), "iface=%p", iface);
    }

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_del(&ep->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_list_add_tail(ep_list, &ep->list);
    ucs_debug("tcp_iface %p: tcp_ep %p added to %p list",
              iface, ep, ep_list);

    return UCS_OK;
}

void uct_tcp_cm_remove_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep)
{
    ucs_list_link_t *ep_list;
    khiter_t iter;

    iter = kh_get(uct_tcp_cm_eps, &iface->ep_cm_map, ep->peer_addr);
    ucs_assertv(iter != kh_end(&iface->ep_cm_map), "iface=%p", iface);

    ep_list = kh_value(&iface->ep_cm_map, iter);
    ucs_assertv(!ucs_list_is_empty(ep_list), "iface=%p", iface);

    ucs_list_del(&ep->list);
    ucs_debug("tcp_iface %p: tcp_ep %p removed from %p list",
              iface, ep, ep_list);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->ep_list, &ep->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    if (ucs_list_is_empty(ep_list)) {
        kh_del(uct_tcp_cm_eps, &iface->ep_cm_map, iter);
        ucs_debug("tcp_iface %p: %p list removed from map",
                  iface, ep_list);
        ucs_free(ep_list);
    }
}

uct_tcp_ep_t *uct_tcp_cm_search_ep(uct_tcp_iface_t *iface,
                                   const struct sockaddr_in *peer_addr,
                                   uct_tcp_ep_ctx_type_t without_ctx_type)
{
    uct_tcp_ep_t *ep = NULL;
    uct_tcp_ep_t *iter_ep;
    ucs_list_link_t *ep_list;
    khiter_t iter;

    iter = kh_get(uct_tcp_cm_eps, &iface->ep_cm_map, *peer_addr);
    if (iter != kh_end(&iface->ep_cm_map)) {
        ep_list = kh_value(&iface->ep_cm_map, iter);
        ucs_assertv(!ucs_list_is_empty(ep_list), "iface=%p", iface);

        ucs_list_for_each(iter_ep, ep_list, list) {
            if (!(iter_ep->ctx_caps & UCS_BIT(without_ctx_type))) {
                ep = iter_ep;
                break;
            }
        }
    }

    return ep;
}

static unsigned
uct_tcp_cm_simult_conn_accept_remote_conn(uct_tcp_ep_t *accept_ep,
                                          uct_tcp_ep_t *connect_ep,
                                          unsigned *progress_count)
{
    ucs_status_t status;

    /* 1. Close the allocated socket `fd` to avoid reading any
     *    events for this socket and assign the socket `fd` returned
     *    from `accept()` to the found EP */
    uct_tcp_ep_mod_events(connect_ep, 0, connect_ep->events);
    ucs_assertv(connect_ep->events == 0,
                "Requsted epoll events must be 0-ed for ep=%p", connect_ep);

    close(connect_ep->fd);
    connect_ep->fd = accept_ep->fd;

    /* 2. Migrate RX from the EP allocated during accepting connection to
     *    the found EP */
    status = uct_tcp_ep_move_ctx_cap(accept_ep, connect_ep,
                                     UCT_TCP_EP_CTX_TYPE_RX);
    if (status != UCS_OK) {
        return status;
    }

    /* 3. Destroy the EP allocated during accepting connection
     *    (set its socket `fd` to -1 prior to avoid closing this socket) */
    uct_tcp_ep_mod_events(accept_ep, 0, EPOLLIN);
    accept_ep->fd = -1;
    uct_tcp_ep_destroy(&accept_ep->super.super);
    accept_ep = NULL;

    /* 4. Send ACK to the peer */
    status = uct_tcp_cm_send_conn_ack(connect_ep);
    if (status != UCS_OK) {
        return status;
    }

    (*progress_count)++;

    /* 5. Ok, now fully connected to the peer */
    uct_tcp_cm_change_conn_state(connect_ep, UCT_TCP_EP_CONN_STATE_CONNECTED);
    uct_tcp_ep_mod_events(connect_ep, EPOLLIN | EPOLLOUT, 0);

    return UCS_OK;
}

static ucs_status_t uct_tcp_cm_handle_simult_conn(uct_tcp_iface_t *iface,
                                                  uct_tcp_ep_t *accept_ep,
                                                  uct_tcp_ep_t *connect_ep,
                                                  unsigned *progress_count)
{
    ucs_status_t status;
    int cmp;

    cmp = ucs_sockaddr_cmp((const struct sockaddr*)&connect_ep->peer_addr,
                           (const struct sockaddr*)&iface->config.ifaddr,
                           &status);
    if (status != UCS_OK) {
        return status;
    }

    if ((cmp > 0) ||
        (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED)) {
        /* Have to live with the current connection and reject this one */

        /* 1. Migrate RX from the EP allocated during accepting connection to
         *    the found EP. Don't set anything if != CONNECTED, because we need
         *    to handle a connection data */
        status = uct_tcp_ep_move_ctx_cap(accept_ep, connect_ep,
                                         UCT_TCP_EP_CTX_TYPE_RX);   
        if (status != UCS_OK) {
            return status;
        }

        if (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) {
            uct_tcp_ep_mod_events(connect_ep, EPOLLIN, 0);
        }

        /* 2. Destroy the EP allocated during accepting connection */
        uct_tcp_ep_destroy(&accept_ep->super.super);
    } else /* our ifacce address less than remote && we are not connected */ {
        /* Have to accept this connection and close the current one */

        /* 1. If we're still connecting, send connection request to the peer
         *    using new socket `fd` to ensure opening RX on the peer's EP
         *    (i.e. it must be done before any data sent to the peer).
         *    NOTE: the peer must be able to handle the receiving of:
         *    - 1 connection request: the 1st connection request failed to
         *      be received (old socket `fd` is closed), the connection
         *      request recevied only using the new socket `fd`.
         *    - 2 connection requests: the 1st and 2nd connection request are
         *      successful, no action should be done for the 2nd one. */
        if ((connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
            (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) {
            status = uct_tcp_cm_send_conn_req(accept_ep);
            if (status != UCS_OK) {
                return status;
            }

            (*progress_count)++;
        }

        /* 2. Accept the remote connection and close the current one */
        status = uct_tcp_cm_simult_conn_accept_remote_conn(accept_ep, connect_ep,
                                                           progress_count);
        if (status != UCS_OK) {
            return status;
        }

        /* 3. Progress TX, because we may have some pending operations queued */
        *progress_count += uct_tcp_ep_progress(connect_ep,
                                               UCT_TCP_EP_CTX_TYPE_TX);
    }

    return status;
}

unsigned uct_tcp_cm_conn_req_rx_progress(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface  = ucs_derived_of(ep->super.super.iface,
                                             uct_tcp_iface_t);
    unsigned progress_count = 1;
    uct_tcp_ep_t *pair_ep;
    struct sockaddr_in peer_addr;
    ucs_status_t status;

    status = uct_tcp_cm_recv_conn_req(ep, &peer_addr);
    if (status != UCS_OK) {
        /* The peer closed the connection - destroy the EP */
        if (ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) {
            uct_tcp_ep_destroy(&ep->super.super);
        }
        return 0;
    }

    if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) {
        status = uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_CTX_TYPE_RX);
        if (status != UCS_OK) {
            goto err;
        }
        return 1;
    }

    ep->peer_addr = peer_addr;
    status = uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_CTX_TYPE_RX);
    if (status != UCS_OK) {
        goto err;
    }

    if (!uct_tcp_ep_peer_addr_to_itself(ep) &&
        (pair_ep = uct_tcp_cm_search_ep(iface, &peer_addr,
                                        UCT_TCP_EP_CTX_TYPE_RX))) {
        status = uct_tcp_cm_handle_simult_conn(iface, ep, pair_ep,
                                               &progress_count);
        if (status != UCS_OK) {
            goto err;
        }
    } else {
        /* Just accept this connection and make it operational for RX events */
        status = uct_tcp_cm_send_conn_ack(ep);
        if (status != UCS_OK) {
            goto err;
        }

        ucs_assertv(ep->rx.buf == NULL, "ep=%p", ep);

        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);

        progress_count = 2;
    }

    return progress_count;

 err:
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
