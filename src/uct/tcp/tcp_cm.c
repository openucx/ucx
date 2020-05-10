/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"

#include <ucs/async/async.h>


void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep,
                                  uct_tcp_ep_conn_state_t new_conn_state)
{
    int full_log           = 1;
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    char str_ctx_caps[UCT_TCP_EP_CTX_CAPS_STR_MAX];
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
    case UCT_TCP_EP_CONN_STATE_WAITING_REQ:
        ucs_assert(old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK);
        break;
    case UCT_TCP_EP_CONN_STATE_CONNECTED:
        ucs_assert((old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_REQ));
        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_REQ) ||
            /* It may happen when a peer is going to use this EP with socket
             * from accepted connection in case of handling simultaneous
             * connection establishment */
            (old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING)) {
            uct_tcp_iface_outstanding_dec(iface);
        }
        if (ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX)) {
            /* Progress possibly pending TX operations */
            uct_tcp_ep_pending_queue_dispatch(ep);
        }
        break;
    case UCT_TCP_EP_CONN_STATE_CLOSED:
        ucs_assert(old_conn_state != UCT_TCP_EP_CONN_STATE_CLOSED);
        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_REQ)) {
            uct_tcp_iface_outstanding_dec(iface);
        } else if ((old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER)) {
            /* Since ep::peer_addr is 0'ed, we have to print w/o peer's address */
            full_log = 0;
        }
        break;
    default:
        ucs_assert((ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
                   (ep->conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER));
        /* Since ep::peer_addr is 0'ed and client's <address:port>
         * has already been logged, print w/o peer's address */
        full_log = 0;
        break;
    }

    if (full_log) {
        ucs_debug("tcp_ep %p: %s -> %s for the [%s]<->[%s] connection %s",
                  ep, uct_tcp_ep_cm_state[old_conn_state].name,
                  uct_tcp_ep_cm_state[ep->conn_state].name,
                  ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                   str_local_addr, UCS_SOCKADDR_STRING_LEN),
                  ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                   str_remote_addr, UCS_SOCKADDR_STRING_LEN),
                  uct_tcp_ep_ctx_caps_str(ep->ctx_caps, str_ctx_caps));
    } else {
        ucs_debug("tcp_ep %p: %s -> %s",
                  ep, uct_tcp_ep_cm_state[old_conn_state].name,
                  uct_tcp_ep_cm_state[ep->conn_state].name);
    }
}

static ucs_status_t uct_tcp_cm_io_err_handler_cb(void *arg,
                                                 ucs_status_t io_status)
{
    return uct_tcp_ep_handle_dropped_connect((uct_tcp_ep_t*)arg, io_status);
}

/* `fmt_str` parameter has to contain "%s" to write event type */
static void uct_tcp_cm_trace_conn_pkt(const uct_tcp_ep_t *ep,
                                      ucs_log_level_t log_level,
                                      const char *fmt_str,
                                      uct_tcp_cm_conn_event_t event)
{
    char event_str[64] = { 0 };
    char str_addr[UCS_SOCKADDR_STRING_LEN], msg[128], *p;

    p = event_str;
    if (event & UCT_TCP_CM_CONN_REQ) {
        ucs_snprintf_zero(event_str, sizeof(event_str), "%s",
                          UCS_PP_MAKE_STRING(UCT_TCP_CM_CONN_REQ));
        p += strlen(event_str);
    }

    if (event & UCT_TCP_CM_CONN_WAIT_REQ) {
        ucs_assert(p == event_str);
        ucs_snprintf_zero(event_str, sizeof(event_str), "%s",
                          UCS_PP_MAKE_STRING(UCT_TCP_CM_CONN_WAIT_REQ));
        p += strlen(event_str);
    }

    if (event & UCT_TCP_CM_CONN_ACK) {
        if (p != event_str) {
            ucs_snprintf_zero(p, sizeof(event_str) - (p - event_str), " | ");
            p += strlen(p);
        }
        ucs_snprintf_zero(p, sizeof(event_str) - (p - event_str), "%s",
                          UCS_PP_MAKE_STRING(UCT_TCP_CM_CONN_ACK));
        p += strlen(event_str);
    }

    if (event_str == p) {
        ucs_snprintf_zero(event_str, sizeof(event_str), "UNKNOWN (%d)", event);
        log_level = UCS_LOG_LEVEL_ERROR;
    }

    ucs_snprintf_zero(msg, sizeof(msg), fmt_str, event_str);

    ucs_log(log_level, "tcp_ep %p: %s %s", ep, msg,
            ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                             str_addr, UCS_SOCKADDR_STRING_LEN));
}

ucs_status_t uct_tcp_cm_send_event(uct_tcp_ep_t *ep, uct_tcp_cm_conn_event_t event)
{
    uct_tcp_iface_t *iface     = ucs_derived_of(ep->super.super.iface,
                                                uct_tcp_iface_t);
    size_t magic_number_length = 0;
    void *pkt_buf;
    size_t pkt_length, cm_pkt_length;
    uct_tcp_cm_conn_req_pkt_t *conn_pkt;
    uct_tcp_cm_conn_event_t *pkt_event;
    uct_tcp_am_hdr_t *pkt_hdr;
    ucs_status_t status;

    ucs_assertv(!(event & ~(UCT_TCP_CM_CONN_REQ |
                            UCT_TCP_CM_CONN_ACK |
                            UCT_TCP_CM_CONN_WAIT_REQ)),
                "ep=%p", ep);
    ucs_assertv(!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX)) ||
                (ep->conn_state != UCT_TCP_EP_CONN_STATE_CONNECTED),
                "ep=%p", ep);

    pkt_length                  = sizeof(*pkt_hdr);
    if (event == UCT_TCP_CM_CONN_REQ) {
        cm_pkt_length           = sizeof(*conn_pkt);

        if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) {
            magic_number_length = sizeof(uint64_t);
        }
    } else {
        cm_pkt_length           = sizeof(event);
    }

    pkt_length     += cm_pkt_length + magic_number_length;
    pkt_buf         = ucs_alloca(pkt_length);
    pkt_hdr         = (uct_tcp_am_hdr_t*)(UCS_PTR_BYTE_OFFSET(pkt_buf,
                                                              magic_number_length));
    pkt_hdr->am_id  = UCT_AM_ID_MAX;
    pkt_hdr->length = cm_pkt_length;

    if (event == UCT_TCP_CM_CONN_REQ) {
        if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) {
            ucs_assert(magic_number_length == sizeof(uint64_t));
            *(uint64_t*)pkt_buf = UCT_TCP_MAGIC_NUMBER;
        }

        conn_pkt             = (uct_tcp_cm_conn_req_pkt_t*)(pkt_hdr + 1);
        conn_pkt->event      = UCT_TCP_CM_CONN_REQ;
        conn_pkt->iface_addr = iface->config.ifaddr;
    } else {
        pkt_event            = (uct_tcp_cm_conn_event_t*)(pkt_hdr + 1);
        *pkt_event           = event;
    }

    status = ucs_socket_send(ep->fd, pkt_buf, pkt_length,
                             uct_tcp_cm_io_err_handler_cb, ep);
    if (status == UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, UCS_LOG_LEVEL_TRACE,
                                  "%s sent to", event);
    } else {
        uct_tcp_cm_trace_conn_pkt(ep, ((status == UCS_ERR_CANCELED) ?
                                       UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR),
                                  "unable to send %s to", event);
    }
    return status;
}

ucs_status_t uct_tcp_cm_add_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep)
{
    ucs_list_link_t *ep_list;
    khiter_t iter;
    int ret;

    iter = kh_get(uct_tcp_cm_eps, &iface->ep_cm_map, ep->peer_addr);
    if (iter == kh_end(&iface->ep_cm_map)) {
        ep_list = ucs_malloc(sizeof(*ep_list), "tcp_ep_cm_map_entry");
        if (ep_list == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        ucs_list_head_init(ep_list);
        iter = kh_put(uct_tcp_cm_eps, &iface->ep_cm_map, ep->peer_addr, &ret);
        kh_value(&iface->ep_cm_map, iter) = ep_list;

        ucs_debug("tcp_iface %p: %p list added to map", iface, ep_list);
    } else {
        ep_list = kh_value(&iface->ep_cm_map, iter);
        ucs_assertv(!ucs_list_is_empty(ep_list), "iface=%p", iface);
    }

    uct_tcp_iface_remove_ep(ep);

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

    uct_tcp_iface_add_ep(ep);

    if (ucs_list_is_empty(ep_list)) {
        kh_del(uct_tcp_cm_eps, &iface->ep_cm_map, iter);
        ucs_debug("tcp_iface %p: %p list removed from map",
                  iface, ep_list);
        ucs_free(ep_list);
    }
}

uct_tcp_ep_t *uct_tcp_cm_search_ep(uct_tcp_iface_t *iface,
                                   const struct sockaddr_in *peer_addr,
                                   uct_tcp_ep_ctx_type_t with_ctx_type)
{
    uct_tcp_ep_t *ep;
    ucs_list_link_t *ep_list;
    khiter_t iter;

    iter = kh_get(uct_tcp_cm_eps, &iface->ep_cm_map, *peer_addr);
    if (iter != kh_end(&iface->ep_cm_map)) {
        ep_list = kh_value(&iface->ep_cm_map, iter);
        ucs_assertv(!ucs_list_is_empty(ep_list), "iface=%p", iface);

        ucs_list_for_each(ep, ep_list, list) {
            if (ep->ctx_caps & UCS_BIT(with_ctx_type)) {
                return ep;
            }
        }
    }

    return NULL;
}

void uct_tcp_cm_purge_ep(uct_tcp_ep_t *ep)
{
    /* Move from a khash's EP list to iface's EP list */
    ucs_list_del(&ep->list);
    uct_tcp_ep_change_ctx_caps(ep, 0);
    uct_tcp_iface_add_ep(ep);
}

static unsigned
uct_tcp_cm_simult_conn_accept_remote_conn(uct_tcp_ep_t *accept_ep,
                                          uct_tcp_ep_t *connect_ep)
{
    uct_tcp_cm_conn_event_t event;
    ucs_status_t status;

    /* 1. Close the allocated socket `fd` to avoid reading any
     *    events for this socket and assign the socket `fd` returned
     *    from `accept()` to the found EP */
    uct_tcp_ep_mod_events(connect_ep, 0, connect_ep->events);
    ucs_assertv(connect_ep->events == 0,
                "Requested epoll events must be 0-ed for ep=%p", connect_ep);

    close(connect_ep->fd);
    connect_ep->fd = accept_ep->fd;

    /* 2. Migrate RX from the EP allocated during accepting connection to
     *    the found EP */
    status = uct_tcp_ep_move_ctx_cap(accept_ep, connect_ep,
                                     UCT_TCP_EP_CTX_TYPE_RX);
    if (status != UCS_OK) {
        return 0;
    }

    /* 3. The EP allocated during accepting connection has to be destroyed
     *    upon return from this function (set its socket `fd` to -1 prior
     *    to avoid closing this socket) */
    uct_tcp_ep_mod_events(accept_ep, 0, UCS_EVENT_SET_EVREAD);
    accept_ep->fd = -1;
    accept_ep = NULL;

    /* 4. Send ACK to the peer */
    event = UCT_TCP_CM_CONN_ACK;

    /* 5. - If found EP is still connecting, tie REQ with ACK and send
     *      it to the peer using new socket fd to ensure that the peer
     *      will be able to receive the data from us
     *    - If found EP is waiting ACK, tie WAIT_REQ with ACK and send
     *      it to the peer using new socket fd to ensure that the peer
     *      will wait for REQ and after receiving the REQ, peer will
     *      be able to receive the data from us */
    if (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) {
        event |= UCT_TCP_CM_CONN_REQ;
    } else if (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) {
        event |= UCT_TCP_CM_CONN_WAIT_REQ;
    }

    status = uct_tcp_cm_send_event(connect_ep, event);
    if (status != UCS_OK) {
        return 0;
    }
    /* 6. Now fully connected to the peer */
    uct_tcp_ep_mod_events(connect_ep, UCS_EVENT_SET_EVREAD, 0);
    uct_tcp_cm_change_conn_state(connect_ep, UCT_TCP_EP_CONN_STATE_CONNECTED);

    return 1;
}

static unsigned uct_tcp_cm_handle_simult_conn(uct_tcp_iface_t *iface,
                                              uct_tcp_ep_t *accept_ep,
                                              uct_tcp_ep_t *connect_ep)
{
    int accept_conn         = 0;
    unsigned progress_count = 0;
    ucs_status_t status;
    int cmp;

    if ((connect_ep->conn_state != UCT_TCP_EP_CONN_STATE_CONNECTED) &&
        (connect_ep->conn_state != UCT_TCP_EP_CONN_STATE_WAITING_REQ)) {
        cmp = ucs_sockaddr_cmp((const struct sockaddr*)&connect_ep->peer_addr,
                               (const struct sockaddr*)&iface->config.ifaddr,
                               &status);
        if (status != UCS_OK) {
            return 0;
        }

        /* Accept connection from a peer if our iface
         * address is greater than peer's one */
        accept_conn = (cmp < 0);
    }

    if (!accept_conn) {
        /* Migrate RX from the EP allocated during accepting connection to
         * the found EP. */
        status = uct_tcp_ep_move_ctx_cap(accept_ep, connect_ep,
                                         UCT_TCP_EP_CTX_TYPE_RX);
        if (status != UCS_OK) {
            return 0;
        }

        if (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_REQ) {
            uct_tcp_cm_change_conn_state(connect_ep, UCT_TCP_EP_CONN_STATE_CONNECTED);
        }

        uct_tcp_ep_mod_events(connect_ep, UCS_EVENT_SET_EVREAD, 0);
    } else /* our iface address less than remote && we are not connected */ {
        /* Accept the remote connection and close the current one */
        ucs_assertv(cmp != 0, "peer addresses for accepted tcp_ep %p and "
                    "found tcp_ep %p mustn't be equal", accept_ep, connect_ep);
        progress_count = uct_tcp_cm_simult_conn_accept_remote_conn(accept_ep,
                                                                   connect_ep);
    }

    return progress_count;
}

static unsigned
uct_tcp_cm_handle_conn_req(uct_tcp_ep_t **ep_p,
                           const uct_tcp_cm_conn_req_pkt_t *cm_req_pkt)
{
    uct_tcp_ep_t *ep        = *ep_p;
    uct_tcp_iface_t *iface  = ucs_derived_of(ep->super.super.iface,
                                             uct_tcp_iface_t);
    unsigned progress_count = 0;
    ucs_status_t status;
    uct_tcp_ep_t *peer_ep;

    ep->peer_addr = cm_req_pkt->iface_addr;
    uct_tcp_cm_trace_conn_pkt(ep, UCS_LOG_LEVEL_TRACE,
                              "%s received from", UCT_TCP_CM_CONN_REQ);

    status = uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_CTX_TYPE_RX);
    if (status != UCS_OK) {
        goto out;
    }

    if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) {
        return 0;
    }

    ucs_assertv(!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX)),
                "ep %p mustn't have TX cap", ep);

    if (!uct_tcp_ep_is_self(ep) &&
        (peer_ep = uct_tcp_cm_search_ep(iface, &ep->peer_addr,
                                        UCT_TCP_EP_CTX_TYPE_TX))) {
        progress_count = uct_tcp_cm_handle_simult_conn(iface, ep, peer_ep);
        ucs_assert(!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX)));
        goto out;
    } else {
        /* Just accept this connection and make it operational for RX events */
        status = uct_tcp_cm_send_event(ep, UCT_TCP_CM_CONN_ACK);
        if (status != UCS_OK) {
            goto out;
        }

        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);

        progress_count = 1;
    }

    return progress_count;

out:
    if (!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_TX))) {
        uct_tcp_ep_destroy_internal(&ep->super.super);
        *ep_p = NULL;
    }
    return progress_count;
}

void uct_tcp_cm_handle_conn_ack(uct_tcp_ep_t *ep, uct_tcp_cm_conn_event_t cm_event,
                                uct_tcp_ep_conn_state_t new_conn_state)
{
    uct_tcp_cm_trace_conn_pkt(ep, UCS_LOG_LEVEL_TRACE,
                              "%s received from", cm_event);
    if (ep->conn_state != new_conn_state) {
        uct_tcp_cm_change_conn_state(ep, new_conn_state);
    }
}

unsigned uct_tcp_cm_handle_conn_pkt(uct_tcp_ep_t **ep_p, void *pkt, uint32_t length)
{
    ucs_status_t status;
    uct_tcp_cm_conn_event_t cm_event;
    uct_tcp_cm_conn_req_pkt_t *cm_req_pkt;
    uct_tcp_ep_conn_state_t new_conn_state;

    ucs_assertv(length >= sizeof(cm_event), "ep=%p", *ep_p);

    cm_event = *((uct_tcp_cm_conn_event_t*)pkt);

    switch (cm_event) {
    case UCT_TCP_CM_CONN_REQ:
        /* Don't trace received CM packet here, because
         * EP doesn't contain the peer address */
        ucs_assertv(length == sizeof(*cm_req_pkt), "ep=%p", *ep_p);
        cm_req_pkt = (uct_tcp_cm_conn_req_pkt_t*)pkt;
        return uct_tcp_cm_handle_conn_req(ep_p, cm_req_pkt);
    case UCT_TCP_CM_CONN_ACK_WITH_WAIT_REQ:
        if (!((*ep_p)->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_RX))) {
            new_conn_state = UCT_TCP_EP_CONN_STATE_WAITING_REQ;
        } else {
            new_conn_state = UCT_TCP_EP_CONN_STATE_CONNECTED;
        }
        uct_tcp_cm_handle_conn_ack(*ep_p, cm_event, new_conn_state);
        return 0;
    case UCT_TCP_CM_CONN_ACK_WITH_REQ:
        status = uct_tcp_ep_add_ctx_cap(*ep_p, UCT_TCP_EP_CTX_TYPE_RX);
        if (status != UCS_OK) {
            return 0;
        }
        /* fall through */
    case UCT_TCP_CM_CONN_ACK:
        uct_tcp_cm_handle_conn_ack(*ep_p, cm_event,
                                   UCT_TCP_EP_CONN_STATE_CONNECTED);
        return 0;
    case UCT_TCP_CM_CONN_WAIT_REQ:
        ucs_error("tcp_ep %p: CM event for waiting REQ (%d) "
                  "must be sent along with ACK", *ep_p, cm_event);
        return 0;
    }

    ucs_error("tcp_ep %p: unknown CM event received %d", *ep_p, cm_event);
    return 0;
}

static ucs_status_t uct_tcp_cm_conn_complete(uct_tcp_ep_t *ep,
                                             unsigned *progress_count_p)
{
    ucs_status_t status;

    status = uct_tcp_cm_send_event(ep, UCT_TCP_CM_CONN_REQ);
    if (status != UCS_OK) {
        goto out;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_WAITING_ACK);
    uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVREAD, 0);

    ucs_assertv((ep->tx.length == 0) && (ep->tx.offset == 0) &&
                (ep->tx.buf == NULL), "ep=%p", ep);
out:
    if (progress_count_p != NULL) {
        *progress_count_p = (status == UCS_OK);
    }
    return status;
}

unsigned uct_tcp_cm_conn_progress(uct_tcp_ep_t *ep)
{
    unsigned progress_count;

    if (!ucs_socket_is_connected(ep->fd)) {
        ucs_error("tcp_ep %p: connection establishment for "
                  "socket fd %d was unsuccessful", ep, ep->fd);
        goto err;
    }

    uct_tcp_cm_conn_complete(ep, &progress_count);
    return progress_count;

err:
    uct_tcp_ep_set_failed(ep);
    return 0;
}

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    ucs_status_t status;

    if (ep->conn_retries++ > iface->config.max_conn_retries) {
        ucs_error("tcp_ep %p: reached maximum number of connection retries "
                  "(%u)", ep, iface->config.max_conn_retries);
        return UCS_ERR_TIMED_OUT;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTING);

    status = ucs_socket_connect(ep->fd, (const struct sockaddr*)&ep->peer_addr);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    } else if (status == UCS_INPROGRESS) {
        uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVWRITE, 0);
        return UCS_OK;
    }

    ucs_assert(status == UCS_OK);

    if (!iface->config.conn_nb) {
        status = ucs_sys_fcntl_modfl(ep->fd, O_NONBLOCK, 0);
        if (status != UCS_OK) {
            return status;
        }
    }

    return uct_tcp_cm_conn_complete(ep, NULL);
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

    if (!ucs_socket_is_connected(fd)) {
        ucs_warn("tcp_iface %p: connection establishment for socket fd %d "
                 "from %s to %s was unsuccessful", iface, fd,
                 ucs_sockaddr_str((const struct sockaddr*)&peer_addr,
                                  str_remote_addr, UCS_SOCKADDR_STRING_LEN),
                 ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                  str_local_addr, UCS_SOCKADDR_STRING_LEN));
        return UCS_ERR_UNREACHABLE;
    }

    status = uct_tcp_ep_init(iface, fd, NULL, &ep);
    if (status != UCS_OK) {
        return status;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER);
    uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVREAD, 0);

    ucs_debug("tcp_iface %p: accepted connection from "
              "%s on %s to tcp_ep %p (fd %d)", iface,
              ucs_sockaddr_str((const struct sockaddr*)peer_addr,
                               str_remote_addr, UCS_SOCKADDR_STRING_LEN),
              ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                               str_local_addr, UCS_SOCKADDR_STRING_LEN),
              ep, fd);
    return UCS_OK;
}
