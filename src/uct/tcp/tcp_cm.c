/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"
#include "tcp/tcp.h"

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

    old_conn_state = (uct_tcp_ep_conn_state_t)ep->conn_state;
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
        /* old_conn_state could be CONNECTING may happen when a peer is going
         * to use this EP with socket from accepted connection in case of
         * handling simultaneous connection establishment */
        ucs_assert(((old_conn_state == UCT_TCP_EP_CONN_STATE_CLOSED) &&
                    (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP))         ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING)  ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
                   (old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING));
        if (old_conn_state != UCT_TCP_EP_CONN_STATE_CLOSED) {
            /* Decrement iface's outstanding counter only in case of the
             * previous state is not CLOSED. If it is CLOSED it means that
             * iface's outstanding counter wasn't incremented prior */
            uct_tcp_iface_outstanding_dec(iface);
        }

        if (ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX) {
            /* Progress possibly pending TX operations */
            uct_tcp_ep_pending_queue_dispatch(ep);
        }
        break;
    case UCT_TCP_EP_CONN_STATE_CLOSED:
        ucs_assert(ep->events == 0);
        if (old_conn_state == UCT_TCP_EP_CONN_STATE_CLOSED) {
            return;
        }

        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) {
            uct_tcp_iface_outstanding_dec(iface);
        }

        if ((old_conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
            (old_conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER)) {
            /* Since ep::peer_addr is 0'ed, we have to print w/o peer's address */
            full_log = 0;
        }
        break;
    case UCT_TCP_EP_CONN_STATE_ACCEPTING:
        ucs_assert((old_conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER) ||
                   ((old_conn_state == UCT_TCP_EP_CONN_STATE_CLOSED) &&
                    (ep->conn_retries == 0) &&
                    (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP)));
        uct_tcp_iface_outstanding_inc(iface);
        /* fall through */
    default:
        ucs_assert((ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
                   (ep->conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER));
        /* Since ep::peer_addr is 0'ed and client's <address:port>
         * has already been logged, print w/o peer's address */
        full_log = 0;
        break;
    }

    if (full_log) {
        ucs_debug("tcp_ep %p: %s -> %s for the [%s]<->[%s]:%"PRIu64" connection %s",
                  ep, uct_tcp_ep_cm_state[old_conn_state].name,
                  uct_tcp_ep_cm_state[ep->conn_state].name,
                  ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                   str_local_addr, UCS_SOCKADDR_STRING_LEN),
                  ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                   str_remote_addr, UCS_SOCKADDR_STRING_LEN),
                  uct_tcp_ep_get_cm_id(ep),
                  uct_tcp_ep_ctx_caps_str(ep->flags, str_ctx_caps));
    } else {
        ucs_debug("tcp_ep %p: %s -> %s",
                  ep, uct_tcp_ep_cm_state[old_conn_state].name,
                  uct_tcp_ep_cm_state[ep->conn_state].name);
    }
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

    ucs_log(log_level, "tcp_ep %p: %s [%s]:%"PRIu64, ep, msg,
            ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                             str_addr, UCS_SOCKADDR_STRING_LEN),
            uct_tcp_ep_get_cm_id(ep));
}

ucs_status_t uct_tcp_cm_send_event(uct_tcp_ep_t *ep,
                                   uct_tcp_cm_conn_event_t event,
                                   int log_error)
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
                            UCT_TCP_CM_CONN_ACK)),
                "ep=%p", ep);
    ucs_assertv(!(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX) ||
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
    pkt_hdr->am_id  = UCT_TCP_EP_CM_AM_ID;
    pkt_hdr->length = cm_pkt_length;

    if (event == UCT_TCP_CM_CONN_REQ) {
        if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) {
            ucs_assert(magic_number_length == sizeof(uint64_t));
            *(uint64_t*)pkt_buf = UCT_TCP_MAGIC_NUMBER;
        }

        conn_pkt             = (uct_tcp_cm_conn_req_pkt_t*)(pkt_hdr + 1);
        conn_pkt->event      = UCT_TCP_CM_CONN_REQ;
        conn_pkt->flags      = (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP) ?
                               UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP : 0;
        conn_pkt->iface_addr = iface->config.ifaddr;
        conn_pkt->cm_id      = ep->cm_id;
        ucs_assert((conn_pkt->flags &
                    UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP) ||
                   (ep->cm_id.conn_sn < UCT_TCP_CM_CONN_SN_MAX));
    } else {
        /* CM events (except CONN_REQ) are not sent for EPs connected with
         * CONNECT_TO_EP connection method */
        ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));
        pkt_event            = (uct_tcp_cm_conn_event_t*)(pkt_hdr + 1);
        *pkt_event           = event;
    }

    status = ucs_socket_send(ep->fd, pkt_buf, pkt_length);
    if (status == UCS_OK) {
        uct_tcp_cm_trace_conn_pkt(ep, UCS_LOG_LEVEL_TRACE,
                                  "%s sent to", event);
    } else {
        ucs_assert(status != UCS_ERR_NO_PROGRESS);
        status = uct_tcp_ep_handle_io_err(ep, "send", status);
        uct_tcp_cm_trace_conn_pkt(ep,
                                  (log_error && (status != UCS_ERR_CANCELED)) ?
                                  UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR,
                                  "unable to send %s to", event);
    }
    return status;
}

static const void*
uct_tcp_cm_conn_match_get_address(const ucs_conn_match_elem_t *elem)
{
    const uct_tcp_ep_t *ep = ucs_container_of(elem, uct_tcp_ep_t, elem);

    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));
    return &ep->peer_addr;
}

static ucs_conn_sn_t
uct_tcp_cm_conn_match_get_conn_sn(const ucs_conn_match_elem_t *elem)
{
    const uct_tcp_ep_t *ep = ucs_container_of(elem, uct_tcp_ep_t, elem);

    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));
    return (ucs_conn_sn_t)ep->cm_id.conn_sn;
}

static const char*
uct_tcp_cm_conn_match_address_str(const ucs_conn_match_ctx_t *conn_match_ctx,
                                  const void *address, char *str,
                                  size_t max_size)
{
    return ucs_sockaddr_str((const struct sockaddr*)address,
                            str, ucs_min(max_size, UCS_SOCKADDR_STRING_LEN));
}

static void
uct_tcp_cm_conn_match_purge_cb(ucs_conn_match_ctx_t *conn_match_ctx,
                               ucs_conn_match_elem_t *elem)
{
    uct_tcp_ep_t *ep = ucs_container_of(elem, uct_tcp_ep_t, elem);

    /* EP was deleted from the connection matching context during cleanup
     * procedure, move EP to the iface's EP list to correctly destroy EP */
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));
    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX);
    ep->flags &= ~UCT_TCP_EP_FLAG_ON_MATCH_CTX;
    uct_tcp_iface_add_ep(ep);

    uct_tcp_ep_destroy_internal(&ep->super.super);
}

const ucs_conn_match_ops_t uct_tcp_cm_conn_match_ops = {
    .get_address = uct_tcp_cm_conn_match_get_address,
    .get_conn_sn = uct_tcp_cm_conn_match_get_conn_sn,
    .address_str = uct_tcp_cm_conn_match_address_str,
    .purge_cb    = uct_tcp_cm_conn_match_purge_cb
};

void uct_tcp_cm_ep_set_conn_sn(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);

    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));

    ep->cm_id.conn_sn = ucs_conn_match_get_next_sn(&iface->conn_match_ctx,
                                                   &ep->peer_addr);
}

uct_tcp_ep_t *uct_tcp_cm_get_ep(uct_tcp_iface_t *iface,
                                const struct sockaddr_in *dest_address,
                                ucs_conn_sn_t conn_sn,
                                uint8_t with_ctx_cap)
{
    ucs_conn_match_queue_type_t queue_type;
    ucs_conn_match_elem_t *elem;
    uct_tcp_ep_t *ep;
    int remove_from_ctx;

    ucs_assert(conn_sn < UCT_TCP_CM_CONN_SN_MAX);
    ucs_assert((with_ctx_cap == UCT_TCP_EP_FLAG_CTX_TYPE_TX) ||
               (with_ctx_cap == UCT_TCP_EP_FLAG_CTX_TYPE_RX));

    if (with_ctx_cap == UCT_TCP_EP_FLAG_CTX_TYPE_TX) {
        /* when getting CONN_REQ we search in both EXP and UNEXP queue.
         * The endpoint could be in EXP queue if it is already created from
         * API, or in UNEXP queue if the connection request message was
         * retransmitted */
        queue_type      = UCS_CONN_MATCH_QUEUE_ANY;
        remove_from_ctx = 0;
    } else {
        /* when creating new endpoint from API, search for the arrived
         * connection requests and remove from the connection matching
         * context, since the EP with RX-only capability will be destroyed
         * or re-used for the EP created through uct_ep_create() and
         * returned to the user (it will be inserted to expected queue) */
        queue_type      = UCS_CONN_MATCH_QUEUE_UNEXP;
        remove_from_ctx = 1;
    }

    elem = ucs_conn_match_get_elem(&iface->conn_match_ctx, dest_address,
                                   conn_sn, queue_type, remove_from_ctx);
    if (elem == NULL) {
        return NULL;
    }

    ep = ucs_container_of(elem, uct_tcp_ep_t, elem);
    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX);
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));

    if ((queue_type == UCS_CONN_MATCH_QUEUE_UNEXP) ||
        !(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX)) {
        ucs_assert(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_RX);
    }

    if (remove_from_ctx) {
        ucs_assert((ep->flags & UCT_TCP_EP_CTX_CAPS) ==
                   UCT_TCP_EP_FLAG_CTX_TYPE_RX);
        ep->flags &= ~UCT_TCP_EP_FLAG_ON_MATCH_CTX;
        /* The EP was removed from connection matching, move it to the EP list
         * on iface to be able to destroy it from EP cleanup correctly that
         * removes the EP from the iface's EP list (an EP has to be either on
         * matching context or in iface's EP list) */
        uct_tcp_iface_add_ep(ep);
    }

    return ep;
}

void uct_tcp_cm_insert_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep)
{
    uint8_t ctx_caps = ep->flags & UCT_TCP_EP_CTX_CAPS;

    ucs_assert(ep->cm_id.conn_sn < UCT_TCP_CM_CONN_SN_MAX);
    ucs_assert((ctx_caps & UCT_TCP_EP_FLAG_CTX_TYPE_TX) ||
               (ctx_caps == UCT_TCP_EP_FLAG_CTX_TYPE_RX));
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX));
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));

    ucs_conn_match_insert(&iface->conn_match_ctx, &ep->peer_addr,
                          ep->cm_id.conn_sn, &ep->elem,
                          (ctx_caps & UCT_TCP_EP_FLAG_CTX_TYPE_TX) ?
                          UCS_CONN_MATCH_QUEUE_EXP :
                          UCS_CONN_MATCH_QUEUE_UNEXP);

    ep->flags |= UCT_TCP_EP_FLAG_ON_MATCH_CTX;
}

void uct_tcp_cm_remove_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep)
{
    uint8_t ctx_caps = ep->flags & UCT_TCP_EP_CTX_CAPS;

    ucs_assert(ep->cm_id.conn_sn < UCT_TCP_CM_CONN_SN_MAX);
    ucs_assert(ctx_caps != 0);
    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX);
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));

    ucs_conn_match_remove_elem(&iface->conn_match_ctx, &ep->elem,
                               (ctx_caps & UCT_TCP_EP_FLAG_CTX_TYPE_TX) ?
                               UCS_CONN_MATCH_QUEUE_EXP :
                               UCS_CONN_MATCH_QUEUE_UNEXP);

    ep->flags &= ~UCT_TCP_EP_FLAG_ON_MATCH_CTX;
}

int uct_tcp_cm_ep_accept_conn(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    int cmp;
    ucs_status_t status;

    if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) {
        return 0;
    }

    cmp = ucs_sockaddr_cmp((const struct sockaddr*)&ep->peer_addr,
                           (const struct sockaddr*)&iface->config.ifaddr,
                           &status);
    ucs_assertv_always(status == UCS_OK, "ucs_sockaddr_cmp(%s, %s) failed",
                       ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                        str_remote_addr, UCS_SOCKADDR_STRING_LEN),
                       ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                        str_local_addr, UCS_SOCKADDR_STRING_LEN));

    /* Accept connection from a peer if the local iface address is greater
     * than peer's one */
    return cmp < 0;
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
    uct_tcp_ep_move_ctx_cap(accept_ep, connect_ep, UCT_TCP_EP_FLAG_CTX_TYPE_RX);

    /* 3. The EP allocated during accepting connection has to be destroyed
     *    upon return from this function (set its socket `fd` to -1 prior
     *    to avoid closing this socket) */
    uct_tcp_ep_mod_events(accept_ep, 0, UCS_EVENT_SET_EVREAD);
    accept_ep->fd = -1;
    accept_ep     = NULL;

    /* 4. Send ACK to the peer */
    event = UCT_TCP_CM_CONN_ACK;

    /* 5. - If found EP is still connecting, tie REQ with ACK and send
     *      it to the peer using new socket fd to ensure that the peer
     *      will be able to receive the data from us
     *    - If found EP is waiting ACK, tie WAIT_REQ with ACK and send
     *      it to the peer using new socket fd to ensure that the peer
     *      will wait for REQ and after receiving the REQ, peer will
     *      be able to receive the data from us */
    if ((connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
        (connect_ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) {
        event |= UCT_TCP_CM_CONN_REQ;
    }

    status = uct_tcp_cm_send_event(connect_ep, event, 0);
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
    unsigned progress_count = 0;

    if (!uct_tcp_cm_ep_accept_conn(connect_ep)) {
        /* Migrate RX from the EP allocated during accepting connection to
         * the found EP. */
        uct_tcp_ep_move_ctx_cap(accept_ep, connect_ep,
                                UCT_TCP_EP_FLAG_CTX_TYPE_RX);
        uct_tcp_ep_mod_events(connect_ep, UCS_EVENT_SET_EVREAD, 0);

        /* If the EP created through API is not connected yet, don't close
         * the fd from accepted connection to avoid possible connection
         * retries from the remote peer. Save the fd to the separate field
         * for futher destroying when the connection is established */
        if (connect_ep->conn_state != UCT_TCP_EP_CONN_STATE_CONNECTED) {
            uct_tcp_ep_mod_events(accept_ep, 0, UCS_EVENT_SET_EVREAD);
            ucs_assert(connect_ep->stale_fd == -1);
            connect_ep->stale_fd = accept_ep->fd;
            accept_ep->fd        = -1;
        }
    } else /* our iface address less than remote && we are not connected */ {
        /* Accept the remote connection and close the current one */
        progress_count = uct_tcp_cm_simult_conn_accept_remote_conn(accept_ep,
                                                                   connect_ep);
    }

    return progress_count;
}

static UCS_F_MAYBE_UNUSED int
uct_tcp_cm_verify_req_connected_ep(uct_tcp_ep_t *ep,
                                   const uct_tcp_cm_conn_req_pkt_t *cm_req_pkt)
{
    /* copy iface_addr to the local variable to avoid potential unaligned access
     * when need to get the address of uct_tcp_cm_conn_req_pkt_t::iface_addr,
     * since uct_tcp_cm_conn_req_pkt_t is a packed structure */
    struct sockaddr_in pkt_addr = cm_req_pkt->iface_addr;
    ucs_status_t status;

    return (ep->cm_id.conn_sn == cm_req_pkt->cm_id.conn_sn) &&
           !ucs_sockaddr_cmp((const struct sockaddr*)&ep->peer_addr,
                             (const struct sockaddr*)&pkt_addr,
                             &status) && (status == UCS_OK);
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
    int connect_to_self;

    ucs_assert(/* EP received the connection request after the TCP
                * connection was accepted */
               (ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
               /* EP is already connected to this peer (conn_sn and address
                * must be the same) */
               ((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) &&
                uct_tcp_cm_verify_req_connected_ep(ep, cm_req_pkt)));

    if (ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) {
        ep->peer_addr = cm_req_pkt->iface_addr;
        ep->cm_id     = cm_req_pkt->cm_id;
        if (cm_req_pkt->flags & UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP) {
            ep->flags |= UCT_TCP_EP_FLAG_CONNECT_TO_EP;
        }
    }

    uct_tcp_cm_trace_conn_pkt(ep, UCS_LOG_LEVEL_TRACE,
                              "%s received from", UCT_TCP_CM_CONN_REQ);

    uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_FLAG_CTX_TYPE_RX);

    if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) {
        return 0;
    }

    ucs_assertv(!(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX),
                "ep %p mustn't have TX cap", ep);

    connect_to_self = uct_tcp_ep_is_self(ep);
    if (connect_to_self) {
        goto accept_conn;
    }

    if (!(cm_req_pkt->flags & UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP)) {
        peer_ep = uct_tcp_cm_get_ep(iface, &ep->peer_addr,
                                    cm_req_pkt->cm_id.conn_sn,
                                    UCT_TCP_EP_FLAG_CTX_TYPE_TX);
        if (peer_ep != NULL) {
            progress_count = uct_tcp_cm_handle_simult_conn(iface, ep, peer_ep);
            ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX));
            goto out_destroy_ep;
        }
    } else {
        ucs_assert(uct_tcp_cm_ep_accept_conn(ep));
        peer_ep = uct_tcp_ep_ptr_map_retrieve(iface, ep->cm_id.ptr_map_key);
        if (peer_ep == NULL) {
            /* Local user-exposed EP was destroyed before receiving CONN_REQ
             * from a peer, drop the connection */
            goto out_destroy_ep;
        }

        peer_ep->peer_addr = ep->peer_addr;
        peer_ep->conn_retries++;
        uct_tcp_ep_add_ctx_cap(peer_ep, UCT_TCP_EP_FLAG_CTX_TYPE_TX);
        uct_tcp_ep_move_ctx_cap(ep, peer_ep, UCT_TCP_EP_FLAG_CTX_TYPE_RX);
        uct_tcp_ep_replace_ep(peer_ep, ep);
        uct_tcp_cm_change_conn_state(peer_ep,
                                     UCT_TCP_EP_CONN_STATE_CONNECTED);
        goto out_destroy_ep;
    }

accept_conn:
    ucs_assert(!(cm_req_pkt->flags &
                 UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP) || connect_to_self);

    /* Just accept this connection and make it operational for RX events */
    if (!(cm_req_pkt->flags & UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP)) {
        status = uct_tcp_cm_send_event(ep, UCT_TCP_CM_CONN_ACK, 1);
        if (status != UCS_OK) {
            goto out_destroy_ep;
        }
    }

    if (!connect_to_self) {
        uct_tcp_iface_remove_ep(ep);
        uct_tcp_cm_insert_ep(iface, ep);
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);
    return 1;

out_destroy_ep:
    if (!(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX)) {
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

    ucs_close_fd(&ep->stale_fd);
    if (ep->conn_state != new_conn_state) {
        uct_tcp_cm_change_conn_state(ep, new_conn_state);
    }
}

unsigned uct_tcp_cm_handle_conn_pkt(uct_tcp_ep_t **ep_p, void *pkt, uint32_t length)
{
    uct_tcp_cm_conn_event_t cm_event;
    uct_tcp_cm_conn_req_pkt_t *cm_req_pkt;

    ucs_assertv(length >= sizeof(cm_event), "ep=%p", *ep_p);

    cm_event = *((uct_tcp_cm_conn_event_t*)pkt);

    switch (cm_event) {
    case UCT_TCP_CM_CONN_REQ:
        /* Don't trace received CM packet here, because
         * EP doesn't contain the peer address */
        ucs_assertv(length == sizeof(*cm_req_pkt), "ep=%p", *ep_p);
        cm_req_pkt = (uct_tcp_cm_conn_req_pkt_t*)pkt;
        return uct_tcp_cm_handle_conn_req(ep_p, cm_req_pkt);
    case UCT_TCP_CM_CONN_ACK_WITH_REQ:
        uct_tcp_ep_add_ctx_cap(*ep_p, UCT_TCP_EP_FLAG_CTX_TYPE_RX);
        /* fall through */
    case UCT_TCP_CM_CONN_ACK:
        uct_tcp_cm_handle_conn_ack(*ep_p, cm_event,
                                   UCT_TCP_EP_CONN_STATE_CONNECTED);
        return 0;
    }

    ucs_error("tcp_ep %p: unknown CM event received %d", *ep_p, cm_event);
    return 0;
}

static void uct_tcp_cm_conn_complete(uct_tcp_ep_t *ep)
{
    ucs_status_t status;

    status = uct_tcp_cm_send_event(ep, UCT_TCP_CM_CONN_REQ, 1);
    if (status != UCS_OK) {
        /* error handling was done inside sending event operation */
        return;
    }

    if (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP) {
        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);
    } else {
        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_WAITING_ACK);
    }

    uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVREAD, 0);

    ucs_assertv((ep->tx.length == 0) && (ep->tx.offset == 0) &&
                (ep->tx.buf == NULL), "ep=%p", ep);
}

unsigned uct_tcp_cm_conn_progress(void *arg)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;

    if (!ucs_socket_is_connected(ep->fd)) {
        ucs_error("tcp_ep %p: connection establishment for "
                  "socket fd %d was unsuccessful", ep, ep->fd);
        goto err;
    }

    uct_tcp_cm_conn_complete(ep);
    return 1;

err:
    uct_tcp_ep_set_failed(ep);
    return 0;
}

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    ucs_status_t status;

    ep->conn_retries++;
    if (ep->conn_retries > iface->config.max_conn_retries) {
        ucs_error("tcp_ep %p: reached maximum number of connection retries "
                  "(%u)", ep, iface->config.max_conn_retries);
        return UCS_ERR_TIMED_OUT;
    }

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTING);

    status = ucs_socket_connect(ep->fd, (const struct sockaddr*)&ep->peer_addr);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    } else if (status == UCS_INPROGRESS) {
        ucs_assert(iface->config.conn_nb);
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

    uct_tcp_cm_conn_complete(ep);
    return UCS_OK;
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

    /* set non-blocking flag, since this is a fd from accept(), i.e.
     * connection was already established */
    status = uct_tcp_iface_set_sockopt(iface, fd, 1);
    if (status != UCS_OK) {
        return status;
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
