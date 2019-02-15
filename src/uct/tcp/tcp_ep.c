/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <sys/poll.h>
#include <ucs/async/async.h>


static void uct_tcp_ep_epoll_ctl(uct_tcp_ep_t *ep, int op)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    struct epoll_event epoll_event;
    int ret;

    memset(&epoll_event, 0, sizeof(epoll_event));
    epoll_event.data.ptr = ep;
    epoll_event.events   = ep->events;
    ret = epoll_ctl(iface->epfd, op, ep->fd, &epoll_event);
    if (ret < 0) {
        ucs_fatal("epoll_ctl(epfd=%d, op=%d, fd=%d) failed: %m",
                  iface->epfd, op, ep->fd);
    }
}

static inline int uct_tcp_ep_can_send(uct_tcp_ep_t *ep)
{
    ucs_assert(ep->offset <= ep->length);
    /* TODO optimize to allow partial sends/message coalescing */
    return ep->length == 0;
}

static inline ucs_status_t uct_tcp_ep_connected(uct_tcp_ep_t *ep)
{
    if (ucs_likely(ep->conn_state == UCT_TCP_EP_CONN_CONNECTED)) {
        return UCS_OK;
    } else if (ep->conn_state == UCT_TCP_EP_CONN_IN_PROGRESS) {
        return UCS_ERR_NO_RESOURCE;
    }
    return UCS_ERR_UNREACHABLE;
}

static void uct_tcp_ep_change_conn_state(uct_tcp_ep_t *ep,
                                         uct_tcp_ep_conn_state_t new_conn_state)
{
    char *str_addr;

    ep->conn_state = new_conn_state;

    str_addr = uct_tcp_sockaddr_2_string(&ep->peer, NULL, NULL);
    if (!str_addr) {
        return;
    }

    switch(ep->conn_state) {
    case UCT_TCP_EP_CONN_IN_PROGRESS:
        ucs_debug("tcp_ep %p: connection establishment in progress to %s",
                  ep, str_addr);
        break;
    case UCT_TCP_EP_CONN_CONNECTED:
        ucs_debug("tcp_ep %p: connected to %s",
                  ep, str_addr);
        break;
    case UCT_TCP_EP_CONN_REFUSED:
        ucs_debug("tcp_ep %p: connection refused to %s",
                  ep, str_addr);
        break;
    }

    ucs_free(str_addr);
}

static void uct_tcp_ep_connect_handler(int conn_fd, void *arg)
{
    uct_tcp_ep_t *ep = arg;
    socklen_t conn_status_sz = sizeof(int);
    int ret, conn_status;
    ucs_status_t status;

    ret = getsockopt(conn_fd, SOL_SOCKET, SO_ERROR,
                     &conn_status, &conn_status_sz);
    if (ret < 0) {
        ucs_error("Failed to get SO_ERROR on fd %d: %m", conn_fd);
        return;
    }

    status = ucs_async_remove_handler(conn_fd, 0);
    if (status != UCS_OK) {
        ucs_warn("failed to remove handler for client socket fd=%d", conn_fd);
    }

    if (conn_status != 0) {
        uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_REFUSED);
        ucs_error("Non-blocking connect(%s:%d) failed: %d",
                  inet_ntoa(ep->peer.sin_addr),
                  ntohs(ep->peer.sin_port), conn_status);
        return;
    }

    uct_tcp_ep_change_conn_state(ep, UCT_TCP_EP_CONN_CONNECTED);

    uct_tcp_ep_mod_events(ep, EPOLLOUT, 0);
}

static UCS_CLASS_INIT_FUNC(uct_tcp_ep_t, uct_tcp_iface_t *iface,
                           int fd, const struct sockaddr_in *dest_addr)
{
    socklen_t addr_len;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)

    self->buf = ucs_malloc(iface->config.buf_size, "tcp_buf");
    if (self->buf == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    self->events = 0;
    self->offset = 0;
    self->length = 0;
    ucs_queue_head_init(&self->pending_q);

    if (fd == -1) {
        status = ucs_tcpip_socket_create(&self->fd);
        if (status != UCS_OK) {
            goto err;
        }

        self->peer = *dest_addr;

        status = ucs_sys_fcntl_modfl(self->fd, O_NONBLOCK, 0);
        if (status != UCS_OK) {
            goto err_close;
        }

        status = uct_tcp_socket_connect(self->fd, dest_addr);
        if (status == UCS_INPROGRESS) {
            /* Register event handler for handling connection
             * establishment procedure */
            UCS_ASYNC_BLOCK(iface->super.worker->async);
            status = ucs_async_set_event_handler(iface->super.worker->async->mode,
                                                 self->fd, POLLOUT|POLLERR,
                                                 uct_tcp_ep_connect_handler, self,
                                                 iface->super.worker->async);
            UCS_ASYNC_UNBLOCK(iface->super.worker->async);
            if (status != UCS_OK) {
                goto err_close;
            }

            uct_tcp_ep_change_conn_state(self, UCT_TCP_EP_CONN_IN_PROGRESS);
        } else if (status != UCS_OK) {
            goto err_close;
        } else {
            uct_tcp_ep_change_conn_state(self, UCT_TCP_EP_CONN_CONNECTED);
        }
    } else {
        self->fd = fd;

        status = ucs_sys_fcntl_modfl(self->fd, O_NONBLOCK, 0);
        if (status != UCS_OK) {
            goto err_close;
        }

        addr_len = sizeof(&self->peer);
        if (getpeername(fd, (struct sockaddr *)&self->peer, &addr_len) < 0) {
            goto err_close;
        }

        uct_tcp_ep_change_conn_state(self, UCT_TCP_EP_CONN_CONNECTED);
    }

    status = uct_tcp_iface_set_sockopt(iface, self->fd);
    if (status != UCS_OK) {
        goto err_close;
    }

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->ep_list, &self->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_debug("tcp_ep %p: created on iface %p, fd %d", self, iface, self->fd);
    return UCS_OK;

err_close:
    close(self->fd);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_ep_t)
{
    uct_tcp_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                            uct_tcp_iface_t);

    ucs_debug("tcp_ep %p: destroying", self);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_del(&self->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_free(self->buf);
    close(self->fd);
}

UCS_CLASS_DEFINE(uct_tcp_ep_t, uct_base_ep_t);

UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_tcp_ep_create, uct_tcp_ep_t, uct_tcp_ep_t,
                                uct_tcp_iface_t*, int,
                                const struct sockaddr_in*)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(uct_tcp_ep_destroy, uct_tcp_ep_t, uct_ep_t)

ucs_status_t uct_tcp_ep_create_connected(const uct_ep_params_t *params,
                                         uct_ep_h *ep_p)
{
    uct_tcp_iface_t *iface = ucs_derived_of(params->iface, uct_tcp_iface_t);
    uct_tcp_ep_t *tcp_ep = NULL;
    struct sockaddr_in dest_addr;
    ucs_status_t status;

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port   = *(in_port_t*)params->iface_addr;
    dest_addr.sin_addr   = *(struct in_addr*)params->dev_addr;

    /* TODO try to reuse existing connection */
    status = uct_tcp_ep_create(iface, -1, &dest_addr, &tcp_ep);
    if (status == UCS_OK) {
        *ep_p = &tcp_ep->super.super;
    }
    return status;
}

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, uint32_t add, uint32_t remove)
{
    int old_events = ep->events;
    int new_events = (ep->events | add) & ~remove;

    if (new_events != ep->events) {
        ep->events = new_events;
        ucs_trace("tcp_ep %p: set events to %c%c", ep,
                  (new_events & EPOLLIN)  ? 'i' : '-',
                  (new_events & EPOLLOUT) ? 'o' : '-');
        if (new_events == 0) {
            uct_tcp_ep_epoll_ctl(ep, EPOLL_CTL_DEL);
        } else if (old_events != 0) {
            uct_tcp_ep_epoll_ctl(ep, EPOLL_CTL_MOD);
        } else {
            uct_tcp_ep_epoll_ctl(ep, EPOLL_CTL_ADD);
        }
    }
}

static unsigned uct_tcp_ep_send(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_tcp_iface_t);
    size_t send_length;
    ucs_status_t status;

    send_length = ep->length - ep->offset;
    ucs_assert(send_length > 0);

    status = uct_tcp_send(ep->fd, ep->buf + ep->offset, &send_length);
    if (status < 0) {
        return 0;
    }

    ucs_trace_data("tcp_ep %p: sent %zu bytes", ep, send_length);

    iface->outstanding -= send_length;
    ep->offset         += send_length;
    if (ep->offset == ep->length) {
        ep->offset = 0;
        ep->length = 0;
    }

    return send_length > 0;
}

unsigned uct_tcp_ep_progress_tx(uct_tcp_ep_t *ep)
{
    unsigned                     count = 0;
    uct_pending_req_priv_queue_t *priv;

    ucs_trace_func("ep=%p", ep);

    if (ep->length > 0) {
        count += uct_tcp_ep_send(ep);
    }

    uct_pending_queue_dispatch(priv, &ep->pending_q, uct_tcp_ep_can_send(ep));

    if (uct_tcp_ep_can_send(ep)) {
        ucs_assert(ucs_queue_is_empty(&ep->pending_q));
        uct_tcp_ep_mod_events(ep, 0, EPOLLOUT);
    }

    return count;
}

unsigned uct_tcp_ep_progress_rx(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_am_hdr_t *hdr;
    ucs_status_t status;
    size_t recv_length;
    ssize_t remainder;

    ucs_trace_func("ep=%p", ep);

    /* Receive next chunk of data */
    recv_length = iface->config.buf_size - ep->length;
    ucs_assertv(recv_length > 0, "ep=%p", ep);

    status = uct_tcp_recv(ep->fd, ep->buf + ep->length, &recv_length);
    if (status != UCS_OK) {
        if (status == UCS_ERR_CANCELED) {
            ucs_debug("tcp_ep %p: remote disconnected", ep);
            uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
            uct_tcp_ep_destroy(&ep->super.super);
        }
        return 0;
    }

    ep->length += recv_length;
    ucs_trace_data("tcp_ep %p: recvd %zu bytes", ep, recv_length);

    /* Parse received active messages */
    while ((remainder = ep->length - ep->offset) >= sizeof(*hdr)) {
        hdr = ep->buf + ep->offset;
        ucs_assert(hdr->length <= (iface->config.buf_size - sizeof(uct_tcp_am_hdr_t)));

        if (remainder < sizeof(*hdr) + hdr->length) {
            break;
        }

        /* Full message was received */
        ep->offset += sizeof(*hdr) + hdr->length;

        if (hdr->am_id >= UCT_AM_ID_MAX) {
            ucs_error("invalid am id: %d", hdr->am_id);
            continue;
        }

        uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_RECV, hdr->am_id,
                           hdr + 1, hdr->length, "RECV fd %d", ep->fd);
        uct_iface_invoke_am(&iface->super, hdr->am_id, hdr + 1,
                            hdr->length, 0);
    }

    /* Move the remaining data to the beginning of the buffer
     * TODO avoid extra copy on partial receive
     */
    ucs_assert(remainder >= 0);
    memmove(ep->buf, ep->buf + ep->offset, remainder);
    ep->offset = 0;
    ep->length = remainder;

    return recv_length > 0;
}

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_tcp_ep_t *ep = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_am_hdr_t *hdr;
    size_t packed_length;
    ucs_status_t status;

    status = uct_tcp_ep_connected(ep);
    if (ucs_likely(status != UCS_OK)) {
        return status;
    }

    UCT_CHECK_AM_ID(am_id);

    if (!uct_tcp_ep_can_send(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    hdr         = ep->buf;
    hdr->am_id  = am_id;
    hdr->length = packed_length = pack_cb(hdr + 1, arg);
    ep->length  = sizeof(*hdr) + packed_length;

    UCT_CHECK_LENGTH(hdr->length, 0,
                     iface->config.buf_size - sizeof(uct_tcp_am_hdr_t),
                     "am_bcopy");
    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, hdr->length);
    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, hdr->am_id,
                       hdr + 1, hdr->length, "SEND fd %d", ep->fd);
    iface->outstanding += ep->length;

    uct_tcp_ep_send(ep);
    if (ep->length > 0) {
        uct_tcp_ep_mod_events(ep, EPOLLOUT, 0);
    }
    return packed_length;
}

ucs_status_t uct_tcp_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req,
                                    unsigned flags)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);

    if (uct_tcp_ep_can_send(ep)) {
        return UCS_ERR_BUSY;
    }

    uct_pending_req_queue_push(&ep->pending_q, req);
    UCT_TL_EP_STAT_PEND(&ep->super);
    return UCS_OK;
}

void uct_tcp_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_tcp_ep_t                 *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    uct_pending_req_priv_queue_t *priv;

    uct_pending_queue_purge(priv, &ep->pending_q, 1, cb, arg);
}

ucs_status_t uct_tcp_ep_flush(uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);

    if (!uct_tcp_ep_can_send(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    UCT_TL_EP_STAT_FLUSH(&ep->super);
    return UCS_OK;
}

