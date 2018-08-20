/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

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

static UCS_CLASS_INIT_FUNC(uct_tcp_ep_t, uct_tcp_iface_t *iface,
                           int fd, const struct sockaddr_in *dest_addr)
{
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

        /* TODO use non-blocking connect */
        status = uct_tcp_socket_connect(self->fd, dest_addr);
        if (status != UCS_OK) {
            goto err_close;
        }
    } else {
        self->fd = fd;
    }

    status = ucs_sys_fcntl_modfl(self->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close;
    }

    status = uct_tcp_iface_set_sockopt(iface, self->fd);
    if (status != UCS_OK) {
        goto err_close;
    }

    uct_tcp_ep_epoll_ctl(self, EPOLL_CTL_ADD);

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

ucs_status_t uct_tcp_ep_create_connected(uct_iface_t *tl_iface,
                                         const uct_device_addr_t *dev_addr,
                                         const uct_iface_addr_t *iface_addr,
                                         uct_ep_h *ep_p)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    uct_tcp_ep_t *tcp_ep = NULL;
    struct sockaddr_in dest_addr;
    ucs_status_t status;

    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port   = *(in_port_t*)iface_addr;
    dest_addr.sin_addr   = *(struct in_addr*)dev_addr;

    /* TODO try to reuse existing connection */
    status = uct_tcp_ep_create(iface, -1, &dest_addr, &tcp_ep);
    if (status == UCS_OK) {
        ucs_debug("tcp_ep %p: connected to %s:%d", tcp_ep,
                  inet_ntoa(dest_addr.sin_addr), ntohs(dest_addr.sin_port));
        *ep_p = &tcp_ep->super.super;
    }
    return status;
}

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, uint32_t add, uint32_t remove)
{
    int new_events = (ep->events | add) & ~remove;

    if (new_events != ep->events) {
        ep->events = new_events;
        ucs_trace("tcp_ep %p: set events to %c%c", ep,
                  (new_events & EPOLLIN)  ? 'i' : '-',
                  (new_events & EPOLLOUT) ? 'o' : '-');
        uct_tcp_ep_epoll_ctl(ep, EPOLL_CTL_MOD);
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
    uct_pending_req_priv_t *priv;
    unsigned count = 0;

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

ucs_status_t uct_tcp_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);

    if (uct_tcp_ep_can_send(ep)) {
        return UCS_ERR_BUSY;
    }

    uct_pending_req_push(&ep->pending_q, req);
    return UCS_OK;
}

void uct_tcp_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    uct_pending_req_priv_t *priv;

    uct_pending_queue_purge(priv, &ep->pending_q, 1, cb, arg);
}

ucs_status_t uct_tcp_ep_flush(uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);

    if (!uct_tcp_ep_can_send(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    ucs_assert(ucs_queue_is_empty(&ep->pending_q));
    UCT_TL_EP_STAT_FLUSH(&ep->super);
    return UCS_OK;
}

