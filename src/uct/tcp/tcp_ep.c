/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>

#define UCT_TCP_AM_SHORT_PACK_DATA(_pack_f, _target_buf, _target_length, \
                                   _am_payload, _payload_length,  _am_header) \
    do { \
        *((uint64_t*)(_target_buf)) = (_am_header); \
        _pack_f((uint8_t*)(_target_buf) + sizeof(_am_header), \
                _am_payload, _payload_length); \
        _target_length = sizeof(_am_header) + _payload_length; \
    } while (0)

#define UCT_TCP_AM_BCOPY_PACK_DATA(_pack_f, _target_buf, _target_length, \
                                   _am_arg, ...) \
    _target_length = _pack_f(_target_buf, _am_arg)

#define UCT_TCP_AM_PREPARE(_ep, _id, _len_thr, _hdr, \
                           _pack_f, _am_payload, _payload_length, \
                           _am_header, _method, _name) \
    do { \
        UCT_CHECK_AM_ID(_id); \
        \
        if (!uct_tcp_ep_can_send(_ep)) { \
            return UCS_ERR_NO_RESOURCE; \
        } \
        \
        (_hdr)        = (_ep)->tx.buf; \
        (_hdr)->am_id = _id; \
        \
        UCT_TCP_AM_ ## _method ## _PACK_DATA(_pack_f, (_hdr) + 1, (_hdr)->length, \
                                             _am_payload, _payload_length, \
                                             _am_header); \
        \
        UCT_CHECK_LENGTH((_hdr)->length, 0, _len_thr, _name); \
        UCT_TL_EP_STAT_OP(&(_ep)->super, AM, _method, (_hdr)->length); \
    } while (0)


static void uct_tcp_ep_epoll_ctl(uct_tcp_ep_t *ep, int op)
{
    uct_tcp_iface_t *iface         = ucs_derived_of(ep->super.super.iface,
                                                    uct_tcp_iface_t);
    struct epoll_event epoll_event = {
        .data.ptr                  = ep,
        .events                    = ep->events,
    };
    int ret;

    ret = epoll_ctl(iface->epfd, op, ep->fd, &epoll_event);
    if (ret < 0) {
        ucs_fatal("epoll_ctl(epfd=%d, op=%d, fd=%d) failed: %m",
                  iface->epfd, op, ep->fd);
    }
}

static inline int uct_tcp_ep_can_send(uct_tcp_ep_t *ep)
{
    ucs_assert(ep->tx.offset <= ep->tx.length);
    /* TODO optimize to allow partial sends/message coalescing */
    return ep->tx.length == 0;
}

static inline void uct_tcp_ep_ctx_rewind(uct_tcp_ep_ctx_t *ctx)
{
    ctx->offset = 0;
    ctx->length = 0;
}

static ucs_status_t uct_tcp_ep_ctx_init(uct_tcp_iface_t *iface,
                                        uct_tcp_ep_ctx_t *ctx)
{
    ucs_assertv(ctx->buf == NULL, "ctx=%p", ctx);

    ctx->buf = ucs_malloc(ucs_max(iface->config.buf_size,
                                  iface->config.short_size), "tcp_buf");
    if (ctx->buf == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    uct_tcp_ep_ctx_rewind(ctx);

    return UCS_OK;
}

static void uct_tcp_ep_ctx_cleanup(uct_tcp_ep_ctx_t *ctx)
{
    ucs_free(ctx->buf);

    ctx->buf = NULL;

    uct_tcp_ep_ctx_rewind(ctx);
}

void uct_tcp_ep_ctx_migrate(uct_tcp_ep_ctx_t *to_ctx, uct_tcp_ep_ctx_t *from_ctx)
{
    ucs_assertv(to_ctx->buf == NULL, "to_ctx=%p", to_ctx);
    ucs_assertv(from_ctx->buf != NULL, "from_ctx=%p", from_ctx);

    *to_ctx = *from_ctx;

    from_ctx->buf = NULL;

    uct_tcp_ep_ctx_cleanup(from_ctx);
}

static void uct_tcp_ep_addr_cleanup(ucs_sock_addr_t *sock_addr)
{
    ucs_free((void*)sock_addr->addr);

    sock_addr->addr    = NULL;
    sock_addr->addrlen = 0;
}

static ucs_status_t uct_tcp_ep_addr_init(ucs_sock_addr_t *sock_addr,
                                         const struct sockaddr *addr)
{
    /* TODO: handle IPv4 and IPv6 */
    socklen_t addr_len = sizeof(struct sockaddr_in);
    struct sockaddr *new_addr;

    if (addr == NULL) {
        sock_addr->addr    = NULL;
        sock_addr->addrlen = 0;
    } else {
        new_addr = ucs_malloc(addr_len, "sock_addr");
        if (new_addr == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        sock_addr->addr    = new_addr;
        sock_addr->addrlen = addr_len;
    }

    return UCS_OK;
}

static void uct_tcp_ep_close_fd(int *fd_p)
{
    if (*fd_p != -1) {
        close(*fd_p);
        *fd_p = -1;
    }
}

/* Must be called with `iface::worker::async` blocked */
static unsigned uct_tcp_ep_in_iface(uct_tcp_ep_t *ep)
{
    return !ucs_list_is_empty(&ep->list);
}

/* Must be called with `iface::worker::async` blocked */
static void uct_tcp_ep_del_from_iface(uct_tcp_iface_t *iface,
                                      uct_tcp_ep_t *ep)
{
    if (uct_tcp_ep_in_iface(ep)) {
        ucs_list_del(&ep->list);
    }
}

/* Must be called with `iface::worker::async` blocked */
static void uct_tcp_ep_add_to_iface(uct_tcp_iface_t *iface,
                                    uct_tcp_ep_t *ep)
{
    ucs_list_add_tail(&iface->ep_list, &ep->list);
}

static void uct_tcp_ep_cleanup(uct_tcp_ep_t *ep)
{ 
    uct_tcp_ep_addr_cleanup(&ep->peer_addr);

    uct_tcp_ep_ctx_cleanup(&ep->tx);
    uct_tcp_ep_ctx_cleanup(&ep->rx);

    uct_tcp_ep_close_fd(&ep->fd);
}

static UCS_CLASS_INIT_FUNC(uct_tcp_ep_t, uct_tcp_iface_t *iface,
                           int fd, const struct sockaddr *dest_addr)
{
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)

    self->peer_addr.addr = NULL;

    status = uct_tcp_ep_addr_init(&self->peer_addr, dest_addr);
    if (status != UCS_OK) {
        return UCS_OK;
    }

    self->tx.buf      = NULL;
    self->tx.progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero;

    self->rx.buf      = NULL;
    self->rx.progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero;

    ucs_queue_head_init(&self->pending_q);
    self->events = 0;
    self->fd     = -1;
    ucs_list_head_init(&self->list);

    if (fd == -1) {
        status = ucs_socket_create(AF_INET, SOCK_STREAM, &self->fd);
        if (status != UCS_OK) {
            goto err_cleanup;
        }

        /* TODO use non-blocking connect */
        status = ucs_socket_connect(self->fd, dest_addr);
        if (status != UCS_OK) {
            goto err_cleanup;
        }

        status = uct_tcp_ep_ctx_init(iface, &self->tx);
        if (status != UCS_OK) {
            goto err_cleanup;
        }

        self->tx.progress = uct_tcp_ep_progress_tx;
    } else {
        self->fd = fd;

        status = uct_tcp_ep_ctx_init(iface, &self->rx);
        if (status != UCS_OK) {
            /* to be closed by this function caller */
            self->fd = -1;
            goto err_cleanup;
        }

        self->rx.progress = uct_tcp_ep_progress_rx;
    }

    status = ucs_sys_fcntl_modfl(self->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        if (fd != -1) {
            /* to be closed by this function caller */
            self->fd = -1;
        }
        goto err_cleanup;
    }

    status = uct_tcp_iface_set_sockopt(iface, self->fd);
    if (status != UCS_OK) {
        if (fd != -1) {
            /* to be closed by this function caller */
            self->fd = -1;
        }
        goto err_cleanup;
    }

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    uct_tcp_ep_add_to_iface(iface, self);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_debug("tcp_ep %p: created on iface %p, fd %d", self, iface, self->fd);
    return UCS_OK;

err_cleanup:
    uct_tcp_ep_cleanup(self);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_ep_t)
{
    uct_tcp_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                            uct_tcp_iface_t);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    uct_tcp_ep_del_from_iface(iface, self);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    uct_tcp_ep_cleanup(self);

    ucs_debug("tcp_ep %p: destroyed on iface %p", self, iface);
}

UCS_CLASS_DEFINE(uct_tcp_ep_t, uct_base_ep_t);

UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_tcp_ep_create, uct_tcp_ep_t, uct_tcp_ep_t,
                                uct_tcp_iface_t*, int,
                                const struct sockaddr*)
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
    status = uct_tcp_ep_create(iface, -1, (struct sockaddr*)&dest_addr, &tcp_ep);
    if (status == UCS_OK) {
        ucs_debug("tcp_ep %p: connected to %s:%d", tcp_ep,
                  inet_ntoa(dest_addr.sin_addr), ntohs(dest_addr.sin_port));
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
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    size_t send_length;
    ucs_status_t status;

    send_length = ep->tx.length - ep->tx.offset;
    ucs_assert(send_length > 0);

    status = uct_tcp_send(ep->fd, ep->tx.buf + ep->tx.offset, &send_length);
    if (status < 0) {
        return 0;
    }

    ucs_trace_data("tcp_ep %p: sent %zu bytes", ep, send_length);

    iface->outstanding -= send_length;
    ep->tx.offset      += send_length;
    if (ep->tx.offset == ep->tx.length) {
        uct_tcp_ep_ctx_rewind(&ep->tx);
    }

    return send_length > 0;
}

unsigned uct_tcp_ep_progress_tx(uct_tcp_ep_t *ep)
{
    unsigned                     count = 0;
    uct_pending_req_priv_queue_t *priv;

    ucs_trace_func("ep=%p", ep);

    if (ep->tx.length > 0) {
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
    recv_length = iface->config.buf_size - ep->rx.length;
    ucs_assertv(recv_length > 0, "ep=%p", ep);

    status = uct_tcp_recv(ep->fd, ep->rx.buf + ep->rx.length, &recv_length);
    if (status != UCS_OK) {
        if (status == UCS_ERR_CANCELED) {
            ucs_debug("tcp_ep %p: remote disconnected", ep);
            uct_tcp_ep_mod_events(ep, 0, EPOLLIN);
            uct_tcp_ep_destroy(&ep->super.super);
        }
        return 0;
    }

    ep->rx.length += recv_length;
    ucs_trace_data("tcp_ep %p: recvd %zu bytes", ep, recv_length);

    /* Parse received active messages */
    while ((remainder = ep->rx.length - ep->rx.offset) >= sizeof(*hdr)) {
        hdr = ep->rx.buf + ep->rx.offset;
        ucs_assert(hdr->length <= (iface->config.buf_size - sizeof(uct_tcp_am_hdr_t)));

        if (remainder < sizeof(*hdr) + hdr->length) {
            break;
        }

        /* Full message was received */
        ep->rx.offset += sizeof(*hdr) + hdr->length;

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
    memmove(ep->rx.buf, ep->rx.buf + ep->rx.offset, remainder);
    ep->rx.offset = 0;
    ep->rx.length = remainder;

    return recv_length > 0;
}


static inline void uct_tcp_ep_am_send(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep,
                                      const uct_tcp_am_hdr_t *hdr)
{
    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, hdr->am_id,
                       hdr + 1, hdr->length, "SEND fd %d", ep->fd);

    ep->tx.length       = sizeof(*hdr) + hdr->length;
    iface->outstanding += ep->tx.length;

    uct_tcp_ep_send(ep);
    if (ep->tx.length > 0) {
        uct_tcp_ep_mod_events(ep, EPOLLOUT, 0);
    }
}

ucs_status_t uct_tcp_ep_am_short(uct_ep_h uct_ep, uint8_t am_id, uint64_t header,
                                 const void *payload, unsigned length)
{
    uct_tcp_ep_t *ep       = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_am_hdr_t *hdr;

    UCT_TCP_AM_PREPARE(ep, am_id, iface->config.short_size - sizeof(*hdr),
                       hdr, memcpy, payload, length, header, SHORT, "am_short");

    uct_tcp_ep_am_send(iface, ep, hdr);
    return UCS_OK;
}

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_tcp_ep_t *ep = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_am_hdr_t *hdr;

    UCT_TCP_AM_PREPARE(ep, am_id, iface->config.buf_size - sizeof(*hdr),
                       hdr, pack_cb, arg, NULL, NULL, BCOPY, "am_bcopy");

    uct_tcp_ep_am_send(iface, ep, hdr);
    return hdr->length;
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

