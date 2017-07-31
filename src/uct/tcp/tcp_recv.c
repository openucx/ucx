/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/arch/atomic.h>
#include <ucs/async/async.h>
#include <sys/poll.h>


static inline void *
uct_tcp_recv_data(uct_tcp_iface_t *iface, uct_tcp_recv_sock_t *rsock)
{
    return (void*)uct_tcp_desc_hdr(iface, rsock->desc) + rsock->offset;
}

static inline void *
uct_tcp_desc_tl_desc(uct_tcp_iface_t *iface, uct_tcp_am_desc_t *desc)
{
    return (void*)desc + iface->config.headroom_offset;
}

static void uct_tcp_iface_invoke_am(uct_tcp_iface_t *iface,
                                    uct_tcp_am_desc_t *desc)
{
    uct_tcp_am_hdr_t *hdr = uct_tcp_desc_hdr(iface, desc);
    ucs_status_t status;
    void *tl_desc;

    status = uct_iface_invoke_am(&iface->super, hdr->am_id, hdr + 1,
                                 hdr->length, UCT_CB_FLAG_DESC);
    if (status == UCS_OK) {
        ucs_mpool_put(desc);
    } else if (status == UCS_INPROGRESS) {
        tl_desc = uct_tcp_desc_tl_desc(iface, desc);
        uct_recv_desc(tl_desc) = &iface->release_desc;
    } else {
        ucs_error("unexpected error from active message hander: %s",
                  ucs_status_string(status));
    }
}

static ucs_status_t uct_tcp_iface_recv_sock_add(uct_tcp_iface_t *iface, int fd,
                                                uct_tcp_recv_sock_t *rsock)
{
    int hash_extra_status;
    khiter_t hash_it;

    hash_it = kh_put(uct_tcp_fd_hash, &iface->fd_hash, fd, &hash_extra_status);
    if (hash_extra_status == -1) {
        ucs_error("failed to add socket to hash");
        return UCS_ERR_NO_MEMORY;
    } else {
        ucs_assert_always(hash_it != kh_end(&iface->fd_hash));
        if (hash_extra_status == 0) {
            ucs_error("TCP rsock %d already exists [old: %p new: %p]", fd,
                      kh_value(&iface->fd_hash, hash_it), rsock);
            return UCS_ERR_ALREADY_EXISTS;
        } else {
            ucs_trace("added rsock %d [%p] to hash", fd, rsock);
            kh_value(&iface->fd_hash, hash_it) = rsock;
            return UCS_OK;
        }
    }
}

static void uct_tcp_iface_recv_sock_remove(uct_tcp_iface_t *iface, int fd)
{
    khiter_t hash_it = kh_get(uct_tcp_fd_hash, &iface->fd_hash, fd);
    if (hash_it != kh_end(&iface->fd_hash)) {
        ucs_trace("removing rsock %d [%p] from hash", fd,
                  kh_value(&iface->fd_hash, hash_it));
        kh_del(uct_tcp_fd_hash, &iface->fd_hash, hash_it);
    }
}

static uct_tcp_recv_sock_t *uct_tcp_iface_recv_sock_get(uct_tcp_iface_t *iface,
                                                        int fd)
{
    khiter_t hash_it = kh_get(uct_tcp_fd_hash, &iface->fd_hash, fd);
    if (hash_it != kh_end(&iface->fd_hash)) {
        return kh_value(&iface->fd_hash, hash_it);
    } else {
        return NULL;
    }
}

ucs_status_t uct_tcp_iface_connection_accepted(uct_tcp_iface_t *iface, int fd)
{
    struct epoll_event epoll_event;
    uct_tcp_recv_sock_t *rsock;
    ucs_status_t status;
    int ret;

    status = ucs_sys_fcntl_modfl(fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close;
    }

    status = uct_tcp_iface_set_sockopt(iface, fd);
    if (status != UCS_OK) {
        goto err_close;
    }

    rsock = ucs_malloc(sizeof(*rsock), "tcp_recv");
    if (rsock == NULL) {
        ucs_error("Failed to allocate TCP receive socket");
        status = UCS_ERR_NO_MEMORY;
        goto err_close;
    }

    rsock->desc   = NULL;
    rsock->offset = 0;

    status = uct_tcp_iface_recv_sock_add(iface, fd, rsock);
    if (status != UCS_OK) {
        goto err_free;
    }

    memset(&epoll_event, 0, sizeof(epoll_event));
    epoll_event.data.fd = fd;
    epoll_event.events  = EPOLLIN|EPOLLERR;
    ret = epoll_ctl(iface->recv_epfd, EPOLL_CTL_ADD, fd, &epoll_event);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, ADD, fd=%d) failed: %m",
                  iface->recv_epfd, fd);
        status = UCS_ERR_IO_ERROR;
        goto err_remove_evfd;
    }

    ucs_atomic_add32(&iface->recv_sock_count, +1);
    return UCS_OK;

err_remove_evfd:
    uct_tcp_iface_recv_sock_remove(iface, fd);
err_free:
    ucs_free(rsock);
err_close:
    close(fd);
    return status;
}

static void uct_tcp_iface_recv_sock_destroy(uct_tcp_iface_t *iface,
                                            uct_tcp_recv_sock_t *rsock, int fd,
                                            int sync)
{
    if (rsock->desc != NULL) {
        ucs_mpool_put(rsock->desc);
        rsock->desc = NULL;
    }

    ucs_atomic_add32(&iface->recv_sock_count, -1);
    ucs_async_remove_handler(fd, sync);
    ucs_free(rsock);
    close(fd);
}

static unsigned uct_tcp_iface_poll_rx(uct_tcp_iface_t *iface, int fd)
{
    uct_tcp_recv_sock_t *rsock;
    uct_tcp_am_desc_t *desc;
    uct_tcp_am_hdr_t *hdr;
    ucs_status_t status;
    size_t recv_length;

    rsock = uct_tcp_iface_recv_sock_get(iface, fd);
    if (rsock == NULL) {
        ucs_debug("got event on unknown fd: %d", fd);
        return 0;
    }

    /* Allocate receive descriptor if needed */
    if (rsock->desc == NULL) {
        rsock->desc = ucs_mpool_get(&iface->mp);
        if (rsock->desc == NULL) {
            ucs_debug("failed to allocate tcp receive descriptor");
            return 0;
        }
        ucs_assert(rsock->offset == 0);
    }
    desc = rsock->desc;

    hdr = uct_tcp_desc_hdr(iface, desc);
    if (rsock->offset < sizeof(uct_tcp_am_hdr_t)) {
        /* Receive header */
        recv_length = sizeof(uct_tcp_am_hdr_t) - rsock->offset;
    } else {
        /* Receive payload */
        recv_length = hdr->length + sizeof(uct_tcp_am_hdr_t) - rsock->offset;
    }

    /* Receive next chunk of data */
    status = uct_tcp_recv(fd, uct_tcp_recv_data(iface, rsock), &recv_length);
    if (status != UCS_OK) {
        if (status == UCS_ERR_CANCELED) {
            uct_tcp_iface_recv_sock_remove(iface, fd);
            uct_tcp_iface_recv_sock_destroy(iface, rsock, fd, 0);
        }
        return 0;
    }
    rsock->offset += recv_length;

    /* Check if all data is there */
    if ((rsock->offset < sizeof(uct_tcp_am_hdr_t)) ||
        (rsock->offset < sizeof(uct_tcp_am_hdr_t) + hdr->length) )
    {
        return 0;
    }
    ucs_assertv(rsock->offset == sizeof(uct_tcp_am_hdr_t) + hdr->length,
                "recv_length=%zu offset=%zu hlen=%zu length=%d", recv_length,
                rsock->offset, sizeof(uct_tcp_am_hdr_t), hdr->length);

    /* Full message was received */
    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_RECV, hdr->am_id, hdr + 1,
                       hdr->length, "RECV fd %d [%p]" UCS_DEBUG_DATA(" sn %u"),
                       fd, rsock UCS_DEBUG_DATA_ARG(hdr->msn));

    uct_tcp_iface_invoke_am(iface, desc);

    rsock->desc   = NULL;
    rsock->offset = 0;
    return 1;
}

unsigned uct_tcp_iface_progress(uct_iface_h tl_iface)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    struct epoll_event events[UCT_TCP_MAX_EVENTS];
    unsigned count, poll_count;
    int idx, nevents;
    int max_events;

    /* Wait for events in epoll set */
    if (iface->recv_sock_count == 0) {
        nevents = 0;
    } else {
        max_events = ucs_min(UCT_TCP_MAX_EVENTS, iface->config.max_poll);
        nevents = epoll_wait(iface->recv_epfd, events, max_events, 0);
        if ((nevents < 0) && (errno != EINTR)) {
            ucs_error("epoll_wait(epfd=%d max=%d) failed: %m", iface->recv_epfd,
                      max_events);
            return 0;
        }
    }

    /* Poll on new incoming events */
    count = 0;
    idx   = 0;
    while ((idx < nevents) && (count < iface->config.max_poll)) {
        poll_count = uct_tcp_iface_poll_rx(iface, events[idx].data.fd);
        count += poll_count;
        if (poll_count == 0) {
            ++idx; /* Poll on next socket */
        }
    }

    return count;
}

void uct_tcp_iface_recv_cleanup(uct_tcp_iface_t *iface)
{
    uct_tcp_recv_sock_t *rsock;
    UCS_LIST_HEAD(desc_list);
    int fd;

    /* Destroy receive sockets */
    kh_foreach(&iface->fd_hash, fd, rsock, {
        uct_tcp_iface_recv_sock_destroy(iface, rsock, fd, 1);
    });
    kh_clear(uct_tcp_fd_hash, &iface->fd_hash);
}
