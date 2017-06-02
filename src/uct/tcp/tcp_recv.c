/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <uct/base/uct_worker.h>
#include <ucs/async/async.h>
#include <sys/poll.h>


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

ucs_status_t uct_tcp_iface_connection_accepted(uct_tcp_iface_t *iface, int fd)
{
    uct_tcp_recv_sock_t *rsock;
    ucs_status_t status;

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

    status = uct_tcp_iface_recv_sock_add(iface, fd, rsock);
    if (status != UCS_OK) {
        goto err_free;
    }

    return UCS_OK;

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
    ucs_free(rsock);
    close(fd);
}

void uct_tcp_iface_recv_cleanup(uct_tcp_iface_t *iface)
{
    uct_tcp_recv_sock_t *rsock;
    int fd;

    /* Destroy receive sockets */
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    kh_foreach(&iface->fd_hash, fd, rsock, {
        uct_tcp_iface_recv_sock_destroy(iface, rsock, fd, 1);
    });
    kh_clear(uct_tcp_fd_hash, &iface->fd_hash);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}
