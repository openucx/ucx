/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"

#include <ucs/async/async.h>


static UCS_CLASS_INIT_FUNC(uct_tcp_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(tl_iface, uct_tcp_iface_t);
    struct sockaddr_in dest_addr;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)

    status = uct_tcp_socket_create(&self->fd);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_tcp_iface_set_sockopt(iface, self->fd);
    if (status != UCS_OK) {
        goto err_close;
    }

    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port   = *(in_port_t*)iface_addr;
    dest_addr.sin_addr   = *(struct in_addr*)dev_addr;

    status = uct_tcp_socket_connect(self->fd, &dest_addr);
    if (status != UCS_OK) {
        goto err_close;
    }

    ucs_debug("connected to %s:%d", inet_ntoa(dest_addr.sin_addr),
              ntohs(dest_addr.sin_port));
    return UCS_OK;

err_close:
    close(self->fd);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_ep_t)
{
    ucs_trace_func("self=%p", self);
    close(self->fd);
}

UCS_CLASS_DEFINE(uct_tcp_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_ep_t, uct_ep_t, uct_iface_t *,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_ep_t, uct_ep_t);
