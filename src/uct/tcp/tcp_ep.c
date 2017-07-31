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

    UCS_DEBUG_DATA(self->msn = 0);

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

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_tcp_ep_t *ep = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_am_desc_t *desc;
    uct_tcp_am_hdr_t *hdr;
    ucs_status_t status;
    size_t total_length, send_length;
    ssize_t ret;
    void *ptr;

    desc = ucs_mpool_get(&iface->mp);

    if (desc == NULL) {
        ret = UCS_ERR_NO_RESOURCE;
        goto out;
    }

    hdr = uct_tcp_desc_hdr(iface, desc);

    hdr->am_id  = am_id;
    hdr->length = pack_cb(hdr + 1, arg);
    UCT_CHECK_LENGTH(hdr->length, 0, iface->config.max_bcopy, "am_bcopy");

    UCS_DEBUG_DATA(hdr->msn = ep->msn);

    /*
     * Push out all data to the socket.
     * TODO make this non-blocking by keeping the unsent data on the ep.
     */
    total_length = sizeof(*hdr) + hdr->length;
    ptr          = hdr;
    do {
        send_length = total_length;
        status = uct_tcp_send(ep->fd, ptr, &send_length);
        if (status < 0) {
            ret = status;
            goto out_put;
        }

        ptr          += send_length;
        total_length -= send_length;
    } while (total_length > 0);

    UCS_DEBUG_DATA(++ep->msn);
    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, hdr->am_id,
                        hdr + 1, hdr->length, "SEND fd %d" UCS_DEBUG_DATA(" sn %u"),
                        ep->fd UCS_DEBUG_DATA_ARG(hdr->msn));
    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, hdr->length);
    ret = hdr->length;

out_put:
    ucs_mpool_put(desc);
out:
    return ret;
}
