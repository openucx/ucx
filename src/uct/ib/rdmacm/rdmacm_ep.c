/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rdmacm_ep.h"

ucs_status_t uct_rdmacm_ep_resolve_addr(uct_rdmacm_ep_t *ep)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rdmacm_iface_t);

    return uct_rdmacm_resolve_addr(iface->cm_id, (struct sockaddr *)&(ep->remote_addr),
                                   UCS_MSEC_PER_SEC * iface->config.addr_resolve_timeout,
                                   UCS_LOG_LEVEL_ERROR);
}

static UCS_CLASS_INIT_FUNC(uct_rdmacm_ep_t, uct_iface_t *tl_iface,
                           const ucs_sock_addr_t *sockaddr,
                           uint32_t cb_flags,
                           const void *priv_data, size_t length)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(tl_iface, uct_rdmacm_iface_t);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    uct_rdmacm_priv_data_hdr_t hdr;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    if (iface->is_server) {
        /* TODO allow an interface to be used both for server and client */
        return UCS_ERR_UNSUPPORTED;
    }

    /* Initialize these fields before calling rdma_resolve_addr to avoid a race
     * where they are used before being initialized (from the async thread
     * - after an RDMA_CM_EVENT_ROUTE_RESOLVED event) */
    hdr.length           = length;
    self->priv_data      = ucs_malloc(sizeof(hdr) + length, "client private data");
    if (self->priv_data == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    memcpy(self->priv_data, &hdr, sizeof(hdr));
    memcpy(self->priv_data + sizeof(hdr), priv_data, length);

    /* TODO When UCT_CB_FLAG_SYNC is supported, return UCS_INPROGRESS since for
     * librdmacm the reply_cb still needs to be invoked and it will be from the
     * main thread after getting a reply from the server side */
    ucs_assertv_always((cb_flags & UCT_CB_FLAG_ASYNC), "UCT_CB_FLAG_SYNC is not supported");

    /* If the user's callbacks are called from the async thread, we cannot
     * tell if by the end of this function they were already invoked and if this
     * client's ep would already be connected to the server */
    self->cb_flags = cb_flags;

    /* Save the remote address */
    if (sockaddr->addr->sa_family == AF_INET) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in));
    } else if (sockaddr->addr->sa_family == AF_INET6) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in6));
    } else {
        ucs_error("rdmacm ep: unknown remote sa_family=%d", sockaddr->addr->sa_family);
        status = UCS_ERR_IO_ERROR;
        goto err_free_mem;
    }

    /* The interface can point at one endpoint at a time and therefore, the
     * connection establishment cannot be done in parallel for several endpoints */
    /* TODO support connection establishment on parallel endpoints on the same iface */
    if (iface->ep == NULL) {
        iface->ep = self;
        self->is_on_pending = 0;

        /* After rdma_resolve_addr(), the client will wait for an
         * RDMA_CM_EVENT_ADDR_RESOLVED event on the event_channel
         * to proceed with the connection establishment.
         * This event will be retrieved from the event_channel by the async thread.
         * All endpoints share the interface's event_channel but can use it serially. */
        status = uct_rdmacm_ep_resolve_addr(self);
        if (status != UCS_OK) {
            goto err_free_mem;
        }
    } else {
        /* Add the ep to the pending queue */
        UCS_ASYNC_BLOCK(iface->super.worker->async);
        ucs_list_add_tail(&iface->pending_eps_list, &self->list_elem);
        self->is_on_pending = 1;
        UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    }

    ucs_debug("created an RDMACM endpoint on iface %p. event_channel: %p, "
              "iface cm_id: %p remote addr: %s",
               iface, iface->event_ch, iface->cm_id,
               ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                ip_port_str, UCS_SOCKADDR_STRING_LEN));

    return UCS_INPROGRESS;

err_free_mem:
    ucs_free(self->priv_data);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_ep_t)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_rdmacm_iface_t);

    ucs_debug("rdmacm_ep %p: destroying", self);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    if (self->is_on_pending) {
        ucs_list_del(&self->list_elem);
        self->is_on_pending = 0;
    }

    /* if the destroyeed ep is the active one on the iface, mark it as destroyed
     * so that arriving events on the iface won't try to access this ep */
    if (iface->ep == self) {
        iface->ep = UCT_RDMACM_IFACE_BLOCKED_NO_EP;
    }
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_free(self->priv_data);
}

UCS_CLASS_DEFINE(uct_rdmacm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, uct_iface_t*,
                          const ucs_sock_addr_t *,
                          uint32_t, const void *, size_t);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

void uct_rdmacm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status)
{
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_rdmacm_ep_t), ep, iface, status);
}
