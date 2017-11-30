/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rdmacm_ep.h"

static UCS_CLASS_INIT_FUNC(uct_rdmacm_ep_t, uct_iface_t *tl_iface,
                           const ucs_sock_addr_t *sockaddr,
                           uct_sockaddr_conn_reply_callback_t reply_cb,
                           void *arg, uint32_t cb_flags,
                           const void *priv_data, size_t length)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(tl_iface, uct_rdmacm_iface_t);
    size_t ip_len = ucs_max(INET_ADDRSTRLEN, INET6_ADDRSTRLEN);
    char *ip_str  = ucs_alloca(ip_len);
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
    self->conn_reply_cb  = reply_cb;
    self->conn_reply_arg = arg;
    self->priv_data      = ucs_malloc(sizeof(hdr) + length, "client private data");
    if (self->priv_data == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    memcpy(self->priv_data, &hdr, sizeof(hdr));
    memcpy(self->priv_data + sizeof(hdr), priv_data, length);

    /* The interface can point at one endpoint at a time and therefore, the
     * connection establishment cannot be done in parallel for several endpoints */
    /* TODO support connection establishment on parallel endpoints on the same iface */
    if (iface->ep == NULL) {
        iface->ep = self;

        /* After rdma_resolve_addr(), the client will wait for an
         * RDMA_CM_EVENT_ADDR_RESOLVED event on the event_channel
         * to proceed with the connection establishment.
         * This event will be retrieved from the event_channel by the async thread.
         * All endpoints share the interface's event_channel but can use it serially. */
         if (rdma_resolve_addr(iface->cm_id, NULL, (struct sockaddr *)sockaddr->addr,
                               UCS_MSEC_PER_SEC * iface->config.addr_resolve_timeout)) {
             ucs_error("rdma_resolve_addr(addr=%s) failed: %m",
                       ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                        ip_str, ip_len));
             status = UCS_ERR_IO_ERROR;
             goto err_free_mem;
         }
    } else {
        /* Add the ep to the pending queue */
        self->remote_addr = (struct sockaddr *)(sockaddr->addr);
        UCS_ASYNC_BLOCK(iface->super.worker->async);
        ucs_queue_push(&iface->pending_eps_q, &self->queue);
        UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    }

    ucs_debug("created an RDMACM endpoint on iface %p. event_channel: %p, "
              "iface cm_id: %p remote addr: %s",
               iface, iface->event_ch, iface->cm_id,
               ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                ip_str, ip_len));

    if (cb_flags == UCT_CB_FLAG_ASYNC) {
        self->cb_flags = cb_flags;
        /* If the user's callbacks are called from the async thread, we cannot
         * tell at this point if they were already invoked and if this client's
         * ep is already connected to the server */
        status = UCS_INPROGRESS;
    } else {
        /* TODO When sync is supported, return UCS_INPROGRESS since for librdmacm the
         * reply_cb still needs to be invoked and it will be from the main thread
         * after getting a reply from the server side */
        ucs_fatal("UCT_CB_FLAG_SYNC is not supported");
    }

    return status;

err_free_mem:
    ucs_free(self->priv_data);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_ep_t)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_rdmacm_iface_t);
    ucs_queue_iter_t iter;
    uct_rdmacm_ep_t *ep;

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_queue_for_each_safe(ep, iter, &iface->pending_eps_q, queue) {
        if (ep == self) {
            ucs_queue_del_iter(&iface->pending_eps_q, iter);
            uct_rdmacm_iface_client_start_next_ep(iface);
            break;
        }
    }
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    ucs_free(self->priv_data);
}

UCS_CLASS_DEFINE(uct_rdmacm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, uct_iface_t*,
                          const ucs_sock_addr_t *,
                          uct_sockaddr_conn_reply_callback_t ,
                          void *, uint32_t, const void *, size_t);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

void uct_rdmacm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status)
{
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_rdmacm_ep_t), ep, iface, status);
}
