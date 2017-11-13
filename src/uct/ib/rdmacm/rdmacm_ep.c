/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
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

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    ucs_assert(iface->is_server == 0);

    /* Initialize these fields before calling rdma_resolve_addr to avoid a race
     * where they are used before being initialized (from the async thread
     * - after an RDMA_CM_EVENT_ROUTE_RESOLVED event) */
    hdr.length           = length;
    self->conn_reply_cb  = reply_cb;
    self->conn_reply_arg = arg;
    self->priv_data      = ucs_malloc(sizeof(uct_rdmacm_priv_data_hdr_t) + length,
                                      "client private data");

    memcpy(self->priv_data, &hdr, sizeof(uct_rdmacm_priv_data_hdr_t));
    memcpy(self->priv_data + sizeof(uct_rdmacm_priv_data_hdr_t), priv_data, length);

    /* The interface can point at one endpoint at a time and therefore, the
     * connection establishment cannot be done in parallel for several endpoints */
    if (iface->ep == NULL) {
        iface->ep = self;
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
        /* If the user's callbacks are called from the async thread, we cannot
         * tell at this point if they were already invoked and if this client's
         * ep is already connected to the server */
        return UCS_INPROGRESS;
    } else {
        /* TODO When sync is supported, return UCS_INPROGRESS since for librdmacm the
         * reply_cb still needs to be invoked and it will be from the main thread
         * after getting a reply from the server side */
        ucs_fatal("UCT_CB_FLAG_SYNC is not supported");
    }
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_ep_t)
{
    ucs_free(self->priv_data);
}

UCS_CLASS_DEFINE(uct_rdmacm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, uct_iface_t*,
                          const ucs_sock_addr_t *,
                          uct_sockaddr_conn_reply_callback_t ,
                          void *, uint32_t, const void *, size_t);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);
