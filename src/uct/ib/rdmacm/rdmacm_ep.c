/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rdmacm_ep.h"

ucs_status_t uct_rdmacm_ep_resolve_addr(uct_rdmacm_ep_t *ep)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rdmacm_iface_t);
    ucs_status_t status;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    status = uct_rdmacm_resolve_addr(ep->rdmacm_cm_id->cm_id,
                                    (struct sockaddr *)&(ep->remote_addr),
                                    UCS_MSEC_PER_SEC * iface->config.addr_resolve_timeout,
                                    UCS_LOG_LEVEL_ERROR);

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return status;
}

ucs_status_t uct_rdmacm_ep_set_cm_id(uct_rdmacm_iface_t *iface, uct_rdmacm_ep_t *ep)
{
    ucs_status_t status;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    /* create a cm_id for the client side */
    if ((iface->num_cm_id_in_quota <= iface->config.cm_id_quota_size) &&
        (iface->num_cm_id_in_quota > 0)) {
        /* Create an id for this interface. Events associated with this id will be
         * reported on the event_channel that was created on iface init. */
        ep->rdmacm_cm_id = ucs_malloc(sizeof(uct_rdmacm_cm_id_t), "client rdmacm_cm_id");
        if (ep->rdmacm_cm_id == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }
        if (rdma_create_id(iface->event_ch, &ep->rdmacm_cm_id->cm_id,
                           ep->rdmacm_cm_id, RDMA_PS_UDP)) {
            ucs_error("rdma_create_id() failed: %m");
            ucs_free(ep->rdmacm_cm_id);
            status = UCS_ERR_IO_ERROR;
            goto out;
        }
        ep->rdmacm_cm_id->ep = ep;
        ucs_list_add_tail(&iface->used_cm_ids_list, &ep->rdmacm_cm_id->list);
        ep->is_on_cm_ids_list = 1;
        iface->num_cm_id_in_quota--;
        ucs_debug("ep %p, new cm_id %p. cm_id_in_quota %d", ep,
                   ep->rdmacm_cm_id->cm_id, iface->num_cm_id_in_quota);
        status = UCS_OK;
    } else {
        ep->is_on_cm_ids_list = 0;
        status = UCS_ERR_NO_RESOURCE;
    }

out:
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return status;
}

static UCS_CLASS_INIT_FUNC(uct_rdmacm_ep_t, uct_iface_t *tl_iface,
                           const ucs_sock_addr_t *sockaddr,
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

    /* Save the remote address */
    if (sockaddr->addr->sa_family == AF_INET) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in));
    } else if (sockaddr->addr->sa_family == AF_INET6) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in6));
    } else {
        ucs_error("rdmacm ep: unknown remote sa_family=%d", sockaddr->addr->sa_family);
        status = UCS_ERR_IO_ERROR;
        goto err_free_priv_data;
    }

    self->slow_prog_id = UCS_CALLBACKQ_ID_NULL;

    status = uct_rdmacm_ep_set_cm_id(iface, self);
    if (status == UCS_ERR_NO_RESOURCE) {
        goto add_to_pending;
    } else if (status != UCS_OK) {
        goto err_free_priv_data;
    }

    self->is_on_pending = 0;

    /* After rdma_resolve_addr(), the client will wait for an
     * RDMA_CM_EVENT_ADDR_RESOLVED event on the event_channel
     * to proceed with the connection establishment.
     * This event will be retrieved from the event_channel by the async thread.
     * All endpoints share the interface's event_channel. */
    status = uct_rdmacm_ep_resolve_addr(self);
    if (status != UCS_OK) {
        goto err_free_priv_data;
    }

    goto out;

add_to_pending:
    /* Add the ep to the pending queue of eps which since there is no
     * available cm_id for it */
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->pending_eps_list, &self->list_elem);
    self->is_on_pending = 1;
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

out:
    ucs_debug("created an RDMACM endpoint on iface %p. event_channel: %p, "
              "iface cm_id: %p remote addr: %s",
               iface, iface->event_ch, iface->cm_id,
               ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                ip_port_str, UCS_SOCKADDR_STRING_LEN));
    return UCS_OK;

err_free_priv_data:
    ucs_free(self->priv_data);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_ep_t)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_rdmacm_iface_t);
    uct_rdmacm_cm_id_t *rdmacm_cm_id;

    ucs_debug("rdmacm_ep %p: destroying", self);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    if (self->is_on_pending) {
        ucs_list_del(&self->list_elem);
        self->is_on_pending = 0;
    }

    /* remove the slow progress function in case it was placed on the slow progress
     * chain but wasn't invoked yet */
    uct_worker_progress_unregister_safe(&iface->super.worker->super,
                                        &self->slow_prog_id);

    /* mark this ep as destroyed so that arriving events on it won't try to
     * use it */
    if (self->is_on_cm_ids_list) {
        rdmacm_cm_id = (uct_rdmacm_cm_id_t *)self->rdmacm_cm_id->cm_id->context;
        rdmacm_cm_id->ep = UCT_RDMACM_IFACE_BLOCKED_NO_EP;
        ucs_debug("ep destroy: cm_id %p", rdmacm_cm_id->cm_id);
    }
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);

    ucs_free(self->priv_data);
}

UCS_CLASS_DEFINE(uct_rdmacm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, uct_iface_t*,
                          const ucs_sock_addr_t *,
                          const void *, size_t);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

static unsigned uct_rdmacm_client_err_handle_progress(void *arg)
{
    uct_rdmacm_ep_t *ep = arg;
    ucs_trace_func("err_handle ep=%p",ep);

    ep->slow_prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_rdmacm_ep_t), &ep->super.super,
                      ep->super.super.iface, UCS_ERR_IO_ERROR);
    return 0;
}

void uct_rdmacm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep)
{
    uct_rdmacm_iface_t *rdmacm_iface = ucs_derived_of(iface, uct_rdmacm_iface_t);
    uct_rdmacm_ep_t *rdmacm_ep = ucs_derived_of(ep, uct_rdmacm_ep_t);

    /* invoke the error handling flow from the main thread */
    uct_worker_progress_register_safe(&rdmacm_iface->super.worker->super,
                                      uct_rdmacm_client_err_handle_progress,
                                      rdmacm_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &rdmacm_ep->slow_prog_id);
}
