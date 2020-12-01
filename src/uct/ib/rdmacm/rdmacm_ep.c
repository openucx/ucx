/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_ep.h"


#define UCT_RDMACM_CB_FLAGS_CHECK(_flags) \
    do { \
        UCT_CB_FLAGS_CHECK(_flags); \
        if (!((_flags) & UCT_CB_FLAG_ASYNC)) { \
            return UCS_ERR_UNSUPPORTED; \
        } \
    } while (0)


ucs_status_t uct_rdmacm_ep_resolve_addr(uct_rdmacm_ep_t *ep)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rdmacm_iface_t);
    ucs_status_t status;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    status = uct_rdmacm_resolve_addr(ep->cm_id_ctx->cm_id,
                                    (struct sockaddr *)&ep->remote_addr,
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
    if (iface->cm_id_quota > 0) {
        /* Create an id for this interface. Events associated with this id will be
         * reported on the event_channel that was created on iface init. */
        ep->cm_id_ctx = ucs_malloc(sizeof(*ep->cm_id_ctx), "client cm_id_ctx");
        if (ep->cm_id_ctx == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        if (rdma_create_id(iface->event_ch, &ep->cm_id_ctx->cm_id,
                           ep->cm_id_ctx, RDMA_PS_UDP)) {
            ucs_error("rdma_create_id() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto out_free;
        }

        ep->cm_id_ctx->ep = ep;
        ucs_list_add_tail(&iface->used_cm_ids_list, &ep->cm_id_ctx->list);
        iface->cm_id_quota--;
        ucs_debug("ep %p, new cm_id %p. cm_id_in_quota %d", ep,
                   ep->cm_id_ctx->cm_id, iface->cm_id_quota);
        status = UCS_OK;
        goto out;
    } else {
        ep->cm_id_ctx = NULL;
        status = UCS_ERR_NO_RESOURCE;
        goto out;
    }

out_free:
    ucs_free(ep->cm_id_ctx);
out:
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return status;
}

static inline void uct_rdmacm_ep_add_to_pending(uct_rdmacm_iface_t *iface, uct_rdmacm_ep_t *ep)
{
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->pending_eps_list, &ep->list_elem);
    ep->is_on_pending = 1;
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

static UCS_CLASS_INIT_FUNC(uct_rdmacm_ep_t, const uct_ep_params_t *params)
{
    uct_rdmacm_iface_t *iface       = ucs_derived_of(params->iface,
                                                     uct_rdmacm_iface_t);
    const ucs_sock_addr_t *sockaddr = params->sockaddr;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    if (iface->is_server) {
        /* TODO allow an interface to be used both for server and client */
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR)) {
        return UCS_ERR_INVALID_PARAM;
    }

    UCT_RDMACM_CB_FLAGS_CHECK((params->field_mask &
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS) ?
                              params->sockaddr_cb_flags : 0);

    /* Initialize these fields before calling rdma_resolve_addr to avoid a race
     * where they are used before being initialized (from the async thread
     * - after an RDMA_CM_EVENT_ROUTE_RESOLVED event) */
    self->pack_cb       = (params->field_mask &
                           UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB) ?
                          params->sockaddr_pack_cb : NULL;
    self->pack_cb_arg   = (params->field_mask &
                           UCT_EP_PARAM_FIELD_USER_DATA) ?
                          params->user_data : NULL;
    self->pack_cb_flags = (params->field_mask &
                           UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS) ?
                          params->sockaddr_cb_flags : 0;
    pthread_mutex_init(&self->ops_mutex, NULL);
    ucs_queue_head_init(&self->ops);

    /* Save the remote address */
    if (sockaddr->addr->sa_family == AF_INET) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in));
    } else if (sockaddr->addr->sa_family == AF_INET6) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in6));
    } else {
        ucs_error("rdmacm ep: unknown remote sa_family=%d", sockaddr->addr->sa_family);
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    self->slow_prog_id = UCS_CALLBACKQ_ID_NULL;

    status = uct_rdmacm_ep_set_cm_id(iface, self);
    if (status == UCS_ERR_NO_RESOURCE) {
        goto add_to_pending;
    } else if (status != UCS_OK) {
        goto err;
    }

    self->is_on_pending = 0;

    /* After rdma_resolve_addr(), the client will wait for an
     * RDMA_CM_EVENT_ADDR_RESOLVED event on the event_channel
     * to proceed with the connection establishment.
     * This event will be retrieved from the event_channel by the async thread.
     * All endpoints share the interface's event_channel. */
    status = uct_rdmacm_ep_resolve_addr(self);
    if (status != UCS_OK) {
        goto err;
    }

    goto out;

add_to_pending:
    /* Add the ep to the pending queue of eps since there is no
     * available cm_id for it */
    uct_rdmacm_ep_add_to_pending(iface, self);
out:
    ucs_debug("created an RDMACM endpoint on iface %p. event_channel: %p, "
              "iface cm_id: %p remote addr: %s",
               iface, iface->event_ch, iface->cm_id,
               ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                ip_port_str, UCS_SOCKADDR_STRING_LEN));
    self->status = UCS_INPROGRESS;
    return UCS_OK;

err:
    pthread_mutex_destroy(&self->ops_mutex);

    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_ep_t)
{
    uct_rdmacm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_rdmacm_iface_t);
    uct_rdmacm_ctx_t *cm_id_ctx;

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

    pthread_mutex_destroy(&self->ops_mutex);
    if (!ucs_queue_is_empty(&self->ops)) {
        ucs_warn("destroying endpoint %p with not completed operations", self);
    }

    /* mark this ep as destroyed so that arriving events on it won't try to
     * use it */
    if (self->cm_id_ctx != NULL) {
        cm_id_ctx     = self->cm_id_ctx->cm_id->context;
        cm_id_ctx->ep = NULL;
        ucs_debug("ep destroy: cm_id %p", cm_id_ctx->cm_id);
    }
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

UCS_CLASS_DEFINE(uct_rdmacm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

static unsigned uct_rdmacm_client_err_handle_progress(void *arg)
{
    uct_rdmacm_ep_t *rdmacm_ep = arg;
    uct_rdmacm_iface_t *iface = ucs_derived_of(rdmacm_ep->super.super.iface,
                                               uct_rdmacm_iface_t);

    ucs_trace_func("err_handle ep=%p", rdmacm_ep);
    UCS_ASYNC_BLOCK(iface->super.worker->async);

    rdmacm_ep->slow_prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_iface_handle_ep_err(&iface->super.super, &rdmacm_ep->super.super,
                            rdmacm_ep->status);

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return 0;
}

void uct_rdmacm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status)
{
    uct_rdmacm_iface_t *rdmacm_iface = ucs_derived_of(iface, uct_rdmacm_iface_t);
    uct_rdmacm_ep_t *rdmacm_ep       = ucs_derived_of(ep, uct_rdmacm_ep_t);

    rdmacm_ep->status = status;

    if (rdmacm_iface->super.err_handler_flags & UCT_CB_FLAG_ASYNC) {
        uct_iface_handle_ep_err(iface, ep, status);
    } else {
        /* invoke the error handling flow from the main thread */
        uct_worker_progress_register_safe(&rdmacm_iface->super.worker->super,
                                          uct_rdmacm_client_err_handle_progress,
                                          rdmacm_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                          &rdmacm_ep->slow_prog_id);
    }
}

/**
 * Caller must lock ep->ops_mutex
 */
void uct_rdmacm_ep_invoke_completions(uct_rdmacm_ep_t *ep, ucs_status_t status)
{
    uct_rdmacm_ep_op_t *op;

    ucs_assert(pthread_mutex_trylock(&ep->ops_mutex) == EBUSY);

    ucs_queue_for_each_extract(op, &ep->ops, queue_elem, 1) {
        pthread_mutex_unlock(&ep->ops_mutex);
        uct_invoke_completion(op->user_comp, status);
        ucs_free(op);
        pthread_mutex_lock(&ep->ops_mutex);
    }
    /* coverity[missing_unlock] */
}
