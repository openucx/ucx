/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcpcm_ep.h"


#define UCT_TCPCM_CB_FLAGS_CHECK(_flags) \
    do { \
        UCT_CB_FLAGS_CHECK(_flags); \
        if (!((_flags) & UCT_CB_FLAG_ASYNC)) { \
            return UCS_ERR_UNSUPPORTED; \
        } \
    } while (0)

ucs_status_t uct_tcpcm_ep_set_sock_id(uct_tcpcm_iface_t *iface, uct_tcpcm_ep_t *ep)
{
    ucs_status_t status;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    /* create a sock_id for the client side */
    if (iface->sock_id_quota > 0) {
        /* Create an id for this interface. Events associated with this id will be
         * reported on the event_channel that was created on iface init. */
        ep->sock_id_ctx = ucs_malloc(sizeof(*ep->sock_id_ctx), "client sock_id_ctx");
        if (ep->sock_id_ctx == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        /* Populate &ep->cm_id_ctx->sock_id */
        if (1) {
            goto out_free;
        }
        //ep->sock_id_ctx->sock_id = socket(); // how to get destination
                                           // address and socket type?

        ep->sock_id_ctx->ep = ep;
        ucs_list_add_tail(&iface->used_sock_ids_list, &ep->sock_id_ctx->list);
        iface->sock_id_quota--;
        ucs_debug("ep %p, new sock_id %p. sock_id_in_quota %d", ep,
                   ep->sock_id_ctx->sock_id, iface->sock_id_quota);
        status = UCS_OK;
        goto out;
    } else {
        ep->sock_id_ctx = NULL;
        status = UCS_ERR_NO_RESOURCE;
        goto out;
    }

out_free:
    ucs_free(ep->sock_id_ctx);
out:
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return status;
}

static inline void uct_tcpcm_ep_add_to_pending(uct_tcpcm_iface_t *iface, uct_tcpcm_ep_t *ep)
{
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->pending_eps_list, &ep->list_elem);
    ep->is_on_pending = 1;
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

static UCS_CLASS_INIT_FUNC(uct_tcpcm_ep_t, const uct_ep_params_t *params)
{
    uct_tcpcm_iface_t *iface       = ucs_derived_of(params->iface,
                                                    uct_tcpcm_iface_t);
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

    UCT_TCPCM_CB_FLAGS_CHECK((params->field_mask &
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS) ?
                              params->sockaddr_cb_flags : 0);

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
        ucs_error("tcpcm ep: unknown remote sa_family=%d", sockaddr->addr->sa_family);
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    self->slow_prog_id = UCS_CALLBACKQ_ID_NULL;

    status = uct_tcpcm_ep_set_sock_id(iface, self);
    if (status == UCS_ERR_NO_RESOURCE) {
        goto add_to_pending;
    } else if (status != UCS_OK) {
        goto err;
    }

    self->is_on_pending = 0;

    /* FIXME: do we need to resolve tcp addr? */

    goto out;

add_to_pending:
    /* Add the ep to the pending queue of eps since there is no
     * available cm_id for it */
    uct_tcpcm_ep_add_to_pending(iface, self);
out:
    ucs_debug("created an TCPCM endpoint on iface %p, "
              "iface sock_id: %p remote addr: %s",
               iface, iface->sock_id,
               ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                ip_port_str, UCS_SOCKADDR_STRING_LEN));
    self->status = UCS_INPROGRESS;
    return UCS_OK;

err:
    pthread_mutex_destroy(&self->ops_mutex);

    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcpcm_ep_t)
{
    uct_tcpcm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_tcpcm_iface_t);
    uct_tcpcm_ctx_t *sock_id_ctx;

    ucs_debug("tcpcm_ep %p: destroying", self);

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
    if (self->sock_id_ctx != NULL) {
        sock_id_ctx     = self->sock_id_ctx; // FIXME!
        sock_id_ctx->ep = NULL;
        ucs_debug("ep destroy: cm_id %p", sock_id_ctx->sock_id);
    }
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

UCS_CLASS_DEFINE(uct_tcpcm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcpcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcpcm_ep_t, uct_ep_t);

static unsigned uct_tcpcm_client_err_handle_progress(void *arg)
{
    uct_tcpcm_ep_t *tcpcm_ep = arg;
    uct_tcpcm_iface_t *iface = ucs_derived_of(tcpcm_ep->super.super.iface,
                                               uct_tcpcm_iface_t);

    ucs_trace_func("err_handle ep=%p", tcpcm_ep);
    UCS_ASYNC_BLOCK(iface->super.worker->async);

    tcpcm_ep->slow_prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_tcpcm_ep_t), &tcpcm_ep->super.super,
                      tcpcm_ep->super.super.iface, tcpcm_ep->status);

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return 0;
}

void uct_tcpcm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status)
{
    uct_tcpcm_iface_t *tcpcm_iface = ucs_derived_of(iface, uct_tcpcm_iface_t);
    uct_tcpcm_ep_t *tcpcm_ep       = ucs_derived_of(ep, uct_tcpcm_ep_t);

    if (tcpcm_iface->super.err_handler_flags & UCT_CB_FLAG_ASYNC) {
        uct_set_ep_failed(&UCS_CLASS_NAME(uct_tcpcm_ep_t), &tcpcm_ep->super.super,
                          &tcpcm_iface->super.super, status);
    } else {
        /* invoke the error handling flow from the main thread */
        tcpcm_ep->status = status;
        uct_worker_progress_register_safe(&tcpcm_iface->super.worker->super,
                                          uct_tcpcm_client_err_handle_progress,
                                          tcpcm_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                          &tcpcm_ep->slow_prog_id);
    }
}

/**
 * Caller must lock ep->ops_mutex
 */
void uct_tcpcm_ep_invoke_completions(uct_tcpcm_ep_t *ep, ucs_status_t status)
{
    uct_tcpcm_ep_op_t *op;

    ucs_assert(pthread_mutex_trylock(&ep->ops_mutex) == EBUSY);

    ucs_queue_for_each_extract(op, &ep->ops, queue_elem, 1) {
        pthread_mutex_unlock(&ep->ops_mutex);
        uct_invoke_completion(op->user_comp, status);
        ucs_free(op);
        pthread_mutex_lock(&ep->ops_mutex);
    }
}
