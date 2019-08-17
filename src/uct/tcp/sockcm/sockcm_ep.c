/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "sockcm_ep.h"

#define UCT_SOCKCM_CB_FLAGS_CHECK(_flags) \
    do { \
        UCT_CB_FLAGS_CHECK(_flags); \
        if (!((_flags) & UCT_CB_FLAG_ASYNC)) { \
            return UCS_ERR_UNSUPPORTED; \
        } \
    } while (0)

ucs_status_t uct_sockcm_ep_set_sock_id(uct_sockcm_iface_t *iface, uct_sockcm_ep_t *ep)
{
    ucs_status_t status;
    struct sockaddr *dest_addr = NULL;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    ep->sock_id_ctx = ucs_malloc(sizeof(*ep->sock_id_ctx), "client sock_id_ctx");
    if (ep->sock_id_ctx == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }
    ep->sock_id_ctx->handler_added = 0;
    dest_addr = (struct sockaddr *) &(ep->remote_addr);

    status = ucs_socket_create(dest_addr->sa_family, SOCK_STREAM,
                               &ep->sock_id_ctx->sock_id);
    if (status != UCS_OK) {
        ucs_debug("unable to create client socket for sockcm");
        goto out_free;
    }

    ucs_list_add_tail(&iface->used_sock_ids_list, &ep->sock_id_ctx->list);
    ucs_debug("ep %p, new sock_id %d", ep, ep->sock_id_ctx->sock_id);
    status = UCS_OK;
    goto out;

out_free:
    ucs_free(ep->sock_id_ctx);
out:
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return status;
}

static inline void uct_sockcm_ep_add_to_pending(uct_sockcm_iface_t *iface, uct_sockcm_ep_t *ep)
{
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->pending_eps_list, &ep->list_elem);
    ep->is_on_pending = 1;
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

static void uct_sockcm_ep_event_handler(int fd, void *arg)
{
    ucs_debug("not implemented");
}

static UCS_CLASS_INIT_FUNC(uct_sockcm_ep_t, const uct_ep_params_t *params)
{
    const ucs_sock_addr_t *sockaddr = params->sockaddr;
    uct_sockcm_iface_t    *iface    = NULL;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;

    iface = ucs_derived_of(params->iface, uct_sockcm_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    if (iface->is_server) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR)) {
        return UCS_ERR_INVALID_PARAM;
    }

    UCT_SOCKCM_CB_FLAGS_CHECK((params->field_mask &
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

    if (sockaddr->addr->sa_family == AF_INET) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in));
    } else if (sockaddr->addr->sa_family == AF_INET6) {
        memcpy(&self->remote_addr, sockaddr->addr, sizeof(struct sockaddr_in6));
    } else {
        ucs_error("sockcm ep: unknown remote sa_family=%d", sockaddr->addr->sa_family);
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    self->slow_prog_id = UCS_CALLBACKQ_ID_NULL;
    self->is_on_pending = 0;

    status = uct_sockcm_ep_set_sock_id(iface, self);
    if (status == UCS_ERR_NO_RESOURCE) {
        goto add_to_pending;
    } else if (status != UCS_OK) {
        goto err;
    }

    status = ucs_sys_fcntl_modfl(self->sock_id_ctx->sock_id, O_NONBLOCK, 0); 
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_socket_connect(self->sock_id_ctx->sock_id, sockaddr->addr);
    if ((status != UCS_OK) && (status != UCS_INPROGRESS)) {
        ucs_debug("%d: connect fail\n", self->sock_id_ctx->sock_id);
        self->conn_state = UCT_SOCKCM_EP_CONN_STATE_CLOSED;
        goto err;
    } else {
        ucs_debug("%d: connect pass/pending\n", self->sock_id_ctx->sock_id);
        /* no need to look for connection completion */
        self->conn_state = (status == UCS_INPROGRESS) ? 
            UCT_SOCKCM_EP_CONN_STATE_SOCK_CONNECTING : 
            UCT_SOCKCM_EP_CONN_STATE_SOCK_CONNECTED;
    }

    status = ucs_async_set_event_handler(iface->super.worker->async->mode,
                                         self->sock_id_ctx->sock_id,
                                         UCS_EVENT_SET_EVWRITE,
                                         uct_sockcm_ep_event_handler,
                                         self, iface->super.worker->async);
    if (status != UCS_OK) {
        goto err;
    }

    goto out;

add_to_pending:
    uct_sockcm_ep_add_to_pending(iface, self);
out:
    ucs_debug("created an SOCKCM endpoint on iface %p, "
              "remote addr: %s", iface,
               ucs_sockaddr_str((struct sockaddr *)sockaddr->addr,
                                ip_port_str, UCS_SOCKADDR_STRING_LEN));
    self->status = UCS_INPROGRESS;
    return UCS_OK;

err:
    ucs_debug("error in sock connect\n");
    pthread_mutex_destroy(&self->ops_mutex);

    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sockcm_ep_t)
{
    uct_sockcm_iface_t *iface = NULL;

    iface = ucs_derived_of(self->super.super.iface, uct_sockcm_iface_t);

    ucs_debug("sockcm_ep %p: destroying", self);

    ucs_async_remove_handler(self->sock_id_ctx->sock_id, 0);

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    if (self->is_on_pending) {
        ucs_list_del(&self->list_elem);
        self->is_on_pending = 0;
    }

    uct_worker_progress_unregister_safe(&iface->super.worker->super,
                                        &self->slow_prog_id);

    pthread_mutex_destroy(&self->ops_mutex);
    if (!ucs_queue_is_empty(&self->ops)) {
        ucs_warn("destroying endpoint %p with not completed operations", self);
    }

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}
UCS_CLASS_DEFINE(uct_sockcm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sockcm_ep_t, uct_ep_t);

static unsigned uct_sockcm_client_err_handle_progress(void *arg)
{
    uct_sockcm_ep_t *sockcm_ep = arg;
    uct_sockcm_iface_t *iface = ucs_derived_of(sockcm_ep->super.super.iface,
                                               uct_sockcm_iface_t);

    ucs_trace_func("err_handle ep=%p", sockcm_ep);
    UCS_ASYNC_BLOCK(iface->super.worker->async);

    sockcm_ep->slow_prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_set_ep_failed(&UCS_CLASS_NAME(uct_sockcm_ep_t), &sockcm_ep->super.super,
                      sockcm_ep->super.super.iface, sockcm_ep->status);

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return 0;
}

void uct_sockcm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status)
{
    uct_sockcm_iface_t *sockcm_iface = ucs_derived_of(iface, uct_sockcm_iface_t);
    uct_sockcm_ep_t *sockcm_ep       = ucs_derived_of(ep, uct_sockcm_ep_t);

    if (sockcm_iface->super.err_handler_flags & UCT_CB_FLAG_ASYNC) {
        uct_set_ep_failed(&UCS_CLASS_NAME(uct_sockcm_ep_t), &sockcm_ep->super.super,
                          &sockcm_iface->super.super, status);
    } else {
        sockcm_ep->status = status;
        uct_worker_progress_register_safe(&sockcm_iface->super.worker->super,
                                          uct_sockcm_client_err_handle_progress,
                                          sockcm_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                          &sockcm_ep->slow_prog_id);
    }
}

void uct_sockcm_ep_invoke_completions(uct_sockcm_ep_t *ep, ucs_status_t status)
{
    uct_sockcm_ep_op_t *op;

    ucs_assert(pthread_mutex_trylock(&ep->ops_mutex) == EBUSY);

    ucs_queue_for_each_extract(op, &ep->ops, queue_elem, 1) {
        pthread_mutex_unlock(&ep->ops_mutex);
        uct_invoke_completion(op->user_comp, status);
        ucs_free(op);
        pthread_mutex_lock(&ep->ops_mutex);
    }
}
