/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcpcm_iface.h"
#include "tcpcm_ep.h"
#include <uct/base/uct_worker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>


enum uct_tcpcm_process_event_flags {
    UCT_TCPCM_PROCESS_EVENT_DESTROY_SOCK_ID_FLAG = UCS_BIT(0),
    UCT_TCPCM_PROCESS_EVENT_ACK_EVENT_FLAG       = UCS_BIT(1)
};

static ucs_config_field_t uct_tcpcm_iface_config_table[] = {
    {"BACKLOG", "1024",
     "Maximum number of pending connections for a (listening?) socket.",
     ucs_offsetof(uct_tcpcm_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

    {"SOCK_ID_QUOTA", "64",
     "How many tcpcm connections can progress simultaneously.",
     ucs_offsetof(uct_tcpcm_iface_config_t, sock_id_quota), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcpcm_iface_t, uct_iface_t);

static ucs_status_t uct_tcpcm_iface_query(uct_iface_h tl_iface,
                                           uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    iface_attr->iface_addr_len  = sizeof(ucs_sock_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR |
                                  UCT_IFACE_FLAG_CB_ASYNC            |
                                  UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    /* User's private data size is UCT_TCPCM_UDP_PRIV_DATA_LEN minus room for
     * the private_data header (to hold the length of the data) */
    iface_attr->max_conn_priv   = UCT_TCPCM_MAX_CONN_PRIV;

    return UCS_OK;
}

static int uct_tcpcm_iface_is_reachable(const uct_iface_h tl_iface,
                                         const uct_device_addr_t *dev_addr,
                                         const uct_iface_addr_t *iface_addr)
{
    /* Reachability can be checked with the uct_md_is_sockaddr_accessible API call */
    return 1;
}

static ucs_status_t uct_tcpcm_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    ucs_sock_addr_t *tcpcm_addr = (ucs_sock_addr_t *)iface_addr;

    tcpcm_addr->addr    = NULL;
    tcpcm_addr->addrlen = 0;
    return UCS_OK;
}

#if 0
static ucs_status_t uct_tcpcm_accept(int sock_id)
{
    /* The server will not send any reply data back to the client */
    //struct tcp_conn_param conn_param = {0};

    /* FIXME:
    accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
    if (accept(id, &conn_param)) {
        ucs_error("rdma_accept(to id=%p) failed: %m", id);
        return UCS_ERR_IO_ERROR;
    }
    */

    return UCS_OK;
}
#endif

static ucs_status_t uct_tcpcm_iface_accept(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    // struct rdma_cm_event *event = conn_request; FIXME
    ucs_status_t         status = UCS_OK;

    ucs_trace("accepting at FIXME");
    // FIXME
    //status = uct_tcpcm_accept(*(conn_request->sock_id));

    return status;
}

static ucs_status_t uct_tcpcm_iface_reject(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    // struct rdma_cm_event       *event = conn_request; FIXME
    ucs_status_t               status = UCS_OK;
    //uct_tcpcm_priv_data_hdr_t hdr    = {
    //    .length = 0,
    //    .status = UCS_ERR_REJECTED
    //};

    ucs_trace("rejecting at FIXME");
    /* FIXME
    if (rdma_reject(event->id, &hdr, sizeof(hdr))) {
        ucs_warn("rdma_reject(id=%p) failed: %m", event->id);
        status = UCS_ERR_IO_ERROR;
    }
    */

    return status;
}

static ucs_status_t uct_tcpcm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    uct_tcpcm_ep_t    *ep = ucs_derived_of(tl_ep, uct_tcpcm_ep_t);
    ucs_status_t       status;
    uct_tcpcm_ep_op_t *op;

    pthread_mutex_lock(&ep->ops_mutex);
    status = ep->status;
    if ((status == UCS_INPROGRESS) && (comp != NULL)) {
        op = ucs_malloc(sizeof(*op), "uct_tcpcm_ep_flush op");
        if (op != NULL) {
            op->user_comp = comp;
            ucs_queue_push(&ep->ops, &op->queue_elem);
        } else {
            status = UCS_ERR_NO_MEMORY;
        }
    }
    pthread_mutex_unlock(&ep->ops_mutex);

    return status;
}

static uct_iface_ops_t uct_tcpcm_iface_ops = {
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_tcpcm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_tcpcm_ep_t),
    .ep_flush                 = uct_tcpcm_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_pending_purge         = ucs_empty_function,
    .iface_accept             = uct_tcpcm_iface_accept,
    .iface_reject             = uct_tcpcm_iface_reject,
    .iface_progress_enable    = (void*)ucs_empty_function_return_success,
    .iface_progress_disable   = (void*)ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_tcpcm_iface_t),
    .iface_query              = uct_tcpcm_iface_query,
    .iface_is_reachable       = uct_tcpcm_iface_is_reachable,
    .iface_get_device_address = (void*)ucs_empty_function_return_success,
    .iface_get_address        = uct_tcpcm_iface_get_address
};

void uct_tcpcm_iface_client_start_next_ep(uct_tcpcm_iface_t *iface)
{
    ucs_status_t status;
    uct_tcpcm_ep_t *ep, *tmp;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    /* try to start an ep from the pending eps list */
    ucs_list_for_each_safe(ep, tmp, &iface->pending_eps_list, list_elem) {
        status = uct_tcpcm_ep_set_sock_id(iface, ep);
        if (status != UCS_OK) {
            continue;
        }

        ucs_list_del(&ep->list_elem);
        ep->is_on_pending = 0;

        uct_tcpcm_ep_set_failed(&iface->super.super, &ep->super.super, status);
    }

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

#if 0
static void uct_tcpcm_client_handle_failure(uct_tcpcm_iface_t *iface,
                                             uct_tcpcm_ep_t *ep,
                                             ucs_status_t status)
{
    ucs_assert(!iface->is_server);
    if (ep != NULL) {
        pthread_mutex_lock(&ep->ops_mutex);
        uct_tcpcm_ep_set_failed(&iface->super.super, &ep->super.super, status);
        uct_tcpcm_ep_invoke_completions(ep, status);
        pthread_mutex_unlock(&ep->ops_mutex);
    }
}


static void uct_tcpcm_iface_process_conn_req(uct_tcpcm_iface_t *iface,
                                              void *event,
                                              struct sockaddr *remote_addr)
{
    uct_tcpcm_priv_data_hdr_t *hdr;

    hdr = (uct_tcpcm_priv_data_hdr_t*) event->param.ud.private_data;
    ucs_assert(hdr->status == UCS_OK);
    iface->conn_request_cb(&iface->super.super, iface->conn_request_arg,
                           event,
                           UCS_PTR_BYTE_OFFSET(event->param.ud.private_data,
                                               sizeof(uct_tcpcm_priv_data_hdr_t)),
                           hdr->length);
}


/**
 * Release sock_id. This function should be called when the async context
 * is locked.
 */
static void uct_tcpcm_iface_release_sock_id(uct_tcpcm_iface_t *iface,
                                            uct_tcpcm_ctx_t *sock_id_ctx)
{
    ucs_trace("destroying sock_id %d", sock_id_ctx->sock_id);

    ucs_list_del(&sock_id_ctx->list);
    if (sock_id_ctx->ep != NULL) {
        sock_id_ctx->ep->sock_id_ctx = NULL;
    }
    close(sock_id_ctx->sock_id); // FIXME review
    ucs_free(sock_id_ctx);
    iface->sock_id_quota++;
}

static void uct_tcpcm_iface_sock_id_to_dev_name(int *sock_id, char *dev_name)
{
    ucs_snprintf_zero(dev_name, UCT_DEVICE_NAME_MAX, "%s:%d",
                      "42"/*ibv_get_device_name(cm_id->verbs->device) FIXME*/,
                      42); //FIXME
}

#endif

static void uct_tcpcm_iface_event_handler(int fd, void *arg)
{
    //uct_tcpcm_iface_t             *iface     = arg;
    //uct_tcpcm_ctx_t               *sock_id_ctx = NULL;
    //int                            ret;

    fprintf(stderr, "event handler noticed something!");
    for (;;) {

        /* FIXME
        ret = rdma_get_cm_event(iface->event_ch, &event);
        if (ret) {
            if (errno != EAGAIN) {
                ucs_warn("rdma_get_cm_event() failed: %m");
            }
            return;
        }

        proc_event_flags = uct_tcpcm_iface_process_event(iface, event);
        if (!iface->is_server) {
            cm_id_ctx = (uct_tcpcm_ctx_t *)event->id->context;
        }

        if (proc_event_flags & UCT_TCPCM_PROCESS_EVENT_ACK_EVENT_FLAG) {
            ret = rdma_ack_cm_event(event);
            if (ret) {
                ucs_warn("rdma_ack_cm_event() failed: %m");
            }
        }

        if ((proc_event_flags & UCT_TCPCM_PROCESS_EVENT_DESTROY_CM_ID_FLAG) &&
            (cm_id_ctx != NULL)) {
            uct_tcpcm_iface_release_cm_id(iface, cm_id_ctx);
            uct_tcpcm_iface_client_start_next_ep(iface);
        }
        */
    }
}

static UCS_CLASS_INIT_FUNC(uct_tcpcm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_tcpcm_iface_config_t *config = ucs_derived_of(tl_config, uct_tcpcm_iface_config_t);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    uct_tcpcm_md_t *tcpcm_md;
    ucs_status_t status;
    int ret = 0;
    struct sockaddr *param_sockaddr;
    int param_sockaddr_len;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");

    UCT_CHECK_PARAM((params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) ||
                    (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT),
                    "Invalid open mode %zu", params->open_mode);

    UCT_CHECK_PARAM(!(params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) ||
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_SOCKADDR),
                    "UCT_IFACE_PARAM_FIELD_SOCKADDR is not defined for UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER");

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_tcpcm_iface_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG((params->field_mask &
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(UCT_TCPCM_TL_NAME));

    tcpcm_md = ucs_derived_of(self->super.md, uct_tcpcm_md_t);

    if (self->super.worker->async == NULL) {
        ucs_error("tcpcm must have async != NULL");
        return UCS_ERR_INVALID_PARAM;
    }
    if (self->super.worker->async->mode == UCS_ASYNC_MODE_SIGNAL) {
        ucs_warn("tcpcm does not support SIGIO");
    }

    self->config.addr_resolve_timeout = tcpcm_md->addr_resolve_timeout;

    if (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) {

        param_sockaddr = (struct sockaddr *) params->mode.sockaddr.listen_sockaddr.addr;
        param_sockaddr_len = params->mode.sockaddr.listen_sockaddr.addrlen;

        status = ucs_socket_create(param_sockaddr->sa_family, SOCK_STREAM,
                                   &self->sock_id);
        if (status != UCS_OK) {
            return status;
        }

        status = ucs_sys_fcntl_modfl(self->sock_id, O_NONBLOCK, 0);
        if (status != UCS_OK) {
            goto err_close_sock;
        }

        ret = bind(self->sock_id, param_sockaddr, param_sockaddr_len);
        if (ret < 0) {
            ucs_error("bind(fd=%d) failed: %m", self->sock_id);
            status = UCS_ERR_IO_ERROR;
            goto err_close_sock;
        }

        ret = listen(self->sock_id, config->backlog);
        if (ret < 0) {
            ucs_error("listen(fd=%d; backlog=%d)", self->sock_id, config->backlog);
            status = UCS_ERR_IO_ERROR;
            goto err_close_sock;
        }

        /* Register event handler for incoming connections */
        status = ucs_async_set_event_handler(self->super.worker->async->mode,
                                             self->sock_id, POLLIN | POLLERR,
                                             uct_tcpcm_iface_event_handler,
                                             self, self->super.worker->async);
        if (status != UCS_OK) {
            goto err_close_sock;
        }

        ucs_debug("tcpcm id %d listening on %s", self->sock_id,
                  ucs_sockaddr_str(param_sockaddr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));

        if (!(params->mode.sockaddr.cb_flags & UCT_CB_FLAG_ASYNC)) {
            ucs_fatal("Synchronous callback is not supported");
        }

        self->cb_flags         = params->mode.sockaddr.cb_flags;
        self->conn_request_cb  = params->mode.sockaddr.conn_request_cb;
        self->conn_request_arg = params->mode.sockaddr.conn_request_arg;
        self->is_server        = 1;
    } else {
        self->sock_id          = -1;
        self->is_server        = 0;
    }

    self->sock_id_quota = config->sock_id_quota;
    ucs_list_head_init(&self->pending_eps_list);
    ucs_list_head_init(&self->used_sock_ids_list);

    return UCS_OK;

 err_close_sock:
    close(self->sock_id);
    return status;

}

static UCS_CLASS_CLEANUP_FUNC(uct_tcpcm_iface_t)
{
    uct_tcpcm_ctx_t *sock_id_ctx;

    ucs_async_remove_handler(self->sock_id, 1);
    if (self->is_server) {
        close(self->sock_id);
    }

    UCS_ASYNC_BLOCK(self->super.worker->async);

    while (!ucs_list_is_empty(&self->used_sock_ids_list)) {
        sock_id_ctx = ucs_list_extract_head(&self->used_sock_ids_list,
                                            uct_tcpcm_ctx_t, list);
        close(sock_id_ctx->sock_id);
        ucs_free(sock_id_ctx);
        self->sock_id_quota++;
    }

    UCS_ASYNC_UNBLOCK(self->super.worker->async);
}

UCS_CLASS_DEFINE(uct_tcpcm_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_tcpcm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcpcm_iface_t, uct_iface_t);

static ucs_status_t uct_tcpcm_query_tl_resources(uct_md_h md,
                                                  uct_tl_resource_desc_t **resource_p,
                                                  unsigned *num_resources_p)
{
    *num_resources_p = 0;
    *resource_p      = NULL;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_tcpcm_tl,
                        uct_tcpcm_query_tl_resources,
                        uct_tcpcm_iface_t,
                        UCT_TCPCM_TL_NAME,
                        "TCPCM_",
                        uct_tcpcm_iface_config_table,
                        uct_tcpcm_iface_config_t);
UCT_MD_REGISTER_TL(&uct_tcpcm_mdc, &uct_tcpcm_tl);
