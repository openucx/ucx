/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "sockcm_iface.h"
#include "sockcm_ep.h"
#include <uct/base/uct_worker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>

enum uct_sockcm_process_event_flags {
    UCT_SOCKCM_PROCESS_EVENT_DESTROY_SOCK_ID_FLAG = UCS_BIT(0),
    UCT_SOCKCM_PROCESS_EVENT_ACK_EVENT_FLAG       = UCS_BIT(1)
};

static ucs_config_field_t uct_sockcm_iface_config_table[] = {
    {"BACKLOG", "1024",
     "Maximum number of pending connections for a listening socket.",
     ucs_offsetof(uct_sockcm_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_sockcm_iface_t, uct_iface_t);

static ucs_status_t uct_sockcm_iface_query(uct_iface_h tl_iface,
                                           uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    iface_attr->iface_addr_len  = sizeof(ucs_sock_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR    |
                                  UCT_IFACE_FLAG_CB_ASYNC               |
                                  UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    iface_attr->max_conn_priv   = UCT_SOCKCM_MAX_CONN_PRIV;

    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    ucs_sock_addr_t *sockcm_addr = (ucs_sock_addr_t *)iface_addr;

    sockcm_addr->addr    = NULL;
    sockcm_addr->addrlen = 0;
    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_accept(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_reject(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    return UCS_OK;
}

static ucs_status_t uct_sockcm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    uct_sockcm_ep_t    *ep = ucs_derived_of(tl_ep, uct_sockcm_ep_t);
    ucs_status_t       status;
    uct_sockcm_ep_op_t *op;

    pthread_mutex_lock(&ep->ops_mutex);
    status = ep->status;
    if ((status == UCS_INPROGRESS) && (comp != NULL)) {
        op = ucs_malloc(sizeof(*op), "uct_sockcm_ep_flush op");
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

static uct_iface_ops_t uct_sockcm_iface_ops = {
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_sockcm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_sockcm_ep_t),
    .ep_flush                 = uct_sockcm_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_pending_purge         = ucs_empty_function,
    .iface_accept             = uct_sockcm_iface_accept,
    .iface_reject             = uct_sockcm_iface_reject,
    .iface_progress_enable    = (void*)ucs_empty_function_return_success,
    .iface_progress_disable   = (void*)ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_sockcm_iface_t),
    .iface_query              = uct_sockcm_iface_query,
    .iface_is_reachable       = (void*)ucs_empty_function_return_zero,
    .iface_get_device_address = (void*)ucs_empty_function_return_success,
    .iface_get_address        = uct_sockcm_iface_get_address
};

void uct_sockcm_iface_client_start_next_ep(uct_sockcm_iface_t *iface)
{
    ucs_status_t status;
    uct_sockcm_ep_t *ep, *tmp;

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    /* try to start an ep from the pending eps list */
    ucs_list_for_each_safe(ep, tmp, &iface->pending_eps_list, list_elem) {
        status = uct_sockcm_ep_set_sock_id(iface, ep);
        if (status != UCS_OK) {
            continue;
        }

        ucs_list_del(&ep->list_elem);
        ep->is_on_pending = 0;

        uct_sockcm_ep_set_failed(&iface->super.super, &ep->super.super, status);
    }

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

static void uct_sockcm_iface_process_conn_req(uct_sockcm_iface_t *iface,
                                             uct_sockcm_conn_param_t conn_param)
{
    iface->conn_request_cb(&iface->super.super, iface->conn_request_arg, NULL,
			   conn_param.private_data, conn_param.private_data_len);
}


static void uct_sockcm_iface_event_handler(int fd, void *arg)
{
    ssize_t recv_len          = 0;
    ssize_t sent_len          = 0;
    int     connect_confirm   = 42;
    uct_sockcm_iface_t *iface = arg;
    struct sockaddr peer_addr;
    socklen_t addrlen;
    int accept_fd;
    uct_sockcm_conn_param_t conn_param;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    // accept client connection
    accept_fd = accept(iface->listen_fd, (struct sockaddr*)&peer_addr, &addrlen);
    if (accept_fd < 0) {
        if ((errno != EAGAIN) && (errno != EINTR)) {
            ucs_error("accept() failed: %m");
            return;
            // FIXME uct_tcp_iface_listen_close(iface);
            //close(iface->sock_id);
        }
    }

    ucs_debug("tcp_iface %p: accepted connection from %s at fd %d %m", iface,
              ucs_sockaddr_str(&peer_addr, ip_port_str,
                               UCS_SOCKADDR_STRING_LEN), accept_fd);

    sent_len = send(accept_fd, (char *) &connect_confirm, sizeof(int), 0);
    ucs_debug("send_len = %d bytes %m", (int) sent_len);

    recv_len = recv(accept_fd, (char *) &conn_param,
                    sizeof(uct_sockcm_conn_param_t), 0);
    ucs_debug("recv len = %d\n", (int) recv_len);

    // schedule connection req callback
    if (recv_len == sizeof(uct_sockcm_conn_param_t)) {
        uct_sockcm_iface_process_conn_req(iface, conn_param);
    }

    return;

}

static UCS_CLASS_INIT_FUNC(uct_sockcm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_sockcm_iface_config_t *config = ucs_derived_of(tl_config, uct_sockcm_iface_config_t);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
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

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_sockcm_iface_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG((params->field_mask &
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(UCT_SOCKCM_TL_NAME));

    if (self->super.worker->async == NULL) {
        ucs_error("sockcm must have async != NULL");
        return UCS_ERR_INVALID_PARAM;
    }
    if (self->super.worker->async->mode == UCS_ASYNC_MODE_SIGNAL) {
        ucs_warn("sockcm does not support SIGIO");
    }

    self->sock_id = -1;
    self->listen_fd = -1;

    if (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) {

        param_sockaddr = (struct sockaddr *) params->mode.sockaddr.listen_sockaddr.addr;
        param_sockaddr_len = params->mode.sockaddr.listen_sockaddr.addrlen;

        status = ucs_socket_create(param_sockaddr->sa_family, SOCK_STREAM,
                                   &self->listen_fd);
        if (status != UCS_OK) {
            return status;
        }

        status = ucs_sys_fcntl_modfl(self->listen_fd, O_NONBLOCK, 0);
        if (status != UCS_OK) {
            goto err_close_sock;
        }

        ret = bind(self->listen_fd, param_sockaddr, param_sockaddr_len);
        if (ret < 0) {
            ucs_error("bind(fd=%d) failed: %m", self->listen_fd);
            status = UCS_ERR_IO_ERROR;
            goto err_close_sock;
        }

        ret = listen(self->listen_fd, config->backlog);
        if (ret < 0) {
            ucs_error("listen(fd=%d; backlog=%d)", self->listen_fd, config->backlog);
            status = UCS_ERR_IO_ERROR;
            goto err_close_sock;
        }

        status = ucs_async_set_event_handler(self->super.worker->async->mode,
                                             self->listen_fd, POLLIN | POLLERR,
                                             uct_sockcm_iface_event_handler,
                                             self, self->super.worker->async);
        if (status != UCS_OK) {
            goto err_close_sock;
        }

        ucs_debug("iface (%p) sockcm id %d listening on %s", self, self->listen_fd,
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

    ucs_list_head_init(&self->pending_eps_list);
    ucs_list_head_init(&self->used_sock_ids_list);

    return UCS_OK;

 err_close_sock:
    close(self->listen_fd);
    return status;

}

static UCS_CLASS_CLEANUP_FUNC(uct_sockcm_iface_t)
{
    uct_sockcm_ctx_t *sock_id_ctx;

    if (self->is_server) {
        if (-1 != self->listen_fd) {
            ucs_debug("cleaning listen_fd = %d", self->listen_fd);
            ucs_async_remove_handler(self->listen_fd, 1);
            close(self->listen_fd);
        }
    }

    UCS_ASYNC_BLOCK(self->super.worker->async);

    while (!ucs_list_is_empty(&self->used_sock_ids_list)) {
        sock_id_ctx = ucs_list_extract_head(&self->used_sock_ids_list,
                                            uct_sockcm_ctx_t, list);
        close(sock_id_ctx->sock_id);
        ucs_free(sock_id_ctx);
    }

    UCS_ASYNC_UNBLOCK(self->super.worker->async);
}

UCS_CLASS_DEFINE(uct_sockcm_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_sockcm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sockcm_iface_t, uct_iface_t);

static ucs_status_t uct_sockcm_query_tl_resources(uct_md_h md,
                                                  uct_tl_resource_desc_t **resource_p,
                                                  unsigned *num_resources_p)
{
    *num_resources_p = 0;
    *resource_p      = NULL;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_sockcm_tl,
                        uct_sockcm_query_tl_resources,
                        uct_sockcm_iface_t,
                        UCT_SOCKCM_TL_NAME,
                        "SOCKCM_",
                        uct_sockcm_iface_config_table,
                        uct_sockcm_iface_config_t);
UCT_MD_REGISTER_TL(&uct_sockcm_mdc, &uct_sockcm_tl);
