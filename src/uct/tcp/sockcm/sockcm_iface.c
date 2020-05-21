/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "sockcm_iface.h"
#include "sockcm_ep.h"

#include <uct/base/uct_worker.h>
#include <uct/tcp/tcp.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>


enum uct_sockcm_process_event_flags {
    UCT_SOCKCM_PROCESS_EVENT_DESTROY_SOCK_ID_FLAG = UCS_BIT(0),
    UCT_SOCKCM_PROCESS_EVENT_ACK_EVENT_FLAG       = UCS_BIT(1)
};

static ucs_config_field_t uct_sockcm_iface_config_table[] = {    
    {"", "", NULL,
     ucs_offsetof(uct_sockcm_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"BACKLOG", "1024",
     "Maximum number of pending connections for a listening socket.",
     ucs_offsetof(uct_sockcm_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_sockcm_iface_t, uct_iface_t);

static ucs_status_t uct_sockcm_iface_query(uct_iface_h tl_iface,
                                           uct_iface_attr_t *iface_attr)
{
    uct_sockcm_iface_t *iface = ucs_derived_of(tl_iface, uct_sockcm_iface_t);
    struct sockaddr_storage addr;
    ucs_status_t status;

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len  = sizeof(ucs_sock_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR    |
                                  UCT_IFACE_FLAG_CB_ASYNC               |
                                  UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    iface_attr->max_conn_priv   = UCT_SOCKCM_MAX_CONN_PRIV;

    if (iface->is_server) {
        socklen_t len = sizeof(struct sockaddr_storage);
        if (getsockname(iface->listen_fd, (struct sockaddr *)&addr, &len)) {
            ucs_error("sockcm_iface: getsockname failed %m");
            return UCS_ERR_IO_ERROR;
        }

        status = ucs_sockaddr_copy((struct sockaddr *)&iface_attr->listen_sockaddr,
                                   (const struct sockaddr *)&addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    ucs_sock_addr_t *sockcm_addr = (ucs_sock_addr_t *)iface_addr;

    sockcm_addr->addr    = NULL;
    sockcm_addr->addrlen = 0;
    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_notify_client(int notif_val,
                                                   uct_conn_request_h conn_request)
{
    char notif = notif_val;
    int fd;
    
    fd = ((uct_sockcm_ctx_t *) conn_request)->sock_fd;

    return ucs_socket_send(fd, &notif, sizeof(notif), NULL, NULL);
}

static ucs_status_t uct_sockcm_iface_accept(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    return uct_sockcm_iface_notify_client(UCT_SOCKCM_IFACE_NOTIFY_ACCEPT, conn_request);
}

static ucs_status_t uct_sockcm_iface_reject(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    return uct_sockcm_iface_notify_client(UCT_SOCKCM_IFACE_NOTIFY_REJECT, conn_request);
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
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)ucs_empty_function_return_success,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_sockcm_iface_t),
    .iface_query              = uct_sockcm_iface_query,
    .iface_is_reachable       = (uct_iface_is_reachable_func_t)ucs_empty_function_return_zero,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_sockcm_iface_get_address
};

static ucs_status_t uct_sockcm_iface_process_conn_req(uct_sockcm_ctx_t *sock_id_ctx)
{
    uct_sockcm_iface_t      *iface      = sock_id_ctx->iface;
    uct_sockcm_conn_param_t *conn_param = &sock_id_ctx->conn_param;

    ucs_debug("process conn req conn_param = %p, conn_param->length = %ld", 
              conn_param, conn_param->length);
    iface->conn_request_cb(&iface->super.super, iface->conn_request_arg, sock_id_ctx, 
                           conn_param->private_data, conn_param->length);
    return UCS_OK;
}

static void uct_sockcm_iface_recv_handler(int fd, int events, void *arg)
{
    uct_sockcm_ctx_t *sock_id_ctx = (uct_sockcm_ctx_t *) arg;
    ucs_status_t status;
    size_t recv_len;

    /* attempt another receive only if initial receive was not successful */
    recv_len = sizeof(uct_sockcm_conn_param_t) - sock_id_ctx->recv_len;
    if (recv_len == 0) {
        goto out_remove_handler;
    }

    status = ucs_socket_recv_nb(sock_id_ctx->sock_fd,
                                UCS_PTR_BYTE_OFFSET(&sock_id_ctx->conn_param,
                                                    sock_id_ctx->recv_len),
                                &recv_len, NULL, NULL);
    if ((status == UCS_ERR_CANCELED) || (status == UCS_ERR_IO_ERROR)) {
        ucs_warn("recv failed in recv handler");
        /* TODO: clean up resources allocated for client endpoint? */
        return;
    }

    sock_id_ctx->recv_len += ((UCS_ERR_NO_PROGRESS == status) ? 0 : recv_len);
    if (sock_id_ctx->recv_len != sizeof(uct_sockcm_conn_param_t)) {
        /* handler should be notified when remaining pieces show up */
        return;
    }

    if (UCS_OK != uct_sockcm_iface_process_conn_req((uct_sockcm_ctx_t*)arg)) {
        ucs_error("unable to process connection request");
    }

out_remove_handler:
    status = ucs_async_modify_handler(fd, 0);
    if (status != UCS_OK) {
        ucs_debug("unable to modify handler");
    }
}

static void uct_sockcm_iface_event_handler(int fd, int events, void *arg)
{
    size_t recv_len               = 0;
    uct_sockcm_iface_t *iface     = arg;
    uct_sockcm_ctx_t *sock_id_ctx = NULL;
    struct sockaddr peer_addr;
    socklen_t addrlen;
    int accept_fd;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;

    addrlen   = sizeof(struct sockaddr);
    accept_fd = accept(iface->listen_fd, (struct sockaddr*)&peer_addr, &addrlen);
    if (accept_fd == -1) {
         if ((errno == EAGAIN) || (errno == EINTR)) {
              ucs_debug("accept(fd=%d) failed: %m", iface->listen_fd);
         } else {
              /* accept failed here, let the client try again */
              ucs_warn("accept(fd=%d) failed with non-recoverable error %m",
                       iface->listen_fd);
         }
         return;
    }

    ucs_debug("sockcm_iface %p: accepted connection from %s at fd %d %m", iface,
              ucs_sockaddr_str(&peer_addr, ip_port_str,
                               UCS_SOCKADDR_STRING_LEN), accept_fd);

    /* Unlike rdmacm, socket connect/accept does not permit exchange of
     * connection parameters but we need to use send/recv on top of that
     * We simulate that with an explicit receive */

    sock_id_ctx = ucs_malloc(sizeof(uct_sockcm_ctx_t), "accepted sock_id_ctx");
    if (sock_id_ctx == NULL) {
        ucs_error("sockcm_listener: unable to create mem for accepted fd");
        close(accept_fd);
        return;
    }

    sock_id_ctx->recv_len = 0;
    sock_id_ctx->sock_fd  = accept_fd;
    sock_id_ctx->iface    = iface;

    status = ucs_sys_fcntl_modfl(sock_id_ctx->sock_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        ucs_error("sockcm_listener: unable make accepted fd non-blocking");
        goto err;
    }

    recv_len = sizeof(sock_id_ctx->conn_param);

    status = ucs_socket_recv_nb(accept_fd, &sock_id_ctx->conn_param, &recv_len,
                                NULL, NULL);
    if (UCS_OK != status) {
        sock_id_ctx->recv_len = ((UCS_ERR_NO_PROGRESS == status) ? 0: recv_len);
        status = ucs_async_set_event_handler(iface->super.worker->async->mode,
                                             sock_id_ctx->sock_fd,
                                             UCS_EVENT_SET_EVREAD, 
                                             uct_sockcm_iface_recv_handler,
                                             sock_id_ctx,
                                             iface->super.worker->async);
        if (status != UCS_OK) {
            ucs_fatal("sockcm_listener: unable to create handler for new connection");
            goto err;
        }
        ucs_debug("assigning recv handler for message from client");
    } else {
        ucs_debug("not assigning recv handler for message from client");
        if (UCS_OK != uct_sockcm_iface_process_conn_req(sock_id_ctx)) {
            ucs_error("Unable to process connection request");
        }
    }

    UCS_ASYNC_BLOCK(iface->super.worker->async);
    ucs_list_add_tail(&iface->used_sock_ids_list, &sock_id_ctx->list);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    
    return;

err:
    uct_sockcm_ep_put_sock_id(sock_id_ctx);
    return;
}

static UCS_CLASS_INIT_FUNC(uct_sockcm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_sockcm_iface_config_t *config = ucs_derived_of(tl_config,
                                                       uct_sockcm_iface_config_t);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    struct sockaddr *param_sockaddr;
    int param_sockaddr_len;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");

    UCT_CHECK_PARAM((params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) ||
                    (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT),
                    "Invalid open mode %zu", params->open_mode);

    UCT_CHECK_PARAM(!(params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) ||
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_SOCKADDR),
                    "UCT_IFACE_PARAM_FIELD_SOCKADDR is not defined "
                    "for UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER");

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

    self->listen_fd = -1;

    if (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) {

        if (!(params->mode.sockaddr.cb_flags & UCT_CB_FLAG_ASYNC)) {
            return UCS_ERR_INVALID_PARAM;
        }

        param_sockaddr = (struct sockaddr *)params->mode.sockaddr.listen_sockaddr.addr;
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

        if (0 > bind(self->listen_fd, param_sockaddr, param_sockaddr_len)) {
            ucs_error("bind(fd=%d) failed: %m", self->listen_fd);
            status = (errno == EADDRINUSE) ? UCS_ERR_BUSY : UCS_ERR_IO_ERROR;
            goto err_close_sock;
        }

        if (0 > listen(self->listen_fd, config->backlog)) {
            ucs_error("listen(fd=%d; backlog=%d)", self->listen_fd,
                      config->backlog);
            status = UCS_ERR_IO_ERROR;
            goto err_close_sock;
        }

        status = ucs_async_set_event_handler(self->super.worker->async->mode,
                                             self->listen_fd,
                                             UCS_EVENT_SET_EVREAD | 
                                             UCS_EVENT_SET_EVERR,
                                             uct_sockcm_iface_event_handler,
                                             self, self->super.worker->async);
        if (status != UCS_OK) {
            goto err_close_sock;
        }

        ucs_debug("iface (%p) sockcm id %d listening on %s", self,
                  self->listen_fd,
                  ucs_sockaddr_str(param_sockaddr, ip_port_str,
                                   UCS_SOCKADDR_STRING_LEN));

        self->cb_flags         = params->mode.sockaddr.cb_flags;
        self->conn_request_cb  = params->mode.sockaddr.conn_request_cb;
        self->conn_request_arg = params->mode.sockaddr.conn_request_arg;
        self->is_server        = 1;
    } else {
        self->is_server        = 0;
    }

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
        ucs_debug("cleaning server fd = %d", sock_id_ctx->sock_fd);
        ucs_async_remove_handler(sock_id_ctx->sock_fd, 1);
        uct_sockcm_ep_put_sock_id(sock_id_ctx);
    }

    UCS_ASYNC_UNBLOCK(self->super.worker->async);
}

UCS_CLASS_DEFINE(uct_sockcm_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_sockcm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sockcm_iface_t, uct_iface_t);

static ucs_status_t
uct_sockcm_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p)
{
    *num_tl_devices_p = 0;
    *tl_devices_p     = NULL;
    return UCS_OK;
}

UCT_TL_DEFINE(&uct_sockcm_component, sockcm, uct_sockcm_query_tl_devices,
              uct_sockcm_iface_t, "SOCKCM_", uct_sockcm_iface_config_table,
              uct_sockcm_iface_config_t);
