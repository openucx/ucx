/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_sockcm_ep.h"

#include <ucs/sys/sock.h>
#include <ucs/async/async.h>


static void uct_tcp_listener_conn_req_handler(int fd, void *arg)
{
    uct_tcp_listener_t *listener = (uct_tcp_listener_t *)arg;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    struct sockaddr_storage client_addr;
    ucs_async_context_t *async_ctx;
    uct_tcp_sockcm_ep_t *ep;
    uct_ep_params_t params;
    ucs_status_t status;
    socklen_t addrlen;
    int conn_fd;

    ucs_assert(fd == listener->listen_fd);

    addrlen   = sizeof(struct sockaddr_storage);
    status    = ucs_socket_accept(listener->listen_fd,
                                  (struct sockaddr*)&client_addr,
                                  &addrlen, &conn_fd);
    if (status != UCS_OK) {
        return;
    }

    ucs_assert(conn_fd != -1);

    ucs_trace("server accepted a connection request (fd=%d) from client %s",
              conn_fd, ucs_sockaddr_str((struct sockaddr*)&client_addr,
                                        ip_port_str, UCS_SOCKADDR_STRING_LEN));

    /* Set the accept_fd to non-blocking mode
     * (so that send/recv won't be blocking) */
    status = ucs_sys_fcntl_modfl(conn_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err;
    }

    /* create the server's endpoint here. uct_ep_create() will return this one */
    params.field_mask        = UCT_EP_PARAM_FIELD_CM               |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST     |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS;
    params.cm                = listener->super.cm;
    params.conn_request      = NULL;
    params.sockaddr_cb_flags = UCT_CB_FLAG_ASYNC;

    status = UCS_CLASS_NEW(uct_tcp_sockcm_ep_t, &ep, &params);
    if (status != UCS_OK) {
        ucs_error("failed to create a new tcp_sockcm ep");
        goto err;
    }

    /* coverity[uninit_use] */
    ep->fd       = conn_fd;
    ep->state   |= UCT_TCP_SOCKCM_EP_CONNECTED;
    ep->listener = listener;

    /* Adding the ep to a list on the cm for cleanup purposes */
    ucs_list_add_tail(&listener->sockcm->ep_list, &ep->list);

    async_ctx = listener->super.cm->iface.worker->async;
    status = ucs_async_set_event_handler(async_ctx->mode, conn_fd,
                                         UCS_EVENT_SET_EVREAD |
                                         UCS_EVENT_SET_EVERR,
                                         uct_tcp_sa_data_handler,
                                         ep, async_ctx);
    if (status != UCS_OK) {
        goto err_delete_ep;
    }

    return;

err_delete_ep:
    UCS_CLASS_DELETE(uct_tcp_sockcm_ep_t, ep);
err:
    close(conn_fd);
}

UCS_CLASS_INIT_FUNC(uct_tcp_listener_t, uct_cm_h cm,
                    const struct sockaddr *saddr, socklen_t socklen,
                    const uct_listener_params_t *params)
{
    ucs_async_context_t *async_ctx;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    int backlog;

    UCS_CLASS_CALL_SUPER_INIT(uct_listener_t, cm);

    self->sockcm          = ucs_derived_of(cm, uct_tcp_sockcm_t);
    self->conn_request_cb = params->conn_request_cb;
    self->user_data       = (params->field_mask & UCT_LISTENER_PARAM_FIELD_USER_DATA) ?
                            params->user_data : NULL;
    backlog               = (params->field_mask & UCT_LISTENER_PARAM_FIELD_BACKLOG) ?
                            params->backlog : ucs_socket_max_conn();

    status = ucs_socket_server_init(saddr, socklen, backlog, &self->listen_fd);
    if (status != UCS_OK) {
        goto err;
    }

    async_ctx = self->sockcm->super.iface.worker->async;
    status = ucs_async_set_event_handler(async_ctx->mode, self->listen_fd,
                                         UCS_EVENT_SET_EVREAD |
                                         UCS_EVENT_SET_EVERR,
                                         uct_tcp_listener_conn_req_handler, self,
                                         async_ctx);
    if (status != UCS_OK) {
        goto err_close_socket;
    }

    ucs_debug("created a TCP listener %p on cm %p with fd: %d "
              "listening on %s", self, cm, self->listen_fd,
              ucs_sockaddr_str(saddr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    return UCS_OK;

err_close_socket:
    close(self->listen_fd);
err:
    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_tcp_listener_t)
{
    ucs_status_t status;

    status = ucs_async_remove_handler(self->listen_fd, 1);
    if (status != UCS_OK) {
        ucs_warn("failed to remove event handler for fd %d: %s",
                 self->listen_fd, ucs_status_string(status));
    }

    close(self->listen_fd);
}

ucs_status_t uct_tcp_listener_reject(uct_listener_h listener,
                                     uct_conn_request_h conn_request)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t uct_tcp_listener_query(uct_listener_h listener,
                                    uct_listener_attr_t *listener_attr)
{
    uct_tcp_listener_t *tcp_listener = ucs_derived_of(listener,
                                                      uct_tcp_listener_t);
    struct sockaddr_storage addr;
    ucs_status_t status;
    socklen_t sock_len;

    if (listener_attr->field_mask & UCT_LISTENER_ATTR_FIELD_SOCKADDR) {
        sock_len = sizeof(struct sockaddr_storage);
        if (getsockname(tcp_listener->listen_fd, (struct sockaddr *)&addr,
                        &sock_len)) {
            ucs_error("getsockname failed (listener=%p) %m", tcp_listener);
            return UCS_ERR_IO_ERROR;
        }

        status = ucs_sockaddr_copy((struct sockaddr *)&listener_attr->sockaddr,
                                   (const struct sockaddr *)&addr);
        if (status != UCS_OK) {
            return status;
        }

    }

    return UCS_OK;
}

UCS_CLASS_DEFINE(uct_tcp_listener_t, uct_listener_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_listener_t, uct_listener_t,
                          uct_cm_h , const struct sockaddr *, socklen_t ,
                          const uct_listener_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_listener_t, uct_listener_t);
