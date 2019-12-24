/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_sockcm_ep.h"
#include <ucs/sys/sock.h>
#include <ucs/async/async.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/string.h>


static UCS_F_ALWAYS_INLINE
uct_tcp_sockcm_t *uct_tcp_sockcm_ep_get_cm(uct_tcp_sockcm_ep_t *cep)
{
    /* return the tcp sockcm connection manager this ep is using */
    return ucs_container_of(cep->super.super.iface, uct_tcp_sockcm_t, super.iface);
}

ucs_status_t uct_tcp_sockcm_ep_disconnect(uct_ep_h ep, unsigned flags)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t uct_tcp_sockcm_ep_server_init(uct_tcp_sockcm_ep_t *cep,
                                                  const uct_ep_params_t *params)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t uct_tcp_sockcm_ep_client_init(uct_tcp_sockcm_ep_t *cep,
                                                  const uct_ep_params_t *params)
{
    uct_tcp_sockcm_t *tcp_sockcm = uct_tcp_sockcm_ep_get_cm(cep);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    const struct sockaddr *server_addr;
    ucs_async_context_t *async_ctx;
    uct_tcp_sa_arg_t *sa_arg_ctx;
    ucs_status_t status;
    int events;

    cep->state                   |= UCT_TCP_SOCKCM_EP_ON_CLIENT |
                                    UCT_TCP_SOCKCM_EP_INIT;
    cep->wireup.client.connect_cb = params->sockaddr_connect_cb.client;

    server_addr = params->sockaddr->addr;
    status = ucs_socket_create(server_addr->sa_family, SOCK_STREAM, &cep->fd);
    if (status != UCS_OK) {
        goto err;
    }

    /* Set the fd to non-blocking mode. (so that connect() won't be blocking) */
    status = ucs_sys_fcntl_modfl(cep->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_close_socket;
    }

    /* try to connect to the server */
    status = ucs_socket_connect(cep->fd, server_addr);
    if (UCS_STATUS_IS_ERR(status)) {
        goto err_close_socket;
    }
    ucs_assert ((status == UCS_OK) || (status == UCS_INPROGRESS));

    if (status == UCS_OK) {
        cep->state &= ~UCT_TCP_SOCKCM_EP_INIT;
        cep->state |= UCT_TCP_SOCKCM_EP_CONNECTED;
        /* TODO: start sending the user's data */

        goto out_print;
    }

    sa_arg_ctx = ucs_malloc(sizeof(uct_tcp_sa_arg_t), "client sa_arg_ctx");
    if (sa_arg_ctx == NULL) {
        ucs_error("failed to allocate memory for a client_ctx");
        status = UCS_ERR_NO_MEMORY;
        goto err_close_socket;
    }

    sa_arg_ctx->fd = cep->fd;
    sa_arg_ctx->ep = cep;

    /* Adding the arg to a list on the cm for cleanup purposes */
    ucs_list_add_tail(&tcp_sockcm->sa_arg_list, &sa_arg_ctx->list);

    events    = UCS_EVENT_SET_EVWRITE;    /* wait until connect() completes */
    async_ctx = tcp_sockcm->super.iface.worker->async;
    status    = ucs_async_set_event_handler(async_ctx->mode, cep->fd, events,
                                            uct_tcp_sa_data_handler, sa_arg_ctx,
                                            async_ctx);
    if (status != UCS_OK) {
        goto err_close_socket;
    }

out_print:
    ucs_debug("created a TCP SOCKCM endpoint (fd=%d) on tcp cm %p, "
              "remote addr: %s", cep->fd, tcp_sockcm,
              ucs_sockaddr_str(server_addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

    return status;

err_close_socket:
    close(cep->fd);
err:
    return status;
}

UCS_CLASS_INIT_FUNC(uct_tcp_sockcm_ep_t, const uct_ep_params_t *params)
{
    ucs_status_t status;

    status = uct_cm_check_ep_params(params);
    if (status != UCS_OK) {
        return status;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &params->cm->iface);

    self->wireup.priv_pack_cb = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB) ?
                                params->sockaddr_pack_cb : NULL;
    self->disconnect_cb       = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB) ?
                                params->disconnect_cb : NULL;
    self->user_data           = (params->field_mask &
                                 UCT_EP_PARAM_FIELD_USER_DATA) ?
                                params->user_data : NULL;
    self->state               = 0;
    self->send.buf            = NULL;
    self->send.offset         = 0;

    if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR) {
        status = uct_tcp_sockcm_ep_client_init(self, params);
    } else if (params->field_mask & UCT_EP_PARAM_FIELD_CONN_REQUEST) {
        status = uct_tcp_sockcm_ep_server_init(self, params);
    } else {
        ucs_error("either UCT_EP_PARAM_FIELD_SOCKADDR or UCT_EP_PARAM_FIELD_CONN_REQUEST "
                  "has to be provided");
        status = UCS_ERR_INVALID_PARAM;
    }

    if (status == UCS_OK) {
        ucs_debug("created an endpoint on tcp_sockcm %p id: %d",
                  uct_tcp_sockcm_ep_get_cm(self), self->fd);
    }

    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_tcp_sockcm_ep_t)
{
    uct_tcp_sockcm_t *tcp_sockcm = uct_tcp_sockcm_ep_get_cm(self);
    ucs_status_t status;

    UCS_ASYNC_BLOCK(tcp_sockcm->super.iface.worker->async);

    status = ucs_async_remove_handler(self->fd, 1);
    if (status != UCS_OK) {
        ucs_debug("failed to remove event handler for fd %d: %s",
                  self->fd, ucs_status_string(status));
    }

    close(self->fd);
    UCS_ASYNC_UNBLOCK(tcp_sockcm->super.iface.worker->async);
}

UCS_CLASS_DEFINE(uct_tcp_sockcm_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t);
