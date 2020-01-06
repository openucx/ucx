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
    return ucs_container_of(cep->super.super.super.iface, uct_tcp_sockcm_t,
                            super.iface);
}

ucs_status_t uct_tcp_sockcm_ep_disconnect(uct_ep_h ep, unsigned flags)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t uct_tcp_sockcm_ep_server_init(uct_tcp_sockcm_ep_t *cep,
                                                  const uct_ep_params_t *params)
{
    cep->state |= UCT_TCP_SOCKCM_EP_ON_SERVER;
    return UCS_OK;
}

static ucs_status_t uct_tcp_sockcm_ep_client_init(uct_tcp_sockcm_ep_t *cep,
                                                  const uct_ep_params_t *params)
{
    uct_tcp_sockcm_t *tcp_sockcm = uct_tcp_sockcm_ep_get_cm(cep);
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    const struct sockaddr *server_addr;
    ucs_async_context_t *async_ctx;
    ucs_status_t status;

    cep->state |= UCT_TCP_SOCKCM_EP_ON_CLIENT;

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
    ucs_assert((status == UCS_OK) || (status == UCS_INPROGRESS));

    if (status == UCS_OK) {
        cep->state |= UCT_TCP_SOCKCM_EP_CONNECTED;
        /* TODO: start sending the user's data */

        goto out_print;
    }

    async_ctx = tcp_sockcm->super.iface.worker->async;
    status    = ucs_async_set_event_handler(async_ctx->mode, cep->fd,
                                            /* wait until connect() completes */
                                            UCS_EVENT_SET_EVWRITE,
                                            uct_tcp_sa_data_handler, cep,
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

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_base_ep_t, params);

    self->state           = 0;
    self->comm_ctx.buf    = NULL;
    self->comm_ctx.offset = 0;

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
        ucs_debug("created an endpoint on tcp_sockcm %p id: %d state: %d",
                  uct_tcp_sockcm_ep_get_cm(self), self->fd, self->state);
    }

    return status;
}

UCS_CLASS_CLEANUP_FUNC(uct_tcp_sockcm_ep_t)
{
    uct_tcp_sockcm_t *tcp_sockcm = uct_tcp_sockcm_ep_get_cm(self);

    UCS_ASYNC_BLOCK(tcp_sockcm->super.iface.worker->async);

    ucs_async_remove_handler(self->fd, 1);

    if (self->fd != -1) {
        close(self->fd);
    }
    UCS_ASYNC_UNBLOCK(tcp_sockcm->super.iface.worker->async);
}

UCS_CLASS_DEFINE(uct_tcp_sockcm_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t);
