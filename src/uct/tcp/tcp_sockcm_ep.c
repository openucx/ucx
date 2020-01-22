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

static void uct_tcp_sockcm_ep_init_comm_ctx(uct_tcp_sockcm_ep_t *cep)
{
    cep->comm_ctx.offset = 0;
    cep->comm_ctx.length = 0;
}

static void uct_tcp_sockcm_ep_handle_disconnect(uct_tcp_sockcm_ep_t *cep,
                                                ucs_status_t status)
{
    uct_cm_remote_data_t remote_data;

    /* remote peer disconnected */
    ucs_debug("ep %p (fd=%d): remote peer disconnected", cep, cep->fd);
    uct_tcp_sockcm_ep_init_comm_ctx(cep);

    ucs_assert(status != UCS_OK);
    if (cep->state & UCT_TCP_SOCKCM_EP_ON_SERVER) {
        uct_cm_ep_server_connect_cb(&cep->super, status);
    } else {
        ucs_assert(cep->state & UCT_TCP_SOCKCM_EP_ON_CLIENT);
        remote_data.field_mask = 0;
        uct_cm_ep_client_connect_cb(&cep->super, &remote_data, status);
    }

    /* TODO handle disconnect if the ep already invoked the connect_cb */
}

static int uct_tcp_sockcm_ep_is_tx_rx_done(uct_tcp_sockcm_ep_t *cep)
{
    ucs_assert((cep->comm_ctx.length != 0));
    return (cep->comm_ctx.offset == cep->comm_ctx.length);
}

ucs_status_t uct_tcp_sockcm_ep_progress_send(uct_tcp_sockcm_ep_t *cep)
{
    ucs_status_t status;
    size_t sent_length;

    ucs_assert(ucs_test_all_flags(cep->state, UCT_TCP_SOCKCM_EP_ON_CLIENT |
                                              UCT_TCP_SOCKCM_EP_CONNECTED));
    ucs_assert(cep->comm_ctx.offset < cep->comm_ctx.length);

    sent_length = cep->comm_ctx.length - cep->comm_ctx.offset;

    status = ucs_socket_send_nb(cep->fd,
                                UCS_PTR_BYTE_OFFSET(cep->comm_ctx.buf,
                                                    cep->comm_ctx.offset),
                                &sent_length, NULL, NULL);
    if ((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS)) {
        if (status == UCS_ERR_NOT_CONNECTED) {
            uct_tcp_sockcm_ep_handle_disconnect(cep, status);
        } else {
            ucs_error("ep %p failed to send client's data (len=%zu offset=%zu)",
                      cep, cep->comm_ctx.length, cep->comm_ctx.offset);
        }
        return status;
    }

    cep->comm_ctx.offset += sent_length;
    ucs_assert(cep->comm_ctx.offset <= cep->comm_ctx.length);
    cep->state           |= UCT_TCP_SOCKCM_EP_SENDING;

    if (uct_tcp_sockcm_ep_is_tx_rx_done(cep)) {
        cep->state |= UCT_TCP_SOCKCM_EP_DATA_SENT;
        uct_tcp_sockcm_ep_init_comm_ctx(cep);

        /* wait for a reply from the peer */
        status = ucs_async_modify_handler(cep->fd, UCS_EVENT_SET_EVREAD);
        if (status != UCS_OK) {
            ucs_error("failed to modify %d event handler to "
                      "UCS_EVENT_SET_EVREAD: %s", cep->fd,
                      ucs_status_string(status));
        }
    }

    return UCS_OK;
}

ucs_status_t uct_tcp_sockcm_ep_send_priv_data(uct_tcp_sockcm_ep_t *cep)
{
    char ifname_str[UCT_DEVICE_NAME_MAX];
    uct_tcp_sockcm_priv_data_hdr_t *hdr;
    ssize_t priv_data_ret;
    ucs_status_t status;

    /* get interface name associated with the connected client fd */
    status = ucs_sockaddr_get_ifname(cep->fd, ifname_str, sizeof(ifname_str));
    if (UCS_OK != status) {
        goto out;
    }

    hdr           = (uct_tcp_sockcm_priv_data_hdr_t*)cep->comm_ctx.buf;
    priv_data_ret = cep->super.priv_pack_cb(cep->super.user_data, ifname_str,
                                            hdr + 1);
    if (priv_data_ret < 0) {
        ucs_assert(priv_data_ret > UCS_ERR_LAST);
        status = (ucs_status_t)priv_data_ret;
        ucs_error("tcp_sockcm private data pack function failed with error: %s",
                  ucs_status_string(status));
        goto out;
    } else if (priv_data_ret > (uct_tcp_sockcm_ep_get_cm(cep)->priv_data_len)) {
        status = UCS_ERR_EXCEEDS_LIMIT;
        ucs_error("tcp_sockcm private data pack function returned %zd "
                  "(max: %zu)", priv_data_ret,
                  uct_tcp_sockcm_ep_get_cm(cep)->priv_data_len);
        goto out;
    }

    hdr->length          = priv_data_ret;
    cep->comm_ctx.length = sizeof(*hdr) + hdr->length;

    status = uct_tcp_sockcm_ep_progress_send(cep);

out:
    return status;
}

static ucs_status_t uct_tcp_sockcm_ep_server_invoke_conn_req_cb(uct_tcp_sockcm_ep_t *cep)
{
    uct_tcp_sockcm_priv_data_hdr_t *hdr = (uct_tcp_sockcm_priv_data_hdr_t *)
                                          cep->comm_ctx.buf;
    struct sockaddr_storage remote_dev_addr = {0};
    socklen_t               remote_dev_addr_len;
    char peer_str[UCS_SOCKADDR_STRING_LEN];
    char                    ifname_str[UCT_DEVICE_NAME_MAX];
    uct_cm_remote_data_t    remote_data;
    ucs_status_t            status;

    /* get the local interface name associated with the connected fd */
    status = ucs_sockaddr_get_ifname(cep->fd, ifname_str, UCT_DEVICE_NAME_MAX);
    if (UCS_OK != status) {
        return status;
    }

    /* get the device address of the remote peer associated with the connected fd */
    status = ucs_socket_getpeername(cep->fd, &remote_dev_addr, &remote_dev_addr_len);
    if (status != UCS_OK) {
        return status;
    }

    remote_data.field_mask            = UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                                        UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                                        UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH;
    remote_data.dev_addr              = (uct_device_addr_t *)&remote_dev_addr;
    remote_data.dev_addr_length       = remote_dev_addr_len;
    remote_data.conn_priv_data        = hdr + 1;
    remote_data.conn_priv_data_length = hdr->length;

    ucs_debug("fd %d: remote_data: (field_mask=%zu) dev_addr: %s (length=%zu), "
              "conn_priv_data_length=%zu", cep->fd, remote_data.field_mask,
              ucs_sockaddr_str((const struct sockaddr*)remote_data.dev_addr,
                               peer_str, UCS_SOCKADDR_STRING_LEN),
              remote_data.dev_addr_length, remote_data.conn_priv_data_length);

    /* the endpoint, passed as the conn_request to the callback, will be passed
     * to uct_ep_create() which will be invoked by the user and therefore moving
     * over to its responsibility. */
    ucs_list_del(&cep->list);
    cep->listener->conn_request_cb(&cep->listener->super, cep->listener->user_data,
                                   ifname_str, cep, &remote_data);

    return UCS_OK;
}

ucs_status_t uct_tcp_sockcm_ep_handle_data_received(uct_tcp_sockcm_ep_t *cep)
{
    ucs_status_t status;

    cep->state |= UCT_TCP_SOCKCM_EP_DATA_RECEIVED;
    uct_tcp_sockcm_ep_init_comm_ctx(cep);

    status = uct_tcp_sockcm_ep_server_invoke_conn_req_cb(cep);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_async_modify_handler(cep->fd, UCS_EVENT_SET_EVWRITE);
    if (status != UCS_OK) {
        ucs_error("failed to modify %d event handler to UCS_EVENT_SET_EVWRITE: %s",
                  cep->fd, ucs_status_string(status));
        goto out;
    }

out:
    return status;
}

static ucs_status_t uct_tcp_sockcm_ep_recv_nb(uct_tcp_sockcm_ep_t *cep)
{
    size_t recv_length;
    ucs_status_t status;

    recv_length = uct_tcp_sockcm_ep_get_cm(cep)->priv_data_len +
                  sizeof(uct_tcp_sockcm_priv_data_hdr_t) - cep->comm_ctx.offset;
    status = ucs_socket_recv_nb(cep->fd, UCS_PTR_BYTE_OFFSET(cep->comm_ctx.buf,
                                                             cep->comm_ctx.offset),
                                &recv_length, NULL, NULL);
    if ((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS)) {
        if (status == UCS_ERR_NOT_CONNECTED) {
            uct_tcp_sockcm_ep_handle_disconnect(cep, status);
        } else {
            ucs_error("ep %p (fd=%d) failed to recv client's data (offset=%zu)",
                      cep, cep->fd, cep->comm_ctx.offset);
        }
        return status;
    }

    cep->comm_ctx.offset += recv_length;
    ucs_assertv((cep->comm_ctx.length ?
                 cep->comm_ctx.offset <= cep->comm_ctx.length : 1), "%zu > %zu",
                cep->comm_ctx.offset, cep->comm_ctx.length);
    return UCS_OK;
}

ucs_status_t uct_tcp_sockcm_ep_progress_recv(uct_tcp_sockcm_ep_t *cep)
{
    ucs_status_t status;

    status = uct_tcp_sockcm_ep_recv_nb(cep);
    if (status != UCS_OK) {
        return status;
    }

    if (uct_tcp_sockcm_ep_is_tx_rx_done(cep)) {
        status = uct_tcp_sockcm_ep_handle_data_received(cep);
    }

    return status;
}

ucs_status_t uct_tcp_sockcm_ep_recv(uct_tcp_sockcm_ep_t *cep)
{
    uct_tcp_sockcm_priv_data_hdr_t *hdr;
    ucs_status_t status;

    status = uct_tcp_sockcm_ep_recv_nb(cep);
    if (status != UCS_OK) {
        goto out;
    }

    if (cep->comm_ctx.offset < sizeof(*hdr)) {
        goto out;
    }

    hdr                  = (uct_tcp_sockcm_priv_data_hdr_t *)cep->comm_ctx.buf;
    cep->comm_ctx.length = sizeof(*hdr) + hdr->length;
    ucs_assertv(cep->comm_ctx.offset <= cep->comm_ctx.length , "%zu > %zu",
                cep->comm_ctx.offset, cep->comm_ctx.length);

    cep->state          |= UCT_TCP_SOCKCM_EP_RECEIVING;

    if (uct_tcp_sockcm_ep_is_tx_rx_done(cep)) {
        status = uct_tcp_sockcm_ep_handle_data_received(cep);
    }

out:
    return status;
}

static ucs_status_t uct_tcp_sockcm_ep_server_init(uct_tcp_sockcm_ep_t *cep,
                                                  const uct_ep_params_t *params)
{
    cep->state                  |= UCT_TCP_SOCKCM_EP_ON_SERVER;
    cep->super.server.connect_cb = params->sockaddr_connect_cb.server;
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
    cep->super.client.connect_cb = params->sockaddr_connect_cb.client;

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

    async_ctx = tcp_sockcm->super.iface.worker->async;
    status    = ucs_async_set_event_handler(async_ctx->mode, cep->fd,
                                            UCS_EVENT_SET_EVWRITE,
                                            uct_tcp_sa_data_handler, cep,
                                            async_ctx);
    if (status != UCS_OK) {
        goto err_close_socket;
    }

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

    uct_tcp_sockcm_ep_init_comm_ctx(self);
    self->state        = 0;
    self->comm_ctx.buf = ucs_malloc(uct_tcp_sockcm_ep_get_cm(self)->priv_data_len +
                                    sizeof(uct_tcp_sockcm_priv_data_hdr_t),
                                    "tcp_sockcm priv data");
    if (self->comm_ctx.buf == NULL) {
        ucs_error("failed to allocate memory for the ep's send/recv buf");
        return UCS_ERR_NO_MEMORY;
    }

    if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR) {
        status = uct_tcp_sockcm_ep_client_init(self, params);
    } else {
        status = uct_tcp_sockcm_ep_server_init(self, params);
    }

    if (status == UCS_OK) {
        ucs_debug("created an endpoint on tcp_sockcm %p id: %d state: %d",
                  uct_tcp_sockcm_ep_get_cm(self), self->fd, self->state);
    }

    return status;
}

ucs_status_t uct_tcp_sockcm_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p)
{
    uct_tcp_sockcm_ep_t *tcp_ep;

    if (params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR) {
        /* create a new endpoint for the client side */
        return UCS_CLASS_NEW(uct_tcp_sockcm_ep_t, ep_p, params);
    } else if (params->field_mask & UCT_EP_PARAM_FIELD_CONN_REQUEST) {
        /* the server's endpoint was already created by the listener, return it */
        tcp_ep = (uct_tcp_sockcm_ep_t*)(params->conn_request);
        *ep_p  = &tcp_ep->super.super.super;
        return UCS_OK;
    } else {
        ucs_error("either UCT_EP_PARAM_FIELD_SOCKADDR or UCT_EP_PARAM_FIELD_CONN_REQUEST "
                  "has to be provided");
        return UCS_ERR_INVALID_PARAM;
    }
}

UCS_CLASS_CLEANUP_FUNC(uct_tcp_sockcm_ep_t)
{
    uct_tcp_sockcm_t *tcp_sockcm = uct_tcp_sockcm_ep_get_cm(self);

    UCS_ASYNC_BLOCK(tcp_sockcm->super.iface.worker->async);

    ucs_free(self->comm_ctx.buf);

    ucs_async_remove_handler(self->fd, 1);

    if (self->fd != -1) {
        close(self->fd);
    }
    UCS_ASYNC_UNBLOCK(tcp_sockcm->super.iface.worker->async);
}

UCS_CLASS_DEFINE(uct_tcp_sockcm_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t);
