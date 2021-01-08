/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "sockcm_ep.h"
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <netinet/tcp.h>
#include <uct/tcp/tcp.h>

#define UCT_SOCKCM_CB_FLAGS_CHECK(_flags) \
    do { \
        UCT_CB_FLAGS_CHECK(_flags); \
        if (!((_flags) & UCT_CB_FLAG_ASYNC)) { \
            return UCS_ERR_UNSUPPORTED; \
        } \
    } while (0)

ucs_status_t uct_sockcm_ep_set_sock_id(uct_sockcm_ep_t *ep)
{
    ucs_status_t status;
    struct sockaddr *dest_addr = NULL;

    ep->sock_id_ctx = ucs_malloc(sizeof(*ep->sock_id_ctx), "client sock_id_ctx");
    if (ep->sock_id_ctx == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dest_addr = (struct sockaddr *) &(ep->remote_addr);

    status = ucs_socket_create(dest_addr->sa_family, SOCK_STREAM,
                               &ep->sock_id_ctx->sock_fd);
    if (status != UCS_OK) {
        ucs_debug("unable to create client socket for sockcm");
        ucs_free(ep->sock_id_ctx);
        return status;
    }

    return UCS_OK;
}

void uct_sockcm_ep_put_sock_id(uct_sockcm_ctx_t *sock_id_ctx)
{
    close(sock_id_ctx->sock_fd);
    ucs_free(sock_id_ctx);
}

ucs_status_t uct_sockcm_ep_send_client_info(uct_sockcm_ep_t *ep)
{
    uct_sockcm_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                               uct_sockcm_iface_t);
    uct_cm_ep_priv_data_pack_args_t pack_args;
    uct_sockcm_conn_param_t conn_param;
    char dev_name[UCT_DEVICE_NAME_MAX];
    ucs_status_t status;

    memset(&conn_param, 0, sizeof(uct_sockcm_conn_param_t));

    /* get interface name associated with the connected client fd; use that for pack_cb */
    status = ucs_sockaddr_get_ifname(ep->sock_id_ctx->sock_fd, dev_name,
                                     UCT_DEVICE_NAME_MAX);
    if (UCS_OK != status) {
        goto out;
    }

    pack_args.field_mask = UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME;
    ucs_strncpy_safe(pack_args.dev_name, dev_name, UCT_DEVICE_NAME_MAX);

    conn_param.length = ep->pack_cb(ep->pack_cb_arg, &pack_args,
                                    (void*)conn_param.private_data);
    if (conn_param.length < 0) {
        ucs_error("sockcm client (iface=%p, ep = %p) failed to fill "
                  "private data. status: %s",
                  iface, ep, ucs_status_string((ucs_status_t)conn_param.length));
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    ucs_assert(conn_param.length <= UCT_SOCKCM_PRIV_DATA_LEN);

    status = ucs_socket_send(ep->sock_id_ctx->sock_fd, &conn_param,
                             sizeof(uct_sockcm_conn_param_t));

out:
    return status;
}

static const char*
uct_sockcm_ep_conn_state_str(uct_sockcm_ep_conn_state_t state)
{
    switch (state) {
    case UCT_SOCKCM_EP_CONN_STATE_SOCK_CONNECTING:
        return "UCT_SOCKCM_EP_CONN_STATE_SOCK_CONNECTING";
    case UCT_SOCKCM_EP_CONN_STATE_INFO_SENT:
        return "UCT_SOCKCM_EP_CONN_STATE_INFO_SENT";
    case UCT_SOCKCM_EP_CONN_STATE_CLOSED:
        return "UCT_SOCKCM_EP_CONN_STATE_CLOSED";
    case UCT_SOCKCM_EP_CONN_STATE_CONNECTED:
        return "UCT_SOCKCM_EP_CONN_STATE_CONNECTED";
    default:
        ucs_fatal("invaild sockcm endpoint state %d", state);
    }
}

static void uct_sockcm_change_state(uct_sockcm_ep_t *ep,
                                    uct_sockcm_ep_conn_state_t conn_state,
                                    ucs_status_t status)
{
    uct_sockcm_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                               uct_sockcm_iface_t);

    pthread_mutex_lock(&ep->ops_mutex);
    ucs_debug("changing ep with status %s from state %s to state %s, status %s",
              ucs_status_string(ep->status),
              uct_sockcm_ep_conn_state_str(ep->conn_state),
              uct_sockcm_ep_conn_state_str(conn_state),
              ucs_status_string(status));
    if ((ep->status != UCS_OK) &&
        (ep->conn_state == UCT_SOCKCM_EP_CONN_STATE_CLOSED)) {
        /* Do not handle failure twice for closed EP */
        pthread_mutex_unlock(&ep->ops_mutex);
        return;
    }

    ep->status     = status;
    ep->conn_state = conn_state;

    if (conn_state == UCT_SOCKCM_EP_CONN_STATE_CLOSED) {
        uct_sockcm_ep_set_failed(&iface->super.super, &ep->super.super, status);
    }

    uct_sockcm_ep_invoke_completions(ep, status);
    pthread_mutex_unlock(&ep->ops_mutex);
}

static void uct_sockcm_handle_sock_connect(uct_sockcm_ep_t *ep)
{
    char sockaddr_str[UCS_SOCKADDR_STRING_LEN];
    int fd = ep->sock_id_ctx->sock_fd;
    ucs_status_t status;

    if (!ucs_socket_is_connected(fd)) {
        ucs_error("failed to connect to %s",
                  ucs_sockaddr_str((struct sockaddr*)&ep->remote_addr,
                                   sockaddr_str, sizeof(sockaddr_str)));
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CLOSED,
                                UCS_ERR_UNREACHABLE);
        goto err;
    } 

    status = uct_sockcm_ep_send_client_info(ep);
    if (status != UCS_OK) {
        ucs_error("failed to send client info: %s", ucs_status_string(status));
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CLOSED, status);
        goto err;
    }

    ep->conn_state = UCT_SOCKCM_EP_CONN_STATE_INFO_SENT;

    /* Call current handler when server responds to sent message */
    if (UCS_OK != ucs_async_modify_handler(fd, UCS_EVENT_SET_EVREAD)) {
        ucs_error("failed to modify async handler for fd %d", fd);
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CLOSED,
                                UCS_ERR_IO_ERROR);
        goto err;
    }

    return;

err:
    status = ucs_async_modify_handler(fd, 0);
    if (status != UCS_OK) {
        ucs_debug("unable to modify handler");
    }
}

static void uct_sockcm_handle_info_sent(uct_sockcm_ep_t *ep)
{
    ucs_status_t status;
    size_t recv_len;
    char notif_val;

    recv_len = sizeof(notif_val);
    status   = ucs_socket_recv_nb(ep->sock_id_ctx->sock_fd, &notif_val,
                                  &recv_len);
    if (UCS_ERR_NO_PROGRESS == status) {
        /* will call recv again when ready */
        return;
    }

    ucs_async_remove_handler(ep->sock_id_ctx->sock_fd, 0);

    if (UCS_OK != status) {
        /* receive notif failed, close the connection */
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CLOSED, status);
        return;
    }

    if (notif_val == UCT_SOCKCM_IFACE_NOTIFY_ACCEPT) {
        ucs_debug("event_handler OK after accept");
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CONNECTED, UCS_OK);
    } else {
        ucs_debug("event_handler REJECTED after reject");
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CLOSED,
                                UCS_ERR_REJECTED);
    }
}

static void uct_sockcm_ep_event_handler(int fd, ucs_event_set_types_t events,
                                        void *arg)
{
    uct_sockcm_ep_t *ep = (uct_sockcm_ep_t *) arg;

    switch (ep->conn_state) {
    case UCT_SOCKCM_EP_CONN_STATE_SOCK_CONNECTING:
        uct_sockcm_handle_sock_connect(ep);
        break;
    case UCT_SOCKCM_EP_CONN_STATE_INFO_SENT:
        uct_sockcm_handle_info_sent(ep);
        break;
    case UCT_SOCKCM_EP_CONN_STATE_CONNECTED:
        if (UCS_OK != ucs_async_modify_handler(fd, 0)) {
            ucs_warn("unable to turn off event notifications on %d", fd);
        }
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CONNECTED, UCS_OK);
        break;
    case UCT_SOCKCM_EP_CONN_STATE_CLOSED:
    default:
        ucs_debug("handling closed/default state, ep %p fd %d", ep, fd);
        uct_sockcm_change_state(ep, UCT_SOCKCM_EP_CONN_STATE_CLOSED,
                                UCS_ERR_IO_ERROR);
        break;
    }
}

static UCS_CLASS_INIT_FUNC(uct_sockcm_ep_t, const uct_ep_params_t *params)
{
    const ucs_sock_addr_t *sockaddr = params->sockaddr;
    uct_sockcm_iface_t    *iface    = NULL;
    struct sockaddr *param_sockaddr = NULL;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    size_t sockaddr_len;

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

    param_sockaddr = (struct sockaddr *) sockaddr->addr;
    if (UCS_OK != ucs_sockaddr_sizeof(param_sockaddr, &sockaddr_len)) {
       ucs_error("sockcm ep: unknown remote sa_family=%d",
                 sockaddr->addr->sa_family);
       status = UCS_ERR_IO_ERROR;
       goto err;
    }

    memcpy(&self->remote_addr, param_sockaddr, sockaddr_len);

    self->slow_prog_id = UCS_CALLBACKQ_ID_NULL;

    status = uct_sockcm_ep_set_sock_id(self);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_sys_fcntl_modfl(self->sock_id_ctx->sock_fd, O_NONBLOCK, 0); 
    if (status != UCS_OK) {
        goto sock_err;
    }

    status = ucs_socket_connect(self->sock_id_ctx->sock_fd, param_sockaddr);
    if (UCS_STATUS_IS_ERR(status)) {
        self->conn_state = UCT_SOCKCM_EP_CONN_STATE_CLOSED;
        goto sock_err;
    }

    self->conn_state = UCT_SOCKCM_EP_CONN_STATE_SOCK_CONNECTING; 
    self->status     = UCS_INPROGRESS;

    /* set ep->status before event handler call to avoid simultaneous writes to state*/
    status = ucs_async_set_event_handler(iface->super.worker->async->mode,
                                         self->sock_id_ctx->sock_fd,
                                         UCS_EVENT_SET_EVWRITE,
                                         uct_sockcm_ep_event_handler,
                                         self, iface->super.worker->async);
    if (status != UCS_OK) {
        goto sock_err;
    }

    ucs_debug("created an SOCKCM endpoint on iface %p, "
              "remote addr: %s", iface,
               ucs_sockaddr_str(param_sockaddr,
                                ip_port_str, UCS_SOCKADDR_STRING_LEN));
    return UCS_OK;

sock_err:
    uct_sockcm_ep_put_sock_id(self->sock_id_ctx);
err:
    ucs_debug("error in sock connect");
    pthread_mutex_destroy(&self->ops_mutex);

    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sockcm_ep_t)
{
    uct_sockcm_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                               uct_sockcm_iface_t);

    ucs_debug("sockcm_ep %p: destroying", self);

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    ucs_async_remove_handler(self->sock_id_ctx->sock_fd, 1);
    uct_sockcm_ep_put_sock_id(self->sock_id_ctx);

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

    UCS_ASYNC_BLOCK(iface->super.worker->async);

    ucs_trace_func("err_handle ep=%p", sockcm_ep);

    sockcm_ep->slow_prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_iface_handle_ep_err(&iface->super.super, &sockcm_ep->super.super,
                            sockcm_ep->status);

    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
    return 0;
}

void uct_sockcm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status)
{
    uct_sockcm_iface_t *sockcm_iface = ucs_derived_of(iface, uct_sockcm_iface_t);
    uct_sockcm_ep_t *sockcm_ep       = ucs_derived_of(ep, uct_sockcm_ep_t);

    if (sockcm_iface->super.err_handler_flags & UCT_CB_FLAG_ASYNC) {
        uct_iface_handle_ep_err(iface, ep, sockcm_ep->status);
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
