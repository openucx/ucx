/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_listener.h"

#include <ucp/stream/stream.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/debug/log.h>
#include <ucs/sys/sock.h>


static unsigned ucp_listener_accept_cb_progress(void *arg)
{
    ucp_ep_h       ep       = arg;
    ucp_listener_h listener = ucp_ep_ext_gen(ep)->listener;

    /* NOTE: protect union */
    ucs_assert(!(ep->flags & (UCP_EP_FLAG_ON_MATCH_CTX |
                              UCP_EP_FLAG_FLUSH_STATE_VALID)));
    ucs_assert(ep->flags   & UCP_EP_FLAG_LISTENER);

    ep->flags &= ~UCP_EP_FLAG_LISTENER;
    ep->flags |= UCP_EP_FLAG_USED;
    ucp_stream_ep_activate(ep);
    ucp_ep_flush_state_reset(ep);

    /*
     * listener is NULL if the EP was created with UCP_EP_PARAM_FIELD_EP_ADDR
     * and we are here because long address requires wireup protocol
     */
    if (listener && listener->accept_cb) {
        listener->accept_cb(ep, listener->arg);
    }

    return 1;
}

int ucp_listener_accept_cb_remove_filter(const ucs_callbackq_elem_t *elem,
                                                void *arg)
{
    ucp_ep_h ep = elem->arg;

    return (elem->cb == ucp_listener_accept_cb_progress) && (ep == arg);
}

void ucp_listener_schedule_accept_cb(ucp_ep_h ep)
{
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;

    uct_worker_progress_register_safe(ep->worker->uct,
                                      ucp_listener_accept_cb_progress,
                                      ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);
}

static unsigned ucp_listener_conn_request_progress(void *arg)
{
    ucp_conn_request_h               conn_request = arg;
    ucp_listener_h                   listener     = conn_request->listener;
    const ucp_wireup_client_data_t   *client_data = &conn_request->client_data;
    ucp_worker_h                     worker;
    ucp_ep_h                         ep;
    ucs_status_t                     status;
    ucp_worker_iface_t               *listening_wiface;

    ucs_trace_func("listener=%p", listener);

    if (listener->conn_cb) {
        listener->conn_cb(conn_request, listener->arg);
        return 1;
    }

    listening_wiface = &listener->wifaces[conn_request->wiface_idx];
    worker           = listening_wiface->worker;

    UCS_ASYNC_BLOCK(&worker->async);
    /* coverity[overrun-buffer-val] */
    status = ucp_ep_create_accept(worker, client_data, &ep);

    if (status != UCS_OK) {
        goto out;
    }

    if (ep->flags & UCP_EP_FLAG_LISTENER) {
        status = ucp_wireup_send_pre_request(ep);
    } else {
        /* send wireup request message, to connect the client to the server's
           new endpoint */
        ucs_assert(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED));
        status = ucp_wireup_send_request(ep);
    }

    if (status != UCS_OK) {
        goto out;
    }

    status = uct_iface_accept(listening_wiface->iface, conn_request->uct_req);
    if (status != UCS_OK) {
        ucp_ep_destroy_internal(ep);
        goto out;
    }

    if (listener->accept_cb != NULL) {
        if (ep->flags & UCP_EP_FLAG_LISTENER) {
            ucs_assert(!(ep->flags & UCP_EP_FLAG_USED));
            ucp_ep_ext_gen(ep)->listener = listener;
        } else {
            ep->flags |= UCP_EP_FLAG_USED;
            listener->accept_cb(ep, listener->arg);
        }
    }

out:
    if (status != UCS_OK) {
        ucs_error("connection request failed on listener %p with status %s",
                  listener, ucs_status_string(status));
        uct_iface_reject(listening_wiface->iface, conn_request->uct_req);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    ucs_free(conn_request);
    return 1;
}

static int ucp_listener_remove_filter(const ucs_callbackq_elem_t *elem,
                                      void *arg)
{
    ucp_listener_h *listener = elem->arg;

    return (elem->cb == ucp_listener_conn_request_progress) && (listener == arg);
}

static void ucp_listener_conn_request_callback(uct_iface_h tl_iface, void *arg,
                                               uct_conn_request_h uct_req,
                                               const void *conn_priv_data,
                                               size_t length)
{
    ucp_listener_h     listener = arg;
    uct_worker_cb_id_t prog_id  = UCS_CALLBACKQ_ID_NULL;
    ucp_conn_request_h conn_request;
    int                i;

    ucs_trace("listener %p: got connection request", listener);

    /* Defer wireup init and user's callback to be invoked from the main thread */
    conn_request = ucs_malloc(ucs_offsetof(ucp_conn_request_t, client_data) +
                              length, "accept connection request");
    if (conn_request == NULL) {
        ucs_error("failed to allocate connect request, "
                  "rejecting connection request %p on TL iface %p, reason %s",
                  uct_req, tl_iface, ucs_status_string(UCS_ERR_NO_MEMORY));
        uct_iface_reject(tl_iface, uct_req);
        return;
    }

    conn_request->listener = listener;
    conn_request->uct_req  = uct_req;
    memcpy(&conn_request->client_data, conn_priv_data, length);

    /* Find the iface the connection request was received on and save its
     * index for future usage by the listener */
    for (i = 0; i < listener->num_wifaces; i++) {
        if (listener->wifaces[i].iface == tl_iface) {
            conn_request->wiface_idx = i;

            uct_worker_progress_register_safe(listener->wifaces[i].worker->uct,
                                              ucp_listener_conn_request_progress,
                                              conn_request, UCS_CALLBACKQ_FLAG_ONESHOT,
                                              &prog_id);

            /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
             * that he can wake-up on this event */
            ucp_worker_signal_internal(listener->wifaces[i].worker);
            return;
        }
    }

    ucs_error("connection request received on listener %p on an unknown interface",
              listener);
    uct_iface_reject(tl_iface, uct_req);
    ucs_free(conn_request);
}

ucs_status_t ucp_listener_query(ucp_listener_h listener, ucp_listener_attr_t *attr)
{
    int i, port;

    ucs_assert(listener->num_wifaces > 0);
    port = listener->wifaces[0].attr.listen_port;

    /* Make sure that all the listening sockaddr ifaces are listening on the same port */
    for (i = 1; i < listener->num_wifaces; i++) {
        if (port != listener->wifaces[i].attr.listen_port) {
            ucs_error("different ports detected on the listener: %d and %d",
                      port, listener->wifaces[i].attr.listen_port);
            return UCS_ERR_IO_ERROR;
        }
    }

    if (attr->field_mask & UCP_LISTENER_ATTR_FIELD_PORT) {
        attr->port = port;
    }

    return UCS_OK;
}

static void ucp_listener_close_ifaces(ucp_listener_h listener)
{
    int i;

    for (i = 0; i < listener->num_wifaces; i++) {
        /* remove pending slow-path progress in case it wasn't removed yet */
        ucs_callbackq_remove_if(&listener->wifaces[i].worker->uct->progress_q,
                                ucp_listener_remove_filter, listener);
        ucp_worker_iface_cleanup(&listener->wifaces[i]);
    }

    ucs_free(listener->wifaces);
}

ucs_status_t ucp_listener_create(ucp_worker_h worker,
                                 const ucp_listener_params_t *params,
                                 ucp_listener_h *listener_p)
{
    ucp_context_h context   = worker->context;
    ucp_listener_h listener = NULL;
    int sockaddr_tls        = 0;
    char saddr_str[UCS_SOCKADDR_STRING_LEN];
    ucp_tl_resource_desc_t *resource;
    uct_iface_params_t iface_params;
    ucp_worker_iface_t *tmp;
    ucp_rsc_index_t tl_id;
    ucs_status_t status;
    ucp_tl_md_t *tl_md;
    uint16_t port;
    int i;

    if (!(params->field_mask & UCP_LISTENER_PARAM_FIELD_SOCK_ADDR)) {
        ucs_error("Missing sockaddr for listener");
        return UCS_ERR_INVALID_PARAM;
    }

    UCP_CHECK_PARAM_NON_NULL(params->sockaddr.addr, status, return status);

    if (ucs_test_all_flags(params->field_mask,
                           UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER |
                           UCP_LISTENER_PARAM_FIELD_CONN_HANDLER)) {
        ucs_error("Only one accept handler should be provided");
        return UCS_ERR_INVALID_PARAM;
    }

    UCS_ASYNC_BLOCK(&worker->async);

    listener = ucs_calloc(1, sizeof(*listener), "ucp_listener");
    if (listener == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    if (params->field_mask & UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER) {
        UCP_CHECK_PARAM_NON_NULL(params->accept_handler.cb, status,
                                 goto err_free_listener);
        listener->accept_cb = params->accept_handler.cb;
        listener->arg       = params->accept_handler.arg;
    } else if (params->field_mask & UCP_LISTENER_PARAM_FIELD_CONN_HANDLER) {
        UCP_CHECK_PARAM_NON_NULL(params->conn_handler.cb, status,
                                 goto err_free_listener);
        listener->conn_cb   = params->conn_handler.cb;
        listener->arg       = params->conn_handler.arg;
    }

    status = ucs_sockaddr_get_port(params->sockaddr.addr, &port);
    if (status != UCS_OK) {
       goto err_free_listener;
    }

    /* Go through all the available resources and for each one, check if the given
     * sockaddr is accessible from its md. Start listening on all the mds that
     * satisfy this.
     * If the given port is set to 0, i.e. use a random port, the first transport
     * in the sockaddr priority list from the environment configuration will
     * dictate the port to listen on for the other sockaddr transports in the list.
     * */
    for (i = 0; i < context->config.num_sockaddr_tls; i++) {
        tl_id    = context->config.sockaddr_tl_ids[i];
        resource = &context->tl_rscs[tl_id];
        tl_md    = &context->tl_mds[resource->md_index];

        if (!uct_md_is_sockaddr_accessible(tl_md->md, &params->sockaddr,
                                           UCT_SOCKADDR_ACC_LOCAL)) {
            continue;
        }

        tmp = ucs_realloc(listener->wifaces,
                          sizeof(*listener->wifaces) * (sockaddr_tls + 1),
                          "listener wifaces");
        if (tmp == NULL) {
            ucs_error("failed to allocate listener wifaces");
            status = UCS_ERR_NO_MEMORY;
            goto err_close_listener_wifaces;
        }

        listener->wifaces = tmp;

        iface_params.field_mask                     = UCT_IFACE_PARAM_FIELD_OPEN_MODE |
                                                      UCT_IFACE_PARAM_FIELD_SOCKADDR;
        iface_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER;
        iface_params.mode.sockaddr.conn_request_cb  = ucp_listener_conn_request_callback;
        iface_params.mode.sockaddr.conn_request_arg = listener;
        iface_params.mode.sockaddr.listen_sockaddr  = params->sockaddr;
        iface_params.mode.sockaddr.cb_flags         = UCT_CB_FLAG_ASYNC;

        if (port) {
            /* Set the port for the next sockaddr iface. This port was either
             * obtained from the user or generated by the first created sockaddr
             * iface if the port from the user was equal to zero */
            status = ucs_sockaddr_set_port((struct sockaddr *)
                                           iface_params.mode.sockaddr.listen_sockaddr.addr,
                                           port);
            if (status != UCS_OK) {
                ucs_error("failed to set port parameter (%d) for creating %s iface",
                          listener->wifaces[sockaddr_tls].attr.listen_port,
                          resource->tl_rsc.tl_name);
                goto err_close_listener_wifaces;
            }
        }

        status = ucp_worker_iface_open(worker, tl_id, &iface_params,
                                       &listener->wifaces[sockaddr_tls]);
        if (status != UCS_OK) {
            ucs_error("failed to open listener on %s on md %s",
                      ucs_sockaddr_str(iface_params.mode.sockaddr.listen_sockaddr.addr,
                                       saddr_str, sizeof(saddr_str)),
                      tl_md->rsc.md_name);
            goto err_close_listener_wifaces;
        }

        status = ucp_worker_iface_init(worker, tl_id, &listener->wifaces[sockaddr_tls]);
        if ((status != UCS_OK) ||
            ((context->config.features & UCP_FEATURE_WAKEUP) &&
             !(listener->wifaces[sockaddr_tls].attr.cap.flags & UCT_IFACE_FLAG_CB_ASYNC))) {
            ucp_worker_iface_cleanup(&listener->wifaces[sockaddr_tls]);
            goto err_close_listener_wifaces;
        }

        port = listener->wifaces[sockaddr_tls].attr.listen_port;

        sockaddr_tls++;
        listener->num_wifaces = sockaddr_tls;
        ucs_trace("listener %p: accepting connections on %s on %s",
                  listener, tl_md->rsc.md_name,
                  ucs_sockaddr_str(iface_params.mode.sockaddr.listen_sockaddr.addr,
                                   saddr_str, sizeof(saddr_str)));
    }

    if (!sockaddr_tls) {
        ucs_error("none of the available transports can listen for connections on %s",
                  ucs_sockaddr_str(params->sockaddr.addr, saddr_str, sizeof(saddr_str)));
        listener->num_wifaces = 0;
        status = UCS_ERR_UNREACHABLE;
    } else {
        *listener_p           = listener;
        status                = UCS_OK;
        goto out;
    }

err_close_listener_wifaces:
    ucp_listener_close_ifaces(listener);
err_free_listener:
    ucs_free(listener);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

void ucp_listener_destroy(ucp_listener_h listener)
{
    ucs_trace("listener %p: destroying", listener);

    ucp_listener_close_ifaces(listener);
    ucs_free(listener);
}

ucs_status_t ucp_listener_reject(ucp_listener_h listener,
                                 ucp_conn_request_h conn_request)
{
    ucp_worker_h worker = listener->wifaces[conn_request->wiface_idx].worker;

    UCS_ASYNC_BLOCK(&worker->async);

    uct_iface_reject(listener->wifaces[conn_request->wiface_idx].iface,
                     conn_request->uct_req);

    UCS_ASYNC_UNBLOCK(&worker->async);

    ucs_free(conn_request);

    return UCS_OK;
}
