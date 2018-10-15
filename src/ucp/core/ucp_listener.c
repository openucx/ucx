/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_listener.h"

#include <ucp/stream/stream.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>


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

    ucs_trace_func("listener=%p", listener);

    if (listener->conn_cb) {
        listener->conn_cb(conn_request, listener->arg);
        return 1;
    }

    worker = listener->wiface.worker;
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
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

    status = uct_iface_accept(listener->wiface.iface, conn_request->uct_req);
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
        uct_iface_reject(listener->wiface.iface, conn_request->uct_req);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
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

    ucs_trace("listener %p: got connection request", listener);

    /* Defer wireup init and user's callback to be invoked from the main thread */
    conn_request = ucs_malloc(ucs_offsetof(ucp_conn_request_t, client_data) +
                              length, "accept connection request");
    if (conn_request == NULL) {
        ucs_error("failed to allocate connect request, rejecting connection request %p on TL iface %p, reason %s",
                  uct_req, tl_iface, ucs_status_string(UCS_ERR_NO_MEMORY));
        uct_iface_reject(tl_iface, uct_req);
        return;
    }

    conn_request->listener = listener;
    conn_request->uct_req  = uct_req;
    memcpy(&conn_request->client_data, conn_priv_data, length);

    uct_worker_progress_register_safe(listener->wiface.worker->uct,
                                      ucp_listener_conn_request_progress,
                                      conn_request, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);

    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(listener->wiface.worker);
}

ucs_status_t ucp_listener_create(ucp_worker_h worker,
                                 const ucp_listener_params_t *params,
                                 ucp_listener_h *listener_p)
{
    ucp_context_h context = worker->context;
    ucp_tl_resource_desc_t *resource;
    uct_iface_params_t iface_params;
    ucp_listener_h listener = NULL;
    ucp_rsc_index_t tl_id;
    ucs_status_t status;
    ucp_tl_md_t *tl_md;
    char saddr_str[UCS_SOCKADDR_STRING_LEN];

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

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
    UCS_ASYNC_BLOCK(&worker->async);

    /* Go through all the available resources and for each one, check if the given
     * sockaddr is accessible from its md. Start listening on the first md that
     * satisfies this.
     * */
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        resource = &context->tl_rscs[tl_id];
        tl_md    = &context->tl_mds[resource->md_index];

        if (!(tl_md->attr.cap.flags & UCT_MD_FLAG_SOCKADDR) ||
            !uct_md_is_sockaddr_accessible(tl_md->md, &params->sockaddr,
                                           UCT_SOCKADDR_ACC_LOCAL)) {
            continue;
        }

        listener = ucs_calloc(1, sizeof(*listener), "ucp_listener");
        if (listener == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        if (params->field_mask & UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER) {
            UCP_CHECK_PARAM_NON_NULL(params->accept_handler.cb, status,
                                     goto err_free);
            listener->accept_cb = params->accept_handler.cb;
            listener->arg       = params->accept_handler.arg;
        } else if (params->field_mask & UCP_LISTENER_PARAM_FIELD_CONN_HANDLER) {
            UCP_CHECK_PARAM_NON_NULL(params->conn_handler.cb, status,
                                     goto err_free);
            listener->conn_cb   = params->conn_handler.cb;
            listener->arg       = params->conn_handler.arg;
        }

        memset(&iface_params, 0, sizeof(iface_params));
        iface_params.open_mode                      = UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER;
        iface_params.mode.sockaddr.conn_request_cb  = ucp_listener_conn_request_callback;
        iface_params.mode.sockaddr.conn_request_arg = listener;
        iface_params.mode.sockaddr.listen_sockaddr  = params->sockaddr;
        iface_params.mode.sockaddr.cb_flags         = UCT_CB_FLAG_ASYNC;

        status = ucp_worker_iface_init(worker, tl_id, &iface_params,
                                       &listener->wiface);
        if (status != UCS_OK) {
            goto err_free;
        }

        if ((context->config.features & UCP_FEATURE_WAKEUP) &&
            !(listener->wiface.attr.cap.flags & UCT_IFACE_FLAG_CB_ASYNC)) {
            ucp_worker_iface_cleanup(&listener->wiface);
            ucs_free(listener);
            continue;
        }

        ucs_trace("listener %p: accepting connections on %s", listener,
                  tl_md->rsc.md_name);

        *listener_p = listener;
        status      = UCS_OK;
        goto out;
    }

    ucs_error("none of the available transports can listen for connections on %s",
              ucs_sockaddr_str(params->sockaddr.addr, saddr_str, sizeof(saddr_str)));
    status = UCS_ERR_UNREACHABLE;
    goto out;

err_free:
    ucs_free(listener);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;
}

void ucp_listener_destroy(ucp_listener_h listener)
{
    ucs_trace("listener %p: destroying", listener);

    /* remove pending slow-path progress in case it wasn't removed yet */
    ucs_callbackq_remove_if(&listener->wiface.worker->uct->progress_q,
                            ucp_listener_remove_filter, listener);
    ucp_worker_iface_cleanup(&listener->wiface);
    ucs_free(listener);
}

ucs_status_t ucp_listener_reject(ucp_listener_h listener,
                                 ucp_conn_request_h conn_request)
{
    ucp_worker_h worker = listener->wiface.worker;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
    UCS_ASYNC_BLOCK(&worker->async);

    uct_iface_reject(listener->wiface.iface, conn_request->uct_req);

    UCS_ASYNC_UNBLOCK(&worker->async);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    ucs_free(conn_request);

    return UCS_OK;
}
