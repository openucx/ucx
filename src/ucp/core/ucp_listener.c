/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_listener.h"

#include <ucp/wireup/wireup_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>


static unsigned ucp_listener_accept_cb_progress(void *arg)
{
    ucp_ep_h ep = arg;

    ep->flags |= UCP_EP_FLAG_USED;
    ucp_ep_ext_gen(ep)->listener->cb(ep, ucp_ep_ext_gen(ep)->listener->arg);

    return 0;
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
    ucp_listener_accept_t *accept = arg;
    ucp_ep_h ep = accept->ep;
    ucs_status_t status;

    ucs_trace_func("listener=%p ep=%p", accept->listener, ep);

    if (!(ep->flags & UCP_EP_FLAG_LISTENER)) {
        /* send wireup request message, to connect the client to the server's new endpoint */
        ucs_assert(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED));
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
            goto err_destroy_ep;
        }
    } else {
        status = ucp_wireup_send_pre_request(ep);
        if (status != UCS_OK) {
            goto err_destroy_ep;
        }
    }

    if (accept->listener->cb != NULL) {
        if (ep->flags & UCP_EP_FLAG_LISTENER) {
            ep->flags |= UCP_EP_FLAG_HIDDEN;
            ucp_ep_ext_gen(ep)->listener = accept->listener;
        } else {
            ep->flags &= ~UCP_EP_FLAG_HIDDEN;
            accept->listener->cb(ep, accept->listener->arg);
        }
    }


    goto out;

err_destroy_ep:
    ucp_ep_destroy_internal(ep);
out:
    ucs_free(accept);
    return 0;
}

static int ucp_listener_remove_filter(const ucs_callbackq_elem_t *elem,
                                      void *arg)
{
    ucp_listener_h *listener = elem->arg;

    return (elem->cb == ucp_listener_conn_request_progress) && (listener == arg);
}

static ucs_status_t ucp_listener_conn_request_callback(void *arg,
                                                       const void *conn_priv_data,
                                                       size_t length)
{
    const ucp_wireup_sockaddr_priv_t *client_data = conn_priv_data;
    ucp_listener_h listener                       = arg;
    ucp_unpacked_address_t remote_address;
    ucp_listener_accept_t *accept;
    uct_worker_cb_id_t prog_id;
    ucp_ep_params_t params;
    ucs_status_t status;
    ucp_ep_h ep;

    ucs_trace("listener %p: got connection request", listener);

    params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    params.err_mode   = client_data->err_mode;

    status = ucp_address_unpack(client_data + 1, &remote_address);
    if (status != UCS_OK) {
        goto err;
    }

    if (client_data->is_full_addr) {
        /* create endpoint to the worker address we got in the private data */
        status = ucp_ep_create_to_worker_addr(listener->wiface.worker, &params,
                                              &remote_address, UCP_EP_CREATE_AM_LANE,
                                              "listener", &ep);
        if (status != UCS_OK) {
            goto err_free_address;
        }

    } else {
        status = ucp_ep_create_sockaddr_aux(listener->wiface.worker,
                                            &params, &remote_address,
                                            &ep);
        if (status != UCS_OK) {
            goto err_free_address;
        }

        /* the listener's ep should be aware of the sent address from the client */
        ep->flags |= UCP_EP_FLAG_LISTENER;
    }

    ucp_ep_update_dest_ep_ptr(ep, client_data->ep_ptr);

    /* Defer wireup init and user's callback to be invoked from the main thread */
    accept = ucs_malloc(sizeof(*accept), "ucp_listener_accept");
    if (accept == NULL) {
        ucs_error("failed to allocate listener accept context");
        status = UCS_ERR_NO_MEMORY;
        goto err_destroy_ep;
    }

    accept->listener = listener;
    accept->ep       = ep;

    prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_worker_progress_register_safe(listener->wiface.worker->uct,
                                      ucp_listener_conn_request_progress,
                                      accept, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);


    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(listener->wiface.worker);

    ucs_free(remote_address.address_list);

    return UCS_OK;

err_destroy_ep:
    ucp_ep_destroy_internal(ep);
err_free_address:
    ucs_free(remote_address.address_list);
err:
    return status;
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

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCS_ASYNC_BLOCK(&worker->async);

    if (!(params->field_mask & UCP_LISTENER_PARAM_FIELD_SOCK_ADDR)) {
        ucs_error("Missing sockaddr for listener");
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    UCP_CHECK_PARAM_NON_NULL(params->sockaddr.addr, status, goto out);

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

        listener = ucs_malloc(sizeof(*listener), "ucp_listener");
        if (listener == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        if (params->field_mask & UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER) {
            UCP_CHECK_PARAM_NON_NULL(params->accept_handler.cb, status, goto err_free);
            listener->cb  = params->accept_handler.cb;
            listener->arg = params->accept_handler.arg;
        } else {
            listener->cb  = NULL;
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
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
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
