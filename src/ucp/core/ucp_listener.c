/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_listener.h"
#include "ucp_vfs.h"
#include "uct/base/uct_cm.h"

#include <ucp/stream/stream.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucp/wireup/wireup_cm.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/debug/log.h>
#include <ucs/sys/sock.h>
#include <ucs/type/serialize.h>
#include <ucs/vfs/base/vfs_obj.h>


static unsigned ucp_listener_accept_cb_progress(void *arg)
{
    ucp_conn_request_h conn_request = arg;
    ucp_listener_h listener         = conn_request->listener;
    ucp_ep_h ep                     = conn_request->ep;

    ucs_free(conn_request->remote_dev_addr);
    ucs_free(conn_request);

    UCS_ASYNC_BLOCK(&ep->worker->async);

    ucp_ep_update_flags(ep, UCP_EP_FLAG_USED, 0);
    ucp_stream_ep_activate(ep);
    ++ep->worker->counters.ep_creations;

    UCS_ASYNC_UNBLOCK(&ep->worker->async);

    listener->accept_cb(ep, listener->arg);
    return 1;
}

int ucp_listener_accept_cb_remove_filter(const ucs_callbackq_elem_t *elem,
                                         void *arg)
{
    ucp_conn_request_h conn_request = elem->arg;

    return (elem->cb == ucp_listener_accept_cb_progress) &&
           (conn_request->ep == arg);
}

void ucp_listener_schedule_accept_cb(ucp_conn_request_h conn_request)
{
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;

    uct_worker_progress_register_safe(conn_request->ep->worker->uct,
                                      ucp_listener_accept_cb_progress,
                                      conn_request,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);
}

ucs_status_t ucp_conn_request_query(ucp_conn_request_h conn_request,
                                    ucp_conn_request_attr_t *attr)
{
    ucp_wireup_sockaddr_data_base_t *sa_data;
    ucs_status_t status;
    uint8_t sa_data_ver;
    void *ucp_addr;

    if (attr->field_mask & UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR) {
        if (conn_request->client_address.ss_family == 0) {
            return UCS_ERR_UNSUPPORTED;
        }

        status = ucs_sockaddr_copy((struct sockaddr *)&attr->client_address,
                                   (struct sockaddr *)&conn_request->client_address);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (attr->field_mask & UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID) {
        sa_data     = (ucp_wireup_sockaddr_data_base_t*)(conn_request + 1);
        sa_data_ver = sa_data->header >> UCP_SA_DATA_HEADER_VERSION_SHIFT;
        ucp_addr    = UCS_PTR_BYTE_OFFSET(sa_data,
                                          ucp_cm_sa_data_length(sa_data_ver));
         /* coverity[overrun-local] */
        attr->client_id = ucp_address_get_client_id(ucp_addr);
    }

    return UCS_OK;
}

ucs_status_t ucp_listener_query(ucp_listener_h listener,
                                ucp_listener_attr_t *attr)
{
    ucs_status_t status;

    if (attr->field_mask & UCP_LISTENER_ATTR_FIELD_SOCKADDR) {
        status = ucs_sockaddr_copy((struct sockaddr *)&attr->sockaddr,
                                   (struct sockaddr *)&listener->sockaddr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucp_listener_reset_uct_listeners(ucp_listener_h listener)
{
    ucp_rsc_index_t i;

    for (i = 0; i < listener->num_rscs; ++i) {
        uct_listener_destroy(listener->listeners[i]);
        listener->listeners[i] = NULL;
    }

    listener->num_rscs = 0;
}

static void ucp_listener_free_uct_listeners(ucp_listener_h listener)
{
    ucp_listener_reset_uct_listeners(listener);
    ucs_free(listener->listeners);
    listener->listeners = NULL;
}

static ucs_status_t
ucp_listen(ucp_listener_h listener, const ucp_listener_params_t *params)
{
    ucp_worker_h          worker  = listener->worker;
    const ucp_rsc_index_t num_cms = ucp_worker_num_cm_cmpts(worker);
    struct sockaddr_storage addr_storage;
    struct sockaddr       *addr;
    uct_listener_h        *uct_listeners;
    uct_listener_params_t uct_params;
    uct_listener_attr_t   uct_attr;
    uint16_t              port, uct_listen_port;
    ucp_rsc_index_t       cm_index;
    char                  addr_str[UCS_SOCKADDR_STRING_LEN];
    ucp_worker_cm_t       *ucp_cm;
    ucs_status_t          status;
    int                   use_any_port;
    ucs_log_level_t       log_level;

    addr = (struct sockaddr *)&addr_storage;
    status = ucs_sockaddr_copy(addr, params->sockaddr.addr);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_sockaddr_get_port(addr, &port);
    if (status != UCS_OK) {
        return status;
    }

    use_any_port = (port == 0);

    ucs_assert_always(num_cms > 0);

    uct_params.field_mask       = UCT_LISTENER_PARAM_FIELD_CONN_REQUEST_CB |
                                  UCT_LISTENER_PARAM_FIELD_USER_DATA;
    uct_params.conn_request_cb  = ucp_cm_server_conn_request_cb;
    uct_params.user_data        = listener;

    if (worker->context->config.ext.listener_backlog != UCS_ULUNITS_AUTO) {
        uct_params.field_mask |= UCT_LISTENER_PARAM_FIELD_BACKLOG;
        uct_params.backlog     = worker->context->config.ext.listener_backlog;
    }

    listener->num_rscs          = 0;
    uct_listeners               = ucs_calloc(num_cms, sizeof(*uct_listeners),
                                             "uct_listeners_arr");
    if (uct_listeners == NULL) {
        ucs_error("Can't allocate memory for UCT listeners array");
        return UCS_ERR_NO_MEMORY;
    }

    listener->listeners = uct_listeners;

    cm_index = 0;
    while (cm_index < num_cms) {
        ucp_cm = &worker->cms[cm_index++];
        if (ucp_cm->cm == NULL) {
            continue;
        }

        status = uct_listener_create(ucp_cm->cm, addr,
                                     params->sockaddr.addrlen, &uct_params,
                                     &uct_listeners[listener->num_rscs]);
        if (status == UCS_OK) {
            ++listener->num_rscs;
            status = ucs_sockaddr_get_port(addr, &port);
            if (status != UCS_OK) {
                goto err_free_listeners;
            }

            uct_attr.field_mask = UCT_LISTENER_ATTR_FIELD_SOCKADDR;
            status              =
                    uct_listener_query(uct_listeners[listener->num_rscs - 1],
                                       &uct_attr);
            if (status != UCS_OK) {
                goto err_free_listeners;
            }

            status = ucs_sockaddr_get_port((struct sockaddr *)&uct_attr.sockaddr,
                                           &uct_listen_port);
            if (status != UCS_OK) {
                goto err_free_listeners;
            }

            if (port != uct_listen_port) {
                ucs_assert(port == 0);
                status = ucs_sockaddr_set_port(addr, uct_listen_port);
                if (status != UCS_OK) {
                    goto err_free_listeners;
                }
            }
        } else if ((status == UCS_ERR_BUSY) && use_any_port) {
            /* retry another port */
            status = ucs_sockaddr_set_port(addr, 0);
            if (status != UCS_OK) {
                goto err_free_listeners;
            }

            ucp_listener_reset_uct_listeners(listener);
            /* TODO: to reduce probability of "any port busy" need to create
             *       TCP listener first */
            cm_index = 0;
        } else {
            log_level = ((status == UCS_ERR_BUSY) ||
                         (status == UCS_ERR_NO_DEVICE)) ? UCS_LOG_LEVEL_DIAG :
                        UCS_LOG_LEVEL_ERROR;
            ucs_log(log_level,
                    "failed to create UCT listener on CM %p (component %s) "
                    "with address %s status %s", ucp_cm->cm,
                    worker->context->tl_cmpts[ucp_cm->cmpt_idx].attr.name,
                    ucs_sockaddr_str(params->sockaddr.addr, addr_str,
                                     UCS_SOCKADDR_STRING_LEN),
                    ucs_status_string(status));
            if (status != UCS_ERR_NO_DEVICE) {
                goto err_free_listeners;
            }
        }
    }

    if (listener->num_rscs == 0) {
        ucs_assert(status != UCS_OK);
        goto err_free_listeners;
    }

    status = ucs_sockaddr_copy((struct sockaddr *)&listener->sockaddr, addr);
    if (status != UCS_OK) {
        goto err_free_listeners;
    }

    return UCS_OK;

err_free_listeners:
    ucp_listener_free_uct_listeners(listener);
    /* if no listener was created, return the status of the last call of
     * uct_listener_create. else, return the error status that invoked this label. */
    return status;
}

static void ucp_listener_vfs_show_ip(void *obj, ucs_string_buffer_t *strb,
                                     void *arg_ptr, uint64_t arg_u64)
{
    ucp_listener_h listener   = obj;
    struct sockaddr *sockaddr = (struct sockaddr*)&listener->sockaddr;

    ucp_vfs_read_ip(sockaddr, strb);
}

static void ucp_listener_vfs_show_port(void *obj, ucs_string_buffer_t *strb,
                                       void *arg_ptr, uint64_t arg_u64)
{
    ucp_listener_h listener   = obj;
    struct sockaddr *sockaddr = (struct sockaddr*)&listener->sockaddr;

    ucp_vfs_read_port(sockaddr, strb);
}

void ucp_listener_vfs_init(ucp_listener_h listener)
{
    ucs_vfs_obj_add_dir(listener->worker, listener, "listener/%p", listener);
    ucs_vfs_obj_add_ro_file(listener, ucp_listener_vfs_show_ip, NULL, 0, "ip");
    ucs_vfs_obj_add_ro_file(listener, ucp_listener_vfs_show_port, NULL, 0,
                            "port");
}

ucs_status_t ucp_listener_create(ucp_worker_h worker,
                                 const ucp_listener_params_t *params,
                                 ucp_listener_h *listener_p)
{
    const unsigned handlers_mask = UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER |
                                   UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    ucp_listener_h listener;
    ucs_status_t   status;

    if (!(params->field_mask & UCP_LISTENER_PARAM_FIELD_SOCK_ADDR)) {
        ucs_error("missing sockaddr for listener");
        return UCS_ERR_INVALID_PARAM;
    }

    UCP_CHECK_PARAM_NON_NULL(params->sockaddr.addr, status, return status);

    if (ucs_test_all_flags(params->field_mask, handlers_mask) ||
        !(params->field_mask & handlers_mask)) {
        ucs_error("one and only one accept handler should be provided");
        return UCS_ERR_INVALID_PARAM;
    }

    if (ucp_worker_num_cm_cmpts(worker) == 0) {
        ucs_error("cannot create listener: none of the available components supports it");
        return UCS_ERR_UNSUPPORTED;
    }

    listener = ucs_calloc(1, sizeof(*listener), "ucp_listener");
    if (listener == NULL) {
        ucs_error("cannot allocate memory for UCP listener");
        return UCS_ERR_NO_MEMORY;
    }

    UCS_ASYNC_BLOCK(&worker->async);

    listener->worker = worker;

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

    status = ucp_listen(listener, params);
    if (status == UCS_OK) {
        ucp_listener_vfs_init(listener);
        *listener_p = listener;
        goto out;
    }

err_free_listener:
    ucs_free(listener);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

void ucp_listener_destroy(ucp_listener_h listener)
{
    ucs_debug("listener %p: destroying", listener);

    UCS_ASYNC_BLOCK(&listener->worker->async);
    ucs_vfs_obj_remove(listener);
    ucs_callbackq_remove_if(&listener->worker->uct->progress_q,
                            ucp_cm_server_conn_request_progress_cb_pred,
                            listener);
    UCS_ASYNC_UNBLOCK(&listener->worker->async);

    ucp_listener_free_uct_listeners(listener);
    ucs_free(listener);
}

ucs_status_t ucp_listener_reject(ucp_listener_h listener,
                                 ucp_conn_request_h conn_request)
{
    ucp_worker_h worker = listener->worker;

    ucs_trace("listener %p: free conn_request %p", listener, conn_request);

    UCS_ASYNC_BLOCK(&worker->async);
    uct_listener_reject(conn_request->uct_listener, conn_request->uct_req);
    ucs_free(conn_request->remote_dev_addr);
    UCS_ASYNC_UNBLOCK(&worker->async);

    ucs_free(conn_request);

    return UCS_OK;
}
