/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "wireup_cm.h"
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_listener.h>
#include <ucp/wireup/wireup.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>


unsigned
ucp_cm_ep_init_flags(const ucp_worker_h worker, const ucp_ep_params_t *params)
{
    if (!ucp_worker_sockaddr_is_cm_proto(worker)) {
        return 0;
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR) {
        return UCP_EP_INIT_CM_WIREUP_CLIENT;
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_CONN_REQUEST) {
        return UCP_EP_INIT_CM_WIREUP_SERVER;
    }

    return 0;
}

static ucs_status_t ucp_cm_ep_client_do_initial_config(ucp_ep_h ucp_ep,
                                                       const char *dev_name)
{
    ucp_worker_h worker        = ucp_ep->worker;
    ucp_ep_config_key_t key    = ucp_ep_config(ucp_ep)->key;
    uint64_t addr_pack_flags   = UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR |
                                 UCP_ADDRESS_PACK_FLAG_IFACE_ADDR;
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    uint64_t tl_bitmap         = ucp_context_dev_tl_bitmap(worker->context,
                                                           dev_name);
    void *ucp_addr;
    size_t ucp_addr_size;
    ucp_unpacked_address_t unpacked_addr;
    unsigned addr_indices[UCP_MAX_RESOURCES];
    ucs_status_t status;

    ucs_assert_always(wireup_ep != NULL);

    /* Construct local dummy address for lanes selection taking an assumption
     * that server has the transports which are the best from client's
     * perspective. */
    status = ucp_address_pack(worker, NULL, tl_bitmap, addr_pack_flags, NULL,
                              &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_address_unpack(worker, ucp_addr, addr_pack_flags,
                                &unpacked_addr);
    if (status != UCS_OK) {
        goto free_ucp_addr;
    }

    ucs_assert(unpacked_addr.address_count <= UCP_MAX_RESOURCES);
    status = ucp_wireup_select_lanes(ucp_ep, wireup_ep->ep_init_flags,
                                     tl_bitmap, &unpacked_addr, addr_indices,
                                     &key);
    if (status != UCS_OK) {
        goto free_addr_list;
    }

    status = ucp_worker_get_ep_config(worker, &key, 0, &ucp_ep->cfg_index);
    if (status != UCS_OK) {
        goto free_addr_list;
    }

    ucp_ep->am_lane = key.am_lane;

free_addr_list:
    ucs_free(unpacked_addr.address_list);
free_ucp_addr:
    ucs_free(ucp_addr);
out:
    return status;
}

static ssize_t ucp_cm_client_priv_pack_cb(void *arg, const char *dev_name,
                                          void *priv_data)
{
    ucp_wireup_sockaddr_data_t *sa_data = priv_data;
    ucp_ep_h ep                         = arg;
    ucp_worker_h worker                 = ep->worker;
    uct_cm_h cm                         = worker->cms[/*cm_idx = */ 0].cm;
    uint64_t tl_bitmap;
    uct_ep_h tl_ep;
    uct_cm_attr_t cm_attr;
    uct_ep_params_t tl_ep_params;
    void* ucp_addr;
    size_t ucp_addr_size;
    ucs_status_t status;
    ucp_lane_index_t lane_idx;
    ucp_rsc_index_t rsc_idx;

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_cm_ep_client_do_initial_config(ep, dev_name);
    if (status != UCS_OK) {
        goto out;
    }

    cm_attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
    status             = uct_cm_query(cm, &cm_attr);
    if (status != UCS_OK) {
        goto out;
    }

    tl_bitmap = 0;
    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(ep); ++lane_idx) {
        if (lane_idx == ucp_ep_get_cm_lane(ep)) {
            continue;
        }

        rsc_idx = ucp_ep_get_rsc_index(ep, lane_idx);
        if (rsc_idx == UCP_NULL_RESOURCE) {
            continue;
        }

        status = ucp_wireup_ep_create(ep, &ep->uct_eps[lane_idx]);
        if (status != UCS_OK) {
            goto out;
        }

        tl_bitmap |= UCS_BIT(rsc_idx);
        if (ucp_worker_is_tl_p2p(worker, rsc_idx)) {
            tl_ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
            tl_ep_params.iface      = ucp_worker_iface(worker, rsc_idx)->iface;
            status = uct_ep_create(&tl_ep_params, &tl_ep);
            if (status != UCS_OK) {
                goto out;
            }

            ucp_wireup_assign_lane(ep, lane_idx, tl_ep, "sockaddr TL lane");
        } else {
            ucs_assert(ucp_worker_iface_get_attr(worker, rsc_idx)->cap.flags &
                       UCT_IFACE_FLAG_CONNECT_TO_IFACE);
        }
    }

    /* Don't pack the device address to reduce address size, it will be
     * delivered by uct_listener_conn_request_callback_t in
     * uct_cm_remote_data_t */
    status = ucp_address_pack(worker, ep, tl_bitmap,
                              UCP_ADDRESS_PACK_FLAG_IFACE_ADDR |
                              UCP_ADDRESS_PACK_FLAG_EP_ADDR,
                              NULL, &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out;
    }

    if (cm_attr.max_conn_priv < (sizeof(*sa_data) + ucp_addr_size)) {
        ucs_error("CM private data buffer is to small to pack UCP endpoint info, "
                  "ep %p service data %lu, address length %lu, cm %p max_conn_priv %lu",
                  ep, sizeof(*sa_data), ucp_addr_size, cm,
                  cm_attr.max_conn_priv);
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto free_addr;
    }

    sa_data->ep_ptr    = (uintptr_t)ep;
    sa_data->err_mode  = ucp_ep_config(ep)->key.err_mode;
    sa_data->addr_mode = UCP_WIREUP_SA_DATA_CM_ADDR;
    memcpy(sa_data + 1, ucp_addr, ucp_addr_size);

free_addr:
    ucs_free(ucp_addr);
out:
    if (status != UCS_OK) {
        ucp_worker_set_ep_failed(worker, ep,
                                 &ucp_ep_get_cm_wireup_ep(ep)->super.super,
                                 ucp_ep_get_cm_lane(ep), status);
    }
    UCS_ASYNC_UNBLOCK(&worker->async);
    return (status == UCS_OK) ? (sizeof(*sa_data) + ucp_addr_size) : status;
}

static void ucp_cm_client_connect_cb(uct_ep_h ep, void *arg,
                                     const uct_cm_remote_data_t *remote_data,
                                     ucs_status_t status)
{
    ucs_error("UCP CM wireup is not completely implemented");
}

static void ucp_cm_disconnect_cb(uct_ep_h ep, void *arg)
{
    ucs_error("UCP close protocol is not completely implemented");
}

ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params)
{
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucp_worker_h worker        = ucp_ep->worker;
    uct_ep_params_t cm_lane_params;
    ucs_status_t status;

    wireup_ep->ep_init_flags  = ucp_ep_init_flags(ucp_ep->worker, params);

    cm_lane_params.field_mask = UCT_EP_PARAM_FIELD_CM                    |
                                UCT_EP_PARAM_FIELD_USER_DATA             |
                                UCT_EP_PARAM_FIELD_SOCKADDR              |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS     |
                                UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB      |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB   |
                                UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    cm_lane_params.user_data                  = ucp_ep;
    cm_lane_params.sockaddr                   = &params->sockaddr;
    cm_lane_params.sockaddr_cb_flags          = UCT_CB_FLAG_ASYNC;
    cm_lane_params.sockaddr_pack_cb           = ucp_cm_client_priv_pack_cb;
    cm_lane_params.sockaddr_connect_cb.client = ucp_cm_client_connect_cb;
    cm_lane_params.disconnect_cb              = ucp_cm_disconnect_cb;

    ucs_assert_always(ucp_worker_num_cm_cmpts(worker) == 1);
    cm_lane_params.cm = worker->cms[0].cm;
    status = uct_ep_create(&cm_lane_params, &wireup_ep->sockaddr_ep);
    if (status == UCS_OK) {
        ucp_ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    }
    return status;
}

static unsigned ucp_cm_server_conn_request_progress(void *arg)
{
    ucp_conn_request_h               conn_request = arg;
    ucp_listener_h                   listener     = conn_request->listener;
    ucp_worker_h                     worker       = listener->worker;
    ucp_ep_h                         ep;
    ucs_status_t                     status;

    ucs_trace_func("listener %p, connect request %p", listener, conn_request);

    if (listener->conn_cb) {
        listener->conn_cb(conn_request, listener->arg);
        return 1;
    }

    UCS_ASYNC_BLOCK(&worker->async);
    status = ucp_ep_create_server_accept(worker, conn_request, &ep);
    if (status != UCS_OK) {
        ucs_warn("server endpoint creation with connect request %p failed, status %s",
                  conn_request, ucs_status_string(status));
    }
    UCS_ASYNC_UNBLOCK(&worker->async);
    ucs_free(conn_request);
    return 1;
}

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const char *local_dev_name,
                                   uct_conn_request_h conn_request,
                                   const uct_cm_remote_data_t *remote_data)
{
    ucp_listener_h ucp_listener = arg;
    uct_worker_cb_id_t prog_id  = UCS_CALLBACKQ_ID_NULL;
    ucp_conn_request_h ucp_conn_request;
    ucs_status_t status;

    ucp_conn_request = ucs_malloc(ucs_offsetof(ucp_conn_request_t, sa_data) +
                                  remote_data->conn_priv_data_length,
                                  "ucp_conn_request_h");
    if (ucp_conn_request == NULL) {
        ucs_error("failed to allocate connect request, rejecting connection request %p on TL listener %p",
                  conn_request, listener);
        goto err_reject;
    }

    ucp_conn_request->remote_dev_addr = ucs_malloc(remote_data->dev_addr_length,
                                                   "remote device address");
    if (ucp_conn_request->remote_dev_addr == NULL) {
        ucs_error("failed to allocate device address, rejecting connection request %p on TL listener %p",
                  conn_request, listener);
        goto err_free_ucp_conn_request;
    }

    ucp_conn_request->listener     = ucp_listener;
    ucp_conn_request->uct.listener = listener;
    ucp_conn_request->uct_req      = conn_request;
    ucs_strncpy_safe(ucp_conn_request->dev_name, local_dev_name,
                     UCT_DEVICE_NAME_MAX);
    memcpy(ucp_conn_request->remote_dev_addr, remote_data->dev_addr,
           remote_data->dev_addr_length);
    memcpy(&ucp_conn_request->sa_data, remote_data->conn_priv_data,
           remote_data->conn_priv_data_length);

    uct_worker_progress_register_safe(ucp_listener->worker->uct,
                                      ucp_cm_server_conn_request_progress,
                                      ucp_conn_request,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);

    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(ucp_listener->worker);
    return;

err_free_ucp_conn_request:
    ucs_free(ucp_conn_request);
err_reject:
    status = uct_listener_reject(listener, conn_request);
    if (status != UCS_OK) {
        ucs_warn("failed to reject connect request %p on listener %p",
                 conn_request, listener);
    }
}

ucs_status_t
ucp_ep_cm_server_create_connected(ucp_worker_h worker, unsigned ep_init_flags,
                                  const ucp_unpacked_address_t *remote_addr,
                                  ucp_conn_request_h conn_request,
                                  ucp_ep_h *ep_p)
{
    ucp_ep_h ep;
    ucp_wireup_ep_t *wireup_ep;
    ucp_lane_index_t lane_idx;
    ucs_status_t status;

    /* Create and connect TL part */
    status = ucp_ep_create_to_worker_addr(worker, remote_addr, ep_init_flags,
                                          "conn_request on uct_listener", &ep);
    if (status != UCS_OK) {
        return status;
    }

    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(ep); ++lane_idx) {
        if (lane_idx == ucp_ep_get_cm_lane(ep)) {
            status = ucp_ep_cm_connect_server_lane(ep, conn_request);
            if (status != UCS_OK) {
                return status;
            }
        } else {
            wireup_ep = ucp_wireup_ep(ep->uct_eps[lane_idx]);
            if (wireup_ep != NULL) {
                wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
            }
        }
    }

    ucp_listener_schedule_accept_cb(ep);
    *ep_p = ep;
    return UCS_OK;
}

static ssize_t ucp_cm_server_priv_pack_cb(void *arg, const char *dev_name,
                                          void *priv_data)
{
    ucs_error("UCP CM wireup is not completely implemented");
    return UCS_ERR_NOT_IMPLEMENTED;
}

static void ucp_cm_server_connect_cb(uct_ep_h ep, void *arg,
                                     ucs_status_t status)
{
    ucs_error("UCP CM wireup is not completely implemented");
}

ucs_status_t ucp_ep_cm_connect_server_lane(ucp_ep_h ep,
                                           ucp_conn_request_h conn_request)
{
    ucp_worker_h worker   = ep->worker;
    ucp_lane_index_t lane = ucp_ep_get_cm_lane(ep);
    uct_ep_params_t uct_ep_params;
    uct_ep_h uct_ep;
    ucs_status_t status;

    ucs_assert(lane != UCP_NULL_LANE);
    ucs_assert(ep->uct_eps[lane] == NULL);

    /* create a server side CM endpoint */
    ucs_trace("ep %p: uct_ep[%d]", ep, lane);
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                    |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST          |
                               UCT_EP_PARAM_FIELD_USER_DATA             |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS     |
                               UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB      |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB   |
                               UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    ucs_assertv_always(ucp_worker_num_cm_cmpts(worker) == 1,
                       "multiple CMs are not supported");
    uct_ep_params.cm                         = worker->cms[0].cm;
    uct_ep_params.user_data                  = ep;
    uct_ep_params.conn_request               = conn_request->uct_req;
    uct_ep_params.sockaddr_cb_flags          = UCT_CB_FLAG_ASYNC;
    uct_ep_params.sockaddr_pack_cb           = ucp_cm_server_priv_pack_cb;
    uct_ep_params.sockaddr_connect_cb.server = ucp_cm_server_connect_cb;
    uct_ep_params.disconnect_cb              = ucp_cm_disconnect_cb;

    status = uct_ep_create(&uct_ep_params, &uct_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    ucp_wireup_assign_lane(ep, lane, uct_ep, "server side cm lane");
    return UCS_OK;
}
