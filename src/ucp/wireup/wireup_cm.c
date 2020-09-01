/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "wireup_cm.h"
#include <ucp/core/ucp_listener.h>
#include <ucp/core/ucp_request.inl>
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

int ucp_ep_init_flags_has_cm(unsigned ep_init_flags)
{
    return !!(ep_init_flags & (UCP_EP_INIT_CM_WIREUP_CLIENT |
                               UCP_EP_INIT_CM_WIREUP_SERVER));
}

static ucs_status_t
ucp_cm_ep_client_initial_config_get(ucp_ep_h ucp_ep, const char *dev_name,
                                    ucp_ep_config_key_t *key)
{
    ucp_worker_h worker        = ucp_ep->worker;
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
    ucp_ep_config_key_reset(key);
    ucp_ep_config_key_set_err_mode(key, wireup_ep->ep_init_flags);
    status = ucp_wireup_select_lanes(ucp_ep, wireup_ep->ep_init_flags,
                                     tl_bitmap, &unpacked_addr, addr_indices,
                                     key);

    ucs_free(unpacked_addr.address_list);
free_ucp_addr:
    ucs_free(ucp_addr);
out:
    return status;
}

static void ucp_cm_priv_data_pack(ucp_wireup_sockaddr_data_t *sa_data,
                                  ucp_ep_h ep, ucp_rsc_index_t dev_index,
                                  const ucp_address_t *addr, size_t addr_size)
{
    ucs_assert((int)ucp_ep_config(ep)->key.err_mode <= UINT8_MAX);
    ucs_assert(dev_index != UCP_NULL_RESOURCE);

    sa_data->ep_id     = ucp_ep_local_id(ep);
    sa_data->err_mode  = ucp_ep_config(ep)->key.err_mode;
    sa_data->addr_mode = UCP_WIREUP_SA_DATA_CM_ADDR;
    sa_data->dev_index = dev_index;
    memcpy(sa_data + 1, addr, addr_size);
}

static ssize_t ucp_cm_client_priv_pack_cb(void *arg,
                                          const uct_cm_ep_priv_data_pack_args_t
                                          *pack_args, void *priv_data)
{
    ucp_wireup_sockaddr_data_t *sa_data = priv_data;
    ucp_ep_h ep                         = arg;
    ucp_worker_h worker                 = ep->worker;
    uct_cm_h cm                         = worker->cms[/*cm_idx = */ 0].cm;
    ucp_rsc_index_t dev_index           = UCP_NULL_RESOURCE;
    ucp_ep_config_key_t key;
    uint64_t tl_bitmap;
    ucp_wireup_ep_t *cm_wireup_ep;
    uct_cm_attr_t cm_attr;
    void* ucp_addr;
    size_t ucp_addr_size;
    ucs_status_t status;
    ucp_lane_index_t lane_idx;
    ucp_rsc_index_t rsc_idx;
    const char *dev_name;
    ucp_ep_h tmp_ep;
    uint8_t path_index;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_assert_always(pack_args->field_mask &
                      UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME);

    dev_name = pack_args->dev_name;

    status = ucp_cm_ep_client_initial_config_get(ep, dev_name, &key);
    if (status != UCS_OK) {
        goto out;
    }

    /* At this point the ep has only CM lane */
    ucs_assert((ucp_ep_num_lanes(ep) == 1) && ucp_ep_has_cm_lane(ep));
    cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    ucs_assert(cm_wireup_ep != NULL);

    /* Create tmp ep which will hold local tl addresses until connect
     * event arrives, to avoid asynchronous ep reconfiguration. */
    status = ucp_ep_create_base(worker, "tmp_cm", "tmp cm client", &tmp_ep);
    if (status != UCS_OK) {
        goto out;
    }

    tmp_ep->flags       |= UCP_EP_FLAG_TEMPORARY;
    cm_wireup_ep->tmp_ep = tmp_ep;

    status = ucp_worker_get_ep_config(worker, &key, 0, &tmp_ep->cfg_index);
    if (status != UCS_OK) {
        goto out;
    }

    cm_attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
    status             = uct_cm_query(cm, &cm_attr);
    if (status != UCS_OK) {
        goto out;
    }

    tl_bitmap = 0;
    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(tmp_ep); ++lane_idx) {
        if (lane_idx == ucp_ep_get_cm_lane(tmp_ep)) {
            continue;
        }

        rsc_idx = ucp_ep_get_rsc_index(tmp_ep, lane_idx);
        if (rsc_idx == UCP_NULL_RESOURCE) {
            continue;
        }

        status = ucp_wireup_ep_create(tmp_ep, &tmp_ep->uct_eps[lane_idx]);
        if (status != UCS_OK) {
            goto out;
        }

        ucs_assert((dev_index == UCP_NULL_RESOURCE) ||
                   (dev_index == worker->context->tl_rscs[rsc_idx].dev_index));
        dev_index = worker->context->tl_rscs[rsc_idx].dev_index;

        tl_bitmap |= UCS_BIT(rsc_idx);
        if (ucp_ep_config(tmp_ep)->p2p_lanes & UCS_BIT(lane_idx)) {
            path_index = ucp_ep_get_path_index(tmp_ep, lane_idx);
            status     = ucp_wireup_ep_connect(tmp_ep->uct_eps[lane_idx], 0,
                                               rsc_idx, path_index, 0, NULL);
            if (status != UCS_OK) {
                goto out;
            }
        } else {
            ucs_assert(ucp_worker_is_tl_2iface(worker, rsc_idx));
        }
    }

    /* Don't pack the device address to reduce address size, it will be
     * delivered by uct_cm_listener_conn_request_callback_t in
     * uct_cm_remote_data_t */
    status = ucp_address_pack(worker, tmp_ep, tl_bitmap,
                              UCP_ADDRESS_PACK_FLAG_IFACE_ADDR |
                              UCP_ADDRESS_PACK_FLAG_EP_ADDR,
                              NULL, &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out;
    }

    if (cm_attr.max_conn_priv < (sizeof(*sa_data) + ucp_addr_size)) {
        ucs_error("CM private data buffer is too small to pack UCP endpoint info, "
                  "ep %p/%p service data %lu, address length %lu, cm %p max_conn_priv %lu",
                  ep, tmp_ep, sizeof(*sa_data), ucp_addr_size, cm,
                  cm_attr.max_conn_priv);
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto free_addr;
    }

    ucs_debug("client ep %p created on device %s idx %d, tl_bitmap 0x%"PRIx64,
              ep, dev_name, dev_index, tl_bitmap);
    /* Pass real ep (not tmp_ep), because only its pointer and err_mode is
     * taken from the config. */
    ucp_cm_priv_data_pack(sa_data, ep, dev_index, ucp_addr, ucp_addr_size);

free_addr:
    ucs_free(ucp_addr);
out:
    if (status == UCS_OK) {
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    } else {
        ucp_worker_set_ep_failed(worker, ep,
                                 &ucp_ep_get_cm_wireup_ep(ep)->super.super,
                                 ucp_ep_get_cm_lane(ep), status);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    /* coverity[leaked_storage] */
    return (status == UCS_OK) ? (sizeof(*sa_data) + ucp_addr_size) : status;
}

static void
ucp_cm_client_connect_prog_arg_free(ucp_cm_client_connect_progress_arg_t *arg)
{
    ucs_free(arg->sa_data);
    ucs_free(arg->dev_addr);
    ucs_free(arg);
}

static void ucp_cm_client_restore_ep(ucp_wireup_ep_t *wireup_cm_ep,
                                     ucp_ep_h ucp_ep)
{
    ucp_ep_h tmp_ep = wireup_cm_ep->tmp_ep;
    ucp_wireup_ep_t *w_ep;
    ucp_lane_index_t lane_idx;

    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(tmp_ep); ++lane_idx) {
        if (tmp_ep->uct_eps[lane_idx] != NULL) {
            ucs_assert(ucp_ep->uct_eps[lane_idx] == NULL);
            ucp_ep->uct_eps[lane_idx] = tmp_ep->uct_eps[lane_idx];
            w_ep = ucs_derived_of(ucp_ep->uct_eps[lane_idx], ucp_wireup_ep_t);
            w_ep->super.ucp_ep = ucp_ep;
        }
    }

    ucp_ep_destroy_base(tmp_ep); /* not needed anymore */
    wireup_cm_ep->tmp_ep = NULL;
}

/*
 * The main thread progress part of connection establishment on client side
 */
static unsigned ucp_cm_client_connect_progress(void *arg)
{
    ucp_cm_client_connect_progress_arg_t *progress_arg = arg;
    ucp_ep_h ucp_ep                                    = progress_arg->ucp_ep;
    ucp_worker_h worker                                = ucp_ep->worker;
    ucp_context_h context                              = worker->context;
    uct_ep_h uct_cm_ep                                 = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucp_wireup_ep_t *wireup_ep;
    ucp_unpacked_address_t addr;
    uint64_t tl_bitmap;
    ucp_rsc_index_t dev_index;
    ucp_rsc_index_t rsc_index;
    unsigned addr_idx;
    unsigned addr_indices[UCP_MAX_RESOURCES];
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_assert(wireup_ep != NULL);
    ucs_assert(wireup_ep->ep_init_flags & UCP_EP_INIT_CM_WIREUP_CLIENT);

    status = ucp_address_unpack(worker, progress_arg->sa_data + 1,
                                UCP_ADDRESS_PACK_FLAG_IFACE_ADDR |
                                UCP_ADDRESS_PACK_FLAG_EP_ADDR, &addr);
    if (status != UCS_OK) {
        goto out;
    }

    if (addr.address_count == 0) {
        status = UCS_ERR_UNREACHABLE;
        goto out_free_addr;
    }

    for (addr_idx = 0; addr_idx < addr.address_count; ++addr_idx) {
        addr.address_list[addr_idx].dev_addr  = progress_arg->dev_addr;
        addr.address_list[addr_idx].dev_index = progress_arg->sa_data->dev_index;
    }

    ucs_assert(addr.address_count <= UCP_MAX_RESOURCES);
    ucp_ep_update_remote_id(ucp_ep, progress_arg->sa_data->ep_id);

    /* Get tl bitmap from tmp_ep, because it contains initial configuration. */
    tl_bitmap = ucp_ep_get_tl_bitmap(wireup_ep->tmp_ep);
    ucs_assert(tl_bitmap != 0);
    rsc_index = ucs_ffs64(tl_bitmap);
    dev_index = context->tl_rscs[rsc_index].dev_index;

    /* Restore initial configuration from tmp_ep created for packing local
     * addresses. */
    ucp_cm_client_restore_ep(wireup_ep, ucp_ep);

#ifdef ENABLE_ASSERT
    ucs_for_each_bit(rsc_index, tl_bitmap) {
        ucs_assert(dev_index == context->tl_rscs[rsc_index].dev_index);
    }
#endif

    tl_bitmap = ucp_context_dev_idx_tl_bitmap(context, dev_index);
    status    = ucp_wireup_init_lanes(ucp_ep, wireup_ep->ep_init_flags,
                                      tl_bitmap, &addr, addr_indices);
    if (status != UCS_OK) {
        goto out_free_addr;
    }

    status = ucp_wireup_connect_local(ucp_ep, &addr, NULL);
    if (status != UCS_OK) {
        goto out_free_addr;
    }

    status = uct_cm_client_ep_conn_notify(uct_cm_ep);
    if (status != UCS_OK) {
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep->flags &= ~UCP_EP_FLAG_LOCAL_CONNECTED;
        goto out_free_addr;
    }

    ucp_wireup_remote_connected(ucp_ep);

out_free_addr:
    ucs_free(addr.address_list);
out:
    if (status != UCS_OK) {
        ucp_worker_set_ep_failed(worker, ucp_ep, &wireup_ep->super.super,
                                 ucp_ep_get_cm_lane(ucp_ep), status);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    ucp_cm_client_connect_prog_arg_free(progress_arg);
    return 1;
}

static ucs_status_t
ucp_cm_remote_data_check(const uct_cm_remote_data_t *remote_data)
{
    if (ucs_test_all_flags(remote_data->field_mask,
                           UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                           UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                           UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                           UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH)) {
        return UCS_OK;
    }

    ucs_error("incompatible client server connection establishment protocol");
    return UCS_ERR_UNSUPPORTED;
}

/*
 * Async callback on a client side which notifies that server is connected.
 */
static void ucp_cm_client_connect_cb(uct_ep_h uct_cm_ep, void *arg,
                                     const uct_cm_ep_client_connect_args_t
                                     *connect_args)
{
    ucp_ep_h ucp_ep            = (ucp_ep_h)arg;
    ucp_worker_h worker        = ucp_ep->worker;
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;
    ucp_cm_client_connect_progress_arg_t *progress_arg;
    const uct_cm_remote_data_t *remote_data;
    ucs_status_t status;

    ucs_assert_always(ucs_test_all_flags(connect_args->field_mask,
                                         (UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_REMOTE_DATA |
                                          UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_STATUS)));

    remote_data = connect_args->remote_data;
    status      = connect_args->status;

    if (status != UCS_OK) {
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep->flags &= ~UCP_EP_FLAG_LOCAL_CONNECTED;
        goto err_out;
    }

    status = ucp_cm_remote_data_check(remote_data);
    if (status != UCS_OK) {
        goto err_out;
    }

    progress_arg = ucs_malloc(sizeof(*progress_arg),
                              "ucp_cm_client_connect_progress_arg_t");
    if (progress_arg == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_out;
    }

    progress_arg->sa_data = ucs_malloc(remote_data->conn_priv_data_length,
                                        "sa data");
    if (progress_arg->sa_data == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_arg;
    }

    progress_arg->dev_addr = ucs_malloc(remote_data->dev_addr_length,
                                        "device address");
    if (progress_arg->dev_addr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_sa_data;
    }

    progress_arg->ucp_ep = ucp_ep;
    memcpy(progress_arg->dev_addr, remote_data->dev_addr,
           remote_data->dev_addr_length);
    memcpy(progress_arg->sa_data, remote_data->conn_priv_data,
           remote_data->conn_priv_data_length);

    uct_worker_progress_register_safe(worker->uct,
                                      ucp_cm_client_connect_progress,
                                      progress_arg, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);
    ucp_worker_signal_internal(ucp_ep->worker);
    return;

err_free_sa_data:
    ucs_free(progress_arg->sa_data);
err_free_arg:
    ucs_free(progress_arg);
err_out:
    UCS_ASYNC_BLOCK(&worker->async);
    ucp_worker_set_ep_failed(worker, ucp_ep, uct_cm_ep,
                             ucp_ep_get_cm_lane(ucp_ep), status);
    UCS_ASYNC_UNBLOCK(&worker->async);
}

/*
 * Internal flush completion callback which is a part of close protocol,
 * this flush was initiated by remote peer in disconnect callback on CM lane.
 */
static void ucp_ep_cm_disconnect_flushed_cb(ucp_request_t *req)
{
    ucp_ep_h ucp_ep            = req->send.ep;
    /* the EP can be closed/destroyed from err callback */
    ucs_async_context_t *async = &ucp_ep->worker->async;

    UCS_ASYNC_BLOCK(async);
    if (req->status == UCS_OK) {
        ucs_assert(ucp_ep_is_cm_local_connected(ucp_ep));
        ucp_ep_cm_disconnect_cm_lane(ucp_ep);
    } else if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        ucs_assert(!ucp_ep_is_cm_local_connected(ucp_ep));
    } else {
        /* 1) ucp_ep_close(force) is called from err callback which was invoked
              on remote connection reset
              TODO: remove this case when IB flush cancel is fixed (#4743),
                    moving QP to err state should move UCP EP to error state,
                    then ucp_worker_set_ep_failed disconnects CM lane
           2) transport err is also possible on flush
         */
        ucs_assert((req->status == UCS_ERR_CANCELED) ||
                   (req->status == UCS_ERR_ENDPOINT_TIMEOUT));
    }

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_CALLBACK));
    ucp_request_put(req);
    UCS_ASYNC_UNBLOCK(async);
}

static unsigned ucp_ep_cm_remote_disconnect_progress(void *arg)
{
    ucp_ep_h ucp_ep = arg;
    void *req;
    ucs_status_t status;

    ucs_trace("ep %p: flags 0x%x cm_remote_disconnect_progress", ucp_ep,
              ucp_ep->flags);

    ucs_assert(ucp_ep_get_cm_uct_ep(ucp_ep) != NULL);

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
    if (ucs_test_all_flags(ucp_ep->flags, UCP_EP_FLAG_CLOSED |
                                          UCP_EP_FLAG_CLOSE_REQ_VALID)) {
        ucp_request_complete_send(ucp_ep_ext_gen(ucp_ep)->close_req.req, UCS_OK);
        return 1;
    }

    if (ucp_ep->flags & UCP_EP_FLAG_CLOSED) {
        /* the ep is closed by API but close req is not valid yet (checked
         * above), it will be set later from scheduled
         * @ref ucp_ep_close_flushed_callback */
        ucs_debug("ep %p: ep closed but request is not set, waiting for the flush callback",
                  ucp_ep);
        return 1;
    }

    /*
     * TODO: set the ucp_ep to error state to prevent user from sending more
     *       ops.
     */
    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_FLUSH_STATE_VALID);
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_CLOSED));
    req = ucp_ep_flush_internal(ucp_ep, UCT_FLUSH_FLAG_LOCAL, 0,
                                &ucp_request_null_param, NULL,
                                ucp_ep_cm_disconnect_flushed_cb,
                                "cm_disconnected_cb");
    if (req == NULL) {
        /* flush is successfully completed in place, notify remote peer
         * that we are disconnected, the EP will be destroyed from API call */
        ucp_ep_cm_disconnect_cm_lane(ucp_ep);
    } else if (UCS_PTR_IS_ERR(req)) {
        status = UCS_PTR_STATUS(req);
        ucs_error("ucp_ep_flush_internal completed with error: %s",
                  ucs_status_string(status));
        goto err;
    }

    return 1;

err:
    ucp_worker_set_ep_failed(ucp_ep->worker, ucp_ep,
                             ucp_ep_get_cm_uct_ep(ucp_ep),
                             ucp_ep_get_cm_lane(ucp_ep), status);
    return 1;
}

static unsigned ucp_ep_cm_disconnect_progress(void *arg)
{
    ucp_ep_h ucp_ep            = arg;
    uct_ep_h uct_cm_ep         = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucs_async_context_t *async = &ucp_ep->worker->async;
    ucp_request_t *close_req;

    UCS_ASYNC_BLOCK(async);

    ucs_trace("ep %p: got remote disconnect, cm_ep %p, flags 0x%x", ucp_ep,
              uct_cm_ep, ucp_ep->flags);
    ucs_assert(ucp_ep_get_cm_uct_ep(ucp_ep) == uct_cm_ep);

    ucp_ep->flags &= ~UCP_EP_FLAG_REMOTE_CONNECTED;

    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        /* - ignore close event on failed ep, since all lanes are destroyed in
             generic err flow
           - if close req is valid this is ucp_ep_close_nb request and it will
             be completed as the ep is destroyed, i.e. flushed and disconnected
             with any status */
        if (ucp_ep->flags & UCP_EP_FLAG_CLOSE_REQ_VALID) {
            ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CLOSED);
        }
    } else if (ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED) {
        /* if the EP is local connected, need to flush it from main thread first */
        ucp_ep_cm_remote_disconnect_progress(ucp_ep);
        ucp_ep_invoke_err_cb(ucp_ep, UCS_ERR_CONNECTION_RESET);
    } else if (ucp_ep->flags & UCP_EP_FLAG_CLOSE_REQ_VALID) {
        /* if the EP is not local connected, the EP has been closed and flushed,
           CM lane is disconnected, complete close request and destroy EP */
        ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CLOSED);
        close_req = ucp_ep_ext_gen(ucp_ep)->close_req.req;
        ucp_ep_local_disconnect_progress(close_req);
    } else {
        ucs_warn("ep %p: unexpected state on disconnect, flags: 0x%u",
                 ucp_ep, ucp_ep->flags);
    }

    UCS_ASYNC_UNBLOCK(async);
    return 1;
}

static void ucp_cm_disconnect_cb(uct_ep_h uct_cm_ep, void *arg)
{
    ucp_ep_h ucp_ep            = arg;
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;

    ucs_trace("ep %p: CM remote disconnect callback invoked, flags 0x%x",
              ucp_ep, ucp_ep->flags);

    uct_worker_progress_register_safe(ucp_ep->worker->uct,
                                      ucp_ep_cm_disconnect_progress,
                                      ucp_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);
    ucp_worker_signal_internal(ucp_ep->worker);
}

ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params)
{
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucp_worker_h worker        = ucp_ep->worker;
    uct_ep_h cm_ep;
    uct_ep_params_t cm_lane_params;
    ucs_status_t status;

    wireup_ep->ep_init_flags  = ucp_ep_init_flags(ucp_ep->worker, params);

    cm_lane_params.field_mask = UCT_EP_PARAM_FIELD_CM                         |
                                UCT_EP_PARAM_FIELD_USER_DATA                  |
                                UCT_EP_PARAM_FIELD_SOCKADDR                   |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS          |
                                UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB           |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB_CLIENT |
                                UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    cm_lane_params.user_data          = ucp_ep;
    cm_lane_params.sockaddr           = &params->sockaddr;
    cm_lane_params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
    cm_lane_params.sockaddr_pack_cb   = ucp_cm_client_priv_pack_cb;
    cm_lane_params.sockaddr_cb_client = ucp_cm_client_connect_cb;
    cm_lane_params.disconnect_cb      = ucp_cm_disconnect_cb;
    ucs_assert_always(ucp_worker_num_cm_cmpts(worker) == 1);
    cm_lane_params.cm                 = worker->cms[0].cm;

    status = uct_ep_create(&cm_lane_params, &cm_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    ucp_wireup_ep_set_next_ep(&wireup_ep->super.super, cm_ep);
    ucp_ep_flush_state_reset(ucp_ep);

    return UCS_OK;
}

static unsigned ucp_cm_server_conn_request_progress(void *arg)
{
    ucp_conn_request_h               conn_request = arg;
    ucp_listener_h                   listener     = conn_request->listener;
    ucp_worker_h                     worker       = listener->worker;
    ucp_ep_h                         ep;

    ucs_trace_func("listener %p, connect request %p", listener, conn_request);

    if (listener->conn_cb) {
        listener->conn_cb(conn_request, listener->arg);
        return 1;
    }

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_ep_create_server_accept(worker, conn_request, &ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const uct_cm_listener_conn_request_args_t
                                   *conn_req_args)
{
    ucp_listener_h ucp_listener = arg;
    uct_worker_cb_id_t prog_id  = UCS_CALLBACKQ_ID_NULL;
    ucp_conn_request_h ucp_conn_request;
    uct_conn_request_h conn_request;
    const uct_cm_remote_data_t *remote_data;
    ucs_status_t status;

    ucs_assert_always(ucs_test_all_flags(conn_req_args->field_mask,
                                         (UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST |
                                          UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_REMOTE_DATA  |
                                          UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_DEV_NAME     |
                                          UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CLIENT_ADDR)));

    conn_request = conn_req_args->conn_request;
    remote_data  = conn_req_args->remote_data;

    status = ucp_cm_remote_data_check(remote_data);
    if (status != UCS_OK) {
        goto err_reject;
    }

    ucp_conn_request = ucs_malloc(ucs_offsetof(ucp_conn_request_t, sa_data) +
                                  remote_data->conn_priv_data_length,
                                  "ucp_conn_request_h");
    if (ucp_conn_request == NULL) {
        ucs_error("failed to allocate connect request, rejecting connection "
                  "request %p on TL listener %p",
                  conn_request, listener);
        goto err_reject;
    }

    ucp_conn_request->remote_dev_addr = ucs_malloc(remote_data->dev_addr_length,
                                                   "remote device address");
    if (ucp_conn_request->remote_dev_addr == NULL) {
        ucs_error("failed to allocate device address, rejecting connection "
                  "request %p on TL listener %p",
                  conn_request, listener);
        goto err_free_ucp_conn_request;
    }

    ucp_conn_request->listener     = ucp_listener;
    ucp_conn_request->uct.listener = listener;
    ucp_conn_request->uct_req      = conn_request;

    status = ucs_sockaddr_copy((struct sockaddr *)&ucp_conn_request->client_address,
                               conn_req_args->client_address.addr);
    if (status != UCS_OK) {
        goto err_free_remote_dev_addr;
    }

    ucs_strncpy_safe(ucp_conn_request->dev_name, conn_req_args->dev_name,
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

err_free_remote_dev_addr:
    ucs_free(ucp_conn_request->remote_dev_addr);
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
    uint64_t tl_bitmap = ucp_context_dev_tl_bitmap(worker->context,
                                                   conn_request->dev_name);
    ucp_ep_h ep;
    ucs_status_t status;

    /* Create and connect TL part */
    status = ucp_ep_create_to_worker_addr(worker, tl_bitmap, remote_addr,
                                          ep_init_flags,
                                          "conn_request on uct_listener", &ep);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to connect to worker address on "
                 "device %s, tl_bitmap 0x%"PRIx64", status %s",
                 ep, conn_request->dev_name, tl_bitmap,
                 ucs_status_string(status));
        uct_listener_reject(conn_request->uct.listener, conn_request->uct_req);
        goto out;
    }

    status = ucp_wireup_connect_local(ep, remote_addr, NULL);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to connect to remote address on "
                 "device %s, tl_bitmap 0x%"PRIx64", status %s",
                 ep, conn_request->dev_name, tl_bitmap,
                 ucs_status_string(status));
        uct_listener_reject(conn_request->uct.listener, conn_request->uct_req);
        ucp_ep_destroy_internal(ep);
        goto out;
    }

    status = ucp_ep_cm_connect_server_lane(ep, conn_request->uct.listener,
                                           conn_request->uct_req);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to connect CM lane on device %s, "
                 "tl_bitmap 0x%"PRIx64", status %s",
                 ep, conn_request->dev_name, tl_bitmap,
                 ucs_status_string(status));
        ucp_ep_destroy_internal(ep);
        goto out;
    }

    ep->flags                   |= UCP_EP_FLAG_LISTENER;
    ucp_ep_ext_gen(ep)->listener = conn_request->listener;
    ucp_ep_update_remote_id(ep, conn_request->sa_data.ep_id);
    ucp_listener_schedule_accept_cb(ep);
    *ep_p = ep;

out:
    ucs_free(conn_request->remote_dev_addr);
    ucs_free(conn_request);

    return status;
}

static ssize_t ucp_cm_server_priv_pack_cb(void *arg,
                                          const uct_cm_ep_priv_data_pack_args_t
                                          *pack_args, void *priv_data)
{
    ucp_wireup_sockaddr_data_t *sa_data = priv_data;
    ucp_ep_h ep                         = arg;
    ucp_worker_h worker                 = ep->worker;
    uint64_t tl_bitmap;
    uct_cm_attr_t cm_attr;
    void* ucp_addr;
    size_t ucp_addr_size;
    ucp_rsc_index_t rsc_index;
    ucp_rsc_index_t dev_index;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    tl_bitmap = ucp_ep_get_tl_bitmap(ep);
    /* make sure that all lanes are created on correct device */
    ucs_assert_always(pack_args->field_mask &
                      UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME);
    ucs_assert(!(tl_bitmap & ~ucp_context_dev_tl_bitmap(worker->context,
                                                        pack_args->dev_name)));

    status = ucp_address_pack(worker, ep, tl_bitmap,
                              UCP_ADDRESS_PACK_FLAG_IFACE_ADDR |
                              UCP_ADDRESS_PACK_FLAG_EP_ADDR, NULL,
                              &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out;
    }

    cm_attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
    ucs_assert(ucp_worker_num_cm_cmpts(worker) == 1);
    status = uct_cm_query(worker->cms[0].cm, &cm_attr);
    if (status != UCS_OK) {
        goto out;
    }

    if (cm_attr.max_conn_priv < (sizeof(*sa_data) + ucp_addr_size)) {
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto free_addr;
    }

    rsc_index = ucs_ffs64_safe(tl_bitmap);
    ucs_assert(rsc_index != UCP_NULL_RESOURCE);
    dev_index = worker->context->tl_rscs[rsc_index].dev_index;
    ucp_cm_priv_data_pack(sa_data, ep, dev_index, ucp_addr, ucp_addr_size);

free_addr:
    ucs_free(ucp_addr);
out:
    if (status == UCS_OK) {
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    } else {
        ucp_worker_set_ep_failed(worker, ep,
                                 &ucp_ep_get_cm_wireup_ep(ep)->super.super,
                                 ucp_ep_get_cm_lane(ep), status);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);

    return (status == UCS_OK) ? (sizeof(*sa_data) + ucp_addr_size) : status;
}

/*
 * The main thread progress part of connection establishment on server side
 */
static unsigned ucp_cm_server_conn_notify_progress(void *arg)
{
    ucp_ep_h ucp_ep = arg;

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);
    ucp_wireup_remote_connected(ucp_ep);
    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);
    return 1;
}

/*
 * Async callback on a server side which notifies that client is connected.
 */
static void ucp_cm_server_conn_notify_cb(uct_ep_h ep, void *arg,
                                         const uct_cm_ep_server_conn_notify_args_t
                                         *notify_args)
{
    ucp_ep_h ucp_ep            = arg;
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;
    ucp_lane_index_t cm_lane;
    ucs_status_t status;

    ucs_assert_always(notify_args->field_mask &
                      UCT_CM_EP_SERVER_CONN_NOTIFY_ARGS_FIELD_STATUS);

    status = notify_args->status;

    if (status == UCS_OK) {
        uct_worker_progress_register_safe(ucp_ep->worker->uct,
                                          ucp_cm_server_conn_notify_progress,
                                          ucp_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                          &prog_id);
        ucp_worker_signal_internal(ucp_ep->worker);
    } else {
        /* if reject is arrived on server side, then UCT does something wrong */
        ucs_assert(status != UCS_ERR_REJECTED);
        cm_lane = ucp_ep_get_cm_lane(ucp_ep);
        ucp_worker_set_ep_failed(ucp_ep->worker, ucp_ep,
                                 ucp_ep->uct_eps[cm_lane], cm_lane, status);
    }
}

ucs_status_t ucp_ep_cm_connect_server_lane(ucp_ep_h ep,
                                           uct_listener_h uct_listener,
                                           uct_conn_request_h uct_conn_req)
{
    ucp_worker_h worker   = ep->worker;
    ucp_lane_index_t lane = ucp_ep_get_cm_lane(ep);
    uct_ep_params_t uct_ep_params;
    uct_ep_h uct_ep;
    ucs_status_t status;

    ucs_assert(lane != UCP_NULL_LANE);
    ucs_assert(ep->uct_eps[lane] == NULL);

    /* TODO: split CM and wireup lanes */
    status = ucp_wireup_ep_create(ep, &ep->uct_eps[lane]);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to create wireup CM lane, status %s",
                 ep, ucs_status_string(status));
        uct_listener_reject(uct_listener, uct_conn_req);
        return status;
    }

    /* create a server side CM endpoint */
    ucs_trace("ep %p: uct_ep[%d]", ep, lane);
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                        |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST              |
                               UCT_EP_PARAM_FIELD_USER_DATA                 |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS         |
                               UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB          |
                               UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER |
                               UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    ucs_assertv_always(ucp_worker_num_cm_cmpts(worker) == 1,
                       "multiple CMs are not supported");
    uct_ep_params.cm                 = worker->cms[0].cm;
    uct_ep_params.user_data          = ep;
    uct_ep_params.conn_request       = uct_conn_req;
    uct_ep_params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
    uct_ep_params.sockaddr_pack_cb   = ucp_cm_server_priv_pack_cb;
    uct_ep_params.sockaddr_cb_server = ucp_cm_server_conn_notify_cb;
    uct_ep_params.disconnect_cb      = ucp_cm_disconnect_cb;

    status = uct_ep_create(&uct_ep_params, &uct_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    ucp_wireup_ep_set_next_ep(ep->uct_eps[lane], uct_ep);
    return UCS_OK;
}

void ucp_ep_cm_disconnect_cm_lane(ucp_ep_h ucp_ep)
{
    uct_ep_h uct_cm_ep = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucs_status_t status;

    ucs_assert_always(uct_cm_ep != NULL);
    /* No reason to try disconnect twice */
    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_FAILED));

    ucp_ep->flags &= ~UCP_EP_FLAG_LOCAL_CONNECTED;
    /* this will invoke @ref ucp_cm_disconnect_cb on remote side */
    status = uct_ep_disconnect(uct_cm_ep, 0);
    if (status != UCS_OK) {
        ucs_warn("failed to disconnect CM lane %p of ep %p, %s", ucp_ep,
                  uct_cm_ep, ucs_status_string(status));
    }
}

ucp_request_t* ucp_ep_cm_close_request_get(ucp_ep_h ep)
{
    ucp_request_t *request = ucp_request_get(ep->worker);

    if (request == NULL) {
        ucs_error("failed to allocate close request for ep %p", ep);
        return NULL;
    }

    request->status  = UCS_OK;
    request->flags   = 0;
    request->send.ep = ep;
    request->send.flush.uct_flags = UCT_FLUSH_FLAG_LOCAL;

    return request;
}

static int ucp_cm_cbs_remove_filter(const ucs_callbackq_elem_t *elem, void *arg)
{
    ucp_cm_client_connect_progress_arg_t *client_connect_arg;

    if (elem->cb == ucp_cm_client_connect_progress) {
        client_connect_arg = elem->arg;
        if (client_connect_arg->ucp_ep == arg) {
            ucp_cm_client_connect_prog_arg_free(client_connect_arg);
            return 1;
        } else {
            return 0;
        }
    } else if ((elem->cb == ucp_ep_cm_disconnect_progress) ||
               (elem->cb == ucp_cm_server_conn_notify_progress)) {
        return arg == elem->arg;
    } else {
        return 0;
    }
}

void ucp_ep_cm_slow_cbq_cleanup(ucp_ep_h ep)
{
    ucs_callbackq_remove_if(&ep->worker->uct->progress_q,
                            ucp_cm_cbs_remove_filter, ep);
}
