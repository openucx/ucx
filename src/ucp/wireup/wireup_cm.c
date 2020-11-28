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
        return UCP_EP_INIT_CM_WIREUP_CLIENT | UCP_EP_INIT_CM_PHASE;
    }
    if (params->field_mask & UCP_EP_PARAM_FIELD_CONN_REQUEST) {
        return UCP_EP_INIT_CM_WIREUP_SERVER | UCP_EP_INIT_CM_PHASE;
    }

    return 0;
}

int ucp_ep_init_flags_has_cm(unsigned ep_init_flags)
{
    return !!(ep_init_flags & (UCP_EP_INIT_CM_WIREUP_CLIENT |
                               UCP_EP_INIT_CM_WIREUP_SERVER));
}

/*
 * The main thread progress part of attempting connecting the client to the server
 * through the next available cm.
 */
static unsigned ucp_cm_client_try_next_cm_progress(void *arg)
{
    ucp_ep_h ucp_ep       = arg;
    ucp_worker_h worker   = ucp_ep->worker;
    ucp_context_h context = worker->context;
    ucp_wireup_ep_t *cm_wireup_ep;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_assert_always(cm_wireup_ep != NULL);
    ucp_wireup_ep_destroy_next_ep(cm_wireup_ep);

    ucs_debug("client switching from %s to %s in attempt to connect to the server",
              ucp_context_cm_name(context, cm_wireup_ep->cm_idx - 1),
              ucp_context_cm_name(context, cm_wireup_ep->cm_idx));

    status = ucp_ep_client_cm_create_uct_ep(ucp_ep);
    if (status != UCS_OK) {
        ucs_error("failed to create a uct sockaddr endpoint on %s cm %p",
                  ucp_context_cm_name(context, cm_wireup_ep->cm_idx),
                  worker->cms[cm_wireup_ep->cm_idx].cm);

        ucp_worker_set_ep_failed(worker, ucp_ep, &cm_wireup_ep->super.super,
                                 ucp_ep_get_cm_lane(ucp_ep), status);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

static int ucp_cm_client_try_fallback_cms(ucp_ep_h ep)
{
    ucp_worker_h worker           = ep->worker;
    ucp_wireup_ep_t *cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    ucp_rsc_index_t next_cm_idx   = cm_wireup_ep->cm_idx + 1;
    uct_worker_cb_id_t prog_id    = UCS_CALLBACKQ_ID_NULL;

    if (next_cm_idx >= ucp_worker_num_cm_cmpts(worker)) {
        ucs_debug("reached the end of the cms priority list, no cms left to"
                  " check (sockaddr_cm=%s, cm_idx=%d).",
                  ucp_context_cm_name(worker->context, cm_wireup_ep->cm_idx),
                  cm_wireup_ep->cm_idx);
        return 0;
    }

    cm_wireup_ep->cm_idx = next_cm_idx;
    uct_worker_progress_register_safe(worker->uct,
                                      ucp_cm_client_try_next_cm_progress,
                                      ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);
    ucp_worker_signal_internal(worker);
    return 1;
}

static ucp_rsc_index_t
ucp_cm_tl_bitmap_get_dev_idx(ucp_context_h context, uint64_t tl_bitmap)
{   
    ucp_rsc_index_t rsc_index;
    ucp_rsc_index_t dev_index;

    ucs_assert(tl_bitmap != 0);

    rsc_index = ucs_ffs64_safe(tl_bitmap);
    dev_index = context->tl_rscs[rsc_index].dev_index;

    /* check that all TL resources in the TL bitmap have the same dev_index */
    ucs_for_each_bit(rsc_index, tl_bitmap) {
        ucs_assert(dev_index == context->tl_rscs[rsc_index].dev_index);
    }

    return dev_index;
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

    if (tl_bitmap == 0) {
        ucs_debug("tl_bitmap for %s is empty", dev_name);
        return UCS_ERR_UNREACHABLE;
    }

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

static void uct_wireup_cm_tmp_ep_cleanup(ucp_ep_h tmp_ep, ucs_queue_head_t *queue)
{
    ucp_lane_index_t lane_idx;
    uct_ep_h uct_ep;

    if (tmp_ep == NULL) {
        return;
    }

    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(tmp_ep); ++lane_idx) {
        if (lane_idx == ucp_ep_get_cm_lane(tmp_ep)) {
            continue;
        }

        /* transfer the pending queues content from the previous tmp_ep to
         * a temporary queue */
        uct_ep_pending_purge(tmp_ep->uct_eps[lane_idx],
                             ucp_wireup_pending_purge_cb, &queue);

        if (ucp_ep_config(tmp_ep)->p2p_lanes & UCS_BIT(lane_idx)) {
            uct_ep = ucp_wireup_extract_lane(tmp_ep, lane_idx);
            /* destroy the transport ep */
            uct_ep_destroy(uct_ep);
        }

        /* destroy the wireup ep */
        uct_ep_destroy(tmp_ep->uct_eps[lane_idx]);
    }

    ucs_trace("deleting tmp_ep %p", tmp_ep);
    ucp_ep_destroy_base(tmp_ep);
}

static ucs_status_t ucp_cm_ep_init_lanes(ucp_ep_h ep, uint64_t *tl_bitmap,
                                         ucp_rsc_index_t *dev_index)
{
    ucp_worker_h worker = ep->worker;
    ucp_ep_h tmp_ep     = ucp_ep_get_cm_wireup_ep(ep)->tmp_ep;
    ucs_status_t status = UCS_ERR_NO_RESOURCE;
    ucp_lane_index_t lane_idx;
    ucp_rsc_index_t rsc_idx;
    uint8_t path_index;

    *tl_bitmap = 0;
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

        ucs_assert((*dev_index == UCP_NULL_RESOURCE) ||
                   (*dev_index == worker->context->tl_rscs[rsc_idx].dev_index));
        *dev_index = worker->context->tl_rscs[rsc_idx].dev_index;

        *tl_bitmap |= UCS_BIT(rsc_idx);
        if (ucp_ep_config(tmp_ep)->p2p_lanes & UCS_BIT(lane_idx)) {
            path_index = ucp_ep_get_path_index(tmp_ep, lane_idx);
            status     = ucp_wireup_ep_connect(tmp_ep->uct_eps[lane_idx], 0,
                                               rsc_idx, path_index, 0, NULL);
            if (status != UCS_OK) {
                goto out;
            }

            ucp_worker_iface_progress_ep(ucp_worker_iface(worker, rsc_idx));
        } else {
            ucs_assert(ucp_worker_is_tl_2iface(worker, rsc_idx));
        }
    }

out:
    return status;
}

static ssize_t ucp_cm_client_priv_pack_cb(void *arg,
                                          const uct_cm_ep_priv_data_pack_args_t
                                          *pack_args, void *priv_data)
{
    ucp_wireup_sockaddr_data_t *sa_data = priv_data;
    ucp_ep_h ep                         = arg;
    ucp_worker_h worker                 = ep->worker;
    ucp_rsc_index_t dev_index           = UCP_NULL_RESOURCE;
    ucp_ep_config_key_t key;
    uint64_t tl_bitmap;
    ucp_wireup_ep_t *cm_wireup_ep;
    void* ucp_addr;
    size_t ucp_addr_size;
    ucs_status_t status;
    const char *dev_name;
    ucs_queue_head_t tmp_pending_queue;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_assert_always(pack_args->field_mask &
                      UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME);

    dev_name = pack_args->dev_name;

    /* At this point the ep has only CM lane */
    ucs_assert((ucp_ep_num_lanes(ep) == 1) && ucp_ep_has_cm_lane(ep));
    cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    ucs_assert(cm_wireup_ep != NULL);

    status = ucp_cm_ep_client_initial_config_get(ep, dev_name, &key);
    if (status != UCS_OK) {
        if (ucp_cm_client_try_fallback_cms(ep)) {
            goto out;
        } else {
            goto out_check_err;
        }
    }

    ucs_queue_head_init(&tmp_pending_queue);

    /* cleanup the previously created cm_wireup_ep->tmp_ep. the one that was
     * created on the previous call to this client's pack_cb */
    uct_wireup_cm_tmp_ep_cleanup(cm_wireup_ep->tmp_ep, &tmp_pending_queue);
    cm_wireup_ep->tmp_ep = NULL;

    /* Create tmp ep which will hold local tl addresses until connect
     * event arrives, to avoid asynchronous ep reconfiguration. */
    status = ucp_ep_create_base(worker, "tmp_cm", "tmp cm client",
                                &cm_wireup_ep->tmp_ep);
    if (status != UCS_OK) {
        goto out_check_err;
    }

    cm_wireup_ep->tmp_ep->flags |= UCP_EP_FLAG_TEMPORARY;

    status = ucp_worker_get_ep_config(worker, &key, 0,
                                      &cm_wireup_ep->tmp_ep->cfg_index);
    if (status != UCS_OK) {
        goto out_check_err;
    }

    status = ucp_cm_ep_init_lanes(ep, &tl_bitmap, &dev_index);
    if (status != UCS_OK) {
        goto out_check_err;
    }

    /* Replay pending requests from the tmp_pending_queue */
    ucp_wireup_replay_pending_requests(ep, &tmp_pending_queue);

    /* Don't pack the device address to reduce address size, it will be
     * delivered by uct_cm_listener_conn_request_callback_t in
     * uct_cm_remote_data_t */
    status = ucp_address_pack(worker, cm_wireup_ep->tmp_ep, tl_bitmap,
                              UCP_ADDRESS_PACK_FLAGS_CM_DEFAULT,
                              NULL, &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out_check_err;
    }

    if (worker->cms[cm_wireup_ep->cm_idx].attr.max_conn_priv <
        (sizeof(*sa_data) + ucp_addr_size)) {
        ucs_error("CM private data buffer is too small to pack UCP endpoint info, "
                  "ep %p/%p service data %lu, address length %lu, cm %p max_conn_priv %lu",
                  ep, cm_wireup_ep->tmp_ep, sizeof(*sa_data), ucp_addr_size,
                  worker->cms[cm_wireup_ep->cm_idx].cm,
                  worker->cms[cm_wireup_ep->cm_idx].attr.max_conn_priv);
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto free_addr;
    }

    ucs_debug("client ep %p created on device %s idx %d, tl_bitmap 0x%"PRIx64
              "on cm %s", ep, dev_name, dev_index, tl_bitmap,
              ucp_context_cm_name(worker->context, cm_wireup_ep->cm_idx));
    /* Pass real ep (not cm_wireup_ep->tmp_ep), because only its pointer and
     * err_mode is taken from the config. */
    ucp_cm_priv_data_pack(sa_data, ep, dev_index, ucp_addr, ucp_addr_size);

free_addr:
    ucs_free(ucp_addr);
out_check_err:
    if (status == UCS_OK) {
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    } else {
        ucp_worker_set_ep_failed(worker, ep,
                                 &ucp_ep_get_cm_wireup_ep(ep)->super.super,
                                 ucp_ep_get_cm_lane(ep), status);
    }

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
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

    ucp_ep->cfg_index = tmp_ep->cfg_index;

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
    ucp_rsc_index_t UCS_V_UNUSED rsc_index;
    unsigned addr_idx;
    unsigned addr_indices[UCP_MAX_RESOURCES];
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_assert(wireup_ep != NULL);
    ucs_assert(wireup_ep->ep_init_flags & UCP_EP_INIT_CM_WIREUP_CLIENT);

    status = ucp_address_unpack(worker, progress_arg->sa_data + 1,
                                UCP_ADDRESS_PACK_FLAGS_CM_DEFAULT, &addr);
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
    dev_index = ucp_cm_tl_bitmap_get_dev_idx(worker->context, tl_bitmap);

    /* Restore initial configuration from tmp_ep created for packing local
     * addresses. */
    ucp_cm_client_restore_ep(wireup_ep, ucp_ep);

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

    if (!context->config.ext.cm_use_all_devices) {
        ucp_wireup_remote_connected(ucp_ep);
    }

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

    ucs_error("incompatible client server connection establishment protocol "
              "(field_mask %"PRIu64")", remote_data->field_mask);
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


    if (((status == UCS_ERR_NOT_CONNECTED) || (status == UCS_ERR_UNREACHABLE) ||
         (status == UCS_ERR_CONNECTION_RESET)) &&
        /* try connecting through another cm (next one in the priority list) */
        ucp_cm_client_try_fallback_cms(ucp_ep)) {
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep->flags &= ~UCP_EP_FLAG_LOCAL_CONNECTED;
        /* cms fallback has started */
        return;
    } else if (status != UCS_OK) {
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep->flags &= ~UCP_EP_FLAG_LOCAL_CONNECTED;
        ucs_debug("failed status on client connect callback: %s "
                  "(sockaddr_cm=%s, cms_used_idx=%d)", ucs_status_string(status),
                  ucp_context_cm_name(worker->context,
                                      ucp_ep_get_cm_wireup_ep(ucp_ep)->cm_idx),
                  ucp_ep_get_cm_wireup_ep(ucp_ep)->cm_idx);
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
    ucs_status_t status = UCS_ERR_CONNECTION_RESET;
    ucp_ep_h ucp_ep     = arg;
    void *req;

    ucs_trace("ep %p: flags 0x%x cm_remote_disconnect_progress", ucp_ep,
              ucp_ep->flags);

    ucs_assert(ucp_ep_get_cm_uct_ep(ucp_ep) != NULL);

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
    if (ucs_test_all_flags(ucp_ep->flags, UCP_EP_FLAG_CLOSED |
                                          UCP_EP_FLAG_CLOSE_REQ_VALID)) {
        ucp_request_complete_send(ucp_ep_ext_control(ucp_ep)->close_req.req,
                                  UCS_OK);
        return 1;
    }

    if (ucp_ep->flags & UCP_EP_FLAG_CLOSED) {
        /* the ep is closed by API but close req is not valid yet (checked
         * above), it will be set later from scheduled
         * @ref ucp_ep_close_flushed_callback */
        ucs_debug("ep %p: ep closed but request is not set, waiting for"
                  " the flush callback", ucp_ep);
        goto err;
    }

    if (!(ucp_ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
        /* CM disconnect happens during WIREUP MSGs exchange phase, when EP
         * is locally connected to the peer */
        goto err;
    }

    /*
     * TODO: set the ucp_ep to error state to prevent user from sending more
     *       ops.
     */
    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_FLUSH_STATE_VALID);
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_CLOSED));
    req = ucp_ep_flush_internal(ucp_ep, 0, &ucp_request_null_param, NULL,
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
        ucp_ep_cm_remote_disconnect_progress(ucp_ep);
    } else if (ucp_ep->flags & UCP_EP_FLAG_CLOSE_REQ_VALID) {
        /* if the EP is not local connected, the EP has been closed and flushed,
           CM lane is disconnected, complete close request and destroy EP */
        ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CLOSED);
        close_req = ucp_ep_ext_control(ucp_ep)->close_req.req;
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
    ucp_worker_h worker        = ucp_ep->worker;
    uct_ep_h uct_ep;

    ucs_trace("ep %p: CM remote disconnect callback invoked, flags 0x%x",
              ucp_ep, ucp_ep->flags);

    uct_ep = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucs_assertv_always(uct_cm_ep == uct_ep,
                       "%p: uct_cm_ep=%p vs found_uct_ep=%p",
                       ucp_ep, uct_cm_ep, uct_ep);

    uct_worker_progress_register_safe(worker->uct,
                                      ucp_ep_cm_disconnect_progress,
                                      ucp_ep, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);
    ucp_worker_signal_internal(worker);
}

ucs_status_t ucp_ep_client_cm_create_uct_ep(ucp_ep_h ucp_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucp_worker_h worker        = ucp_ep->worker;
    uct_ep_params_t cm_lane_params;
    ucs_sock_addr_t remote_addr;
    size_t sockaddr_size;
    ucs_status_t status;
    uct_ep_h cm_ep;

    cm_lane_params.field_mask = UCT_EP_PARAM_FIELD_CM                         |
                                UCT_EP_PARAM_FIELD_USER_DATA                  |
                                UCT_EP_PARAM_FIELD_SOCKADDR                   |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS          |
                                UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB           |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB_CLIENT |
                                UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    status = ucs_sockaddr_sizeof((struct sockaddr *)&wireup_ep->cm_remote_sockaddr,
                                 &sockaddr_size);
    if (status != UCS_OK) {
        return status;
    }

    remote_addr.addrlen               = sockaddr_size;
    remote_addr.addr                  = (struct sockaddr *)&wireup_ep->cm_remote_sockaddr;

    cm_lane_params.sockaddr           = &remote_addr;
    cm_lane_params.user_data          = ucp_ep;
    cm_lane_params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
    cm_lane_params.sockaddr_pack_cb   = ucp_cm_client_priv_pack_cb;
    cm_lane_params.sockaddr_cb_client = ucp_cm_client_connect_cb;
    cm_lane_params.disconnect_cb      = ucp_cm_disconnect_cb;
    cm_lane_params.cm                 = worker->cms[wireup_ep->cm_idx].cm;

    status = uct_ep_create(&cm_lane_params, &cm_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    ucp_wireup_ep_set_next_ep(&wireup_ep->super.super, cm_ep);
    ucs_trace("created cm_ep %p, wireup_ep %p, uct_ep %p, wireup_ep_from_uct_ep %p",
              cm_ep, wireup_ep, &wireup_ep->super.super, ucp_wireup_ep(&wireup_ep->super.super));
    return status;
}

ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params)
{
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_status_t status;

    wireup_ep->ep_init_flags = ucp_ep_init_flags(ucp_ep->worker, params);
    wireup_ep->cm_idx        = 0;

    /* save the address from the ep_params on the wireup_ep */
    status = ucs_sockaddr_copy((struct sockaddr *)&wireup_ep->cm_remote_sockaddr,
                               params->sockaddr.addr);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_ep_client_cm_create_uct_ep(ucp_ep);
    if (status != UCS_OK) {
        return status;
    }

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

static ucp_rsc_index_t ucp_listener_get_cm_index(uct_listener_h listener,
                                                 ucp_worker_h worker)
{
    const ucp_rsc_index_t num_cms = ucp_worker_num_cm_cmpts(worker);
    ucp_rsc_index_t i;

    for (i = 0; i < num_cms; i++) {
        if (worker->cms[i].cm == listener->cm) {
            return i;
        }
    }

    return UCP_NULL_RESOURCE;
}

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const uct_cm_listener_conn_request_args_t
                                   *conn_req_args)
{
    ucp_listener_h ucp_listener = arg;
    ucp_worker_h worker         = ucp_listener->worker;
    uct_worker_cb_id_t prog_id  = UCS_CALLBACKQ_ID_NULL;
    ucp_conn_request_h ucp_conn_request;
    uct_conn_request_h conn_request;
    const uct_cm_remote_data_t *remote_data;
    ucp_rsc_index_t cm_idx;
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

    cm_idx = ucp_listener_get_cm_index(listener, worker);
    if (cm_idx == UCP_NULL_RESOURCE) {
        ucs_error("failed to find listener's cm (%p) in local cms array",
                  listener->cm);
        goto err_reject;
    }

    ucs_debug("server received a connection request on the %s "
              "sockaddr transport (worker=%p cm=%p worker_cms_index=%d)",
              ucp_context_cm_name(worker->context, cm_idx),
              worker, listener->cm, cm_idx);

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
    ucp_conn_request->cm_idx       = cm_idx;

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

    uct_worker_progress_register_safe(worker->uct,
                                      ucp_cm_server_conn_request_progress,
                                      ucp_conn_request,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);

    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(worker);
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
    char client_addr_str[UCS_SOCKADDR_STRING_LEN];

    ep_init_flags |= UCP_EP_INIT_CM_WIREUP_SERVER | UCP_EP_INIT_CM_PHASE;

    if (tl_bitmap == 0) {
        ucs_error("listener %p: got connection request from %s on a device %s "
                  "which was not present during UCP initialization",
                  conn_request->listener,
                  ucs_sockaddr_str((struct sockaddr*)&conn_request->client_address,
                                   client_addr_str, sizeof(client_addr_str)),
                  conn_request->dev_name);
        status = UCS_ERR_UNREACHABLE;
        goto out;
    }

    /* Create and connect TL part */
    status = ucp_ep_create_to_worker_addr(worker, tl_bitmap, remote_addr,
                                          ep_init_flags,
                                          "conn_request on uct_listener", &ep);
    if (status != UCS_OK) {
        ucs_warn("failed to create server ep and connect to worker address on "
                 "device %s, tl_bitmap 0x%"PRIx64", status %s",
                 conn_request->dev_name, tl_bitmap, ucs_status_string(status));
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
        goto err_destroy_ep;
    }

    status = ucp_ep_cm_connect_server_lane(ep, conn_request->uct.listener,
                                           conn_request->uct_req,
                                           conn_request->cm_idx);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to connect CM lane on device %s, "
                 "tl_bitmap 0x%"PRIx64", status %s",
                 ep, conn_request->dev_name, tl_bitmap,
                 ucs_status_string(status));
        goto err_destroy_ep;
    }

    ep->flags                       |= UCP_EP_FLAG_LISTENER;
    ucp_ep_ext_control(ep)->listener = conn_request->listener;
    ucp_ep_update_remote_id(ep, conn_request->sa_data.ep_id);
    ucp_listener_schedule_accept_cb(ep);
    *ep_p = ep;

out:
    ucs_free(conn_request->remote_dev_addr);
    ucs_free(conn_request);

    return status;

err_destroy_ep:
    ucp_ep_destroy_internal(ep);
    goto out;
}

static ssize_t ucp_cm_server_priv_pack_cb(void *arg,
                                          const uct_cm_ep_priv_data_pack_args_t
                                          *pack_args, void *priv_data)
{
    ucp_wireup_sockaddr_data_t *sa_data = priv_data;
    ucp_ep_h ep                         = arg;
    ucp_worker_h worker                 = ep->worker;
    ucp_wireup_ep_t *cm_wireup_ep       = ucp_ep_get_cm_wireup_ep(ep);
    uint64_t tl_bitmap;
    void* ucp_addr;
    size_t ucp_addr_size;
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
                              UCP_ADDRESS_PACK_FLAGS_CM_DEFAULT, NULL,
                              &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out;
    }

    if (worker->cms[cm_wireup_ep->cm_idx].attr.max_conn_priv <
        (sizeof(*sa_data) + ucp_addr_size)) {
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto free_addr;
    }

    dev_index = ucp_cm_tl_bitmap_get_dev_idx(worker->context, tl_bitmap);
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
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);
    if (!ucp_ep->worker->context->config.ext.cm_use_all_devices) {
        ucp_wireup_remote_connected(ucp_ep);
    } else {
        status = ucp_wireup_send_pre_request(ucp_ep);
        ucs_assert_always(status == UCS_OK);
    }
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
                                           uct_conn_request_h uct_conn_req,
                                           ucp_rsc_index_t cm_idx)
{
    ucp_worker_h worker   = ep->worker;
    ucp_lane_index_t lane = ucp_ep_get_cm_lane(ep);
    uct_ep_params_t uct_ep_params;
    ucp_wireup_ep_t *cm_wireup_ep;
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

    cm_wireup_ep         = ucs_derived_of(ep->uct_eps[lane], ucp_wireup_ep_t);
    cm_wireup_ep->cm_idx = cm_idx;

    /* create a server side CM endpoint */
    ucs_trace("server ep %p: uct_ep[%d], worker %p, cm_idx=%d, cm=%s",
              ep, lane, worker, cm_idx,
              ucp_context_cm_name(worker->context, cm_idx));
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                        |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST              |
                               UCT_EP_PARAM_FIELD_USER_DATA                 |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS         |
                               UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB          |
                               UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER |
                               UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    uct_ep_params.cm                 = worker->cms[cm_idx].cm;
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
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_DISCONNECTED_CM_LANE));
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_FAILED));

    ucp_ep->flags &= ~UCP_EP_FLAG_LOCAL_CONNECTED;
    ucp_ep->flags |= UCP_EP_FLAG_DISCONNECTED_CM_LANE;
    /* this will invoke @ref ucp_cm_disconnect_cb on remote side */
    status = uct_ep_disconnect(uct_cm_ep, 0);
    if (status != UCS_OK) {
        ucs_diag("failed to disconnect CM lane %p of ep %p, %s", ucp_ep,
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
