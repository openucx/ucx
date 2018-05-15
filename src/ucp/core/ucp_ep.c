/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_ep.h"
#include "ucp_worker.h"
#include "ucp_ep.inl"
#include "ucp_request.inl"

#include <ucp/wireup/wireup_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucp/wireup/fin.h>
#include <ucp/tag/eager.h>
#include <ucp/tag/offload.h>
#include <ucp/stream/stream.h>
#include <ucp/core/ucp_listener.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <string.h>


extern const ucp_proto_t ucp_stream_am_proto;

#if ENABLE_STATS
static ucs_stats_class_t ucp_ep_stats_class = {
    .name           = "ucp_ep",
    .num_counters   = UCP_EP_STAT_LAST,
    .counter_names  = {
        [UCP_EP_STAT_TAG_TX_EAGER]      = "tx_eager",
        [UCP_EP_STAT_TAG_TX_EAGER_SYNC] = "tx_eager_sync",
        [UCP_EP_STAT_TAG_TX_RNDV]       = "tx_rndv"
    }
};
#endif


void ucp_ep_config_key_reset(ucp_ep_config_key_t *key)
{
    memset(key, 0, sizeof(*key));
    key->num_lanes        = 0;
    key->am_lane          = UCP_NULL_LANE;
    key->wireup_lane      = UCP_NULL_LANE;
    key->tag_lane         = UCP_NULL_LANE;
    key->rma_bw_md_map    = 0;
    key->reachable_md_map = 0;
    key->err_mode         = UCP_ERR_HANDLING_MODE_NONE;
    key->status           = UCS_OK;
    memset(key->am_bw_lanes,  UCP_NULL_LANE, sizeof(key->am_bw_lanes));
    memset(key->rma_lanes,    UCP_NULL_LANE, sizeof(key->rma_lanes));
    memset(key->rma_bw_lanes, UCP_NULL_LANE, sizeof(key->rma_bw_lanes));
    memset(key->amo_lanes,    UCP_NULL_LANE, sizeof(key->amo_lanes));
}

ucs_status_t ucp_ep_new(ucp_worker_h worker, const char *peer_name,
                        const char *message, ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_config_key_t key;
    ucp_lane_index_t lane;
    ucp_ep_h ep;

    ep = ucs_strided_alloc_get(&worker->ep_alloc, "ucp_ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate ep");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ucp_ep_config_key_reset(&key);
    ep->worker                             = worker;
    ep->cfg_index                          = ucp_worker_get_ep_config(worker, &key);
    ep->am_lane                            = UCP_NULL_LANE;
    ep->flags                              = UCP_EP_FLAG_HIDDEN;
    ucp_ep_ext_gen(ep)->user_data          = NULL;
    ucp_ep_ext_gen(ep)->err_cb             = NULL;
    ucp_ep_ext_proto(ep)->conn.dest_ep_ptr = 0;
    ucp_ep_ext_proto(ep)->conn.conn_sn     = -1;

    ucp_stream_ep_init(ep);

    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        ep->uct_eps[lane] = NULL;
    }

#if ENABLE_DEBUG_DATA
    ucs_snprintf_zero(ep->peer_name, UCP_WORKER_NAME_MAX, "%s", peer_name);
#endif

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&ep->stats, &ucp_ep_stats_class,
                                  worker->stats, "-%p", ep);
    if (status != UCS_OK) {
        goto err_free_ep;
    }

    ucs_list_add_tail(&worker->all_eps, &ucp_ep_ext_gen(ep)->ep_list);
    *ep_p = ep;
    ucs_debug("created ep %p to %s %s", ep, ucp_ep_peer_name(ep), message);
    return UCS_OK;

err_free_ep:
    ucs_free(ep);
err:
    return status;
}

void ucp_ep_delete(ucp_ep_h ep)
{
    UCS_STATS_NODE_FREE(ep->stats);
    ucs_list_del(&ucp_ep_ext_gen(ep)->ep_list);
    ucs_strided_alloc_put(&ep->worker->ep_alloc, ep);
}

ucs_status_t ucp_ep_create_sockaddr_aux(ucp_worker_h worker,
                                        const ucp_ep_params_t *params,
                                        const ucp_unpacked_address_t *remote_address,
                                        ucp_ep_h *ep_p)
{
    ucp_wireup_ep_t *wireup_ep;
    ucs_status_t status;
    ucp_ep_h ep;

    /* allocate endpoint */
    status = ucp_ep_new(worker, remote_address->name, "listener", &ep);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_ep_init_create_wireup(ep, params, &wireup_ep);
    if (status != UCS_OK) {
        goto err_delete;
    }

    status = ucp_wireup_ep_connect_aux(wireup_ep, params,
                                       remote_address->address_count,
                                       remote_address->address_list);
    if (status != UCS_OK) {
        goto err_destroy_wireup_ep;
    }

    *ep_p = ep;
    return status;

err_destroy_wireup_ep:
    uct_ep_destroy(ep->uct_eps[0]);
err_delete:
    ucp_ep_delete(ep);
err:
    return status;
}

void ucp_ep_config_key_set_params(ucp_ep_config_key_t *key,
                                  const ucp_ep_params_t *params)
{
    key->err_mode = UCP_PARAM_VALUE(EP, params, err_mode, ERR_HANDLING_MODE,
                                    UCP_ERR_HANDLING_MODE_NONE);
}

int ucp_ep_is_sockaddr_stub(ucp_ep_h ep)
{
    /* Only a sockaddr client-side endpoint may be created as a "stub" */
    return ucp_ep_get_rsc_index(ep, 0) == UCP_NULL_RESOURCE;
}

static ucs_status_t
ucp_ep_adjust_params(ucp_ep_h ep, const ucp_ep_params_t *params)
{
    /* handle a case where the existing endpoint is incomplete */

    if (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) {
        if (ucp_ep_config(ep)->key.err_mode != params->err_mode) {
            ucs_error("asymmetric endpoint configuration not supported, "
                      "error handling level mismatch");
            return UCS_ERR_UNSUPPORTED;
        }
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLER) {
        ucp_ep_ext_gen(ep)->user_data = params->err_handler.arg;
        ucp_ep_ext_gen(ep)->err_cb    = params->err_handler.cb;
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_USER_DATA) {
        /* user_data overrides err_handler.arg */
        ucp_ep_ext_gen(ep)->user_data = params->user_data;
    }

    return UCS_OK;
}

ucs_status_t ucp_worker_create_mem_type_endpoints(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_unpacked_address_t local_address;
    unsigned i, mem_type, md_index;
    ucs_status_t status;
    void *address_buffer;
    size_t address_length;
    ucp_ep_params_t params;

    for (i = 0; i < UCT_MD_MEM_TYPE_LAST; i++) {
        worker->mem_type_ep[i] = NULL;
    }

    if (!context->num_mem_type_mds) {
        return UCS_OK;
    }

    params.field_mask = 0;

    for (i = 0; i < context->num_mem_type_mds; ++i) {
        md_index = context->mem_type_tl_mds[i];
        mem_type = context->tl_mds[md_index].attr.cap.mem_type;

        status = ucp_address_pack(worker, NULL, context->mem_type_tls[mem_type], NULL,
                                  &address_length, &address_buffer);
        if (status != UCS_OK) {
            goto err_cleanup_eps;
        }

        status = ucp_address_unpack(address_buffer, &local_address);
        if (status != UCS_OK) {
            goto err_free_address_buffer;
        }

        status = ucp_ep_create_to_worker_addr(worker, &params, &local_address,
                                              UCP_EP_INIT_FLAG_MEM_TYPE, "mem type",
                                              &worker->mem_type_ep[mem_type]);
        if (status != UCS_OK) {
            goto err_free_address_list;
        }

        ucs_free(local_address.address_list);
        ucs_free(address_buffer);
    }

    return UCS_OK;

err_free_address_list:
    ucs_free(local_address.address_list);
err_free_address_buffer:
    ucs_free(address_buffer);
err_cleanup_eps:
    for (i = 0; i < UCT_MD_MEM_TYPE_LAST; i++) {
        if (worker->mem_type_ep[i]) {
           ucp_ep_destroy_internal(worker->mem_type_ep[i]);
        }
    }
    return status;
}

ucs_status_t ucp_ep_init_create_wireup(ucp_ep_h ep,
                                       const ucp_ep_params_t *params,
                                       ucp_wireup_ep_t **wireup_ep)
{
    ucp_ep_config_key_t key;
    ucs_status_t status;

    ucp_ep_config_key_reset(&key);
    ucp_ep_config_key_set_params(&key, params);

    /* all operations will use the first lane, which is a stub endpoint */
    key.num_lanes             = 1;
    key.lanes[0].rsc_index    = UCP_NULL_RESOURCE;
    key.lanes[0].dst_md_index = UCP_NULL_RESOURCE;
    key.am_lane               = 0;
    key.wireup_lane           = 0;
    key.tag_lane              = 0;
    key.am_bw_lanes[0]        = 0;
    key.rma_lanes[0]          = 0;
    key.rma_bw_lanes[0]       = 0;
    key.amo_lanes[0]          = 0;

    ep->cfg_index             = ucp_worker_get_ep_config(ep->worker, &key);
    ep->am_lane               = 0;
    ep->flags                |= UCP_EP_FLAG_CONNECT_REQ_QUEUED;

    status = ucp_wireup_ep_create(ep, &ep->uct_eps[0]);
    if (status != UCS_OK) {
        return status;
    }

    *wireup_ep = ucs_derived_of(ep->uct_eps[0], ucp_wireup_ep_t);
    return UCS_OK;
}

ucs_status_t ucp_ep_create_to_worker_addr(ucp_worker_h worker,
                                          const ucp_ep_params_t *params,
                                          const ucp_unpacked_address_t *remote_address,
                                          unsigned ep_init_flags,
                                          const char *message, ucp_ep_h *ep_p)
{
    uint8_t addr_indices[UCP_MAX_LANES];
    ucs_status_t status;
    ucp_ep_h ep;

    /* allocate endpoint */
    status = ucp_ep_new(worker, remote_address->name, message, &ep);
    if (status != UCS_OK) {
        goto err;
    }

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes(ep, params, ep_init_flags,
                                   remote_address->address_count,
                                   remote_address->address_list, addr_indices);
    if (status != UCS_OK) {
        goto err_delete;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        goto err_cleanup_lanes;
    }

    *ep_p = ep;
    return UCS_OK;

err_cleanup_lanes:
    ucp_ep_cleanup_lanes(ep);
err_delete:
    ucp_ep_delete(ep);
err:
    return status;
}

static ucs_status_t ucp_ep_create_to_sock_addr(ucp_worker_h worker,
                                               const ucp_ep_params_t *params,
                                               ucp_ep_h *ep_p)
{
    char peer_name[UCS_SOCKADDR_STRING_LEN];
    ucp_wireup_ep_t *wireup_ep;
    ucs_status_t status;
    ucp_ep_h ep;

    if (!(params->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR)) {
        ucs_error("destination socket address is missing");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    UCP_CHECK_PARAM_NON_NULL(params->sockaddr.addr, status, goto err);

    /* allocate endpoint */
    ucs_sockaddr_str(params->sockaddr.addr, peer_name, sizeof(peer_name));

    status = ucp_ep_new(worker, peer_name, "from api call", &ep);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_ep_init_create_wireup(ep, params, &wireup_ep);
    if (status != UCS_OK) {
        goto err_delete;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        goto err_cleanup_lanes;
    }

    status = ucp_wireup_ep_connect_to_sockaddr(ep->uct_eps[0], params);
    if (status != UCS_OK) {
        goto err_cleanup_lanes;
    }

    *ep_p = ep;
    return UCS_OK;

err_cleanup_lanes:
    ucp_ep_cleanup_lanes(ep);
err_delete:
    ucp_ep_delete(ep);
err:
    return status;
}

static ucs_status_t
ucp_ep_create_api_to_worker_addr(ucp_worker_h worker,
                                 const ucp_ep_params_t *params, ucp_ep_h *ep_p)
{
    ucp_unpacked_address_t remote_address;
    ucp_ep_conn_sn_t conn_sn;
    ucs_status_t status;
    unsigned flags;
    ucp_ep_h ep;

    if (!(params->field_mask & UCP_EP_PARAM_FIELD_REMOTE_ADDRESS)) {
        status = UCS_ERR_INVALID_PARAM;
        ucs_error("remote worker address is missing");
        goto out;
    }

    UCP_CHECK_PARAM_NON_NULL(params->address, status, goto out);

    status = ucp_address_unpack(params->address, &remote_address);
    if (status != UCS_OK) {
        goto out;
    }

    /* Check if there is already an unconnected internal endpoint to the same
     * destination address.
     * In case of loopback connection, search the hash table for an endpoint with
     * even/odd matching, so that every 2 endpoints connected to the local worker
     * with be paired to each other.
     * Note that if a loopback endpoint had the UCP_EP_PARAMS_FLAGS_NO_LOOPBACK
     * flag set, it will not be added to ep_match as an unexpected ep. Because
     * dest_ep_ptr will be initialized, a WIREUP_REQUEST (if sent) will have
     * dst_ep != 0. So, ucp_wireup_request() will not create an unexpected ep
     * in ep_match.
     */
    conn_sn = ucp_ep_match_get_next_sn(&worker->ep_match_ctx, remote_address.uuid);
    ep = ucp_ep_match_retrieve_unexp(&worker->ep_match_ctx, remote_address.uuid,
                                     conn_sn ^ (remote_address.uuid == worker->uuid));
    if (ep != NULL) {
        status = ucp_ep_adjust_params(ep, params);
        if (status != UCS_OK) {
            ucp_ep_destroy_internal(ep);
        }

        ucp_stream_ep_activate(ep);
        goto out_free_address;
    }

    status = ucp_ep_create_to_worker_addr(worker, params, &remote_address, 0,
                                          "from api call", &ep);
    if (status != UCS_OK) {
        goto out_free_address;
    }

    ucp_ep_ext_proto(ep)->conn.conn_sn = conn_sn;

    /*
     * If we are connecting to our own worker, and loopback is allowed, connect
     * the endpoint to itself by updating dest_ep_ptr.
     * Otherwise, add the new ep to the matching context as an expected endpoint,
     * waiting for connection request from the peer endpoint
     */
    flags = UCP_PARAM_VALUE(EP, params, flags, FLAGS, 0);
    if ((remote_address.uuid == worker->uuid) &&
        !(flags & UCP_EP_PARAMS_FLAGS_NO_LOOPBACK)) {
        ucp_ep_update_dest_ep_ptr(ep, (uintptr_t)ep);
    } else {
        ucp_ep_match_insert_exp(&worker->ep_match_ctx, remote_address.uuid, ep);
    }

    /* if needed, send initial wireup message */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        ucs_assert(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED));
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
            ucp_ep_destroy_internal(ep);
            goto out_free_address;
        }
    }

    status = UCS_OK;

out_free_address:
    ucs_free(remote_address.address_list);
out:
    if (status == UCS_OK) {
        *ep_p = ep;
    }
    return status;
}

ucs_status_t ucp_ep_create(ucp_worker_h worker, const ucp_ep_params_t *params,
                           ucp_ep_h *ep_p)
{
    ucs_status_t status;
    unsigned flags;
    ucp_ep_h ep = NULL;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCS_ASYNC_BLOCK(&worker->async);

    flags = UCP_PARAM_VALUE(EP, params, flags, FLAGS, 0);
    if (flags & UCP_EP_PARAMS_FLAGS_CLIENT_SERVER) {
        status = ucp_ep_create_to_sock_addr(worker, params, &ep);
        if (status != UCS_OK) {
            goto out;
        }
    } else {
        status = ucp_ep_create_api_to_worker_addr(worker, params, &ep);
        if (status != UCS_OK) {
            goto out;
        }
    }

    ep->flags &= ~UCP_EP_FLAG_HIDDEN;
    *ep_p = ep;

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return status;
}

ucs_status_ptr_t ucp_ep_modify_nb(ucp_ep_h ep, const ucp_ep_params_t *params)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_t status;

    if (params->field_mask & (UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                              UCP_EP_PARAM_FIELD_SOCK_ADDR      |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE)) {
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_ep_adjust_params(ep, params);

    UCS_ASYNC_UNBLOCK(&worker->async);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);

    return UCS_STATUS_PTR(status);
}

void ucp_ep_err_pending_purge(uct_pending_req_t *self, void *arg)
{
    ucp_request_t *req      = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t  status    = UCS_PTR_STATUS(arg);

    ucp_request_send_state_ff(req, status);
}

static void ucp_destroyed_ep_pending_purge(uct_pending_req_t *self, void *arg)
{
    ucs_bug("pending request %p on ep %p should have been flushed", self, arg);
}

void ucp_ep_destroy_internal(ucp_ep_h ep)
{
    ucs_debug("ep %p: destroy", ep);
    ucp_ep_cleanup_lanes(ep);
    ucp_ep_delete(ep);
}

void ucp_ep_cleanup_lanes(ucp_ep_h ep)
{
    ucp_lane_index_t lane, proxy_lane;
    uct_ep_h uct_ep;

    ucs_debug("ep %p: cleanup lanes", ep);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = ep->uct_eps[lane];
        if (uct_ep != NULL) {
            ucs_debug("ep %p: purge uct_ep[%d]=%p", ep, lane, uct_ep);
            uct_ep_pending_purge(uct_ep, ucp_destroyed_ep_pending_purge, ep);
        }
    }

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = ep->uct_eps[lane];
        if (uct_ep == NULL) {
            continue;
        }

        proxy_lane = ucp_ep_get_proxy_lane(ep, lane);
        if ((proxy_lane != UCP_NULL_LANE) && (proxy_lane != lane) &&
            (ep->uct_eps[lane] == ep->uct_eps[proxy_lane]))
        {
            /* duplicate of another lane */
            continue;
        }

        ucs_debug("ep %p: destroy uct_ep[%d]=%p", ep, lane, uct_ep);
        uct_ep_destroy(uct_ep);
    }
}

void ucp_ep_disconnected(ucp_ep_h ep, int force)
{
    /* remove pending slow-path progress in case it wasn't removed yet */
    ucs_callbackq_remove_if(&ep->worker->uct->progress_q,
                            ucp_worker_err_handle_remove_filter, ep);

    /* remove pending slow-path function it wasn't removed yet */
    ucs_callbackq_remove_if(&ep->worker->uct->progress_q,
                            ucp_listener_accept_cb_remove_filter, ep);

    ucp_stream_ep_cleanup(ep);

    ep->flags |= UCP_EP_FLAG_HIDDEN;

    if ((ep->flags & (UCP_EP_FLAG_CONNECT_REQ_QUEUED|UCP_EP_FLAG_REMOTE_CONNECTED))
        && !force) {
        /* Endpoints which have remote connection are destroyed only when the
         * worker is destroyed, to enable remote endpoints keep sending
         * TODO negotiate disconnect.
         */
        ucs_trace("not destroying ep %p because of connection from remote", ep);
        return;
    }

    ucp_ep_match_remove_ep(&ep->worker->ep_match_ctx, ep);
    ucp_ep_destroy_internal(ep);
}

static unsigned ucp_ep_do_disconnect(void *arg)
{
    ucp_request_t *req = arg;
    ucp_ep_h      ep   = req->send.ep;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    if (ep->flags & UCP_EP_FLAG_FIN_REQ_QUEUED) {
        ep->flags |= UCP_EP_FLAG_FIN_REQ_COMPLETED;
    }

    if (req->flags & UCP_REQUEST_FLAG_SYNC) {
        if (ucs_test_all_flags(ep->flags, UCP_EP_MASK_FIN_DONE)) {
            ucp_ep_disconnected(ep, 1);
        } else {
            ucp_stream_ep_cleanup(ep);
        }
    } else {
        ucp_ep_disconnected(req->send.ep, req->send.flush.uct_flags &
                                          UCT_FLUSH_FLAG_CANCEL);
    }

    /* Complete send request from here, to avoid releasing the request while
     * slow-path element is still pending */
    ucp_request_complete_send(req, req->status);

    return 0;
}

static void ucp_ep_close_flushed_callback(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;

    /* If a flush is completed from a pending/completion callback, we need to
     * schedule slow-path callback to release the endpoint later, since a UCT
     * endpoint cannot be released from pending/completion callback context.
     */
    ucs_trace("adding slow-path callback to destroy ep %p", ep);
    req->send.disconnect.prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_worker_progress_register_safe(ep->worker->uct, ucp_ep_do_disconnect,
                                      req, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &req->send.disconnect.prog_id);
}

ucs_status_ptr_t ucp_ep_close_nb(ucp_ep_h ep, unsigned mode)
{
    ucp_worker_h worker = ep->worker;
    void *request;

    ucs_trace("start ep %p closing with mode %d", ep, mode);

    if ((mode == UCP_EP_CLOSE_MODE_FORCE) &&
        (ucp_ep_config(ep)->key.err_mode != UCP_ERR_HANDLING_MODE_PEER)) {
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    UCS_ASYNC_BLOCK(&worker->async);

    if (mode == UCP_EP_CLOSE_MODE_SYNC) {
        ucp_fin_msg_send(ep);
    }
    ucs_assert(!(ep->flags & UCP_EP_FLAG_HIDDEN));
    ep->flags |= UCP_EP_FLAG_HIDDEN;
    request = ucp_ep_flush_internal(ep,
                                    (mode == UCP_EP_CLOSE_MODE_FORCE) ?
                                    UCT_FLUSH_FLAG_CANCEL : UCT_FLUSH_FLAG_LOCAL,
                                    NULL,
                                    (mode == UCP_EP_CLOSE_MODE_SYNC) ?
                                    UCP_REQUEST_FLAG_SYNC : 0,
                                    ucp_ep_close_flushed_callback);
    if (!UCS_PTR_IS_PTR(request)) {
        if (mode == UCP_EP_CLOSE_MODE_SYNC) {
            if (ucs_test_all_flags(ep->flags, UCP_EP_MASK_FIN_DONE)) {
                ucp_ep_disconnected(ep, 1);
            } else {
                ucp_stream_ep_cleanup(ep);
            }
        } else {
            ucp_ep_disconnected(ep, mode == UCP_EP_CLOSE_MODE_FORCE);
        }
    }

    UCS_ASYNC_UNBLOCK(&worker->async);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);

    return request;
}

ucs_status_ptr_t ucp_disconnect_nb(ucp_ep_h ep)
{
    return ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_ptr_t *request;
    ucs_status_t status;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    request = ucp_disconnect_nb(ep);
    if (request == NULL) {
        goto out;
    } else if (UCS_PTR_IS_ERR(request)) {
        ucs_warn("disconnect failed: %s",
                 ucs_status_string(UCS_PTR_STATUS(request)));
        goto out;
    } else {
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);

        ucp_request_release(request);
    }

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return;
}

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2)
{
    ucp_lane_index_t lane;


    if ((key1->num_lanes        != key2->num_lanes)                                ||
        memcmp(key1->rma_lanes,    key2->rma_lanes,    sizeof(key1->rma_lanes))    ||
        memcmp(key1->am_bw_lanes,  key2->am_bw_lanes,  sizeof(key1->am_bw_lanes))  ||
        memcmp(key1->rma_bw_lanes, key2->rma_bw_lanes, sizeof(key1->rma_bw_lanes)) ||
        memcmp(key1->amo_lanes,    key2->amo_lanes,    sizeof(key1->amo_lanes))    ||
        (key1->rma_bw_md_map    != key2->rma_bw_md_map)                            ||
        (key1->reachable_md_map != key2->reachable_md_map)                         ||
        (key1->am_lane          != key2->am_lane)                                  ||
        (key1->tag_lane         != key2->tag_lane)                                 ||
        (key1->wireup_lane      != key2->wireup_lane)                              ||
        (key1->err_mode         != key2->err_mode)                                 ||
        (key1->status           != key2->status))
    {
        return 0;
    }

    for (lane = 0; lane < key1->num_lanes; ++lane) {
        if ((key1->lanes[lane].rsc_index != key2->lanes[lane].rsc_index) ||
            (key1->lanes[lane].proxy_lane != key2->lanes[lane].proxy_lane) ||
            (key1->lanes[lane].dst_md_index != key2->lanes[lane].dst_md_index))
        {
            return 0;
        }
    }

    return 1;
}

static size_t ucp_ep_config_calc_rndv_thresh(ucp_context_h context,
                                             uct_iface_attr_t *iface_attr,
                                             uct_md_attr_t *md_attr,
                                             size_t bcopy_bw, int recv_reg_cost)
{
    double numerator, denumerator, md_reg_growth, md_reg_overhead;
    double diff_percent = 1.0 - context->config.ext.rndv_perf_diff / 100.0;

    /* We calculate the Rendezvous threshold by finding the message size at which:
     * AM/RMA rndv's latency is worse than the eager_zcopy
     * latency by a small percentage (that is set by the user).
     * Starting this message size (rndv_thresh), rndv may be used.
     *
     * The latency function for eager_zcopy is:
     * [ reg_cost.overhead + size * md_attr->reg_cost.growth +
     * max(size/bw , size/bcopy_bw) + overhead ]
     *
     * The latency function for Active message Rendezvous is:
     * [ latency + overhead + reg_cost.overhead +
     * size * md_attr->reg_cost.growth + overhead + latency +
     * max(size/bw , size/bcopy_bw) + latency + overhead + latency ]
     *
     * The latency function for RMA (get_zcopy) Rendezvous is:
     * [ reg_cost.overhead + size * md_attr->reg_cost.growth + latency + overhead +
     *   reg_cost.overhead + size * md_attr->reg_cost.growth + overhead + latency +
     *   size/bw + latency + overhead + latency ]
     *
     * Isolating the 'size' yields the rndv_thresh.
     * The used latency functions for eager_zcopy and rndv are also specified in
     * the UCX wiki */

    if (md_attr->cap.flags & UCT_MD_FLAG_REG) {
        md_reg_growth   = md_attr->reg_cost.growth;
        md_reg_overhead = md_attr->reg_cost.overhead;
    } else {
        md_reg_growth   = 0;
        md_reg_overhead = 0;
    }

    numerator = diff_percent * ((4 * ucp_tl_iface_latency(context, iface_attr)) +
                (3 * iface_attr->overhead) +
                (md_reg_overhead * (1 + recv_reg_cost))) -
                 md_reg_overhead - iface_attr->overhead;

    denumerator = md_reg_growth +
                  ucs_max((1.0 / iface_attr->bandwidth), (1.0 / context->config.ext.bcopy_bw)) -
                  (diff_percent * (ucs_max((1.0 / iface_attr->bandwidth), (1.0 / bcopy_bw)) +
                   md_reg_growth * (1 + recv_reg_cost)));

    if ((numerator > 0) && (denumerator > 0)) {
        return (numerator / denumerator);
    } else {
        return context->config.ext.rndv_thresh_fallback;
    }
}

static size_t ucp_ep_thresh(size_t thresh_value, size_t min_value,
                            size_t max_value)
{
    size_t thresh;

    ucs_assert(min_value <= max_value);

    thresh = ucs_max(min_value, thresh_value);
    thresh = ucs_min(max_value, thresh);

    return thresh;
}

static void ucp_ep_config_set_am_rndv_thresh(ucp_context_h context, uct_iface_attr_t *iface_attr,
                                             uct_md_attr_t *md_attr, ucp_ep_config_t *config,
                                             size_t max_rndv_thresh)
{
    size_t rndv_thresh, rndv_nbr_thresh;

    ucs_assert(config->key.am_lane != UCP_NULL_LANE);
    ucs_assert(config->key.lanes[config->key.am_lane].rsc_index != UCP_NULL_RESOURCE);

    if (config->key.err_mode == UCP_ERR_HANDLING_MODE_PEER) {
        /* Disable RNDV */
        rndv_thresh = rndv_nbr_thresh = SIZE_MAX;
    } else if (context->config.ext.rndv_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
        /* auto - Make UCX calculate the AM rndv threshold on its own.*/
        rndv_thresh     = ucp_ep_config_calc_rndv_thresh(context, iface_attr, md_attr,
                                                         context->config.ext.bcopy_bw,
                                                         0);
        rndv_nbr_thresh = context->config.ext.rndv_send_nbr_thresh;
    } else {
        rndv_thresh     = context->config.ext.rndv_thresh;
        rndv_nbr_thresh = context->config.ext.rndv_thresh;
    }

    config->tag.rndv.am_thresh = ucp_ep_thresh(rndv_thresh,
                                               iface_attr->cap.am.min_zcopy,
                                               max_rndv_thresh);

    config->tag.rndv_send_nbr.am_thresh = ucp_ep_thresh(rndv_nbr_thresh,
                                                        iface_attr->cap.am.min_zcopy,
                                                        max_rndv_thresh);

    ucs_trace("Active Message rndv threshold is %zu (send_nbr: %zu)",
              config->tag.rndv.am_thresh, config->tag.rndv_send_nbr.am_thresh);
}

static void ucp_ep_config_set_rndv_thresh(ucp_worker_t *worker,
                                          ucp_ep_config_t *config,
                                          ucp_lane_index_t lane,
                                          uint64_t rndv_cap_flag,
                                          size_t max_rndv_thresh)
{
    ucp_context_t *context = worker->context;
    ucp_rsc_index_t rsc_index;
    size_t rndv_thresh, rndv_nbr_thresh;
    uct_iface_attr_t *iface_attr;
    uct_md_attr_t *md_attr;

    if (lane == UCP_NULL_LANE) {
        ucs_debug("rendezvous (get_zcopy) protocol is not supported");
        return;
    }

    rsc_index = config->key.lanes[lane].rsc_index;
    if (rsc_index == UCP_NULL_RESOURCE) {
        return;
    }

    iface_attr = &worker->ifaces[rsc_index].attr;
    md_attr    = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
    ucs_assert_always(iface_attr->cap.flags & rndv_cap_flag);

    if (context->config.ext.rndv_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
        /* auto - Make UCX calculate the RMA (get_zcopy) rndv threshold on its own.*/
        rndv_thresh     = ucp_ep_config_calc_rndv_thresh(context, iface_attr,
                                                         md_attr, SIZE_MAX, 1);
        rndv_nbr_thresh = context->config.ext.rndv_send_nbr_thresh;
    } else {
        rndv_thresh     = context->config.ext.rndv_thresh;
        rndv_nbr_thresh = context->config.ext.rndv_thresh;
    }

    config->tag.rndv.max_get_zcopy = iface_attr->cap.get.max_zcopy;
    config->tag.rndv.max_put_zcopy = iface_attr->cap.put.max_zcopy;
    config->tag.rndv.rma_thresh    = ucp_ep_thresh(rndv_thresh,
                                                   iface_attr->cap.get.min_zcopy,
                                                   max_rndv_thresh);

    config->tag.rndv_send_nbr.rma_thresh = ucp_ep_thresh(rndv_nbr_thresh,
                                                         iface_attr->cap.get.min_zcopy,
                                                         max_rndv_thresh);

    ucs_trace("rndv threshold is %zu (send_nbr: %zu)",
              config->tag.rndv.rma_thresh, config->tag.rndv_send_nbr.rma_thresh);
}

static void ucp_ep_config_init_attrs(ucp_worker_t *worker, ucp_rsc_index_t rsc_index,
                                     ucp_ep_msg_config_t *config, size_t max_short,
                                     size_t max_bcopy, size_t max_zcopy,
                                     size_t max_iov, uint64_t short_flag,
                                     uint64_t bcopy_flag, uint64_t zcopy_flag,
                                     unsigned hdr_len, size_t adjust_min_val)
{
    ucp_context_t *context       = worker->context;
    uct_iface_attr_t *iface_attr = &worker->ifaces[rsc_index].attr;
    uct_md_attr_t *md_attr       = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
    size_t it;
    size_t zcopy_thresh;
    int mem_type;

    if (iface_attr->cap.flags & short_flag && !context->num_mem_type_mds) {
        config->max_short = max_short - hdr_len;
    } else {
        config->max_short = -1;
    }

    if (iface_attr->cap.flags & bcopy_flag) {
        config->max_bcopy = max_bcopy;
    }

    if (!((iface_attr->cap.flags & zcopy_flag) && (md_attr->cap.flags & UCT_MD_FLAG_REG))) {
        return;
    }

    config->max_zcopy = max_zcopy;
    config->max_iov   = ucs_min(UCP_MAX_IOV, max_iov);

    if (context->config.ext.zcopy_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
        config->zcopy_auto_thresh = 1;
        for (it = 0; it < UCP_MAX_IOV; ++it) {
            zcopy_thresh = ucp_ep_config_get_zcopy_auto_thresh(it + 1,
                                                               &md_attr->reg_cost,
                                                               context,
                                                               iface_attr->bandwidth);
            zcopy_thresh = ucs_min(zcopy_thresh, adjust_min_val);
            config->sync_zcopy_thresh[it] = zcopy_thresh;
            config->zcopy_thresh[it]      = zcopy_thresh;
        }
    } else {
        config->zcopy_auto_thresh    = 0;
        config->sync_zcopy_thresh[0] = config->zcopy_thresh[0] =
                ucs_min(context->config.ext.zcopy_thresh, adjust_min_val);
    }

    for (mem_type = 0; mem_type < UCT_MD_MEM_TYPE_LAST; mem_type++) {
        if (UCP_MEM_IS_HOST(mem_type)) {
            config->mem_type_zcopy_thresh[mem_type] = config->zcopy_thresh[0];
        } else if (md_attr->cap.reg_mem_types & UCS_BIT(mem_type)) {
            config->mem_type_zcopy_thresh[mem_type] = 1;
        }
    }
}

void ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config)
{
    ucp_context_h context = worker->context;
    ucp_ep_rma_config_t *rma_config;
    uct_iface_attr_t *iface_attr;
    uct_md_attr_t *md_attr;
    uct_memory_type_t mem_type;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    size_t it;
    size_t max_rndv_thresh;
    size_t max_am_rndv_thresh;

    /* Default settings */
    for (it = 0; it < UCP_MAX_IOV; ++it) {
        config->am.zcopy_thresh[it]              = SIZE_MAX;
        config->am.sync_zcopy_thresh[it]         = SIZE_MAX;
        config->tag.eager.zcopy_thresh[it]       = SIZE_MAX;
        config->tag.eager.sync_zcopy_thresh[it]  = SIZE_MAX;
    }

    for (mem_type = 0; mem_type < UCT_MD_MEM_TYPE_LAST; mem_type++) {
        config->am.mem_type_zcopy_thresh[mem_type]        = SIZE_MAX;
        config->tag.eager.mem_type_zcopy_thresh[mem_type] = SIZE_MAX;
    }

    config->tag.eager.zcopy_auto_thresh = 0;
    config->am.zcopy_auto_thresh        = 0;
    config->p2p_lanes                   = 0;
    config->bcopy_thresh                = context->config.ext.bcopy_thresh;
    config->tag.lane                    = UCP_NULL_LANE;
    config->tag.proto                   = &ucp_tag_eager_proto;
    config->tag.sync_proto              = &ucp_tag_eager_sync_proto;
    config->tag.rndv.rma_thresh         = SIZE_MAX;
    config->tag.rndv.max_get_zcopy      = SIZE_MAX;
    config->tag.rndv.max_put_zcopy      = SIZE_MAX;
    config->tag.rndv.am_thresh          = SIZE_MAX;
    config->tag.rndv_send_nbr.am_thresh = SIZE_MAX;
    config->tag.rndv_send_nbr.rma_thresh = SIZE_MAX;
    config->tag.rndv.rkey_size          = ucp_rkey_packed_size(context,
                                                               config->key.rma_bw_md_map);
    config->stream.proto                = &ucp_stream_am_proto;
    config->tag.offload.max_eager_short = -1;
    max_rndv_thresh                     = SIZE_MAX;
    max_am_rndv_thresh                  = SIZE_MAX;

    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            config->md_index[lane] = context->tl_rscs[rsc_index].md_index;
            if (ucp_worker_is_tl_p2p(worker, rsc_index)) {
                config->p2p_lanes |= UCS_BIT(lane);
            }
        } else {
            config->md_index[lane] = UCP_NULL_RESOURCE;
        }
    }

    /* Configuration for tag offload */
    if (config->key.tag_lane != UCP_NULL_LANE) {
        lane      = config->key.tag_lane;
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = &worker->ifaces[rsc_index].attr;
            ucp_ep_config_init_attrs(worker, rsc_index, &config->tag.eager,
                                     iface_attr->cap.tag.eager.max_short,
                                     iface_attr->cap.tag.eager.max_bcopy,
                                     iface_attr->cap.tag.eager.max_zcopy,
                                     iface_attr->cap.tag.eager.max_iov,
                                     UCT_IFACE_FLAG_TAG_EAGER_SHORT,
                                     UCT_IFACE_FLAG_TAG_EAGER_BCOPY,
                                     UCT_IFACE_FLAG_TAG_EAGER_ZCOPY, 0,
                                     iface_attr->cap.tag.eager.max_bcopy);

            config->tag.offload.max_rndv_iov    = iface_attr->cap.tag.rndv.max_iov;
            config->tag.offload.max_rndv_zcopy  = iface_attr->cap.tag.rndv.max_zcopy;
            config->tag.offload.max_eager_short = config->tag.eager.max_short;
            config->tag.eager.max_short         = -1;
            config->tag.sync_proto              = &ucp_tag_offload_sync_proto;
            config->tag.proto                   = &ucp_tag_offload_proto;
            config->tag.lane                    = lane;
            max_rndv_thresh                     = iface_attr->cap.tag.eager.max_zcopy;
            max_am_rndv_thresh                  = iface_attr->cap.tag.eager.max_bcopy;

            ucs_assert_always(iface_attr->cap.tag.rndv.max_hdr >=
                              sizeof(ucp_tag_offload_unexp_rndv_hdr_t));
            ucs_assert_always(config->tag.offload.max_eager_short >= 0);

            if (config->key.am_lane != UCP_NULL_LANE) {
                /* Must have active messages for using rendezvous */
                ucp_ep_config_set_rndv_thresh(worker, config, lane,
                                              UCT_IFACE_FLAG_TAG_RNDV_ZCOPY,
                                              max_rndv_thresh);
            }
        }
    }

    if (config->key.am_lane != UCP_NULL_LANE) {
        lane        = config->key.am_lane;
        rsc_index   = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = &worker->ifaces[rsc_index].attr;
            md_attr   = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
            ucp_ep_config_init_attrs(worker, rsc_index, &config->am,
                                     iface_attr->cap.am.max_short,
                                     iface_attr->cap.am.max_bcopy,
                                     iface_attr->cap.am.max_zcopy,
                                     iface_attr->cap.am.max_iov,
                                     UCT_IFACE_FLAG_AM_SHORT,
                                     UCT_IFACE_FLAG_AM_BCOPY,
                                     UCT_IFACE_FLAG_AM_ZCOPY,
                                     sizeof(ucp_eager_hdr_t), SIZE_MAX);

            /* Calculate rndv threshold for AM Rendezvous, which may be used by
             * any tag-matching protocol (AM and offload). */
            ucp_ep_config_set_am_rndv_thresh(context, iface_attr, md_attr, config,
                                             max_am_rndv_thresh);

            /* All keys must fit in RNDV packet.
             * TODO remove some MDs if they don't
             */
            ucs_assert_always(config->tag.rndv.rkey_size <= config->am.max_bcopy);

            if (!ucp_ep_is_tag_offload_enabled(config)) {
                /* Tag offload is disabled, AM will be used for all
                 * tag-matching protocols */
                /* TODO: set threshold level based on all available lanes */
                ucp_ep_config_set_rndv_thresh(worker, config,
                                              config->key.rma_bw_lanes[0],
                                              UCT_IFACE_FLAG_GET_ZCOPY,
                                              max_rndv_thresh);
                config->tag.eager      = config->am;
                config->tag.lane       = lane;
            }
        } else {
            /* Stub endpoint */
            config->am.max_bcopy = UCP_MIN_BCOPY;
        }
    }

    /* Configuration for remote memory access */
    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        if (ucp_ep_config_get_multi_lane_prio(config->key.rma_lanes, lane) == -1) {
            continue;
        }

        rma_config = &config->rma[lane];
        rsc_index  = config->key.lanes[lane].rsc_index;

        rma_config->put_zcopy_thresh = SIZE_MAX;
        rma_config->get_zcopy_thresh = SIZE_MAX;

        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = &worker->ifaces[rsc_index].attr;
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
                rma_config->max_put_short = iface_attr->cap.put.max_short;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
                rma_config->max_put_bcopy = iface_attr->cap.put.max_bcopy;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY) {
                rma_config->max_put_zcopy    = iface_attr->cap.put.max_zcopy;
                /* TODO: formula */
                if (context->config.ext.zcopy_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
                    rma_config->put_zcopy_thresh = 16384;
                } else {
                    rma_config->put_zcopy_thresh = context->config.ext.zcopy_thresh;
                }
                rma_config->put_zcopy_thresh = ucs_max(rma_config->put_zcopy_thresh,
                                                       iface_attr->cap.put.min_zcopy);
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_SHORT) {
                rma_config->max_get_short = iface_attr->cap.get.max_short;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_BCOPY) {
                rma_config->max_get_bcopy = iface_attr->cap.get.max_bcopy;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
                /* TODO: formula */
                rma_config->max_get_zcopy = iface_attr->cap.get.max_zcopy;
                if (context->config.ext.zcopy_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
                    rma_config->get_zcopy_thresh = 16384;
                } else {
                    rma_config->get_zcopy_thresh = context->config.ext.zcopy_thresh;
                }
                rma_config->get_zcopy_thresh = ucs_max(rma_config->get_zcopy_thresh,
                                                       iface_attr->cap.get.min_zcopy);
            }
        } else {
            rma_config->max_put_bcopy = UCP_MIN_BCOPY; /* Stub endpoint */
        }
    }
}

static void ucp_ep_config_print_tag_proto(FILE *stream, const char *name,
                                          size_t max_eager_short,
                                          size_t zcopy_thresh,
                                          size_t rndv_rma_thresh,
                                          size_t rndv_am_thresh)
{
    size_t max_bcopy, min_rndv;

    fprintf(stream, "# %23s: 0", name);
    if (max_eager_short > 0) {
        fprintf(stream, "..<egr/short>..%zu" , max_eager_short + 1);
    }

    min_rndv  = ucs_min(rndv_rma_thresh, rndv_am_thresh);
    max_bcopy = ucs_min(zcopy_thresh, min_rndv);
    if (max_eager_short < max_bcopy) {
        fprintf(stream, "..<egr/bcopy>..");
        if (max_bcopy < SIZE_MAX) {
            fprintf(stream, "%zu", max_bcopy);
        }
    }
    if (zcopy_thresh < min_rndv) {
        fprintf(stream, "..<egr/zcopy>..");
        if (min_rndv < SIZE_MAX) {
            fprintf(stream, "%zu", min_rndv);
        }
    }

    if (min_rndv < SIZE_MAX) {
        fprintf(stream, "..<rndv>..");
    }
    fprintf(stream, "(inf)\n");
}

static void ucp_ep_config_print_rma_proto(FILE *stream, const char *name,
                                          ucp_lane_index_t lane,
                                          size_t bcopy_thresh, size_t zcopy_thresh)
{

    fprintf(stream, "# %20s[%d]: 0", name, lane);
    if (bcopy_thresh > 0) {
        fprintf(stream, "..<short>");
    }
    if (bcopy_thresh < zcopy_thresh) {
        if (bcopy_thresh > 0) {
            fprintf(stream, "..%zu", bcopy_thresh);
        }
        fprintf(stream, "..<bcopy>");
    }
    if (zcopy_thresh < SIZE_MAX) {
        fprintf(stream, "..%zu..<zcopy>", zcopy_thresh);
    }
    fprintf(stream, "..(inf)\n");
}

int ucp_ep_config_get_multi_lane_prio(const ucp_lane_index_t *lanes,
                                      ucp_lane_index_t lane)
{
    int prio;
    for (prio = 0; prio < UCP_MAX_LANES; ++prio) {
        if (lane == lanes[prio]) {
            return prio;
        }
    }
    return -1;
}

void ucp_ep_config_lane_info_str(ucp_context_h context,
                                 const ucp_ep_config_key_t *key,
                                 const uint8_t *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 char *buf, size_t max)
{
    uct_tl_resource_desc_t *rsc;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t proxy_lane;
    char *p, *endp;
    char *desc_str;
    int prio;

    p          = buf;
    endp       = buf + max;
    rsc_index  = key->lanes[lane].rsc_index;
    proxy_lane = key->lanes[lane].proxy_lane;
    rsc        = &context->tl_rscs[rsc_index].tl_rsc;

    if ((proxy_lane == lane) || (proxy_lane == UCP_NULL_LANE)) {
        if (key->lanes[lane].proxy_lane == lane) {
            desc_str = " <proxy>";
        } else {
            desc_str = "";
        }
        snprintf(p, endp - p, "lane[%d]: %2d:" UCT_TL_RESOURCE_DESC_FMT " md[%d]%s %-*c-> ",
                 lane, rsc_index, UCT_TL_RESOURCE_DESC_ARG(rsc),
                 context->tl_rscs[rsc_index].md_index, desc_str,
                 20 - (int)(strlen(rsc->dev_name) + strlen(rsc->tl_name) + strlen(desc_str)),
                 ' ');
        p += strlen(p);

        if (addr_indices != NULL) {
            snprintf(p, endp - p, "addr[%d].", addr_indices[lane]);
            p += strlen(p);
        }

    } else {
        snprintf(p, endp - p, "lane[%d]: proxy to lane[%d] %12c -> ", lane,
                 proxy_lane, ' ');
        p += strlen(p);
    }

    snprintf(p, endp - p, "md[%d]", key->lanes[lane].dst_md_index);
    p += strlen(p);

    prio = ucp_ep_config_get_multi_lane_prio(key->rma_lanes, lane);
    if (prio != -1) {
        snprintf(p, endp - p, " rma#%d", prio);
        p += strlen(p);
    }

    prio = ucp_ep_config_get_multi_lane_prio(key->rma_bw_lanes, lane);
    if (prio != -1) {
        snprintf(p, endp - p, " rma_bw#%d", prio);
        p += strlen(p);
    }

    prio = ucp_ep_config_get_multi_lane_prio(key->amo_lanes, lane);
    if (prio != -1) {
        snprintf(p, endp - p, " amo#%d", prio);
        p += strlen(p);
    }

    if (key->am_lane == lane) {
        snprintf(p, endp - p, " am");
        p += strlen(p);
    }

    prio = ucp_ep_config_get_multi_lane_prio(key->am_bw_lanes, lane);
    if (prio != -1) {
        snprintf(p, endp - p, " am_bw#%d", prio);
        p += strlen(p);
    }

    if (lane == key->tag_lane) {
        snprintf(p, endp - p, " tag_offload");
        p += strlen(p);
    }

    if (key->wireup_lane == lane) {
        snprintf(p, endp - p, " wireup");
        p += strlen(p);
        if (aux_rsc_index != UCP_NULL_RESOURCE) {
            snprintf(p, endp - p, "{" UCT_TL_RESOURCE_DESC_FMT "}",
                     UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[aux_rsc_index].tl_rsc));
            p += strlen(p);
        }
    }
}

static void ucp_ep_config_print(FILE *stream, ucp_worker_h worker,
                                const ucp_ep_config_t *config,
                                const uint8_t *addr_indices,
                                ucp_rsc_index_t aux_rsc_index)
{
    ucp_context_h context = worker->context;
    char lane_info[128]   = {0};
    const ucp_ep_msg_config_t *tag_config;
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;

    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        ucp_ep_config_lane_info_str(context, &config->key, addr_indices, lane,
                                    aux_rsc_index, lane_info, sizeof(lane_info));
        fprintf(stream, "#                 %s\n", lane_info);
    }
    fprintf(stream, "#\n");

    if (context->config.features & UCP_FEATURE_TAG) {
         tag_config = (ucp_ep_is_tag_offload_enabled((ucp_ep_config_t *)config)) ?
                       &config->tag.eager : &config->am;
         ucp_ep_config_print_tag_proto(stream, "tag_send",
                                       tag_config->max_short,
                                       tag_config->zcopy_thresh[0],
                                       config->tag.rndv.rma_thresh,
                                       config->tag.rndv.am_thresh);
         ucp_ep_config_print_tag_proto(stream, "tag_send_nbr",
                                       tag_config->max_short,
                                       /* disable zcopy */
                                       config->tag.rndv_send_nbr.rma_thresh,
                                       config->tag.rndv_send_nbr.rma_thresh,
                                       config->tag.rndv_send_nbr.am_thresh);
         ucp_ep_config_print_tag_proto(stream, "tag_send_sync",
                                       tag_config->max_short,
                                       tag_config->sync_zcopy_thresh[0],
                                       config->tag.rndv.rma_thresh,
                                       config->tag.rndv.am_thresh);
     }

     if (context->config.features & UCP_FEATURE_RMA) {
         for (lane = 0; lane < config->key.num_lanes; ++lane) {
             if (ucp_ep_config_get_multi_lane_prio(config->key.rma_lanes, lane) == -1) {
                 continue;
             }
             ucp_ep_config_print_rma_proto(stream, "put", lane,
                                           ucs_max(config->rma[lane].max_put_short + 1,
                                                   config->bcopy_thresh),
                                           config->rma[lane].put_zcopy_thresh);
             ucp_ep_config_print_rma_proto(stream, "get", lane, 0,
                                           config->rma[lane].get_zcopy_thresh);
         }
     }

     if (context->config.features & (UCP_FEATURE_TAG|UCP_FEATURE_RMA)) {
         fprintf(stream, "#\n");
         fprintf(stream, "# %23s: mds ", "rma_bw");
         ucs_for_each_bit(md_index, config->key.rma_bw_md_map) {
             fprintf(stream, "[%d] ", md_index);
         }
     }

     if (context->config.features & UCP_FEATURE_TAG) {
         fprintf(stream, "rndv_rkey_size %zu\n", config->tag.rndv.rkey_size);
     }
}

void ucp_ep_print_info(ucp_ep_h ep, FILE *stream)
{
    ucp_rsc_index_t aux_rsc_index;
    ucp_lane_index_t wireup_lane;
    uct_ep_h wireup_ep;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP endpoint\n");
    fprintf(stream, "#\n");
    fprintf(stream, "#               peer: %s\n", ucp_ep_peer_name(ep));

    /* if there is a wireup lane, set aux_rsc_index to the stub ep resource */
    aux_rsc_index = UCP_NULL_RESOURCE;
    wireup_lane   = ucp_ep_config(ep)->key.wireup_lane;
    if (wireup_lane != UCP_NULL_LANE) {
        wireup_ep = ep->uct_eps[wireup_lane];
        if (ucp_wireup_ep_test(wireup_ep)) {
            aux_rsc_index = ucp_wireup_ep_get_aux_rsc_index(wireup_ep);
        }
    }

    ucp_ep_config_print(stream, ep->worker, ucp_ep_config(ep), NULL,
                        aux_rsc_index);

    fprintf(stream, "#\n");

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
}

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const uct_linear_growth_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth)
{
    double zcopy_thresh;
    double bcopy_bw = context->config.ext.bcopy_bw;

    zcopy_thresh = (iovcnt * reg_cost->overhead) /
                   ((1.0 / bcopy_bw) - (1.0 / bandwidth) - (iovcnt * reg_cost->growth));

    if ((zcopy_thresh < 0.0) || (zcopy_thresh > SIZE_MAX)) {
        return SIZE_MAX;
    }

    return zcopy_thresh;
}
