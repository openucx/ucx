/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_ep.h"
#include "ucp_worker.h"
#include "ucp_am.h"
#include "ucp_ep.inl"
#include "ucp_request.inl"

#include <ucp/wireup/wireup_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucp/wireup/wireup_cm.h>
#include <ucp/tag/eager.h>
#include <ucp/tag/offload.h>
#include <ucp/tag/rndv.h>
#include <ucp/stream/stream.h>
#include <ucp/core/ucp_listener.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <string.h>


typedef struct {
    double reg_growth;
    double reg_overhead;
    double overhead;
    double latency;
    size_t bw;
} ucp_ep_thresh_params_t;

extern const ucp_request_send_proto_t ucp_stream_am_proto;
extern const ucp_request_send_proto_t ucp_am_proto;
extern const ucp_request_send_proto_t ucp_am_reply_proto;

#ifdef ENABLE_STATS
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
    ucp_lane_index_t i;
    memset(key, 0, sizeof(*key));
    key->num_lanes        = 0;
    for (i = 0; i < UCP_MAX_LANES; ++i) {
        key->lanes[i].rsc_index    = UCP_NULL_RESOURCE;
        key->lanes[i].proxy_lane   = UCP_NULL_LANE;
        key->lanes[i].dst_md_index = UCP_MAX_MDS;
    }
    key->am_lane          = UCP_NULL_LANE;
    key->wireup_lane      = UCP_NULL_LANE;
    key->cm_lane          = UCP_NULL_LANE;
    key->rkey_ptr_lane    = UCP_NULL_LANE;
    key->tag_lane         = UCP_NULL_LANE;
    key->rma_bw_md_map    = 0;
    key->reachable_md_map = 0;
    key->dst_md_cmpts     = NULL;
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
    int ret;

    ep = ucs_strided_alloc_get(&worker->ep_alloc, "ucp_ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate ep");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ucp_ep_config_key_reset(&key);

    status = ucp_worker_get_ep_config(worker, &key, 0, &ep->cfg_index);
    if (status != UCS_OK) {
        goto err_free_ep;
    }

    ep->worker                      = worker;
    ep->am_lane                     = UCP_NULL_LANE;
    ep->flags                       = 0;
    ep->conn_sn                     = (ucp_ep_conn_sn_t)-1;
    ucp_ep_ext_gen(ep)->user_data   = NULL;
    ucp_ep_ext_gen(ep)->dest_ep_ptr = 0;
    ucp_ep_ext_gen(ep)->err_cb      = NULL;
    UCS_STATIC_ASSERT(sizeof(ucp_ep_ext_gen(ep)->ep_match) >=
                      sizeof(ucp_ep_ext_gen(ep)->listener));
    UCS_STATIC_ASSERT(sizeof(ucp_ep_ext_gen(ep)->ep_match) >=
                      sizeof(ucp_ep_ext_gen(ep)->flush_state));
    memset(&ucp_ep_ext_gen(ep)->ep_match, 0,
           sizeof(ucp_ep_ext_gen(ep)->ep_match));

    ucp_stream_ep_init(ep);
    ucp_am_ep_init(ep);

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
    kh_put(ucp_worker_ep_ptrs, &worker->ep_ptrs, (uintptr_t)ep, &ret);
    if (ret == 0) {
        ucs_warn("failed to add ep %p to worker (%p) ep ptrs hash", ep, worker);
    }

    *ep_p = ep;
    ucs_debug("created ep %p to %s %s", ep, ucp_ep_peer_name(ep), message);
    return UCS_OK;

err_free_ep:
    ucs_strided_alloc_put(&worker->ep_alloc, ep);
err:
    return status;
}

void ucp_ep_delete(ucp_ep_h ep)
{
    khiter_t iter;

    ucs_callbackq_remove_if(&ep->worker->uct->progress_q,
                            ucp_wireup_msg_ack_cb_pred, ep);
    UCS_STATS_NODE_FREE(ep->stats);
    ucs_list_del(&ucp_ep_ext_gen(ep)->ep_list);

    iter = kh_get(ucp_worker_ep_ptrs, &ep->worker->ep_ptrs, (uintptr_t)ep);
    if (iter != kh_end(&ep->worker->ep_ptrs)) {
        kh_del(ucp_worker_ep_ptrs, &ep->worker->ep_ptrs, iter);
    } else {
        ucs_warn("ep %p does not exist on worker %p", ep, ep->worker);
    }

    ucs_strided_alloc_put(&ep->worker->ep_alloc, ep);
}

ucs_status_t
ucp_ep_create_sockaddr_aux(ucp_worker_h worker, unsigned ep_init_flags,
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

    status = ucp_ep_init_create_wireup(ep, ep_init_flags, &wireup_ep);
    if (status != UCS_OK) {
        goto err_delete;
    }

    status = ucp_wireup_ep_connect_aux(wireup_ep, ep_init_flags, remote_address);
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

void ucp_ep_config_key_set_err_mode(ucp_ep_config_key_t *key,
                                    unsigned ep_init_flags)
{
    key->err_mode = (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE) ?
                    UCP_ERR_HANDLING_MODE_PEER : UCP_ERR_HANDLING_MODE_NONE;
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
    unsigned i, mem_type;
    ucs_status_t status;
    void *address_buffer;
    size_t address_length;

    for (mem_type = 0; mem_type < UCS_MEMORY_TYPE_LAST; mem_type++) {
        if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type) ||
            !context->mem_type_access_tls[mem_type]) {
            continue;
        }

        status = ucp_address_pack(worker, NULL,
                                  context->mem_type_access_tls[mem_type],
                                  UCP_ADDRESS_PACK_FLAGS_ALL, NULL,
                                  &address_length, &address_buffer);
        if (status != UCS_OK) {
            goto err_cleanup_eps;
        }

        status = ucp_address_unpack(worker, address_buffer,
                                    UCP_ADDRESS_PACK_FLAGS_ALL, &local_address);
        if (status != UCS_OK) {
            goto err_free_address_buffer;
        }

        status = ucp_ep_create_to_worker_addr(worker, UINT64_MAX,
                                              &local_address,
                                              UCP_EP_INIT_FLAG_MEM_TYPE,
                                              "mem type",
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
    for (i = 0; i < UCS_MEMORY_TYPE_LAST; i++) {
        if (worker->mem_type_ep[i]) {
           ucp_ep_destroy_internal(worker->mem_type_ep[i]);
        }
    }
    return status;
}

ucs_status_t ucp_ep_init_create_wireup(ucp_ep_h ep, unsigned ep_init_flags,
                                       ucp_wireup_ep_t **wireup_ep)
{
    ucp_ep_config_key_t key;
    ucs_status_t status;

    ucp_ep_config_key_reset(&key);
    ucp_ep_config_key_set_err_mode(&key, ep_init_flags);

    key.num_lanes             = 1;
    /* all operations will use the first lane, which is a stub endpoint before
     * reconfiguration */
    key.am_lane = 0;
    if (ucp_worker_sockaddr_is_cm_proto(ep->worker)) {
        key.cm_lane = 0;
    } else {
        key.wireup_lane = 0;
    }

    status = ucp_worker_get_ep_config(ep->worker, &key, 0, &ep->cfg_index);
    if (status != UCS_OK) {
        return status;
    }

    ep->am_lane = key.am_lane;
    ep->flags  |= UCP_EP_FLAG_CONNECT_REQ_QUEUED;

    status = ucp_wireup_ep_create(ep, &ep->uct_eps[0]);
    if (status != UCS_OK) {
        return status;
    }

    *wireup_ep = ucs_derived_of(ep->uct_eps[0], ucp_wireup_ep_t);
    return UCS_OK;
}

ucs_status_t ucp_ep_create_to_worker_addr(ucp_worker_h worker,
                                          uint64_t local_tl_bitmap,
                                          const ucp_unpacked_address_t *remote_address,
                                          unsigned ep_init_flags,
                                          const char *message, ucp_ep_h *ep_p)
{
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_t status;
    ucp_ep_h ep;

    /* allocate endpoint */
    status = ucp_ep_new(worker, remote_address->name, message, &ep);
    if (status != UCS_OK) {
        goto err;
    }

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes(ep, ep_init_flags, local_tl_bitmap,
                                   remote_address, addr_indices);
    if (status != UCS_OK) {
        goto err_delete;
    }

    ucs_assert(!(ucp_ep_get_tl_bitmap(ep) & ~local_tl_bitmap));

    *ep_p = ep;
    return UCS_OK;

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

    status = ucp_ep_init_create_wireup(ep, ucp_ep_init_flags(worker, params),
                                       &wireup_ep);
    if (status != UCS_OK) {
        goto err_delete;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        goto err_cleanup_lanes;
    }

    status = ucp_worker_sockaddr_is_cm_proto(ep->worker) ?
             ucp_ep_client_cm_connect_start(ep, params) :
             ucp_wireup_ep_connect_to_sockaddr(ep->uct_eps[0], params);
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

/**
 * Create an endpoint on the server side connected to the client endpoint.
 */
ucs_status_t ucp_ep_create_server_accept(ucp_worker_h worker,
                                         const ucp_conn_request_h conn_request,
                                         ucp_ep_h *ep_p)
{
    const ucp_wireup_sockaddr_data_t *sa_data = &conn_request->sa_data;
    unsigned ep_init_flags                    = 0;
    ucp_unpacked_address_t           remote_addr;
    uint64_t                         addr_flags;
    unsigned                         i;
    ucs_status_t                     status;

    if (sa_data->err_mode == UCP_ERR_HANDLING_MODE_PEER) {
        ep_init_flags |= UCP_EP_INIT_ERR_MODE_PEER_FAILURE;
    }

    if (sa_data->addr_mode == UCP_WIREUP_SA_DATA_CM_ADDR) {
        addr_flags = UCP_ADDRESS_PACK_FLAG_IFACE_ADDR |
                     UCP_ADDRESS_PACK_FLAG_EP_ADDR;
    } else {
        addr_flags = UCP_ADDRESS_PACK_FLAGS_ALL;
    }

    /* coverity[overrun-local] */
    status = ucp_address_unpack(worker, sa_data + 1, addr_flags, &remote_addr);
    if (status != UCS_OK) {
        goto out;
    }

    switch (sa_data->addr_mode) {
    case UCP_WIREUP_SA_DATA_FULL_ADDR:
        /* create endpoint to the worker address we got in the private data */
        status = ucp_ep_create_to_worker_addr(worker, UINT64_MAX, &remote_addr,
                                              ep_init_flags |
                                              UCP_EP_INIT_CREATE_AM_LANE,
                                              "listener", ep_p);
        if (status != UCS_OK) {
            goto out_free_address;
        }

        ucs_assert(ucp_ep_config(*ep_p)->key.err_mode == sa_data->err_mode);
        ucp_ep_flush_state_reset(*ep_p);
        ucp_ep_update_dest_ep_ptr(*ep_p, sa_data->ep_ptr);
        break;
    case UCP_WIREUP_SA_DATA_PARTIAL_ADDR:
        status = ucp_ep_create_sockaddr_aux(worker, ep_init_flags,
                                            &remote_addr, ep_p);
        if (status != UCS_OK) {
            goto out_free_address;
        }

        ucp_ep_update_dest_ep_ptr(*ep_p, sa_data->ep_ptr);
        /* the server's ep should be aware of the sent address from the client */
        (*ep_p)->flags |= UCP_EP_FLAG_LISTENER;
        /* NOTE: protect union */
        ucs_assert(!((*ep_p)->flags & (UCP_EP_FLAG_ON_MATCH_CTX |
                                       UCP_EP_FLAG_FLUSH_STATE_VALID)));
        break;
    case UCP_WIREUP_SA_DATA_CM_ADDR:
        ucs_assert(ucp_worker_sockaddr_is_cm_proto(worker));
        for (i = 0; i < remote_addr.address_count; ++i) {
            remote_addr.address_list[i].dev_addr  = conn_request->remote_dev_addr;
            remote_addr.address_list[i].dev_index = conn_request->sa_data.dev_index;
        }
        status = ucp_ep_cm_server_create_connected(worker,
                                                   ep_init_flags |
                                                   UCP_EP_INIT_CM_WIREUP_SERVER,
                                                   &remote_addr, conn_request,
                                                   ep_p);
        if (status != UCS_OK) {
            goto out_free_address;
        }

        (*ep_p)->flags                 |= UCP_EP_FLAG_LISTENER;
        ucp_ep_ext_gen(*ep_p)->listener = conn_request->listener;
        break;
    default:
        ucs_fatal("client sockaddr data contains invalid address mode %d",
                  sa_data->addr_mode);
    }

out_free_address:
    ucs_free(remote_addr.address_list);
out:
    return status;
}

static ucs_status_t
ucp_ep_create_api_conn_request(ucp_worker_h worker,
                               const ucp_ep_params_t *params, ucp_ep_h *ep_p)
{
    ucp_conn_request_h conn_request = params->conn_request;
    ucp_ep_h           ep;
    ucs_status_t       status;

    status = ucp_ep_create_server_accept(worker, conn_request, &ep);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        goto out_ep_destroy;
    }

    if (ucp_worker_sockaddr_is_cm_proto(worker)) {
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

    if (status == UCS_OK) {
        goto out;
    }

out_ep_destroy:
    ucp_ep_destroy_internal(ep);
out:
    if (status == UCS_OK) {
        if (!ucp_worker_sockaddr_is_cm_proto(worker)) {
            status = uct_iface_accept(conn_request->uct.iface,
                                      conn_request->uct_req);
        }
    } else {
        if (ucp_worker_sockaddr_is_cm_proto(worker)) {
            uct_listener_reject(conn_request->uct.listener,
                                conn_request->uct_req);
        } else {
            uct_iface_reject(conn_request->uct.iface, conn_request->uct_req);
        }
    }

    if (ucp_worker_sockaddr_is_cm_proto(worker)) {
        ucs_free(conn_request->remote_dev_addr);
    }

    ucs_free(conn_request);

    if (status == UCS_OK) {
        *ep_p = ep;
    }

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

    status = ucp_address_unpack(worker, params->address,
                                UCP_ADDRESS_PACK_FLAGS_ALL, &remote_address);
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

        ucp_ep_flush_state_reset(ep);
        ucp_stream_ep_activate(ep);
        goto out_free_address;
    }

    status = ucp_ep_create_to_worker_addr(worker, UINT64_MAX, &remote_address,
                                          ucp_ep_init_flags(worker, params),
                                          "from api call", &ep);
    if (status != UCS_OK) {
        goto out_free_address;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        ucp_ep_destroy_internal(ep);
        goto out_free_address;
    }

    ep->conn_sn = conn_sn;

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
        ucp_ep_flush_state_reset(ep);
    } else {
        ucp_ep_match_insert_exp(&worker->ep_match_ctx, remote_address.uuid, ep);
    }

    /* if needed, send initial wireup message */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        ucs_assert(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED));
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
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

    UCS_ASYNC_BLOCK(&worker->async);

    flags = UCP_PARAM_VALUE(EP, params, flags, FLAGS, 0);
    if (flags & UCP_EP_PARAMS_FLAGS_CLIENT_SERVER) {
        status = ucp_ep_create_to_sock_addr(worker, params, &ep);
    } else if (params->field_mask & UCP_EP_PARAM_FIELD_CONN_REQUEST) {
        status = ucp_ep_create_api_conn_request(worker, params, &ep);
    } else if (params->field_mask & UCP_EP_PARAM_FIELD_REMOTE_ADDRESS) {
        status = ucp_ep_create_api_to_worker_addr(worker, params, &ep);
    } else {
        status = UCS_ERR_INVALID_PARAM;
    }

    if (status == UCS_OK) {
        ep->flags |= UCP_EP_FLAG_USED;
        *ep_p      = ep;
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
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

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_ep_adjust_params(ep, params);

    UCS_ASYNC_UNBLOCK(&worker->async);

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

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        ep->uct_eps[lane] = NULL;
    }
}

/* Must be called with async lock held */
void ucp_ep_disconnected(ucp_ep_h ep, int force)
{
    /* remove pending slow-path progress in case it wasn't removed yet */
    ucs_callbackq_remove_if(&ep->worker->uct->progress_q,
                            ucp_worker_err_handle_remove_filter, ep);

    /* remove pending slow-path function it wasn't removed yet */
    ucs_callbackq_remove_if(&ep->worker->uct->progress_q,
                            ucp_listener_accept_cb_remove_filter, ep);

    ucp_ep_cm_slow_cbq_cleanup(ep);

    ucp_stream_ep_cleanup(ep);
    ucp_am_ep_cleanup(ep);

    ep->flags &= ~UCP_EP_FLAG_USED;

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

unsigned ucp_ep_local_disconnect_progress(void *arg)
{
    ucp_request_t *req         = arg;
    ucp_ep_h ep                = req->send.ep;
    ucs_async_context_t *async = &ep->worker->async; /* ep becomes invalid */

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_COMPLETED));

    UCS_ASYNC_BLOCK(async);
    ucs_debug("ep %p: disconnected with request %p, %s", ep, req,
              ucs_status_string(req->status));
    ucp_ep_disconnected(ep, req->send.flush.uct_flags & UCT_FLUSH_FLAG_CANCEL);
    UCS_ASYNC_UNBLOCK(async);

    /* Complete send request from here, to avoid releasing the request while
     * slow-path element is still pending */
    ucp_request_complete_send(req, req->status);

    return 0;
}

static void ucp_ep_set_close_request(ucp_ep_h ep, ucp_request_t *request,
                                     const char *debug_msg)
{
    ucs_trace("ep %p: setting close request %p, %s", ep, request, debug_msg);

    ucp_ep_flush_state_invalidate(ep);
    ucp_ep_ext_gen(ep)->close_req.req = request;
    ep->flags                        |= UCP_EP_FLAG_CLOSE_REQ_VALID;
}

static void ucp_ep_close_flushed_callback(ucp_request_t *req)
{
    ucp_ep_h ep                = req->send.ep;
    ucs_async_context_t *async = &ep->worker->async;

    /* in case of force close, schedule ucp_ep_local_disconnect_progress to
     * destroy the ep and all its lanes */
    if (req->send.flush.uct_flags & UCT_FLUSH_FLAG_CANCEL) {
        goto out;
    }

    UCS_ASYNC_BLOCK(async);

    ucs_debug("ep %p: flags 0x%x close flushed callback for request %p", ep,
              ep->flags, req);

    if (ucp_ep_is_cm_local_connected(ep)) {
        /* Now, when close flush is completed and we are still locally connected,
         * we have to notify remote side */
        ucp_ep_cm_disconnect_cm_lane(ep);
        if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
            /* Wait disconnect notification from remote side to complete this
             * request */
            ucp_ep_set_close_request(ep, req, "close flushed callback");
            UCS_ASYNC_UNBLOCK(async);
            return;
        }
    }
    UCS_ASYNC_UNBLOCK(async);

out:
    /* If a flush is completed from a pending/completion callback, we need to
     * schedule slow-path callback to release the endpoint later, since a UCT
     * endpoint cannot be released from pending/completion callback context.
     */
    ucs_trace("adding slow-path callback to destroy ep %p", ep);
    req->send.disconnect.prog_id = UCS_CALLBACKQ_ID_NULL;
    uct_worker_progress_register_safe(ep->worker->uct,
                                      ucp_ep_local_disconnect_progress,
                                      req, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &req->send.disconnect.prog_id);
}

static ucs_status_t ucp_ep_close_nb_check_params(ucp_ep_h ep, unsigned mode)
{
    /* CM lane tracks remote state, so it can be used with any modes of close
     * and error handling */
    if ((mode == UCP_EP_CLOSE_MODE_FLUSH) ||
        (ucp_ep_get_cm_lane(ep) != UCP_NULL_LANE)) {
        return UCS_OK;
    }

    /* In case of close in force mode, remote peer failure detection mechanism
     * should be enabled (CM lane is handled above) to prevent hang or any
     * other undefined behavior */
    if ((mode == UCP_EP_CLOSE_MODE_FORCE) &&
        (ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_PEER)) {
        return UCS_OK;
    }

    return UCS_ERR_INVALID_PARAM;
}

ucs_status_ptr_t ucp_ep_close_nb(ucp_ep_h ep, unsigned mode)
{
    ucp_worker_h  worker = ep->worker;
    void          *request;
    ucp_request_t *close_req;
    ucs_status_t  status;

    status = ucp_ep_close_nb_check_params(ep, mode);
    if (status != UCS_OK) {
        return UCS_STATUS_PTR(status);
    }

    UCS_ASYNC_BLOCK(&worker->async);

    ep->flags |= UCP_EP_FLAG_CLOSED;
    request = ucp_ep_flush_internal(ep,
                                    (mode == UCP_EP_CLOSE_MODE_FLUSH) ?
                                    UCT_FLUSH_FLAG_LOCAL : UCT_FLUSH_FLAG_CANCEL,
                                    NULL, 0, NULL,
                                    ucp_ep_close_flushed_callback, "close");

    if (!UCS_PTR_IS_PTR(request)) {
        if (ucp_ep_is_cm_local_connected(ep) &&
            (mode == UCP_EP_CLOSE_MODE_FLUSH)) {
            /* lanes already flushed, start disconnect on CM lane */
            ucp_ep_cm_disconnect_cm_lane(ep);
            close_req = ucp_ep_cm_close_request_get(ep);
            if (close_req != NULL) {
                request = close_req + 1;
                ucp_ep_set_close_request(ep, close_req, "close");
            } else {
                request = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
            }
        } else {
            ucp_ep_disconnected(ep, mode == UCP_EP_CLOSE_MODE_FORCE);
        }
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
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

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
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
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return;
}

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2)
{
    ucp_lane_index_t lane;
    int i;

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
        (key1->cm_lane          != key2->cm_lane)                                  ||
        (key1->rkey_ptr_lane    != key2->rkey_ptr_lane)                            ||
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

    for (i = 0; i < ucs_popcount(key1->reachable_md_map); ++i) {
        if (key1->dst_md_cmpts[i] != key2->dst_md_cmpts[i]) {
            return 0;
        }
    }

    return 1;
}

static void ucp_ep_config_calc_params(ucp_worker_h worker,
                                      const ucp_ep_config_t *config,
                                      const ucp_lane_index_t *lanes,
                                      ucp_ep_thresh_params_t *params)
{
    ucp_context_h context = worker->context;
    ucp_md_map_t md_map   = 0;
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    uct_iface_attr_t *iface_attr;
    int i;

    memset(params, 0, sizeof(*params));

    for (i = 0; (i < UCP_MAX_LANES) && (lanes[i] != UCP_NULL_LANE); i++) {
        lane      = lanes[i];
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        md_index   = config->md_index[lane];
        iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

        if (!(md_map & UCS_BIT(md_index))) {
            md_map |= UCS_BIT(md_index);
            md_attr = &context->tl_mds[md_index].attr;
            if (md_attr->cap.flags & UCT_MD_FLAG_REG) {
                params->reg_growth   += md_attr->reg_cost.growth;
                params->reg_overhead += md_attr->reg_cost.overhead;
                params->overhead     += iface_attr->overhead;
                params->latency      += ucp_tl_iface_latency(context,
                                                             &iface_attr->latency);
            }
        }

        params->bw += ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);
    }
}

static size_t ucp_ep_config_calc_rndv_thresh(ucp_worker_t *worker,
                                             const ucp_ep_config_t *config,
                                             const ucp_lane_index_t *eager_lanes,
                                             const ucp_lane_index_t *rndv_lanes,
                                             int recv_reg_cost)
{
    ucp_context_h context = worker->context;
    double diff_percent   = 1.0 - context->config.ext.rndv_perf_diff / 100.0;
    ucp_ep_thresh_params_t eager_zcopy;
    ucp_ep_thresh_params_t rndv;
    double numerator, denumerator;
    ucp_rsc_index_t eager_rsc_index;
    uct_iface_attr_t *eager_iface_attr;
    double rts_latency;

    /* All formulas and descriptions are listed at
     * https://github.com/openucx/ucx/wiki/Rendezvous-Protocol-threshold-for-multilane-mode */

    ucp_ep_config_calc_params(worker, config, eager_lanes, &eager_zcopy);
    ucp_ep_config_calc_params(worker, config, rndv_lanes, &rndv);

    if ((eager_zcopy.bw == 0) || (rndv.bw == 0)) {
        goto fallback;
    }

    eager_rsc_index  = config->key.lanes[eager_lanes[0]].rsc_index;
    eager_iface_attr = ucp_worker_iface_get_attr(worker, eager_rsc_index);

    /* RTS/RTR latency is used from lanes[0] */
    rts_latency      = ucp_tl_iface_latency(context, &eager_iface_attr->latency);

    numerator = diff_percent * (rndv.reg_overhead * (1 + recv_reg_cost) +
                                (2 * rts_latency) + (2 * rndv.latency) +
                                (2 * eager_zcopy.overhead) + rndv.overhead) -
                eager_zcopy.reg_overhead - eager_zcopy.overhead;

    denumerator = eager_zcopy.reg_growth +
                  1.0 / ucs_min(eager_zcopy.bw, context->config.ext.bcopy_bw) -
                  diff_percent *
                  (rndv.reg_growth * (1 + recv_reg_cost) + 1.0 / rndv.bw);

    if ((numerator > 0) && (denumerator > 0)) {
        return ucs_max(numerator / denumerator, eager_iface_attr->cap.am.max_bcopy);
    }

fallback:
    return context->config.ext.rndv_thresh_fallback;
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

static size_t ucp_ep_config_calc_rma_zcopy_thresh(ucp_worker_t *worker,
                                                  const ucp_ep_config_t *config,
                                                  const ucp_lane_index_t *rma_lanes)
{
    ucp_context_h context = worker->context;
    double bcopy_bw       = context->config.ext.bcopy_bw;
    ucp_ep_thresh_params_t rma;
    uct_md_attr_t *md_attr;
    double numerator, denumerator;
    double reg_overhead, reg_growth;

    ucp_ep_config_calc_params(worker, config, rma_lanes, &rma);

    if (rma.bw == 0) {
        goto fallback;
    }

    md_attr = &context->tl_mds[config->md_index[rma_lanes[0]]].attr;
    if (md_attr->cap.flags & UCT_MD_FLAG_NEED_MEMH) {
        reg_overhead = rma.reg_overhead;
        reg_growth   = rma.reg_growth;
    } else {
        reg_overhead = 0;
        reg_growth   = 0;
    }

    numerator   = reg_overhead;
    denumerator = (1 / bcopy_bw) - reg_growth;

    if (denumerator > 0) {
        return numerator / denumerator;
    }

fallback:
    return SIZE_MAX;
}

static void ucp_ep_config_adjust_max_short(ssize_t *max_short,
                                           size_t thresh)
{
    *max_short = ucs_min((size_t)(*max_short + 1), thresh) - 1;
    ucs_assert(*max_short >= -1);
}

/* With tag offload, SW RNDV requests are temporarily stored in the receiver
 * user buffer when matched. Thus, minimum message size allowed to be sent with
 * RNDV protocol should be bigger than maximal possible SW RNDV request
 * (i.e. header plus packed keys size). */
size_t ucp_ep_tag_offload_min_rndv_thresh(ucp_ep_config_t *config)
{
    return sizeof(ucp_rndv_rts_hdr_t) + config->tag.rndv.rkey_size;
}

static void ucp_ep_config_set_am_rndv_thresh(ucp_worker_h worker,
                                             uct_iface_attr_t *iface_attr,
                                             uct_md_attr_t *md_attr,
                                             ucp_ep_config_t *config,
                                             size_t min_rndv_thresh,
                                             size_t max_rndv_thresh)
{
    ucp_context_h context = worker->context;
    size_t rndv_thresh, rndv_nbr_thresh, min_thresh;

    ucs_assert(config->key.am_lane != UCP_NULL_LANE);
    ucs_assert(config->key.lanes[config->key.am_lane].rsc_index != UCP_NULL_RESOURCE);

    if (!ucp_ep_config_test_rndv_support(config)) {
        /* Disable RNDV */
        rndv_thresh = rndv_nbr_thresh = SIZE_MAX;
    } else if (context->config.ext.rndv_thresh == UCS_MEMUNITS_AUTO) {
        /* auto - Make UCX calculate the AM rndv threshold on its own.*/
        rndv_thresh     = ucp_ep_config_calc_rndv_thresh(worker, config,
                                                         config->key.am_bw_lanes,
                                                         config->key.am_bw_lanes,
                                                         0);
        rndv_nbr_thresh = context->config.ext.rndv_send_nbr_thresh;
        ucs_trace("active message rendezvous threshold is %zu", rndv_thresh);
    } else {
        rndv_thresh     = context->config.ext.rndv_thresh;
        rndv_nbr_thresh = context->config.ext.rndv_thresh;

        /* adjust max_short if rndv_thresh is set externally */
        ucp_ep_config_adjust_max_short(&config->tag.eager.max_short,
                                       rndv_thresh);
    }

    min_thresh = ucs_max(iface_attr->cap.am.min_zcopy, min_rndv_thresh);

    config->tag.rndv.am_thresh = ucp_ep_thresh(rndv_thresh,
                                               min_thresh,
                                               max_rndv_thresh);

    config->tag.rndv_send_nbr.am_thresh = ucp_ep_thresh(rndv_nbr_thresh,
                                                        min_thresh,
                                                        max_rndv_thresh);

    ucs_trace("Active Message rndv threshold is %zu (send_nbr: %zu)",
              config->tag.rndv.am_thresh, config->tag.rndv_send_nbr.am_thresh);
}

static void ucp_ep_config_set_rndv_thresh(ucp_worker_t *worker,
                                          ucp_ep_config_t *config,
                                          ucp_lane_index_t *lanes,
                                          size_t min_rndv_thresh,
                                          size_t max_rndv_thresh)
{
    ucp_context_t *context = worker->context;
    ucp_lane_index_t lane  = lanes[0];
    ucp_rsc_index_t rsc_index;
    size_t rndv_thresh, rndv_nbr_thresh, min_thresh;
    uct_iface_attr_t *iface_attr;

    if (lane == UCP_NULL_LANE) {
        ucs_debug("rendezvous (get_zcopy) protocol is not supported");
        return;
    }

    rsc_index = config->key.lanes[lane].rsc_index;
    if (rsc_index == UCP_NULL_RESOURCE) {
        return;
    }

    iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

    if (!ucp_ep_config_test_rndv_support(config)) {
        /* Disable RNDV */
        rndv_thresh = rndv_nbr_thresh = SIZE_MAX;
    } else if (context->config.ext.rndv_thresh == UCS_MEMUNITS_AUTO) {
        /* auto - Make UCX calculate the RMA (get_zcopy) rndv threshold on its own.*/
        rndv_thresh     = ucp_ep_config_calc_rndv_thresh(worker, config,
                                                         config->key.am_bw_lanes,
                                                         lanes, 1);
        rndv_nbr_thresh = context->config.ext.rndv_send_nbr_thresh;
    } else {
        rndv_thresh     = context->config.ext.rndv_thresh;
        rndv_nbr_thresh = context->config.ext.rndv_thresh;

        /* adjust max_short if rndv_thresh is set externally */
        ucp_ep_config_adjust_max_short(&config->tag.eager.max_short,
                                       rndv_thresh);
    }

    min_thresh = ucs_max(iface_attr->cap.get.min_zcopy, min_rndv_thresh);

    /* TODO: need to check minimal PUT Zcopy */
    config->tag.rndv.rma_thresh = ucp_ep_thresh(rndv_thresh,
                                                min_thresh,
                                                max_rndv_thresh);

    config->tag.rndv_send_nbr.rma_thresh = ucp_ep_thresh(rndv_nbr_thresh,
                                                         min_thresh,
                                                         max_rndv_thresh);

    ucs_trace("rndv threshold is %zu (send_nbr: %zu)",
              config->tag.rndv.rma_thresh, config->tag.rndv_send_nbr.rma_thresh);
}

static void ucp_ep_config_set_memtype_thresh(ucp_memtype_thresh_t *max_eager_short,
                                             ssize_t max_short, int num_mem_type_mds)
{
    if (!num_mem_type_mds) {
        max_eager_short->memtype_off = max_short;
    }

    max_eager_short->memtype_on = max_short;
}

static void ucp_ep_config_init_attrs(ucp_worker_t *worker, ucp_rsc_index_t rsc_index,
                                     ucp_ep_msg_config_t *config, size_t max_short,
                                     size_t max_bcopy, size_t max_zcopy,
                                     size_t max_iov, uint64_t short_flag,
                                     uint64_t bcopy_flag, uint64_t zcopy_flag,
                                     unsigned hdr_len, size_t adjust_min_val)
{
    ucp_context_t *context = worker->context;
    const uct_md_attr_t *md_attr;
    uct_iface_attr_t *iface_attr;
    size_t it;
    size_t zcopy_thresh;
    int mem_type;

    iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

    if ((iface_attr->cap.flags & short_flag)) {
        config->max_short = max_short - hdr_len;
    } else {
        config->max_short = -1;
    }

    if (iface_attr->cap.flags & bcopy_flag) {
        config->max_bcopy = max_bcopy;
    } else {
        config->max_bcopy = SIZE_MAX;
    }

    md_attr = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
    if (!(iface_attr->cap.flags & zcopy_flag) ||
        ((md_attr->cap.flags & UCT_MD_FLAG_NEED_MEMH) &&
         !(md_attr->cap.flags & UCT_MD_FLAG_REG))) {
        return;
    }

    config->max_zcopy = max_zcopy;
    config->max_iov   = ucs_min(UCP_MAX_IOV, max_iov);

    if (context->config.ext.zcopy_thresh == UCS_MEMUNITS_AUTO) {
        config->zcopy_auto_thresh = 1;
        for (it = 0; it < UCP_MAX_IOV; ++it) {
            zcopy_thresh = ucp_ep_config_get_zcopy_auto_thresh(
                               it + 1, &md_attr->reg_cost, context,
                               ucp_tl_iface_bandwidth(context,
                                                      &iface_attr->bandwidth));
            zcopy_thresh = ucs_min(zcopy_thresh, adjust_min_val);
            config->sync_zcopy_thresh[it] = zcopy_thresh;
            config->zcopy_thresh[it]      = zcopy_thresh;
        }
    } else {
        config->zcopy_auto_thresh    = 0;
        config->sync_zcopy_thresh[0] = config->zcopy_thresh[0] =
                ucs_min(context->config.ext.zcopy_thresh, adjust_min_val);

        /* adjust max_short if zcopy_thresh is set externally */
        ucp_ep_config_adjust_max_short(&config->max_short,
                                       config->zcopy_thresh[0]);
    }

    for (mem_type = 0; mem_type < UCS_MEMORY_TYPE_LAST; mem_type++) {
        if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type)) {
            config->mem_type_zcopy_thresh[mem_type] = config->zcopy_thresh[0];
        } else if (md_attr->cap.reg_mem_types & UCS_BIT(mem_type)) {
            config->mem_type_zcopy_thresh[mem_type] = 1;
        }
    }
}

static ucs_status_t ucp_ep_config_key_copy(ucp_ep_config_key_t *dst,
                                           const ucp_ep_config_key_t *src)
{
    *dst = *src;
    dst->dst_md_cmpts = ucs_calloc(ucs_popcount(src->reachable_md_map),
                                   sizeof(*dst->dst_md_cmpts),
                                   "ucp_dst_md_cmpts");
    if (dst->dst_md_cmpts == NULL) {
        ucs_error("failed to allocate ucp_ep dest component list");
        return UCS_ERR_NO_MEMORY;
    }

    memcpy(dst->dst_md_cmpts, src->dst_md_cmpts,
           ucs_popcount(src->reachable_md_map) * sizeof(*dst->dst_md_cmpts));
    return UCS_OK;
}

ucs_status_t ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config,
                                const ucp_ep_config_key_t *key)
{
    ucp_context_h context              = worker->context;
    ucp_lane_index_t tag_lanes[2]      = {UCP_NULL_LANE, UCP_NULL_LANE};
    ucp_lane_index_t rkey_ptr_lanes[2] = {UCP_NULL_LANE, UCP_NULL_LANE};
    ucp_lane_index_t get_zcopy_lane_count;
    ucp_lane_index_t put_zcopy_lane_count;
    ucp_ep_rma_config_t *rma_config;
    uct_iface_attr_t *iface_attr;
    uct_md_attr_t *md_attr;
    ucs_memory_type_t mem_type;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane, i;
    size_t max_rndv_thresh, max_am_rndv_thresh;
    size_t min_rndv_thresh, min_am_rndv_thresh;
    size_t rma_zcopy_thresh;
    double rndv_max_bw, scale, bw;
    ucs_status_t status;
    size_t it;

    memset(config, 0, sizeof(*config));

    status = ucp_ep_config_key_copy(&config->key, key);
    if (status != UCS_OK) {
        return status;
    }

    /* Default settings */
    for (it = 0; it < UCP_MAX_IOV; ++it) {
        config->am.zcopy_thresh[it]              = SIZE_MAX;
        config->am.sync_zcopy_thresh[it]         = SIZE_MAX;
        config->tag.eager.zcopy_thresh[it]       = SIZE_MAX;
        config->tag.eager.sync_zcopy_thresh[it]  = SIZE_MAX;
    }

    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_HOST == 0);
    for (mem_type = UCS_MEMORY_TYPE_HOST; mem_type < UCS_MEMORY_TYPE_LAST; mem_type++) {
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
    config->tag.rndv.min_get_zcopy      = 0;
    config->tag.rndv.max_get_zcopy      = SIZE_MAX;
    config->tag.rndv.min_put_zcopy      = 0;
    config->tag.rndv.max_put_zcopy      = SIZE_MAX;
    config->tag.rndv.am_thresh          = SIZE_MAX;
    config->tag.rndv_send_nbr.am_thresh = SIZE_MAX;
    config->tag.rndv_send_nbr.rma_thresh = SIZE_MAX;
    config->tag.rndv.rkey_size          = ucp_rkey_packed_size(context,
                                                               config->key.rma_bw_md_map);
    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        config->tag.rndv.get_zcopy_lanes[lane] = UCP_NULL_LANE;
        config->tag.rndv.put_zcopy_lanes[lane] = UCP_NULL_LANE;
    }

    config->tag.rndv.rkey_ptr_dst_mds   = 0;
    config->stream.proto                = &ucp_stream_am_proto;
    config->am_u.proto                  = &ucp_am_proto;
    config->am_u.reply_proto            = &ucp_am_reply_proto;
    max_rndv_thresh                     = SIZE_MAX;
    max_am_rndv_thresh                  = SIZE_MAX;
    min_am_rndv_thresh                  = 0;

    config->tag.offload.max_eager_short.memtype_on   = -1;
    config->tag.offload.max_eager_short.memtype_off  = -1;
    config->tag.max_eager_short.memtype_on           = -1;
    config->tag.max_eager_short.memtype_off          = -1;

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

    /* configuration for rndv */
    get_zcopy_lane_count = 0;
    put_zcopy_lane_count = 0;
    rndv_max_bw          = 0;
    for (i = 0; (i < config->key.num_lanes) &&
                (config->key.rma_bw_lanes[i] != UCP_NULL_LANE); ++i) {
        lane      = config->key.rma_bw_lanes[i];
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
        if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
            /* only GET Zcopy RNDV scheme supports multi-rail */
            bw          = ucp_tl_iface_bandwidth(context,
                                                 &iface_attr->bandwidth);
            rndv_max_bw = ucs_max(rndv_max_bw, bw);
        }
    }

    for (i = 0; (i < config->key.num_lanes) &&
                (config->key.rma_bw_lanes[i] != UCP_NULL_LANE); ++i) {
        lane      = config->key.rma_bw_lanes[i];
        rsc_index = config->key.lanes[lane].rsc_index;

        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

            /* GET Zcopy */
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
                scale = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth) /
                        rndv_max_bw;
                if (scale < (1. / context->config.ext.multi_lane_max_ratio)) {
                    continue;
                }

                config->tag.rndv.min_get_zcopy = ucs_max(config->tag.rndv.min_get_zcopy,
                                                         iface_attr->cap.get.min_zcopy);

                config->tag.rndv.max_get_zcopy = ucs_min(config->tag.rndv.max_get_zcopy,
                                                         iface_attr->cap.get.max_zcopy);
                ucs_assert(get_zcopy_lane_count < UCP_MAX_LANES);
                config->tag.rndv.get_zcopy_lanes[get_zcopy_lane_count++] = lane;
                config->tag.rndv.scale[lane]                             = scale;
            }

            /* PUT Zcopy */
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY) {
                config->tag.rndv.min_put_zcopy = ucs_max(config->tag.rndv.min_put_zcopy,
                                                         iface_attr->cap.put.min_zcopy);

                config->tag.rndv.max_put_zcopy = ucs_min(config->tag.rndv.max_put_zcopy,
                                                         iface_attr->cap.put.max_zcopy);
                ucs_assert(put_zcopy_lane_count < UCP_MAX_LANES);
                config->tag.rndv.put_zcopy_lanes[put_zcopy_lane_count++] = lane;
            }
        }
    }

    if (get_zcopy_lane_count == 0) {
        /* if there are no RNDV RMA BW lanes that support GET Zcopy, reset
         * min/max values to show that the scheme is unsupported */
        config->tag.rndv.min_get_zcopy   = SIZE_MAX;
        config->tag.rndv.max_get_zcopy   = 0;
        config->tag.rndv.get_zcopy_split = 0;
    } else {
        config->tag.rndv.get_zcopy_split = config->tag.rndv.min_get_zcopy <=
                                           (config->tag.rndv.max_get_zcopy / 2);
    }

    if (put_zcopy_lane_count == 0) {
        /* if there are no RNDV RMA BW lanes that support PUT Zcopy, reset
         * min/max values to show that the scheme is unsupported */
        config->tag.rndv.min_put_zcopy   = SIZE_MAX;
        config->tag.rndv.max_put_zcopy   = 0;
        config->tag.rndv.put_zcopy_split = 0;
    } else {
        config->tag.rndv.put_zcopy_split = config->tag.rndv.min_put_zcopy <=
                                           (config->tag.rndv.max_put_zcopy / 2);
    }

    /* Rkey ptr */
    if (key->rkey_ptr_lane != UCP_NULL_LANE) {
        lane      = key->rkey_ptr_lane;
        rsc_index = config->key.lanes[lane].rsc_index;
        md_attr   = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
        ucs_assert_always(md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR);

        config->tag.rndv.rkey_ptr_dst_mds =
                UCS_BIT(config->key.lanes[lane].dst_md_index);
    }

    /* Configuration for tag offload */
    if (config->key.tag_lane != UCP_NULL_LANE) {
        lane      = config->key.tag_lane;
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
            ucp_ep_config_init_attrs(worker, rsc_index, &config->tag.eager,
                                     iface_attr->cap.tag.eager.max_short,
                                     iface_attr->cap.tag.eager.max_bcopy,
                                     iface_attr->cap.tag.eager.max_zcopy,
                                     iface_attr->cap.tag.eager.max_iov,
                                     UCT_IFACE_FLAG_TAG_EAGER_SHORT,
                                     UCT_IFACE_FLAG_TAG_EAGER_BCOPY,
                                     UCT_IFACE_FLAG_TAG_EAGER_ZCOPY, 0,
                                     iface_attr->cap.tag.eager.max_bcopy);

            config->tag.offload.max_rndv_iov   = iface_attr->cap.tag.rndv.max_iov;
            config->tag.offload.max_rndv_zcopy = iface_attr->cap.tag.rndv.max_zcopy;
            config->tag.sync_proto             = &ucp_tag_offload_sync_proto;
            config->tag.proto                  = &ucp_tag_offload_proto;
            config->tag.lane                   = lane;
            max_rndv_thresh                    = iface_attr->cap.tag.eager.max_zcopy;
            max_am_rndv_thresh                 = iface_attr->cap.tag.eager.max_bcopy;
            min_rndv_thresh                    = ucp_ep_tag_offload_min_rndv_thresh(config);
            min_am_rndv_thresh                 = min_rndv_thresh;

            ucs_assert_always(iface_attr->cap.tag.rndv.max_hdr >=
                              sizeof(ucp_tag_offload_unexp_rndv_hdr_t));

            if (config->key.am_lane != UCP_NULL_LANE) {
                /* Must have active messages for using rendezvous */
                tag_lanes[0] = lane;
                ucp_ep_config_set_rndv_thresh(worker, config, tag_lanes,
                                              min_rndv_thresh, max_rndv_thresh);
            }

            /* Max Eager short has to be set after Zcopy and RNDV thresholds */
            ucp_ep_config_set_memtype_thresh(&config->tag.offload.max_eager_short,
                                             config->tag.eager.max_short,
                                             context->num_mem_type_detect_mds);
        }
    }

    /* Configuration for active messages */
    if (config->key.am_lane != UCP_NULL_LANE) {
        lane        = config->key.am_lane;
        rsc_index   = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
            md_attr    = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
            ucp_ep_config_init_attrs(worker, rsc_index, &config->am,
                                     iface_attr->cap.am.max_short,
                                     iface_attr->cap.am.max_bcopy,
                                     iface_attr->cap.am.max_zcopy,
                                     iface_attr->cap.am.max_iov,
                                     UCT_IFACE_FLAG_AM_SHORT,
                                     UCT_IFACE_FLAG_AM_BCOPY,
                                     UCT_IFACE_FLAG_AM_ZCOPY,
                                     sizeof(ucp_eager_hdr_t), SIZE_MAX);

            /* All keys must fit in RNDV packet.
             * TODO remove some MDs if they don't
             */
            ucs_assert_always(config->tag.rndv.rkey_size <= config->am.max_bcopy);

            if (!ucp_ep_is_tag_offload_enabled(config)) {
                /* Tag offload is disabled, AM will be used for all
                 * tag-matching protocols */
                /* TODO: set threshold level based on all available lanes */

                config->tag.eager  = config->am;
                config->tag.lane   = lane;
                min_rndv_thresh    = iface_attr->cap.get.min_zcopy;
                min_am_rndv_thresh = iface_attr->cap.am.min_zcopy;

                if (config->key.rkey_ptr_lane != UCP_NULL_LANE) {
                    rkey_ptr_lanes[0] = config->key.rkey_ptr_lane;
                    ucp_ep_config_set_rndv_thresh(worker, config,
                                                  rkey_ptr_lanes,
                                                  min_rndv_thresh,
                                                  max_rndv_thresh);
                } else {
                    ucp_ep_config_set_rndv_thresh(worker, config,
                                                  config->key.rma_bw_lanes,
                                                  min_rndv_thresh,
                                                  max_rndv_thresh);
                }

                /* Max Eager short has to be set after Zcopy and RNDV thresholds */
                ucp_ep_config_set_memtype_thresh(&config->tag.max_eager_short,
                                                 config->tag.eager.max_short,
                                                 context->num_mem_type_detect_mds);
            }

            /* Calculate rndv threshold for AM Rendezvous, which may be used by
             * any tag-matching protocol (AM and offload). */
            ucp_ep_config_set_am_rndv_thresh(worker, iface_attr, md_attr, config,
                                             min_am_rndv_thresh,
                                             max_am_rndv_thresh);
        } else {
            /* Stub endpoint */
            config->am.max_bcopy        = UCP_MIN_BCOPY;
            config->tag.eager.max_bcopy = UCP_MIN_BCOPY;
       }
    }

    memset(&config->rma, 0, sizeof(config->rma));

    rma_zcopy_thresh = ucp_ep_config_calc_rma_zcopy_thresh(worker, config,
                                                           config->key.rma_lanes);

    /* Configuration for remote memory access */
    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        rma_config                   = &config->rma[lane];
        rma_config->put_zcopy_thresh = SIZE_MAX;
        rma_config->get_zcopy_thresh = SIZE_MAX;
        rma_config->max_put_short    = -1;
        rma_config->max_get_short    = -1;
        rma_config->max_put_bcopy    = SIZE_MAX;
        rma_config->max_get_bcopy    = SIZE_MAX;

        if (ucp_ep_config_get_multi_lane_prio(config->key.rma_lanes, lane) == -1) {
            continue;
        }

        rsc_index  = config->key.lanes[lane].rsc_index;

        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
            /* PUT */
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
                rma_config->max_put_short = iface_attr->cap.put.max_short;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY) {
                rma_config->max_put_zcopy = iface_attr->cap.put.max_zcopy;
                if (context->config.ext.zcopy_thresh == UCS_MEMUNITS_AUTO) {
                    /* TODO: Use calculated value for PUT Zcopy threshold */
                    rma_config->put_zcopy_thresh = 16384;
                } else {
                    rma_config->put_zcopy_thresh = context->config.ext.zcopy_thresh;

                    ucp_ep_config_adjust_max_short(&rma_config->max_put_short,
                                                   rma_config->put_zcopy_thresh);
                }
                rma_config->put_zcopy_thresh = ucs_max(rma_config->put_zcopy_thresh,
                                                       iface_attr->cap.put.min_zcopy);
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
                rma_config->max_put_bcopy = ucs_min(iface_attr->cap.put.max_bcopy,
                                                    rma_config->put_zcopy_thresh);
            }

            /* GET */
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_SHORT) {
                rma_config->max_get_short = iface_attr->cap.get.max_short;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
                rma_config->max_get_zcopy = iface_attr->cap.get.max_zcopy;
                if (context->config.ext.zcopy_thresh == UCS_MEMUNITS_AUTO) {
                    rma_config->get_zcopy_thresh = rma_zcopy_thresh;
                } else {
                    rma_config->get_zcopy_thresh = context->config.ext.zcopy_thresh;

                    ucp_ep_config_adjust_max_short(&rma_config->max_get_short,
                                                   rma_config->get_zcopy_thresh);
                }
                rma_config->get_zcopy_thresh = ucs_max(rma_config->get_zcopy_thresh,
                                                       iface_attr->cap.get.min_zcopy);
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_BCOPY) {
                rma_config->max_get_bcopy = ucs_min(iface_attr->cap.get.max_bcopy,
                                                    rma_config->get_zcopy_thresh);
            }
        }
    }

    return UCS_OK;
}

void ucp_ep_config_cleanup(ucp_worker_h worker, ucp_ep_config_t *config)
{
    ucs_free(config->key.dst_md_cmpts);
}

static int ucp_ep_is_short_lower_thresh(ssize_t max_short,
                                        size_t thresh)
{
    return ((max_short < 0) ||
            (((size_t)max_short + 1) < thresh));
}

static void ucp_ep_config_print_tag_proto(FILE *stream, const char *name,
                                          ssize_t max_eager_short,
                                          size_t zcopy_thresh,
                                          size_t rndv_rma_thresh,
                                          size_t rndv_am_thresh)
{
    size_t max_bcopy, min_rndv, max_short;

    min_rndv  = ucs_min(rndv_rma_thresh, rndv_am_thresh);
    max_bcopy = ucs_min(zcopy_thresh, min_rndv);

    fprintf(stream, "# %23s: 0", name);

    /* print eager short */
    if (max_eager_short > 0) {
        max_short = max_eager_short;
        ucs_assert(max_short <= SSIZE_MAX);
        fprintf(stream, "..<egr/short>..%zu" , max_short + 1);
    } else if (!max_eager_short) {
        fprintf(stream, "..<egr/short>..%zu" , max_eager_short);
    }

    /* print eager bcopy */
    if (ucp_ep_is_short_lower_thresh(max_eager_short, max_bcopy) && max_bcopy) {
        fprintf(stream, "..<egr/bcopy>..");
        if (max_bcopy < SIZE_MAX) {
            fprintf(stream, "%zu", max_bcopy);
        }
    }

    /* print eager zcopy */
    if (ucp_ep_is_short_lower_thresh(max_eager_short, min_rndv) &&
        (zcopy_thresh < min_rndv)) {
        fprintf(stream, "..<egr/zcopy>..");
        if (min_rndv < SIZE_MAX) {
            fprintf(stream, "%zu", min_rndv);
        }
    }

    /* print rendezvous */
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
        if (zcopy_thresh > 0) {
            fprintf(stream, "..%zu", zcopy_thresh);
        }
        fprintf(stream, "..<zcopy>");
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
                                 const unsigned *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 char *buf, size_t max)
{
    uct_tl_resource_desc_t *rsc;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t proxy_lane;
    ucp_md_index_t dst_md_index;
    ucp_rsc_index_t cmpt_index;
    unsigned path_index;
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
        path_index = key->lanes[lane].path_index;
        snprintf(p, endp - p, "lane[%d]: %2d:" UCT_TL_RESOURCE_DESC_FMT ".%u md[%d]%s %-*c-> ",
                 lane, rsc_index, UCT_TL_RESOURCE_DESC_ARG(rsc), path_index,
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

    dst_md_index = key->lanes[lane].dst_md_index;
    cmpt_index   = ucp_ep_config_get_dst_md_cmpt(key, dst_md_index);
    snprintf(p, endp - p, "md[%d]/%-8s", dst_md_index,
             context->tl_cmpts[cmpt_index].attr.name);
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

    if (key->rkey_ptr_lane == lane) {
        snprintf(p, endp - p, " rkey_ptr");
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
        }
    }
}

static void ucp_ep_config_print(FILE *stream, ucp_worker_h worker,
                                const ucp_ep_config_t *config,
                                const unsigned *addr_indices,
                                ucp_rsc_index_t aux_rsc_index)
{
    ucp_context_h context = worker->context;
    char lane_info[128]   = {0};
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;

    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        ucp_ep_config_lane_info_str(context, &config->key, addr_indices, lane,
                                    aux_rsc_index, lane_info, sizeof(lane_info));
        fprintf(stream, "#                 %s\n", lane_info);
    }
    fprintf(stream, "#\n");

    if (context->config.features & UCP_FEATURE_TAG) {
        ucp_ep_config_print_tag_proto(stream, "tag_send",
                                      config->tag.eager.max_short,
                                      config->tag.eager.zcopy_thresh[0],
                                      config->tag.rndv.rma_thresh,
                                      config->tag.rndv.am_thresh);
        ucp_ep_config_print_tag_proto(stream, "tag_send_nbr",
                                      config->tag.eager.max_short,
                                      /* disable zcopy */
                                      ucs_min(config->tag.rndv_send_nbr.rma_thresh,
                                              config->tag.rndv_send_nbr.am_thresh),
                                      config->tag.rndv_send_nbr.rma_thresh,
                                      config->tag.rndv_send_nbr.am_thresh);
        ucp_ep_config_print_tag_proto(stream, "tag_send_sync",
                                      config->tag.eager.max_short,
                                      config->tag.eager.sync_zcopy_thresh[0],
                                      config->tag.rndv.rma_thresh,
                                      config->tag.rndv.am_thresh);
    }

     if (context->config.features & UCP_FEATURE_RMA) {
         for (lane = 0; lane < config->key.num_lanes; ++lane) {
             if (ucp_ep_config_get_multi_lane_prio(config->key.rma_lanes, lane) == -1) {
                 continue;
             }
             ucp_ep_config_print_rma_proto(stream, "put", lane,
                                           config->rma[lane].max_put_short + 1,
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

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

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

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
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

    if (zcopy_thresh < 0.0) {
        return SIZE_MAX;
    }

    return zcopy_thresh;
}

ucp_wireup_ep_t * ucp_ep_get_cm_wireup_ep(ucp_ep_h ep)
{
    const ucp_lane_index_t lane = ucp_ep_get_cm_lane(ep);

    if (lane == UCP_NULL_LANE) {
        return NULL;
    }

    return ucp_wireup_ep_test(ep->uct_eps[lane]) ?
           ucs_derived_of(ep->uct_eps[lane], ucp_wireup_ep_t) : NULL;
}

uct_ep_h ucp_ep_get_cm_uct_ep(ucp_ep_h ep)
{
    ucp_lane_index_t lane;
    ucp_wireup_ep_t *wireup_ep;

    lane = ucp_ep_get_cm_lane(ep);
    if (lane == UCP_NULL_LANE) {
        return NULL;
    }

    wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    return (wireup_ep == NULL) ? ep->uct_eps[lane] : wireup_ep->super.uct_ep;
}

int ucp_ep_is_cm_local_connected(ucp_ep_h ep)
{
    return (ucp_ep_get_cm_lane(ep) != UCP_NULL_LANE) &&
           (ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
}

uint64_t ucp_ep_get_tl_bitmap(ucp_ep_h ep)
{
    uint64_t tl_bitmap = 0;
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_idx;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (lane == ucp_ep_get_cm_lane(ep)) {
            continue;
        }

        rsc_idx = ucp_ep_get_rsc_index(ep, lane);
        if (rsc_idx == UCP_NULL_RESOURCE) {
            continue;
        }

        tl_bitmap |= UCS_BIT(rsc_idx);
    }

    return tl_bitmap;
}

void ucp_ep_invoke_err_cb(ucp_ep_h ep, ucs_status_t status)
{
    /* Do not invoke error handler if it's not enabled */
    if ((ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_NONE) ||
        /* error callback is not set */
        (ucp_ep_ext_gen(ep)->err_cb == NULL) ||
        /* the EP has been closed by user */
        (ep->flags & UCP_EP_FLAG_CLOSED)) {
        return;
    }

    ucs_assert(ep->flags & UCP_EP_FLAG_USED);
    ucs_debug("ep %p: calling user error callback %p with arg %p and status %s",
              ep, ucp_ep_ext_gen(ep)->err_cb, ucp_ep_ext_gen(ep)->user_data,
              ucs_status_string(status));
    ucp_ep_ext_gen(ep)->err_cb(ucp_ep_ext_gen(ep)->user_data, ep, status);
}

int ucp_ep_config_test_rndv_support(const ucp_ep_config_t *config)
{
    return (config->key.err_mode == UCP_ERR_HANDLING_MODE_NONE) ||
           (config->key.cm_lane  != UCP_NULL_LANE);
}
