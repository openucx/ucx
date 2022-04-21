/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
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
#include "ucp_rkey.h"
#include "ucp_ep.inl"
#include "ucp_request.inl"

#include <ucp/wireup/wireup_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucp/wireup/wireup_cm.h>
#include <ucp/tag/eager.h>
#include <ucp/tag/offload.h>
#include <ucp/proto/proto_common.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/rndv/rndv.h>
#include <ucp/stream/stream.h>
#include <ucp/core/ucp_listener.h>
#include <ucp/rma/rma.inl>
#include <ucp/rma/rma.h>

#include <ucs/datastruct/queue.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug_int.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <string.h>

__KHASH_IMPL(ucp_ep_peer_mem_hash, kh_inline, uint64_t,
             ucp_ep_peer_mem_data_t, 1,
             kh_int64_hash_func, kh_int64_hash_equal);

typedef struct {
    double reg_growth;
    double reg_overhead;
    double overhead;
    double latency;
    size_t bw;
} ucp_ep_thresh_params_t;


/**
 * Argument for the setting UCP endpoint as failed
 */
typedef struct ucp_ep_set_failed_arg {
    ucp_ep_h         ucp_ep; /* UCP endpoint which is failed */
    ucp_lane_index_t lane; /* UCP endpoint lane which is failed  */
    ucs_status_t     status; /* Failure status */
} ucp_ep_set_failed_arg_t;


/**
 * Argument for discarding UCP endpoint's lanes
 */
typedef struct ucp_ep_discard_lanes_arg {
    unsigned     counter; /* How many discarding operations on UCT lanes are
                           * in-progress if purging of the UCP endpoint is
                           * required */
    ucs_status_t status; /* Completion status of operations after discarding is
                          * done */
    ucp_ep_h     ucp_ep; /* UCP endpoint which should be discarded */
} ucp_ep_discard_lanes_arg_t;


extern const ucp_request_send_proto_t ucp_stream_am_proto;
extern const ucp_request_send_proto_t ucp_am_proto;
extern const ucp_request_send_proto_t ucp_am_reply_proto;

#ifdef ENABLE_STATS
static ucs_stats_class_t ucp_ep_stats_class = {
    .name           = "ucp_ep",
    .num_counters   = UCP_EP_STAT_LAST,
    .class_id       = UCS_STATS_CLASS_ID_INVALID,
    .counter_names  = {
        [UCP_EP_STAT_TAG_TX_EAGER]      = "tx_eager",
        [UCP_EP_STAT_TAG_TX_EAGER_SYNC] = "tx_eager_sync",
        [UCP_EP_STAT_TAG_TX_RNDV]       = "tx_rndv"
    }
};
#endif

static uct_iface_t ucp_failed_tl_iface = {
    .ops = {
        .ep_put_short        = (uct_ep_put_short_func_t)ucs_empty_function_return_ep_timeout,
        .ep_put_bcopy        = (uct_ep_put_bcopy_func_t)ucs_empty_function_return_bc_ep_timeout,
        .ep_put_zcopy        = (uct_ep_put_zcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_get_short        = (uct_ep_get_short_func_t)ucs_empty_function_return_ep_timeout,
        .ep_get_bcopy        = (uct_ep_get_bcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_get_zcopy        = (uct_ep_get_zcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_am_short         = (uct_ep_am_short_func_t)ucs_empty_function_return_ep_timeout,
        .ep_am_short_iov     = (uct_ep_am_short_iov_func_t)ucs_empty_function_return_ep_timeout,
        .ep_am_bcopy         = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_bc_ep_timeout,
        .ep_am_zcopy         = (uct_ep_am_zcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_atomic_cswap64   = (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_ep_timeout,
        .ep_atomic_cswap32   = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_ep_timeout,
        .ep_atomic64_post    = (uct_ep_atomic64_post_func_t)ucs_empty_function_return_ep_timeout,
        .ep_atomic32_post    = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_ep_timeout,
        .ep_atomic64_fetch   = (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_ep_timeout,
        .ep_atomic32_fetch   = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_ep_timeout,
        .ep_tag_eager_short  = (uct_ep_tag_eager_short_func_t)ucs_empty_function_return_ep_timeout,
        .ep_tag_eager_bcopy  = (uct_ep_tag_eager_bcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_tag_eager_zcopy  = (uct_ep_tag_eager_zcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_tag_rndv_zcopy   = (uct_ep_tag_rndv_zcopy_func_t)ucs_empty_function_return_ep_timeout,
        .ep_tag_rndv_cancel  = (uct_ep_tag_rndv_cancel_func_t)ucs_empty_function_return_ep_timeout,
        .ep_tag_rndv_request = (uct_ep_tag_rndv_request_func_t)ucs_empty_function_return_ep_timeout,
        .ep_pending_add      = (uct_ep_pending_add_func_t)ucs_empty_function_return_busy,
        .ep_pending_purge    = (uct_ep_pending_purge_func_t)ucs_empty_function_return_success,
        .ep_flush            = (uct_ep_flush_func_t)ucs_empty_function_return_ep_timeout,
        .ep_fence            = (uct_ep_fence_func_t)ucs_empty_function_return_ep_timeout,
        .ep_check            = (uct_ep_check_func_t)ucs_empty_function_return_success,
        .ep_connect_to_ep    = (uct_ep_connect_to_ep_func_t)ucs_empty_function_return_ep_timeout,
        .ep_destroy          = (uct_ep_destroy_func_t)ucs_empty_function,
        .ep_get_address      = (uct_ep_get_address_func_t)ucs_empty_function_return_ep_timeout
    }
};

static uct_ep_t ucp_failed_tl_ep = {
    .iface = &ucp_failed_tl_iface
};


int ucp_is_uct_ep_failed(uct_ep_h uct_ep)
{
    return uct_ep == &ucp_failed_tl_ep;
}

void ucp_ep_config_key_reset(ucp_ep_config_key_t *key)
{
    ucp_lane_index_t i;

    memset(key, 0, sizeof(*key));
    key->num_lanes        = 0;
    for (i = 0; i < UCP_MAX_LANES; ++i) {
        key->lanes[i].rsc_index    = UCP_NULL_RESOURCE;
        key->lanes[i].dst_md_index = UCP_NULL_RESOURCE;
        key->lanes[i].dst_sys_dev  = UCS_SYS_DEVICE_ID_UNKNOWN;
        key->lanes[i].path_index   = 0;
        key->lanes[i].lane_types   = 0;
        key->lanes[i].seg_size     = 0;
    }
    key->am_lane          = UCP_NULL_LANE;
    key->wireup_msg_lane  = UCP_NULL_LANE;
    key->cm_lane          = UCP_NULL_LANE;
    key->keepalive_lane   = UCP_NULL_LANE;
    key->rkey_ptr_lane    = UCP_NULL_LANE;
    key->tag_lane         = UCP_NULL_LANE;
    key->rma_bw_md_map    = 0;
    key->reachable_md_map = 0;
    key->dst_md_cmpts     = NULL;
    key->err_mode         = UCP_ERR_HANDLING_MODE_NONE;
    memset(key->am_bw_lanes,  UCP_NULL_LANE, sizeof(key->am_bw_lanes));
    memset(key->rma_lanes,    UCP_NULL_LANE, sizeof(key->rma_lanes));
    memset(key->rma_bw_lanes, UCP_NULL_LANE, sizeof(key->rma_bw_lanes));
    memset(key->amo_lanes,    UCP_NULL_LANE, sizeof(key->amo_lanes));
}

static void ucp_ep_deallocate(ucp_ep_h ep)
{
    UCS_STATS_NODE_FREE(ep->stats);
    ucs_free(ucp_ep_ext_control(ep));
    ucs_strided_alloc_put(&ep->worker->ep_alloc, ep);
}

static ucp_ep_h ucp_ep_allocate(ucp_worker_h worker, const char *peer_name)
{
    ucp_ep_h ep;
    ucp_lane_index_t lane;
    ucs_status_t status;

    ep = ucs_strided_alloc_get(&worker->ep_alloc, "ucp_ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate ep");
        goto err;
    }

    ucp_ep_ext_gen(ep)->control_ext = ucs_calloc(1,
                                                 sizeof(ucp_ep_ext_control_t),
                                                 "ep_control_ext");
    if (ucp_ep_ext_gen(ep)->control_ext == NULL) {
        ucs_error("Failed to allocate ep control extension");
        goto err_free_ep;
    }

    ep->refcount                          = 0;
    ep->cfg_index                         = UCP_WORKER_CFG_INDEX_NULL;
    ep->worker                            = worker;
    ep->am_lane                           = UCP_NULL_LANE;
    ep->flags                             = 0;
    ep->conn_sn                           = UCP_EP_MATCH_CONN_SN_MAX;
#if UCS_ENABLE_ASSERT
    ep->refcounts.create                  =
    ep->refcounts.flush                   =
    ep->refcounts.discard                 = 0;
#endif
    ucp_ep_ext_gen(ep)->user_data         = NULL;
    ucp_ep_ext_control(ep)->cm_idx        = UCP_NULL_RESOURCE;
    ucp_ep_ext_control(ep)->local_ep_id   = UCS_PTR_MAP_KEY_INVALID;
    ucp_ep_ext_control(ep)->remote_ep_id  = UCS_PTR_MAP_KEY_INVALID;
    ucp_ep_ext_control(ep)->err_cb        = NULL;
    ucp_ep_ext_control(ep)->close_req     = NULL;
#if UCS_ENABLE_ASSERT
    ucp_ep_ext_control(ep)->ka_last_round = 0;
#endif
    ucp_ep_ext_control(ep)->peer_mem      = NULL;

    UCS_STATIC_ASSERT(sizeof(ucp_ep_ext_gen(ep)->ep_match) >=
                      sizeof(ucp_ep_ext_gen(ep)->flush_state));
    memset(&ucp_ep_ext_gen(ep)->ep_match, 0,
           sizeof(ucp_ep_ext_gen(ep)->ep_match));

    ucs_hlist_head_init(&ucp_ep_ext_gen(ep)->proto_reqs);

    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        ep->uct_eps[lane] = NULL;
    }
#if ENABLE_DEBUG_DATA
    ucs_snprintf_zero(ep->peer_name, UCP_WORKER_ADDRESS_NAME_MAX, "%s",
                      peer_name);
#endif
    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&ep->stats, &ucp_ep_stats_class,
                                  worker->stats, "-%p", ep);
    if (status != UCS_OK) {
        goto err_free_ep_control_ext;
    }

    return ep;

err_free_ep_control_ext:
    ucs_free(ucp_ep_ext_control(ep));
err_free_ep:
    ucs_strided_alloc_put(&worker->ep_alloc, ep);
err:
    return NULL;
}

static int ucp_ep_shall_use_indirect_id(ucp_context_h context,
                                        unsigned ep_init_flags)
{
    return !(ep_init_flags & UCP_EP_INIT_FLAG_INTERNAL) &&
           ((context->config.ext.proto_indirect_id == UCS_CONFIG_ON) ||
            ((context->config.ext.proto_indirect_id == UCS_CONFIG_AUTO) &&
             (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE)));
}

void ucp_ep_peer_mem_destroy(ucp_context_h context,
                             ucp_ep_peer_mem_data_t *ppln_data)
{
    ucp_md_map_t md_map;
    ucs_status_t UCS_V_UNUSED status;

    md_map = (ppln_data->md_index == UCP_NULL_RESOURCE) ?
             0 : UCS_BIT(ppln_data->md_index);
    status = ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL,
                               UCS_MEMORY_TYPE_UNKNOWN, NULL,
                               &ppln_data->uct_memh, &md_map);
    ucs_assertv(status == UCS_OK, "%s", ucs_status_string(status));

    ucp_rkey_destroy(ppln_data->rkey);
}

ucp_ep_peer_mem_data_t*
ucp_ep_peer_mem_get(ucp_context_h context, ucp_ep_h ep, uint64_t address,
                    size_t size, void *rkey_buf, ucp_md_index_t md_index)
{
    khash_t(ucp_ep_peer_mem_hash) *peer_mem = ucp_ep_ext_control(ep)->peer_mem;
    ucp_ep_peer_mem_data_t *data;
    khiter_t iter;
    int ret;

    if (ucs_unlikely(peer_mem == NULL)) {
        ucp_ep_ext_control(ep)->peer_mem =
        peer_mem                         = kh_init(ucp_ep_peer_mem_hash);
    }

    iter = kh_put(ucp_ep_peer_mem_hash, peer_mem, address, &ret);
    ucs_assert_always(ret != UCS_KH_PUT_FAILED);
    data = &kh_val(ucp_ep_ext_control(ep)->peer_mem, iter);

    if (ucs_likely(ret == UCS_KH_PUT_KEY_PRESENT)) {
        if (ucs_likely(size <= data->size)) {
            return data;
        }

        ucp_ep_peer_mem_destroy(context, data);
    }

    data->size     = size;
    data->uct_memh = NULL;
    ucp_ep_rkey_unpack_internal(ep, rkey_buf, 0, UCS_BIT(md_index),
                                &data->rkey);
    return data;
}

ucs_status_t ucp_ep_create_base(ucp_worker_h worker, unsigned ep_init_flags,
                                const char *peer_name, const char *message,
                                ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_h ep;

    ep = ucp_ep_allocate(worker, peer_name);
    if (ep == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ucp_stream_ep_init(ep);
    ucp_am_ep_init(ep);

    if (ucp_ep_shall_use_indirect_id(ep->worker->context, ep_init_flags)) {
        ucp_ep_update_flags(ep, UCP_EP_FLAG_INDIRECT_ID, 0);
    }

    status = UCS_PTR_MAP_PUT(ep, &worker->ep_map, ep,
                             ep->flags & UCP_EP_FLAG_INDIRECT_ID,
                             &ucp_ep_ext_control(ep)->local_ep_id);
    if ((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS)) {
        ucs_error("ep %p: failed to allocate ID: %s", ep,
                  ucs_status_string(status));
        goto err_ep_deallocate;
    }

    ucp_ep_flush_state_reset(ep);

    /* Create endpoint VFS node on demand to avoid memory bloat */
    ucs_vfs_obj_set_dirty(worker, ucp_worker_vfs_refresh);

    /* Insert new UCP endpoint to the UCP worker */
    if (ep_init_flags & UCP_EP_INIT_FLAG_INTERNAL) {
        ucp_ep_update_flags(ep, UCP_EP_FLAG_INTERNAL, 0);
        ucs_list_add_tail(&worker->internal_eps, &ucp_ep_ext_gen(ep)->ep_list);
    } else {
        ucs_list_add_tail(&worker->all_eps, &ucp_ep_ext_gen(ep)->ep_list);
        ucs_assert(ep->worker->num_all_eps < UINT_MAX);
        ++ep->worker->num_all_eps;
    }

    ucp_ep_refcount_add(ep, create);

    *ep_p = ep;
    ucs_debug("created ep %p to %s %s", ep, ucp_ep_peer_name(ep), message);
    return UCS_OK;

err_ep_deallocate:
    ucp_ep_deallocate(ep);
err:
    return status;
}

static int
ucp_ep_local_disconnect_progress_remove_filter(const ucs_callbackq_elem_t *elem,
                                               void *arg)
{
    ucp_ep_h ep = (ucp_ep_h)arg;
    ucp_request_t *req;

    if (elem->cb != ucp_ep_local_disconnect_progress) {
        return 0;
    }

    req = (ucp_request_t*)elem->arg;
    if (ep != req->send.ep) {
        return 0;
    }

    /* Expect that only EP flush request can be remained in the callback queue,
     * because reply UCP EP created for sending WIREUP_MSG/EP_REMOVED message is
     * not exposed to a user */
    ucs_assert(req->flags & UCP_REQUEST_FLAG_RELEASED);
    ucs_assert(req->send.uct.func == ucp_ep_flush_progress_pending);

    ucp_request_complete_send(req, req->status);
    return 1;
}

static unsigned ucp_ep_set_failed_progress(void *arg)
{
    ucp_ep_set_failed_arg_t *set_ep_failed_arg = arg;
    ucp_ep_h ucp_ep                            = set_ep_failed_arg->ucp_ep;
    ucp_worker_h worker                        = ucp_ep->worker;

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_ep_set_failed(ucp_ep, set_ep_failed_arg->lane,
                      set_ep_failed_arg->status);
    UCS_ASYNC_UNBLOCK(&worker->async);

    ucs_free(set_ep_failed_arg);
    return 1;
}

static int ucp_ep_set_failed_remove_filter(const ucs_callbackq_elem_t *elem,
                                           void *arg)
{
    ucp_ep_set_failed_arg_t *set_ep_failed_arg = elem->arg;

    if ((elem->cb == ucp_ep_set_failed_progress) &&
        (set_ep_failed_arg->ucp_ep == arg)) {
        ucs_free(set_ep_failed_arg);
        return 1;
    }

    return 0;
}

static int ucp_ep_remove_filter(const ucs_callbackq_elem_t *elem, void *arg)
{
    if (ucp_wireup_msg_ack_cb_pred(elem, arg) ||
        ucp_listener_accept_cb_remove_filter(elem, arg) ||
        ucp_ep_local_disconnect_progress_remove_filter(elem, arg) ||
        ucp_ep_set_failed_remove_filter(elem, arg)) {
        return 1;
    }

    return 0;
}

void ucp_ep_destroy_base(ucp_ep_h ep)
{
    ucp_ep_peer_mem_data_t data;
    ucp_ep_refcount_field_assert(ep, refcount, ==, 0);
    ucp_ep_refcount_assert(ep, create, ==, 0);
    ucp_ep_refcount_assert(ep, flush, ==, 0);
    ucp_ep_refcount_assert(ep, discard, ==, 0);
    ucs_assert(ucs_hlist_is_empty(&ucp_ep_ext_gen(ep)->proto_reqs));

    if (!(ep->flags & UCP_EP_FLAG_INTERNAL)) {
        ucs_assert(ep->worker->num_all_eps > 0);
        --ep->worker->num_all_eps;
    }

    ucp_worker_keepalive_remove_ep(ep);
    ucp_ep_release_id(ep);
    ucs_list_del(&ucp_ep_ext_gen(ep)->ep_list);

    ucs_vfs_obj_remove(ep);
    ucs_callbackq_remove_if(&ep->worker->uct->progress_q, ucp_ep_remove_filter,
                            ep);
    UCS_STATS_NODE_FREE(ep->stats);
    if (ucp_ep_ext_control(ep)->peer_mem != NULL) {
        kh_foreach_value(ucp_ep_ext_control(ep)->peer_mem, data, {
            ucp_ep_peer_mem_destroy(ep->worker->context, &data);
        });

        kh_destroy(ucp_ep_peer_mem_hash, ucp_ep_ext_control(ep)->peer_mem);
    }
    ucp_ep_deallocate(ep);
}

void ucp_ep_delete(ucp_ep_h ep)
{
    ucp_ep_refcount_assert(ep, create, ==, 1);
    ucp_ep_refcount_remove(ep, create);
}

void ucp_ep_flush_state_reset(ucp_ep_h ep)
{
    ucp_ep_flush_state_t *flush_state = &ucp_ep_ext_gen(ep)->flush_state;

    ucs_assert(!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX));
    ucs_assert(!(ep->flags & UCP_EP_FLAG_FLUSH_STATE_VALID) ||
               ((flush_state->send_sn == 0) &&
                (flush_state->cmpl_sn == 0) &&
                ucs_hlist_is_empty(&flush_state->reqs)));

    flush_state->send_sn = 0;
    flush_state->cmpl_sn = 0;
    ucs_hlist_head_init(&flush_state->reqs);
    ucp_ep_update_flags(ep, UCP_EP_FLAG_FLUSH_STATE_VALID, 0);
}

void ucp_ep_flush_state_invalidate(ucp_ep_h ep)
{
    ucs_assert(ucs_hlist_is_empty(&ucp_ep_flush_state(ep)->reqs));
    ucp_ep_update_flags(ep, 0, UCP_EP_FLAG_FLUSH_STATE_VALID);
}

/* Since release function resets EP ID to @ref UCS_PTR_MAP_KEY_INVALID and PTR
 * MAP considers @ref UCS_PTR_MAP_KEY_INVALID as direct key, release EP ID is
 * re-entrant function */
void ucp_ep_release_id(ucp_ep_h ep)
{
    ucs_status_t status;

    /* Don't use ucp_ep_local_id() function here to avoid assertion failure,
     * because local_ep_id can be set to @ref UCS_PTR_MAP_KEY_INVALID */
    status = UCS_PTR_MAP_DEL(ep, &ep->worker->ep_map,
                             ucp_ep_ext_control(ep)->local_ep_id);
    if ((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS)) {
        ucs_warn("ep %p local id 0x%" PRIxPTR ": ucs_ptr_map_del failed: %s",
                 ep, ucp_ep_local_id(ep), ucs_status_string(status));
    }

    ucp_ep_ext_control(ep)->local_ep_id = UCS_PTR_MAP_KEY_INVALID;
}

void ucp_ep_config_key_set_err_mode(ucp_ep_config_key_t *key,
                                    unsigned ep_init_flags)
{
    key->err_mode = (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE) ?
                    UCP_ERR_HANDLING_MODE_PEER : UCP_ERR_HANDLING_MODE_NONE;
}

ucs_status_t
ucp_ep_config_err_mode_check_mismatch(ucp_ep_h ep,
                                      ucp_err_handling_mode_t err_mode)
{
    if (ucp_ep_config(ep)->key.err_mode != err_mode) {
        ucs_error("ep %p: asymmetric endpoint configuration is not supported,"
                  " error handling level mismatch (expected: %d, got: %d)",
                  ep, ucp_ep_config(ep)->key.err_mode, err_mode);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}


/* Handles a case where the existing endpoint is incomplete */
static ucs_status_t
ucp_ep_adjust_params(ucp_ep_h ep, const ucp_ep_params_t *params)
{
    ucs_status_t status;

    if (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) {
        status = ucp_ep_config_err_mode_check_mismatch(ep, params->err_mode);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLER) {
        ucp_ep_ext_gen(ep)->user_data  = params->err_handler.arg;
        ucp_ep_ext_control(ep)->err_cb = params->err_handler.cb;
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_USER_DATA) {
        /* user_data overrides err_handler.arg */
        ucp_ep_ext_gen(ep)->user_data = params->user_data;
    }

    return UCS_OK;
}

ucs_status_t ucp_ep_evaluate_perf(ucp_ep_h ep,
                                  const ucp_ep_evaluate_perf_param_t *param,
                                  ucp_ep_evaluate_perf_attr_t *attr)
{
    const ucp_worker_h worker               = ep->worker;
    const ucp_context_h context             = worker->context;
    const ucp_ep_config_key_t *key          = &ucp_ep_config(ep)->key;
    double max_bandwidth                    = 0;
    ucp_rsc_index_t max_bandwidth_rsc_index = 0;
    ucp_rsc_index_t rsc_index;
    double bandwidth;
    ucp_lane_index_t lane;
    ucp_worker_iface_t *wiface;
    uct_iface_attr_t *iface_attr;
    ucs_linear_func_t estimated_time;

    if (!ucs_test_all_flags(attr->field_mask,
                            UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME &
                            UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE)) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (lane == key->cm_lane) {
            /* Skip CM lanes for bandwidth calculation */
            continue;
        }

        rsc_index = key->lanes[lane].rsc_index;
        wiface    = worker->ifaces[rsc_index];
        bandwidth = ucp_tl_iface_bandwidth(context,
                                            &wiface->attr.bandwidth);
        if (bandwidth > max_bandwidth) {
            max_bandwidth           = bandwidth;
            max_bandwidth_rsc_index = rsc_index;
        }
    }

    iface_attr           = ucp_worker_iface_get_attr(worker,
                                                     max_bandwidth_rsc_index);
    estimated_time.c     = ucp_tl_iface_latency(context, &iface_attr->latency);
    estimated_time.m     = param->message_size / max_bandwidth;
    attr->estimated_time = estimated_time.c + estimated_time.m;

    return UCS_OK;
}

ucs_status_t ucp_worker_mem_type_eps_create(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    unsigned pack_flags   = ucp_worker_default_address_pack_flags(worker);
    ucp_unpacked_address_t local_address;
    ucs_memory_type_t mem_type;
    ucs_status_t status;
    void *address_buffer;
    size_t address_length;
    ucp_tl_bitmap_t mem_access_tls;
    char ep_name[UCP_WORKER_ADDRESS_NAME_MAX];

    ucs_memory_type_for_each(mem_type) {
        ucp_context_get_mem_access_tls(context, mem_type, &mem_access_tls);
        if (UCP_MEM_IS_HOST(mem_type) ||
            UCS_BITMAP_IS_ZERO_INPLACE(&mem_access_tls)) {
            continue;
        }

        status = ucp_address_pack(worker, NULL, &mem_access_tls, pack_flags,
                                  context->config.ext.worker_addr_version, NULL,
                                  &address_length, &address_buffer);
        if (status != UCS_OK) {
            goto err_cleanup_eps;
        }

        status = ucp_address_unpack(worker, address_buffer, pack_flags,
                                    &local_address);
        if (status != UCS_OK) {
            goto err_free_address_buffer;
        }

        ucs_snprintf_zero(ep_name, UCP_WORKER_ADDRESS_NAME_MAX,
                          "mem_type_ep:%s", ucs_memory_type_names[mem_type]);

        /* create memtype UCP EPs after blocking async context, because they set
         * INTERNAL flag (setting EP flags is expected to be guarded) */
        UCS_ASYNC_BLOCK(&worker->async);
        status = ucp_ep_create_to_worker_addr(worker, &ucp_tl_bitmap_max,
                                              &local_address,
                                              UCP_EP_INIT_FLAG_MEM_TYPE |
                                              UCP_EP_INIT_FLAG_INTERNAL,
                                              ep_name,
                                              &worker->mem_type_ep[mem_type]);
        if (status != UCS_OK) {
            UCS_ASYNC_UNBLOCK(&worker->async);
            goto err_free_address_list;
        }

        UCS_ASYNC_UNBLOCK(&worker->async);

        ucs_free(local_address.address_list);
        ucs_free(address_buffer);
    }

    return UCS_OK;

err_free_address_list:
    ucs_free(local_address.address_list);
err_free_address_buffer:
    ucs_free(address_buffer);
err_cleanup_eps:
    ucp_worker_mem_type_eps_destroy(worker);
    return status;
}

void ucp_worker_mem_type_eps_destroy(ucp_worker_h worker)
{
    ucs_memory_type_t mem_type;
    ucp_ep_h ep;

    /* Destroy memtype UCP EPs after blocking async context, because cleanup
     * lanes set FAILED flag (setting EP flags is expected to be guarded) */
    UCS_ASYNC_BLOCK(&worker->async);

    ucs_memory_type_for_each(mem_type) {
        ep = worker->mem_type_ep[mem_type];
        if (ep == NULL) {
            continue;
        }

        ucs_debug("memtype ep %p: destroy", ep);
        ucs_assert(ep->flags & UCP_EP_FLAG_INTERNAL);

        ucp_ep_destroy_internal(ep);
        worker->mem_type_ep[mem_type] = NULL;
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_ep_init_create_wireup(ucp_ep_h ep, unsigned ep_init_flags,
                                       ucp_wireup_ep_t **wireup_ep)
{
    ucp_ep_config_key_t key;
    ucs_status_t status;

    ucs_assert(ep_init_flags & UCP_EP_INIT_CM_WIREUP_CLIENT);
    ucs_assert(ucp_worker_num_cm_cmpts(ep->worker) != 0);

    ucp_ep_config_key_reset(&key);
    ucp_ep_config_key_set_err_mode(&key, ep_init_flags);

    key.num_lanes = 1;
    /* all operations will use the first lane, which is a stub endpoint before
     * reconfiguration */
    key.am_lane = 0;
    if (ucp_ep_init_flags_has_cm(ep_init_flags)) {
        key.cm_lane = 0;
        /* Send keepalive on wireup_ep (which will send on aux_ep) */
        if (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE) {
            key.keepalive_lane = 0;
        }
    } else {
        key.wireup_msg_lane = 0;
    }

    status = ucp_worker_get_ep_config(ep->worker, &key, ep_init_flags,
                                      &ep->cfg_index);
    if (status != UCS_OK) {
        return status;
    }

    ep->am_lane = key.am_lane;
    if (!ucp_ep_has_cm_lane(ep)) {
        ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_REQ_QUEUED, 0);
    }

    status = ucp_wireup_ep_create(ep, &ep->uct_eps[0]);
    if (status != UCS_OK) {
        return status;
    }

    *wireup_ep = ucs_derived_of(ep->uct_eps[0], ucp_wireup_ep_t);
    return UCS_OK;
}

ucs_status_t
ucp_ep_create_to_worker_addr(ucp_worker_h worker,
                             const ucp_tl_bitmap_t *local_tl_bitmap,
                             const ucp_unpacked_address_t *remote_address,
                             unsigned ep_init_flags, const char *message,
                             ucp_ep_h *ep_p)
{
    unsigned addr_indices[UCP_MAX_LANES];
    ucp_tl_bitmap_t ep_tl_bitmap;
    ucs_status_t status;
    ucp_ep_h ep;

    /* allocate endpoint */
    status = ucp_ep_create_base(worker, ep_init_flags, remote_address->name,
                                message, &ep);
    if (status != UCS_OK) {
        goto err;
    }

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes(ep, ep_init_flags, local_tl_bitmap,
                                   remote_address, addr_indices);
    if (status != UCS_OK) {
        goto err_delete;
    }

    ucp_ep_get_tl_bitmap(ep, &ep_tl_bitmap);
    ucp_tl_bitmap_validate(&ep_tl_bitmap, local_tl_bitmap);

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
    unsigned ep_init_flags;

    if (!(params->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR)) {
        ucs_error("destination socket address is missing");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    UCP_CHECK_PARAM_NON_NULL(params->sockaddr.addr, status, goto err);

    /* allocate endpoint */
    ucs_sockaddr_str(params->sockaddr.addr, peer_name, sizeof(peer_name));
    ep_init_flags = ucp_ep_init_flags(worker, params) |
                    ucp_cm_ep_init_flags(params);

    status = ucp_ep_create_base(worker, ep_init_flags, peer_name,
                                "from api call", &ep);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_ep_init_create_wireup(ep, ep_init_flags, &wireup_ep);
    if (status != UCS_OK) {
        goto err_delete;
    }

    if (UCP_PARAM_VALUE(EP, params, flags, FLAGS, 0) &
        UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID) {
        wireup_ep->flags |= UCP_WIREUP_EP_FLAG_SEND_CLIENT_ID;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        goto err_cleanup_lanes;
    }

    status = ucp_ep_client_cm_connect_start(ep, params);
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
ucp_sa_data_v1_unpack(const ucp_wireup_sockaddr_data_base_t *sa_data,
                      unsigned *ep_init_flags_p,
                      const void** worker_addr_p)
{
    const ucp_wireup_sockaddr_data_v1_t *sa_data_v1 =
            ucs_derived_of(sa_data, ucp_wireup_sockaddr_data_v1_t);

    if (sa_data_v1->addr_mode != UCP_WIREUP_SA_DATA_CM_ADDR) {
        ucs_error("sa_data_v1 contains unsupported address mode %u",
                  sa_data_v1->addr_mode);
        return UCS_ERR_UNSUPPORTED;
    }

    *ep_init_flags_p = (sa_data->header == UCP_ERR_HANDLING_MODE_PEER) ?
                       UCP_EP_INIT_ERR_MODE_PEER_FAILURE : 0;
    *worker_addr_p   = sa_data_v1 + 1;
    return UCS_OK;
}

static ucs_status_t
ucp_sa_data_v2_unpack(const ucp_wireup_sockaddr_data_base_t *sa_data,
                      unsigned *ep_init_flags_p,
                      const void** worker_addr_p)
{
    *ep_init_flags_p = (sa_data->header & UCP_SA_DATA_FLAG_ERR_MODE_PEER) ?
                       UCP_EP_INIT_ERR_MODE_PEER_FAILURE : 0;
    *worker_addr_p   = sa_data + 1;
    return UCS_OK;
}

static ucs_status_t
ucp_conn_request_unpack_sa_data(const ucp_conn_request_h conn_request,
                                unsigned *ep_init_flags_p,
                                const void** worker_addr_p)
{
    const ucp_wireup_sockaddr_data_base_t *sa_data =
            UCS_PTR_TYPE_OFFSET(conn_request, *conn_request);
    uint8_t sa_data_version                        =
            sa_data->header >> UCP_SA_DATA_HEADER_VERSION_SHIFT;

    switch (sa_data_version) {
    case UCP_OBJECT_VERSION_V1:
        return ucp_sa_data_v1_unpack(sa_data, ep_init_flags_p, worker_addr_p);
    case UCP_OBJECT_VERSION_V2:
        return ucp_sa_data_v2_unpack(sa_data, ep_init_flags_p, worker_addr_p);
    default:
        ucs_error("conn_request %p: unsupported sa_data version: %u",
                  conn_request, sa_data_version);
        return UCS_ERR_UNSUPPORTED;
    }
}

/**
 * Create an endpoint on the server side connected to the client endpoint.
 */
ucs_status_t ucp_ep_create_server_accept(ucp_worker_h worker,
                                         const ucp_conn_request_h conn_request,
                                         ucp_ep_h *ep_p)
{
    ucp_unpacked_address_t remote_addr;
    unsigned ep_init_flags;
    uint64_t addr_flags;
    ucs_status_t status;
    const void *worker_addr;
    unsigned i;

    status = ucp_conn_request_unpack_sa_data(conn_request, &ep_init_flags,
                                             &worker_addr);
    if (status != UCS_OK) {
        return status;
    }

    addr_flags = ucp_worker_common_address_pack_flags(worker) |
                 UCP_ADDRESS_PACK_FLAGS_CM_DEFAULT;

    if (ucp_address_is_am_only(worker_addr)) {
        ep_init_flags |= UCP_EP_INIT_CREATE_AM_LANE_ONLY;
    }

    /* coverity[overrun-local] */
    status = ucp_address_unpack(worker, worker_addr, addr_flags, &remote_addr);
    if (status != UCS_OK) {
        ucp_listener_reject(conn_request->listener, conn_request);
        return status;
    }

    for (i = 0; i < remote_addr.address_count; ++i) {
        remote_addr.address_list[i].dev_addr  = conn_request->remote_dev_addr;
        remote_addr.address_list[i].dev_index = 0; /* CM addr contains only 1
                                                      device */
    }

    status = ucp_ep_cm_server_create_connected(worker, ep_init_flags,
                                               &remote_addr, conn_request,
                                               ep_p);
    ucs_free(remote_addr.address_list);
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
        return status;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status == UCS_OK) {
        *ep_p = ep;
    } else {
        ucp_ep_destroy_internal(ep);
    }

    return status;
}

static ucs_status_t
ucp_ep_create_api_to_worker_addr(ucp_worker_h worker,
                                 const ucp_ep_params_t *params, ucp_ep_h *ep_p)
{
    ucp_context_h context  = worker->context;
    unsigned ep_init_flags = ucp_ep_init_flags(worker, params);
    ucp_unpacked_address_t remote_address;
    ucp_ep_match_conn_sn_t conn_sn;
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
                                ucp_worker_default_address_pack_flags(worker),
                                &remote_address);
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
    conn_sn = ucp_ep_match_get_sn(worker, remote_address.uuid);
    ep      = ucp_ep_match_retrieve(worker, remote_address.uuid,
                                    conn_sn ^
                                    (remote_address.uuid == worker->uuid),
                                    UCS_CONN_MATCH_QUEUE_UNEXP);
    if (ep != NULL) {
        status = ucp_ep_adjust_params(ep, params);
        if (status != UCS_OK) {
            goto err_destroy_ep;
        }

        ucp_stream_ep_activate(ep);
        goto out_resolve_remote_id;
    }

    status = ucp_ep_create_to_worker_addr(worker, &ucp_tl_bitmap_max,
                                          &remote_address, ep_init_flags,
                                          "from api call", &ep);
    if (status != UCS_OK) {
        goto out_free_address;
    }

    status = ucp_ep_adjust_params(ep, params);
    if (status != UCS_OK) {
        goto err_destroy_ep;
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
        ucp_ep_update_remote_id(ep, ucp_ep_local_id(ep));
    } else if (!ucp_ep_match_insert(worker, ep, remote_address.uuid, conn_sn,
                                    UCS_CONN_MATCH_QUEUE_EXP)) {
        if (context->config.features & UCP_FEATURE_STREAM) {
            status = UCS_ERR_EXCEEDS_LIMIT;
            ucs_error("worker %p: failed to create the endpoint without"
                      "connection matching and Stream API support", worker);
            goto err_destroy_ep;
        }
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

out_resolve_remote_id:
    if ((context->config.ext.resolve_remote_ep_id == UCS_CONFIG_ON) ||
        ((context->config.ext.resolve_remote_ep_id == UCS_CONFIG_AUTO) &&
         (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE) &&
         ucp_worker_keepalive_is_enabled(worker))) {
        /* If resolving remote ID forced by configuration or PEER_FAILURE
         * and keepalive were requested, resolve remote endpoint ID prior to
         * communicating with a peer to make sure that remote peer's endpoint
         * won't be changed during runtime */
        status = ucp_ep_resolve_remote_id(ep, ep->am_lane);
        if (ucs_unlikely(status != UCS_OK)) {
            goto out_free_address;
        }
    }
out_free_address:
    ucs_free(remote_address.address_list);
out:
    if (status == UCS_OK) {
        *ep_p = ep;
    }
    return status;

err_destroy_ep:
    ucp_ep_destroy_internal(ep);
    goto out_free_address;
}

static void ucp_ep_params_check_err_handling(ucp_ep_h ep,
                                             const ucp_ep_params_t *params)
{
    ucp_err_handling_mode_t err_mode =
            UCP_PARAM_VALUE(EP, params, err_mode, ERR_HANDLING_MODE,
                            UCP_ERR_HANDLING_MODE_NONE);

    if (err_mode == UCP_ERR_HANDLING_MODE_NONE) {
        return;
    }

    if (ucp_worker_keepalive_is_enabled(ep->worker) &&
        ucp_ep_use_indirect_id(ep)) {
        return;
    }

    ucs_diag("ep %p: creating endpoint with error handling but without "
             "keepalive and indirect id", ep);
}

ucs_status_t ucp_ep_create(ucp_worker_h worker, const ucp_ep_params_t *params,
                           ucp_ep_h *ep_p)
{
    ucp_ep_h ep    = NULL;
    unsigned flags = UCP_PARAM_VALUE(EP, params, flags, FLAGS, 0);
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

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
#if ENABLE_DEBUG_DATA
        if ((params->field_mask & UCP_EP_PARAM_FIELD_NAME) &&
            (params->name != NULL)) {
            ucs_snprintf_zero(ep->name, UCP_ENTITY_NAME_MAX, "%s",
                              params->name);
        } else {
            ucs_snprintf_zero(ep->name, UCP_ENTITY_NAME_MAX, "%p", ep);
        }
#endif

        ucp_ep_params_check_err_handling(ep, params);
        ucp_ep_update_flags(ep, UCP_EP_FLAG_USED, 0);
        *ep_p = ep;
    } else {
        ++worker->counters.ep_creation_failures;
    }
    ++worker->counters.ep_creations;

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

    /* TODO: check for context->config.ext.proto_enable when all protocols are
     *       implemented, such as flush, AM/RNDV, etc */
    if (req->flags & UCP_REQUEST_FLAG_PROTO_SEND) {
        ucp_proto_request_abort(req, status);
    } else {
        ucp_request_send_state_ff(req, status);
    }
}

void ucp_destroyed_ep_pending_purge(uct_pending_req_t *self, void *arg)
{
    ucs_bug("pending request %p (%s) on ep %p should have been flushed",
            self, ucs_debug_get_symbol_name(self->func), arg);
}

void
ucp_ep_purge_lanes(ucp_ep_h ep, uct_pending_purge_callback_t purge_cb,
                   void *purge_arg)
{
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = ep->uct_eps[lane];
        if ((lane == ucp_ep_get_cm_lane(ep)) || (uct_ep == NULL)) {
            continue;
        }

        ucs_debug("ep %p: purge uct_ep[%d]=%p", ep, lane, uct_ep);
        uct_ep_pending_purge(uct_ep, purge_cb, purge_arg);
    }
}

void ucp_ep_destroy_internal(ucp_ep_h ep)
{
    ucs_debug("ep %p: destroy", ep);
    ucp_ep_cleanup_lanes(ep);
    ucp_ep_delete(ep);
}

static void ucp_ep_check_lanes(ucp_ep_h ep)
{
#if UCS_ENABLE_ASSERT
    uint8_t num_inprog       = ep->refcounts.discard + ep->refcounts.flush +
                               ep->refcounts.create;
    uint8_t num_failed_tl_ep = 0;
    ucp_lane_index_t lane;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        num_failed_tl_ep += ucp_is_uct_ep_failed(ep->uct_eps[lane]);
    }

    ucs_assert((num_failed_tl_ep == 0) ||
               (ucp_ep_num_lanes(ep) == num_failed_tl_ep));
    ucp_ep_refcount_field_assert(ep, refcount, ==, num_inprog);
#endif
}

static void ucp_ep_set_lanes_failed(ucp_ep_h ep, uct_ep_h *uct_eps)
{
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    ucp_ep_check_lanes(ep);
    ucp_ep_release_id(ep);
    ucp_ep_update_flags(ep, UCP_EP_FLAG_FAILED, UCP_EP_FLAG_LOCAL_CONNECTED);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep        = ep->uct_eps[lane];
        uct_eps[lane] = uct_ep;

        /* Set UCT EP to failed UCT EP to make sure if UCP EP won't be destroyed
         * due to some UCT EP discarding procedures are in-progress and UCP EP
         * may get some operation completions which could try to dereference its
         * lanes */
        ep->uct_eps[lane] = &ucp_failed_tl_ep;
    }
}

void ucp_ep_unprogress_uct_ep(ucp_ep_h ep, uct_ep_h uct_ep,
                              ucp_rsc_index_t rsc_index)
{
    ucp_worker_iface_t *wiface;

    if ((rsc_index == UCP_NULL_RESOURCE) ||
        !ep->worker->context->config.ext.adaptive_progress ||
        /* Do not unprogress an already failed lane */
        ucp_is_uct_ep_failed(uct_ep) ||
        ucp_wireup_ep_test(uct_ep)) {
        return;
    }

    wiface = ucp_worker_iface(ep->worker, rsc_index);
    ucs_debug("ep %p: unprogress iface %p " UCT_TL_RESOURCE_DESC_FMT,
              ep, wiface->iface,
              UCT_TL_RESOURCE_DESC_ARG(
              &(ep->worker->context->tl_rscs[rsc_index].tl_rsc)));
    ucp_worker_iface_unprogress_ep(wiface);
}

static void ucp_ep_discard_lanes_callback(void *request, ucs_status_t status,
                                          void *user_data)
{
    ucp_ep_discard_lanes_arg_t *arg = (ucp_ep_discard_lanes_arg_t*)user_data;

    ucs_assert(arg != NULL);
    ucs_assert(arg->counter > 0);

    if (--arg->counter > 0) {
        return;
    }

    ucp_ep_reqs_purge(arg->ucp_ep, arg->status);
    ucs_free(arg);
}

static void ucp_ep_discard_lanes(ucp_ep_h ep, ucs_status_t discard_status)
{
    unsigned ep_flush_flags         = (ucp_ep_config(ep)->key.err_mode ==
                                       UCP_ERR_HANDLING_MODE_NONE) ?
                                      UCT_FLUSH_FLAG_LOCAL :
                                      UCT_FLUSH_FLAG_CANCEL;
    uct_ep_h uct_eps[UCP_MAX_LANES] = { NULL };
    ucp_ep_discard_lanes_arg_t *discard_arg;
    ucs_status_t status;
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    if (ep->flags & UCP_EP_FLAG_FAILED) {
        /* Avoid calling ucp_ep_discard_lanes_callback() that will purge UCP
         * endpoint's requests, if we already started discard and purge process
         * this endpoint. Doing so could complete send requests before UCT lanes
         * using them are flushed and destroyed. */
        return;
    }

    discard_arg = ucs_malloc(sizeof(*discard_arg), "discard_lanes_arg");
    if (discard_arg == NULL) {
        ucs_error("ep %p: failed to allocate memory for discarding lanes"
                  " argument", ep);
        ucp_ep_cleanup_lanes(ep); /* Just close all UCT endpoints */
        ucp_ep_reqs_purge(ep, discard_status);
        return;
    }

    discard_arg->ucp_ep  = ep;
    discard_arg->status  = discard_status;
    discard_arg->counter = 1;

    ucs_debug("ep %p: discarding lanes", ep);
    ucp_ep_set_lanes_failed(ep, uct_eps);
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = uct_eps[lane];
        if (uct_ep == NULL) {
            continue;
        }

        ucs_debug("ep %p: discard uct_ep[%d]=%p", ep, lane, uct_ep);
        status = ucp_worker_discard_uct_ep(ep, uct_ep,
                                           ucp_ep_get_rsc_index(ep, lane),
                                           ep_flush_flags,
                                           ucp_ep_err_pending_purge,
                                           UCS_STATUS_PTR(discard_status),
                                           ucp_ep_discard_lanes_callback,
                                           discard_arg);
        if (status == UCS_INPROGRESS) {
            ++discard_arg->counter;
        }
    }

    ucp_ep_discard_lanes_callback(NULL, UCS_OK, discard_arg);
}

ucs_status_t
ucp_ep_set_failed(ucp_ep_h ucp_ep, ucp_lane_index_t lane, ucs_status_t status)
{
    UCS_STRING_BUFFER_ONSTACK(lane_info_strb, 64);
    ucp_ep_ext_control_t *ep_ext_control = ucp_ep_ext_control(ucp_ep);
    ucp_err_handling_mode_t err_mode;
    ucs_log_level_t log_level;
    ucp_request_t *close_req;

    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(ucp_ep->worker);
    ucs_assert(UCS_STATUS_IS_ERR(status));
    ucs_assert(!ucs_async_is_from_async(&ucp_ep->worker->async));

    ucs_debug("ep %p: set_ep_failed status %s on lane[%d]=%p", ucp_ep,
              ucs_status_string(status), lane,
              (lane != UCP_NULL_LANE) ? ucp_ep->uct_eps[lane] : NULL);

    /* In case if this is a local failure we need to notify remote side */
    if (ucp_ep_is_cm_local_connected(ucp_ep)) {
        ucp_ep_cm_disconnect_cm_lane(ucp_ep);
    }

    /* set endpoint to failed to prevent wireup_ep switch */
    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        return UCS_OK;
    }

    ++ucp_ep->worker->counters.ep_failures;

    /* The EP can be closed from last completion callback */
    ucp_ep_discard_lanes(ucp_ep, status);
    ucp_stream_ep_cleanup(ucp_ep, status);

    if (ucp_ep->flags & UCP_EP_FLAG_USED) {
        if (ucp_ep->flags & UCP_EP_FLAG_CLOSED) {
            if (ep_ext_control->close_req != NULL) {
                /* Promote close operation to CANCEL in case of transport error,
                 * since the disconnect event may never arrive. */
                close_req                        = ep_ext_control->close_req;
                close_req->send.flush.uct_flags |= UCT_FLUSH_FLAG_CANCEL;
                ucp_ep_local_disconnect_progress(close_req);
            }
            return UCS_OK;
        } else if (ep_ext_control->err_cb == NULL) {
            /* Print error if user requested error handling support but did not
               install a valid error handling callback */
            err_mode  = ucp_ep_config(ucp_ep)->key.err_mode;
            log_level = (err_mode == UCP_ERR_HANDLING_MODE_NONE) ?
                                UCS_LOG_LEVEL_DIAG :
                                UCS_LOG_LEVEL_ERROR;

            ucp_ep_get_lane_info_str(ucp_ep, lane, &lane_info_strb);
            ucs_log(log_level,
                    "ep %p: error '%s' on %s will not be handled"
                    " since no error callback is installed",
                    ucp_ep, ucs_status_string(status),
                    ucs_string_buffer_cstr(&lane_info_strb));
            return UCS_ERR_UNSUPPORTED;
        } else {
            ucp_ep_invoke_err_cb(ucp_ep, status);
            return UCS_OK;
        }
    } else if (ucp_ep->flags & (UCP_EP_FLAG_INTERNAL | UCP_EP_FLAG_CLOSED)) {
        /* No additional actions are required, this is already closed EP or
         * an internal one for sending WIREUP/EP_REMOVED message to a peer.
         * So, close operation was already scheduled, this EP will be deleted
         * after all lanes will be discarded successfully */
        ucs_debug("ep %p: detected peer failure on internal endpoint", ucp_ep);
        return UCS_OK;
    } else {
        ucs_debug("ep %p: destroy endpoint which is not exposed to a user due"
                  " to peer failure", ucp_ep);
        ucp_ep_disconnected(ucp_ep, 1);
        return UCS_OK;
    }
}

void ucp_ep_set_failed_schedule(ucp_ep_h ucp_ep, ucp_lane_index_t lane,
                                ucs_status_t status)
{
    ucp_worker_h worker        = ucp_ep->worker;
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;
    ucp_ep_set_failed_arg_t *set_ep_failed_arg;

    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(worker);

    set_ep_failed_arg = ucs_malloc(sizeof(*set_ep_failed_arg),
                                   "set_ep_failed_arg");
    if (set_ep_failed_arg == NULL) {
        ucs_error("failed to allocate set_ep_failed argument");
        return;
    }

    set_ep_failed_arg->ucp_ep = ucp_ep;
    set_ep_failed_arg->lane   = lane;
    set_ep_failed_arg->status = status;

    uct_worker_progress_register_safe(worker->uct, ucp_ep_set_failed_progress,
                                      set_ep_failed_arg,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);

    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(worker);
}

void ucp_ep_cleanup_lanes(ucp_ep_h ep)
{
    uct_ep_h uct_eps[UCP_MAX_LANES] = { NULL };
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    ucs_debug("ep %p: cleanup lanes", ep);

    ucp_ep_set_lanes_failed(ep, uct_eps);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = uct_eps[lane];
        if (uct_ep == NULL) {
            continue;
        }

        ucs_debug("ep %p: pending & destroy uct_ep[%d]=%p", ep, lane, uct_ep);
        uct_ep_pending_purge(uct_ep, ucp_destroyed_ep_pending_purge, ep);
        ucp_ep_unprogress_uct_ep(ep, uct_ep, ucp_ep_get_rsc_index(ep, lane));

        /* coverity wrongly resolves ucp_failed_tl_ep's no-op EP destroy
         * function to 'ucp_proxy_ep_destroy' */
        /* coverity[incorrect_free] */
        uct_ep_destroy(uct_ep);
    }
}

void ucp_ep_disconnected(ucp_ep_h ep, int force)
{
    ucp_worker_h worker = ep->worker;

    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(worker);

    ucp_ep_cm_slow_cbq_cleanup(ep);

    ucp_stream_ep_cleanup(ep, UCS_ERR_CANCELED);
    ucp_am_ep_cleanup(ep);

    ucp_ep_update_flags(ep, 0, UCP_EP_FLAG_USED);

    if ((ep->flags & (UCP_EP_FLAG_CONNECT_REQ_QUEUED |
                      UCP_EP_FLAG_REMOTE_CONNECTED)) && !force) {
        /* Endpoints which have remote connection are destroyed only when the
         * worker is destroyed, to enable remote endpoints keep sending
         * TODO negotiate disconnect.
         */
        ucs_trace("not destroying ep %p because of connection from remote", ep);
        return;
    }

    ucp_ep_match_remove_ep(worker, ep);
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
    ucs_assertv(ucp_ep_ext_control(ep)->close_req == NULL,
                "ep=%p: close_req=%p", ep, ucp_ep_ext_control(ep)->close_req);
    ucs_trace("ep %p: setting close request %p, %s", ep, request, debug_msg);
    ucp_ep_ext_control(ep)->close_req = request;
}

void ucp_ep_register_disconnect_progress(ucp_request_t *req)
{
    ucp_ep_h ep                = req->send.ep;
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;

    /* If a flush is completed from a pending/completion callback, we need to
     * schedule slow-path callback to release the endpoint later, since a UCT
     * endpoint cannot be released from pending/completion callback context.
     */
    ucs_trace("adding slow-path callback to destroy ep %p", ep);
    uct_worker_progress_register_safe(ep->worker->uct,
                                      ucp_ep_local_disconnect_progress, req,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &prog_id);
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
    ucp_ep_register_disconnect_progress(req);
}

ucs_status_ptr_t ucp_ep_close_nb(ucp_ep_h ep, unsigned mode)
{
    const ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
        .flags        = (mode == UCP_EP_CLOSE_MODE_FORCE) ?
                        UCP_EP_CLOSE_FLAG_FORCE : 0
    };

    return ucp_ep_close_nbx(ep, &param);
}

ucs_status_ptr_t ucp_ep_close_nbx(ucp_ep_h ep, const ucp_request_param_t *param)
{
    ucp_worker_h  worker = ep->worker;
    void          *request = NULL;
    ucp_request_t *close_req;

    if ((ucp_request_param_flags(param) & UCP_EP_CLOSE_FLAG_FORCE) &&
        (ucp_ep_config(ep)->key.err_mode != UCP_ERR_HANDLING_MODE_PEER)) {
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_debug("ep %p flags 0x%x cfg_index %d: close_nbx(flags=0x%x)", ep,
              ep->flags, ep->cfg_index, ucp_request_param_flags(param));

    if (ep->flags & UCP_EP_FLAG_CLOSED) {
        ucs_error("ep %p has already been closed", ep);
        request = UCS_STATUS_PTR(UCS_ERR_NOT_CONNECTED);
        goto out;
    }

    ucp_ep_update_flags(ep, UCP_EP_FLAG_CLOSED, 0);

    if (ucp_request_param_flags(param) & UCP_EP_CLOSE_FLAG_FORCE) {
        ucp_ep_discard_lanes(ep, UCS_ERR_CANCELED);
        ucp_ep_disconnected(ep, 1);
    } else {
        request = ucp_ep_flush_internal(ep, 0, param, NULL,
                                        ucp_ep_close_flushed_callback, "close");
        if (!UCS_PTR_IS_PTR(request)) {
            if (ucp_ep_is_cm_local_connected(ep)) {
                /* lanes already flushed, start disconnect on CM lane */
                ucp_ep_cm_disconnect_cm_lane(ep);
                close_req = ucp_ep_cm_close_request_get(ep, param);
                if (close_req != NULL) {
                    request = close_req + 1;
                    ucp_ep_set_close_request(ep, close_req, "close");
                } else {
                    request = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                }
            } else {
                ucp_ep_disconnected(ep, 0);
            }
        }
    }

    ++worker->counters.ep_closures;

out:
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
        ucs_debug("ep_close request %p completed with status %s", request,
                  ucs_status_string(status));
        ucp_request_release(request);
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return;
}

ucp_lane_index_t ucp_ep_lookup_lane(ucp_ep_h ucp_ep, uct_ep_h uct_ep)
{
    ucp_lane_index_t lane;

    for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
        if ((uct_ep == ucp_ep->uct_eps[lane]) ||
            ucp_wireup_ep_is_owner(ucp_ep->uct_eps[lane], uct_ep)) {
            return lane;
        }
    }

    return UCP_NULL_LANE;
}

static int ucp_ep_lane_is_dst_index_match(ucp_rsc_index_t dst_index1,
                                          ucp_rsc_index_t dst_index2)
{
    return (dst_index1 == UCP_NULL_RESOURCE) ||
           (dst_index2 == UCP_NULL_RESOURCE) || (dst_index1 == dst_index2);
}

int ucp_ep_config_lane_is_peer_match(const ucp_ep_config_key_t *key1,
                                     ucp_lane_index_t lane1,
                                     const ucp_ep_config_key_t *key2,
                                     ucp_lane_index_t lane2)
{
    const ucp_ep_config_key_lane_t *config_lane1 = &key1->lanes[lane1];
    const ucp_ep_config_key_lane_t *config_lane2 = &key2->lanes[lane2];

    return (config_lane1->rsc_index == config_lane2->rsc_index) &&
           (config_lane1->path_index == config_lane2->path_index) &&
           ucp_ep_lane_is_dst_index_match(config_lane1->dst_md_index,
                                          config_lane2->dst_md_index);
}

static ucp_lane_index_t
ucp_ep_config_find_match_lane(const ucp_ep_config_key_t *key1,
                              const ucp_rsc_index_t *dst_rsc_indices1,
                              ucp_lane_index_t lane1,
                              const ucp_ep_config_key_t *key2,
                              const ucp_rsc_index_t *dst_rsc_indices2)
{
    ucp_lane_index_t lane_idx;

    for (lane_idx = 0; lane_idx < key2->num_lanes; ++lane_idx) {
        if (ucp_ep_config_lane_is_peer_match(key1, lane1, key2, lane_idx) &&
            ucp_ep_lane_is_dst_index_match(dst_rsc_indices1[lane1],
                                           dst_rsc_indices2[lane_idx])) {
            return lane_idx;
        }
    }

    return UCP_NULL_LANE;
}

/* Go through the first configuration and check if the lanes selected
 * for this configuration could be used for the second configuration */
void ucp_ep_config_lanes_intersect(const ucp_ep_config_key_t *key1,
                                   const ucp_rsc_index_t *dst_rsc_indices1,
                                   const ucp_ep_config_key_t *key2,
                                   const ucp_rsc_index_t *dst_rsc_indices2,
                                   ucp_lane_index_t *lane_map)
{
    ucp_lane_index_t lane1_idx;

    for (lane1_idx = 0; lane1_idx < key1->num_lanes; ++lane1_idx) {
        lane_map[lane1_idx] = ucp_ep_config_find_match_lane(key1,
                                                            dst_rsc_indices1,
                                                            lane1_idx, key2,
                                                            dst_rsc_indices2);
    }
}

static int ucp_ep_config_lane_is_equal(const ucp_ep_config_key_t *key1,
                                       const ucp_ep_config_key_t *key2,
                                       ucp_lane_index_t lane)
{
    const ucp_ep_config_key_lane_t *config_lane1 = &key1->lanes[lane];
    const ucp_ep_config_key_lane_t *config_lane2 = &key2->lanes[lane];

    return (config_lane1->rsc_index == config_lane2->rsc_index) &&
           (config_lane1->path_index == config_lane2->path_index) &&
           (config_lane1->dst_md_index == config_lane2->dst_md_index) &&
           (config_lane1->dst_sys_dev == config_lane2->dst_sys_dev) &&
           (config_lane1->lane_types == config_lane2->lane_types) &&
           (config_lane1->seg_size == config_lane2->seg_size);
}

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2)
{
    ucp_lane_index_t lane;
    int i;

    if ((key1->num_lanes != key2->num_lanes) ||
        memcmp(key1->rma_lanes, key2->rma_lanes, sizeof(key1->rma_lanes)) ||
        memcmp(key1->am_bw_lanes, key2->am_bw_lanes,
               sizeof(key1->am_bw_lanes)) ||
        memcmp(key1->rma_bw_lanes, key2->rma_bw_lanes,
               sizeof(key1->rma_bw_lanes)) ||
        memcmp(key1->amo_lanes, key2->amo_lanes, sizeof(key1->amo_lanes)) ||
        (key1->rma_bw_md_map != key2->rma_bw_md_map) ||
        (key1->reachable_md_map != key2->reachable_md_map) ||
        (key1->am_lane != key2->am_lane) ||
        (key1->tag_lane != key2->tag_lane) ||
        (key1->wireup_msg_lane != key2->wireup_msg_lane) ||
        (key1->cm_lane != key2->cm_lane) ||
        (key1->keepalive_lane != key2->keepalive_lane) ||
        (key1->rkey_ptr_lane != key2->rkey_ptr_lane) ||
        (key1->err_mode != key2->err_mode)) {
        return 0;
    }

    for (lane = 0; lane < key1->num_lanes; ++lane) {
        if (!ucp_ep_config_lane_is_equal(key1, key2, lane)) {
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

static ucs_status_t ucp_ep_config_calc_params(ucp_worker_h worker,
                                              const ucp_ep_config_t *config,
                                              const ucp_lane_index_t *lanes,
                                              ucp_ep_thresh_params_t *params,
                                              int eager)
{
    ucp_context_h context = worker->context;
    ucp_md_map_t md_map   = 0;
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    uct_iface_attr_t *iface_attr;
    ucp_worker_iface_t *wiface;
    uct_perf_attr_t perf_attr;
    ucs_status_t status;
    double bw;
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
                params->reg_growth   += md_attr->reg_cost.m;
                params->reg_overhead += md_attr->reg_cost.c;
                params->overhead     += iface_attr->overhead;
                params->latency      += ucp_tl_iface_latency(context,
                                                             &iface_attr->latency);
            }
        }

        bw = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);
        if (eager && (iface_attr->cap.am.max_bcopy > 0)) {
            /* Eager protocol has overhead for each fragment */
            perf_attr.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD;
            perf_attr.operation  = UCT_EP_OP_AM_ZCOPY;

            wiface = ucp_worker_iface(worker, rsc_index);
            status = uct_iface_estimate_perf(wiface->iface, &perf_attr);
            if (status != UCS_OK) {
                return status;
            }

            params->bw += 1.0 / ((1.0 / bw) + ((perf_attr.send_pre_overhead +
                                                perf_attr.send_post_overhead) /
                                               iface_attr->cap.am.max_bcopy));
        } else {
            params->bw += bw;
        }
    }

    return UCS_OK;
}

static ucs_status_t
ucp_ep_config_calc_rndv_thresh(ucp_worker_t *worker,
                               const ucp_ep_config_t *config,
                               const ucp_lane_index_t *eager_lanes,
                               const ucp_lane_index_t *rndv_lanes,
                               int recv_reg_cost, size_t *thresh_p)
{
    ucp_context_h context = worker->context;
    double diff_percent   = 1.0 - context->config.ext.rndv_perf_diff / 100.0;
    ucp_ep_thresh_params_t eager_zcopy;
    ucp_ep_thresh_params_t rndv;
    double numerator, denominator;
    ucp_rsc_index_t eager_rsc_index;
    uct_iface_attr_t *eager_iface_attr;
    ucs_status_t status;
    double rts_latency;

    /* All formulas and descriptions are listed at
     * https://github.com/openucx/ucx/wiki/Rendezvous-Protocol-threshold-for-multilane-mode */

    status = ucp_ep_config_calc_params(worker, config, eager_lanes,
                                       &eager_zcopy, 1);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_ep_config_calc_params(worker, config, rndv_lanes, &rndv, 0);
    if (status != UCS_OK) {
        return status;
    }

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

    denominator = eager_zcopy.reg_growth +
                  1.0 / ucs_min(eager_zcopy.bw, context->config.ext.bcopy_bw) -
                  diff_percent *
                  (rndv.reg_growth * (1 + recv_reg_cost) + 1.0 / rndv.bw);

    if ((numerator <= 0) || (denominator <= 0)) {
        goto fallback;
    }

    *thresh_p = ucs_max(numerator / denominator,
                        eager_iface_attr->cap.am.max_bcopy);
    return UCS_OK;

fallback:
    *thresh_p = context->config.ext.rndv_thresh_fallback;
    return UCS_OK;

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

static ucs_status_t
ucp_ep_config_calc_rma_zcopy_thresh(ucp_worker_t *worker,
                                    const ucp_ep_config_t *config,
                                    const ucp_lane_index_t *rma_lanes,
                                    ssize_t *thresh_p)
{
    ucp_context_h context = worker->context;
    double bcopy_bw       = context->config.ext.bcopy_bw;
    ucp_ep_thresh_params_t rma;
    uct_md_attr_t *md_attr;
    double numerator, denominator;
    double reg_overhead, reg_growth;
    ucs_status_t status;

    status = ucp_ep_config_calc_params(worker, config, rma_lanes, &rma, 0);
    if (status != UCS_OK) {
        return status;
    }

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
    denominator = (1 / bcopy_bw) - reg_growth;

    if (denominator <= 0) {
        goto fallback;
    }

    *thresh_p = numerator / denominator;
    return UCS_OK;

fallback:
    *thresh_p = SIZE_MAX;
    return UCS_OK;
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
    return sizeof(ucp_rndv_rts_hdr_t) + config->rndv.rkey_size;
}

static void ucp_ep_config_init_short_thresh(ucp_memtype_thresh_t *thresh)
{
    thresh->memtype_on  = -1;
    thresh->memtype_off = -1;
}

static ucs_status_t ucp_ep_config_set_am_rndv_thresh(
        ucp_worker_h worker, uct_iface_attr_t *iface_attr,
        uct_md_attr_t *md_attr, ucp_ep_config_t *config, size_t min_rndv_thresh,
        size_t max_rndv_thresh, ucp_rndv_thresh_t *thresh)
{
    ucp_context_h context = worker->context;
    size_t rndv_thresh, rndv_local_thresh, min_thresh;
    ucs_status_t status;

    ucs_assert(config->key.am_lane != UCP_NULL_LANE);
    ucs_assert(config->key.lanes[config->key.am_lane].rsc_index != UCP_NULL_RESOURCE);

    if (context->config.ext.rndv_thresh == UCS_MEMUNITS_AUTO) {
        /* auto - Make UCX calculate the AM rndv threshold on its own.*/
        status = ucp_ep_config_calc_rndv_thresh(worker, config,
                                                config->key.am_bw_lanes,
                                                config->key.am_bw_lanes,
                                                0, &rndv_thresh);
        if (status != UCS_OK) {
            return status;
        }

        rndv_local_thresh = context->config.ext.rndv_send_nbr_thresh;
        ucs_trace("active message rendezvous threshold is %zu", rndv_thresh);
    } else {
        rndv_thresh       = context->config.ext.rndv_thresh;
        rndv_local_thresh = context->config.ext.rndv_thresh;
    }

    min_thresh     = ucs_max(iface_attr->cap.am.min_zcopy, min_rndv_thresh);
    thresh->remote = ucp_ep_thresh(rndv_thresh, min_thresh, max_rndv_thresh);
    thresh->local  = ucp_ep_thresh(rndv_local_thresh, min_thresh, max_rndv_thresh);

    ucs_trace("Active Message rndv threshold is %zu (fast local compl: %zu)",
              thresh->remote, thresh->local);

    return UCS_OK;
}

static void
ucp_ep_config_set_rndv_thresh(ucp_worker_t *worker, ucp_ep_config_t *config,
                              ucp_lane_index_t *lanes, size_t min_rndv_thresh,
                              size_t max_rndv_thresh, ucp_rndv_thresh_t *thresh)
{
    ucp_context_t *context = worker->context;
    ucp_lane_index_t lane  = lanes[0];
    ucp_rsc_index_t rsc_index;
    size_t rndv_thresh, rndv_local_thresh, min_thresh;
    uct_iface_attr_t *iface_attr;
    ucs_status_t status;

    if (lane == UCP_NULL_LANE) {
        goto out_not_supported;
    }

    rsc_index = config->key.lanes[lane].rsc_index;
    if (rsc_index == UCP_NULL_RESOURCE) {
        goto out_not_supported;
    }

    iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

    if (context->config.ext.rndv_thresh == UCS_MEMUNITS_AUTO) {
        /* auto - Make UCX calculate the RMA (get_zcopy) rndv threshold on its own.*/
        status = ucp_ep_config_calc_rndv_thresh(worker, config,
                                                config->key.am_bw_lanes,
                                                lanes, 1, &rndv_thresh);
        if (status != UCS_OK) {
            goto out_not_supported;
        }

        rndv_local_thresh = context->config.ext.rndv_send_nbr_thresh;
    } else {
        rndv_thresh       = context->config.ext.rndv_thresh;
        rndv_local_thresh = context->config.ext.rndv_thresh;
    }

    min_thresh = ucs_max(iface_attr->cap.get.min_zcopy, min_rndv_thresh);

    /* TODO: need to check minimal PUT Zcopy */
    thresh->remote = ucp_ep_thresh(rndv_thresh, min_thresh, max_rndv_thresh);
    thresh->local  = ucp_ep_thresh(rndv_local_thresh, min_thresh, max_rndv_thresh);

    ucs_trace("rndv threshold is %zu (fast local compl: %zu)",
              thresh->remote, thresh->local);

    return;

out_not_supported:
    ucs_trace("rendezvous (get_zcopy) protocol is not supported");
}

static void ucp_ep_config_set_memtype_thresh(ucp_memtype_thresh_t *max_eager_short,
                                             ssize_t max_short, int num_mem_type_mds)
{
    if (!num_mem_type_mds) {
        max_eager_short->memtype_off = max_short;
    }

    max_eager_short->memtype_on = max_short;
}

/* Coverity assumes that mem_type_index could have value >= UCS_MEMORY_TYPE_LAST,
 * a caller of this function should suppress this false-positive warning */
static void
ucp_ep_config_rndv_zcopy_max_bw_update(ucp_context_t *context,
                                       const uct_md_attr_t *md_attr,
                                       const uct_iface_attr_t *iface_attr,
                                       uint64_t cap_flag,
                                       double max_bw[UCS_MEMORY_TYPE_LAST])
{
    uint8_t mem_type_index;
    double bw;

    if (!(iface_attr->cap.flags & cap_flag)) {
        return;
    }

    bw = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth);
    ucs_for_each_bit(mem_type_index, md_attr->cap.reg_mem_types) {
        ucs_assert(mem_type_index < UCS_MEMORY_TYPE_LAST);
        max_bw[mem_type_index] = ucs_max(max_bw[mem_type_index], bw);
    }
}

static void
ucp_ep_config_rndv_zcopy_set(ucp_context_t *context, uint64_t cap_flag,
                             ucp_lane_index_t lane,
                             const uct_md_attr_t *md_attr,
                             const uct_iface_attr_t *iface_attr,
                             double max_bw[UCS_MEMORY_TYPE_LAST],
                             ucp_ep_rndv_zcopy_config_t *rndv_zcopy,
                             ucp_lane_index_t *lanes_count_p)
{
    const double min_scale = 1. / context->config.ext.multi_lane_max_ratio;
    uint8_t mem_type_index;
    double scale;
    size_t min, max;

    if (!(iface_attr->cap.flags & cap_flag)) {
        return;
    }

    if (cap_flag == UCT_IFACE_FLAG_GET_ZCOPY) {
        min = iface_attr->cap.get.min_zcopy;
        max = iface_attr->cap.get.max_zcopy;
    } else {
        ucs_assert(cap_flag == UCT_IFACE_FLAG_PUT_ZCOPY);
        min = iface_attr->cap.put.min_zcopy;
        max = iface_attr->cap.put.max_zcopy;
    }

    ucs_for_each_bit(mem_type_index, md_attr->cap.reg_mem_types) {
        ucs_assert(mem_type_index < UCS_MEMORY_TYPE_LAST);
        scale = ucp_tl_iface_bandwidth(context, &iface_attr->bandwidth) /
                max_bw[mem_type_index];
        if ((scale - min_scale) < -ucp_calc_epsilon(scale, min_scale)) {
            continue;
        }

        rndv_zcopy->min = ucs_max(rndv_zcopy->min, min);
        rndv_zcopy->max = ucs_min(rndv_zcopy->max, max);
        ucs_assert(*lanes_count_p < UCP_MAX_LANES);
        rndv_zcopy->lanes[(*lanes_count_p)++] = lane;
        rndv_zcopy->scale[lane]               = scale;
        break;
    }
}

void ucp_ep_config_rndv_zcopy_commit(ucp_lane_index_t lanes_count,
                                     ucp_ep_rndv_zcopy_config_t *rndv_zcopy)
{
    if (lanes_count == 0) {
        /* if there are no RNDV RMA BW lanes that support Zcopy operation, reset
         * min/max values to show that the scheme is unsupported */
        rndv_zcopy->min   = SIZE_MAX;
        rndv_zcopy->max   = 0;
        rndv_zcopy->split = 0;
    } else {
        rndv_zcopy->split = rndv_zcopy->min <= (rndv_zcopy->max / 2);
    }
}

static ssize_t
ucp_ep_config_max_short(ucp_context_t *context, uct_iface_attr_t *iface_attr,
                        uint64_t short_flag, size_t max_short, unsigned hdr_len,
                        size_t zcopy_thresh,
                        const ucp_rndv_thresh_t *rndv_thresh)
{
    ssize_t cfg_max_short;

    if (!(iface_attr->cap.flags & short_flag)) {
        return -1;
    }

    cfg_max_short = max_short - hdr_len;

    if ((context->config.ext.zcopy_thresh != UCS_MEMUNITS_AUTO)) {
        /* Adjust max_short if zcopy_thresh is set externally */
        ucp_ep_config_adjust_max_short(&cfg_max_short, zcopy_thresh);
    }

    if ((rndv_thresh != NULL) &&
        (context->config.ext.rndv_thresh != UCS_MEMUNITS_AUTO)) {
        /* Adjust max_short if rndv_thresh is set externally. Note local and
         * remote threshold values are the same if set externally, so can
         * compare with just one of them. */
        ucs_assert(rndv_thresh->remote == rndv_thresh->local);
        ucp_ep_config_adjust_max_short(&cfg_max_short, rndv_thresh->remote);
    }

    return cfg_max_short;
}

static void
ucp_ep_config_init_attrs(ucp_worker_t *worker, ucp_rsc_index_t rsc_index,
                         ucp_ep_msg_config_t *config, size_t max_bcopy,
                         size_t max_zcopy, size_t max_iov, size_t max_hdr,
                         uint64_t bcopy_flag, uint64_t zcopy_flag,
                         size_t adjust_min_val, size_t max_seg_size)
{
    ucp_context_t *context = worker->context;
    const uct_md_attr_t *md_attr;
    uct_iface_attr_t *iface_attr;
    size_t it;
    size_t zcopy_thresh;
    size_t mem_type_zcopy_thresh;
    int mem_type;

    iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

    if (iface_attr->cap.flags & bcopy_flag) {
        config->max_bcopy = ucs_min(max_bcopy, max_seg_size);
    } else {
        config->max_bcopy = SIZE_MAX;
    }

    md_attr = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;
    if (!(iface_attr->cap.flags & zcopy_flag) ||
        ((md_attr->cap.flags & UCT_MD_FLAG_NEED_MEMH) &&
         !(md_attr->cap.flags & UCT_MD_FLAG_REG))) {
        return;
    }

    config->max_zcopy = ucs_min(max_zcopy, max_seg_size);
    config->max_hdr   = max_hdr;
    config->max_iov   = ucs_min(UCP_MAX_IOV, max_iov);

    if (context->config.ext.zcopy_thresh == UCS_MEMUNITS_AUTO) {
        config->zcopy_auto_thresh = 1;
        mem_type_zcopy_thresh     = 1;
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
        mem_type_zcopy_thresh        = config->zcopy_thresh[0];
    }

    ucs_memory_type_for_each(mem_type) {
        if (UCP_MEM_IS_HOST(mem_type)) {
            config->mem_type_zcopy_thresh[mem_type] = config->zcopy_thresh[0];
        } else if (md_attr->cap.reg_mem_types & UCS_BIT(mem_type)) {
            config->mem_type_zcopy_thresh[mem_type] = mem_type_zcopy_thresh;
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
    size_t am_max_eager_short;
    double get_zcopy_max_bw[UCS_MEMORY_TYPE_LAST];
    double put_zcopy_max_bw[UCS_MEMORY_TYPE_LAST];
    ucs_status_t status;
    size_t it;

    memset(config, 0, sizeof(*config));

    status = ucp_ep_config_key_copy(&config->key, key);
    if (status != UCS_OK) {
        goto err;
    }

    /* Default settings */
    for (it = 0; it < UCP_MAX_IOV; ++it) {
        config->am.zcopy_thresh[it]              = SIZE_MAX;
        config->am.sync_zcopy_thresh[it]         = SIZE_MAX;
        config->tag.eager.zcopy_thresh[it]       = SIZE_MAX;
        config->tag.eager.sync_zcopy_thresh[it]  = SIZE_MAX;
    }

    ucs_memory_type_for_each(mem_type) {
        config->am.mem_type_zcopy_thresh[mem_type]        = SIZE_MAX;
        config->tag.eager.mem_type_zcopy_thresh[mem_type] = SIZE_MAX;
    }

    config->tag.eager.zcopy_auto_thresh = 0;
    config->am.zcopy_auto_thresh        = 0;
    config->p2p_lanes                   = 0;
    config->uct_rkey_pack_flags         = 0;
    if (context->config.ext.bcopy_thresh == UCS_MEMUNITS_AUTO) {
        config->bcopy_thresh = 0;
    } else {
        config->bcopy_thresh = context->config.ext.bcopy_thresh;
    }
    config->tag.lane                    = UCP_NULL_LANE;
    config->tag.proto                   = &ucp_tag_eager_proto;
    config->tag.sync_proto              = &ucp_tag_eager_sync_proto;
    config->tag.rndv.rma_thresh.remote  = SIZE_MAX;
    config->tag.rndv.rma_thresh.local   = SIZE_MAX;
    config->tag.rndv.am_thresh          = config->tag.rndv.rma_thresh;
    config->rndv.rma_thresh             = config->tag.rndv.rma_thresh;
    config->rndv.am_thresh              = config->tag.rndv.am_thresh;
    /* use 1 instead of 0, since messages passed to RNDV PUT/GET Zcopy are always > 0
     * and make sure that multi-rail chunks are adjusted to not be 0-length */
    config->rndv.get_zcopy.min          = 1;
    config->rndv.get_zcopy.max          = SIZE_MAX;
    config->rndv.put_zcopy.min          = 1;
    config->rndv.put_zcopy.max          = SIZE_MAX;
    config->rndv.rkey_size              = ucp_rkey_packed_size(context,
                                                               config->key.rma_bw_md_map,
                                                               UCS_SYS_DEVICE_ID_UNKNOWN, 0);
    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        config->rndv.get_zcopy.lanes[lane] =
                config->rndv.put_zcopy.lanes[lane] = UCP_NULL_LANE;
    }

    config->rndv.rkey_ptr_dst_mds       = 0;
    config->stream.proto                = &ucp_stream_am_proto;
    config->am_u.proto                  = &ucp_am_proto;
    config->am_u.reply_proto            = &ucp_am_reply_proto;

    ucp_ep_config_init_short_thresh(&config->tag.offload.max_eager_short);
    ucp_ep_config_init_short_thresh(&config->tag.max_eager_short);
    ucp_ep_config_init_short_thresh(&config->am_u.max_eager_short);
    ucp_ep_config_init_short_thresh(&config->am_u.max_reply_eager_short);

    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            config->md_index[lane] = context->tl_rscs[rsc_index].md_index;
            if (ucp_ep_config_connect_p2p(worker, &config->key, rsc_index)) {
                config->p2p_lanes |= UCS_BIT(lane);
            } else if (config->key.err_mode == UCP_ERR_HANDLING_MODE_PEER) {
                config->uct_rkey_pack_flags |= UCT_MD_MKEY_PACK_FLAG_INVALIDATE;
            }
        } else {
            config->md_index[lane] = UCP_NULL_RESOURCE;
        }
    }

    /* Memory domains of lanes that require registration and support AM_ZCOPY */
    for (i = 0; (i < config->key.num_lanes) &&
                (config->key.am_bw_lanes[i] != UCP_NULL_LANE);
         ++i) {
        lane = config->key.am_bw_lanes[i];
        if (config->md_index[lane] == UCP_NULL_RESOURCE) {
            continue;
        }

        md_attr = &context->tl_mds[config->md_index[lane]].attr;
        if (!ucs_test_all_flags(md_attr->cap.flags,
                                UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_MEMH)) {
            continue;
        }

        rsc_index  = config->key.lanes[lane].rsc_index;
        iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
        if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
            config->am_bw_prereg_md_map |= UCS_BIT(config->md_index[lane]);
        }
    }

    /* configuration for rndv */
    get_zcopy_lane_count = put_zcopy_lane_count = 0;

    ucs_memory_type_for_each(i) {
        get_zcopy_max_bw[i] = put_zcopy_max_bw[i] = 0;
    }

    for (i = 0; (i < config->key.num_lanes) &&
                (config->key.rma_bw_lanes[i] != UCP_NULL_LANE); ++i) {
        lane      = config->key.rma_bw_lanes[i];
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        md_attr    = &context->tl_mds[config->md_index[lane]].attr;
        iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

        /* GET Zcopy */
        /* coverity[overrun-buffer-val] */
        ucp_ep_config_rndv_zcopy_max_bw_update(context, md_attr, iface_attr,
                                               UCT_IFACE_FLAG_GET_ZCOPY,
                                               get_zcopy_max_bw);

        /* PUT Zcopy */
        /* coverity[overrun-buffer-val] */
        ucp_ep_config_rndv_zcopy_max_bw_update(context, md_attr, iface_attr,
                                               UCT_IFACE_FLAG_PUT_ZCOPY,
                                               put_zcopy_max_bw);
    }

    for (i = 0; (i < config->key.num_lanes) &&
                (config->key.rma_bw_lanes[i] != UCP_NULL_LANE); ++i) {
        lane      = config->key.rma_bw_lanes[i];
        rsc_index = config->key.lanes[lane].rsc_index;

        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
            md_attr    = &context->tl_mds[config->md_index[lane]].attr;

            /* GET Zcopy */
            ucp_ep_config_rndv_zcopy_set(context, UCT_IFACE_FLAG_GET_ZCOPY,
                                         lane, md_attr, iface_attr,
                                         get_zcopy_max_bw,
                                         &config->rndv.get_zcopy,
                                         &get_zcopy_lane_count);

            /* PUT Zcopy */
            ucp_ep_config_rndv_zcopy_set(context, UCT_IFACE_FLAG_PUT_ZCOPY,
                                         lane, md_attr, iface_attr,
                                         put_zcopy_max_bw,
                                         &config->rndv.put_zcopy,
                                         &put_zcopy_lane_count);
        }
    }

    /* GET Zcopy */
    ucp_ep_config_rndv_zcopy_commit(get_zcopy_lane_count,
                                    &config->rndv.get_zcopy);

    /* PUT Zcopy */
    ucp_ep_config_rndv_zcopy_commit(put_zcopy_lane_count,
                                    &config->rndv.put_zcopy);

    /* Rkey ptr */
    if (key->rkey_ptr_lane != UCP_NULL_LANE) {
        lane      = key->rkey_ptr_lane;
        md_attr   = &context->tl_mds[config->md_index[lane]].attr;
        ucs_assert_always(md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR);

        config->rndv.rkey_ptr_dst_mds =
                UCS_BIT(config->key.lanes[lane].dst_md_index);
    }

    /* Configuration for tag offload */
    if (config->key.tag_lane != UCP_NULL_LANE) {
        lane      = config->key.tag_lane;
        rsc_index = config->key.lanes[lane].rsc_index;
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);
            ucp_ep_config_init_attrs(worker, rsc_index, &config->tag.eager,
                                     iface_attr->cap.tag.eager.max_bcopy,
                                     iface_attr->cap.tag.eager.max_zcopy,
                                     iface_attr->cap.tag.eager.max_iov, 0,
                                     UCT_IFACE_FLAG_TAG_EAGER_BCOPY,
                                     UCT_IFACE_FLAG_TAG_EAGER_ZCOPY,
                                     iface_attr->cap.tag.eager.max_bcopy,
                                     UINT_MAX);

            config->tag.offload.max_rndv_iov   = iface_attr->cap.tag.rndv.max_iov;
            config->tag.offload.max_rndv_zcopy = iface_attr->cap.tag.rndv.max_zcopy;
            config->tag.sync_proto             = &ucp_tag_offload_sync_proto;
            config->tag.proto                  = &ucp_tag_offload_proto;
            config->tag.lane                   = lane;
            max_rndv_thresh                    = iface_attr->cap.tag.eager.max_zcopy;
            max_am_rndv_thresh                 = iface_attr->cap.tag.eager.max_bcopy;
            min_rndv_thresh                    = ucp_ep_tag_offload_min_rndv_thresh(config);
            min_am_rndv_thresh                 = min_rndv_thresh;

            ucs_assertv_always(iface_attr->cap.tag.rndv.max_hdr >=
                               sizeof(ucp_tag_offload_unexp_rndv_hdr_t),
                               "rndv.max_hdr %zu, offload_unexp_rndv_hdr %zu",
                               iface_attr->cap.tag.rndv.max_hdr,
                               sizeof(ucp_tag_offload_unexp_rndv_hdr_t));

            /* Must have active messages for using rendezvous */
            if (config->key.am_lane != UCP_NULL_LANE) {
                tag_lanes[0] = lane;
                ucp_ep_config_set_rndv_thresh(worker, config, tag_lanes,
                                              min_rndv_thresh, max_rndv_thresh,
                                              &config->tag.rndv.rma_thresh);

                md_attr = &context->tl_mds[config->md_index[lane]].attr;
                status = ucp_ep_config_set_am_rndv_thresh(worker, iface_attr,
                        md_attr, config, min_am_rndv_thresh,
                        max_am_rndv_thresh, &config->tag.rndv.am_thresh);
                if (status != UCS_OK) {
                    goto err_free_dst_mds;
                }
            }

            config->tag.eager.max_short = ucp_ep_config_max_short(
                    worker->context, iface_attr, UCT_IFACE_FLAG_TAG_EAGER_SHORT,
                    iface_attr->cap.tag.eager.max_short, 0,
                    config->tag.eager.zcopy_thresh[0],
                    &config->tag.rndv.am_thresh);

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
            md_attr    = &context->tl_mds[config->md_index[lane]].attr;
            ucp_ep_config_init_attrs(worker, rsc_index, &config->am,
                                     iface_attr->cap.am.max_bcopy,
                                     iface_attr->cap.am.max_zcopy,
                                     iface_attr->cap.am.max_iov,
                                     iface_attr->cap.am.max_hdr,
                                     UCT_IFACE_FLAG_AM_BCOPY,
                                     UCT_IFACE_FLAG_AM_ZCOPY, SIZE_MAX,
                                     config->key.lanes[lane].seg_size);

            /* Configuration stored in config->am is used by TAG, UCP AM and
             * STREAM protocol implementations, do not adjust max_short value by
             * zcopy and rndv thresholds. */
            config->am.max_short = ucp_ep_config_max_short(
                    worker->context, iface_attr, UCT_IFACE_FLAG_AM_SHORT,
                    iface_attr->cap.am.max_short, sizeof(ucp_eager_hdr_t),
                    SIZE_MAX, NULL);

            /* Calculate rendezvous thresholds which may be used by UCP AM
             * protocol. */
            if (config->key.rkey_ptr_lane != UCP_NULL_LANE) {
                rkey_ptr_lanes[0] = config->key.rkey_ptr_lane;
                ucp_ep_config_set_rndv_thresh(worker, config, rkey_ptr_lanes,
                                              iface_attr->cap.get.min_zcopy,
                                              SIZE_MAX,
                                              &config->rndv.rma_thresh);
            } else {
                ucp_ep_config_set_rndv_thresh(worker, config,
                                              config->key.rma_bw_lanes,
                                              iface_attr->cap.get.min_zcopy,
                                              SIZE_MAX,
                                              &config->rndv.rma_thresh);
            }

            status = ucp_ep_config_set_am_rndv_thresh(worker, iface_attr,
                    md_attr, config, iface_attr->cap.am.min_zcopy, SIZE_MAX,
                    &config->rndv.am_thresh);
            if (status != UCS_OK) {
                goto err_free_dst_mds;
            }

            am_max_eager_short = ucp_ep_config_max_short(
                    worker->context, iface_attr, UCT_IFACE_FLAG_AM_SHORT,
                    iface_attr->cap.am.max_short, sizeof(ucp_am_hdr_t),
                    config->am.zcopy_thresh[0], &config->rndv.am_thresh);

            ucp_ep_config_set_memtype_thresh(&config->am_u.max_eager_short,
                                             am_max_eager_short,
                                             context->num_mem_type_detect_mds);

            /* All keys must fit in RNDV packet.
             * TODO remove some MDs if they don't
             */
            ucs_assertv_always(config->rndv.rkey_size <= config->am.max_bcopy,
                               "rkey_size %zu, am.max_bcopy %zu",
                               config->rndv.rkey_size, config->am.max_bcopy);

            if (!ucp_ep_config_key_has_tag_lane(&config->key)) {
                /* Tag offload is disabled, AM will be used for all
                 * tag-matching protocols */
                /* TODO: set threshold level based on all available lanes */

                config->tag.eager           = config->am;
                config->tag.eager.max_short = am_max_eager_short;
                config->tag.lane            = lane;
                config->tag.rndv.am_thresh  = config->rndv.am_thresh;
                config->tag.rndv.rma_thresh = config->rndv.rma_thresh;

                /* Max Eager short has to be set after Zcopy and RNDV thresholds */
                ucp_ep_config_set_memtype_thresh(&config->tag.max_eager_short,
                                                 config->tag.eager.max_short,
                                                 context->num_mem_type_detect_mds);
            }

            /* Calculate max short threshold for UCP AM short reply protocol */
            am_max_eager_short = ucp_ep_config_max_short(
                    worker->context, iface_attr, UCT_IFACE_FLAG_AM_SHORT,
                    iface_attr->cap.am.max_short,
                    sizeof(ucp_am_hdr_t) + sizeof(ucp_am_reply_ftr_t),
                    config->am.zcopy_thresh[0], &config->rndv.am_thresh);

            ucp_ep_config_set_memtype_thresh(&config->am_u.max_reply_eager_short,
                                             am_max_eager_short,
                                             context->num_mem_type_detect_mds);
        } else {
            /* Stub endpoint */
            config->am.max_bcopy        = UCP_MIN_BCOPY;
            config->tag.eager.max_bcopy = UCP_MIN_BCOPY;
            config->tag.lane            = lane;
       }
    }

    memset(&config->rma, 0, sizeof(config->rma));

    status = ucp_ep_config_calc_rma_zcopy_thresh(worker, config,
                                                 config->key.rma_lanes,
                                                 &rma_zcopy_thresh);
    if (status != UCS_OK) {
        goto err_free_dst_mds;
    }

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

    status = ucp_proto_select_init(&config->proto_select);
    if (status != UCS_OK) {
        goto err_free_dst_mds;
    }

    return UCS_OK;

err_free_dst_mds:
    ucs_free(config->key.dst_md_cmpts);
err:
    return status;
}

void ucp_ep_config_cleanup(ucp_worker_h worker, ucp_ep_config_t *config)
{
    ucp_proto_select_cleanup(&config->proto_select);
    ucs_free(config->key.dst_md_cmpts);
}

static int ucp_ep_is_short_lower_thresh(ssize_t max_short,
                                        size_t thresh)
{
    return ((max_short < 0) ||
            (((size_t)max_short + 1) < thresh));
}

static void ucp_ep_config_print_short(FILE *stream, const char *proto_name,
                                      ssize_t max_config_short)
{
    size_t max_short;

    if (max_config_short > 0) {
        max_short = max_config_short;
        ucs_assert(max_short <= SSIZE_MAX);
        fprintf(stream, "..<%s>..%zu", proto_name, max_short + 1);
    } else if (max_config_short == 0) {
        fprintf(stream, "..<%s>..0", proto_name);
    }
}

static void ucp_ep_config_print_proto_middle(FILE *stream,
                                             const char *proto_name,
                                             ssize_t max_config_short,
                                             size_t min_current_proto,
                                             size_t min_next_proto)
{
    if (ucp_ep_is_short_lower_thresh(max_config_short, min_next_proto) &&
        (min_current_proto < min_next_proto)) {
        fprintf(stream, "..<%s>..", proto_name);
        if (min_next_proto < SIZE_MAX) {
            fprintf(stream, "%zu", min_next_proto);
        }
    }
}

static void ucp_ep_config_print_proto_last(FILE *stream, const char *name,
                                           size_t min_proto)
{
    if (min_proto < SIZE_MAX) {
        fprintf(stream, "..<%s>..", name);
    }

    fprintf(stream, "(inf)\n");
}

static void
ucp_ep_config_print_proto(FILE *stream, const char *name,
                          ssize_t max_eager_short, size_t zcopy_thresh,
                          size_t rndv_rma_thresh, size_t rndv_am_thresh)
{
    size_t min_zcopy, min_rndv;

    min_rndv  = ucs_min(rndv_rma_thresh, rndv_am_thresh);
    min_zcopy = ucs_min(zcopy_thresh, min_rndv);

    fprintf(stream, "# %23s: 0", name);

    /* print eager short */
    ucp_ep_config_print_short(stream, "egr/short", max_eager_short);

    /* print eager bcopy */
    ucp_ep_config_print_proto_middle(stream, "egr/bcopy", max_eager_short, 0,
                                     min_zcopy);

    /* print eager zcopy */
    ucp_ep_config_print_proto_middle(stream, "egr/zcopy", max_eager_short,
                                     min_zcopy, min_rndv);

    /* print rendezvous */
    ucp_ep_config_print_proto_last(stream, "rndv", min_rndv);
}

static void ucp_ep_config_print_rma_proto(FILE *stream, const char *name,
                                          ucp_lane_index_t lane,
                                          ssize_t max_rma_short,
                                          size_t zcopy_thresh)
{
    fprintf(stream, "# %20s[%d]: 0", name, lane);

    /* print short */
    ucp_ep_config_print_short(stream, "short", max_rma_short);

    /* print bcopy */
    ucp_ep_config_print_proto_middle(stream, "bcopy", max_rma_short, 0,
                                     zcopy_thresh);

    /* print zcopy */
    ucp_ep_config_print_proto_last(stream, "zcopy", zcopy_thresh);
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

void ucp_ep_config_cm_lane_info_str(ucp_worker_h worker,
                                    const ucp_ep_config_key_t *key,
                                    ucp_lane_index_t lane,
                                    ucp_rsc_index_t cm_index,
                                    ucs_string_buffer_t *strbuf)
{
    ucs_string_buffer_appendf(strbuf, "lane[%d]: cm %s", lane,
                              (cm_index != UCP_NULL_RESOURCE) ?
                              ucp_context_cm_name(worker->context, cm_index) :
                              "<unknown>");
}

void ucp_ep_config_lane_info_str(ucp_worker_h worker,
                                 const ucp_ep_config_key_t *key,
                                 const unsigned *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 ucs_string_buffer_t *strbuf)
{
    ucp_context_h context = worker->context;
    uct_tl_resource_desc_t *rsc;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t dst_md_index;
    ucp_rsc_index_t cmpt_index;
    unsigned path_index;
    int prio;

    rsc_index  = key->lanes[lane].rsc_index;
    rsc        = &context->tl_rscs[rsc_index].tl_rsc;

    path_index = key->lanes[lane].path_index;
    ucs_string_buffer_appendf(strbuf,
            "lane[%d]: %2d:" UCT_TL_RESOURCE_DESC_FMT ".%u md[%d] %-*c-> ",
            lane, rsc_index, UCT_TL_RESOURCE_DESC_ARG(rsc), path_index,
            context->tl_rscs[rsc_index].md_index,
            20 - (int)(strlen(rsc->dev_name) + strlen(rsc->tl_name)),
            ' ');

    if (addr_indices != NULL) {
        ucs_string_buffer_appendf(strbuf, "addr[%d].", addr_indices[lane]);
    }

    dst_md_index = key->lanes[lane].dst_md_index;
    cmpt_index   = ucp_ep_config_get_dst_md_cmpt(key, dst_md_index);
    ucs_string_buffer_appendf(strbuf, "md[%d]/%s/sysdev[%d]", dst_md_index,
                              context->tl_cmpts[cmpt_index].attr.name,
                              key->lanes[lane].dst_sys_dev);

    prio = ucp_ep_config_get_multi_lane_prio(key->rma_bw_lanes, lane);
    if (prio != -1) {
        ucs_string_buffer_appendf(strbuf, " rma_bw#%d", prio);
    }

    prio = ucp_ep_config_get_multi_lane_prio(key->amo_lanes, lane);
    if (prio != -1) {
        ucs_string_buffer_appendf(strbuf, " amo#%d", prio);
    }

    if (key->am_lane == lane) {
        ucs_string_buffer_appendf(strbuf, " am");
    }

    if (key->rkey_ptr_lane == lane) {
        ucs_string_buffer_appendf(strbuf, " rkey_ptr");
    }

    prio = ucp_ep_config_get_multi_lane_prio(key->am_bw_lanes, lane);
    if (prio != -1) {
        ucs_string_buffer_appendf(strbuf, " am_bw#%d", prio);
    }

    if (lane == key->tag_lane) {
        ucs_string_buffer_appendf(strbuf, " tag_offload");
    }

    if (key->keepalive_lane == lane) {
        ucs_string_buffer_appendf(strbuf, " keepalive");
    }

    if (key->wireup_msg_lane == lane) {
        ucs_string_buffer_appendf(strbuf, " wireup");
        if (aux_rsc_index != UCP_NULL_RESOURCE) {
            ucs_string_buffer_appendf(strbuf, "{" UCT_TL_RESOURCE_DESC_FMT "}",
                     UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[aux_rsc_index].tl_rsc));
        }
    }
}

static void ucp_ep_config_print(FILE *stream, ucp_worker_h worker,
                                const ucp_ep_h ep, const unsigned *addr_indices,
                                ucp_rsc_index_t aux_rsc_index)
{
    ucp_context_h context   = worker->context;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;
    ucp_rsc_index_t cm_idx;

    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        UCS_STRING_BUFFER_ONSTACK(strb, 128);
        if (lane == config->key.cm_lane) {
            cm_idx = ucp_ep_ext_control(ep)->cm_idx;
            ucp_ep_config_cm_lane_info_str(worker, &config->key, lane, cm_idx,
                                           &strb);
        } else {
            ucp_ep_config_lane_info_str(worker, &config->key, addr_indices,
                                        lane, aux_rsc_index, &strb);
        }
        fprintf(stream, "#                 %s\n", ucs_string_buffer_cstr(&strb));
    }
    fprintf(stream, "#\n");

    if (context->config.features & UCP_FEATURE_TAG) {
        ucp_ep_config_print_proto(stream, "tag_send",
                                  config->tag.eager.max_short,
                                  config->tag.eager.zcopy_thresh[0],
                                  config->tag.rndv.rma_thresh.remote,
                                  config->tag.rndv.am_thresh.remote);
        ucp_ep_config_print_proto(stream, "tag_send_nbr",
                                  config->tag.eager.max_short,
                                  /* disable zcopy */
                                  ucs_min(config->tag.rndv.rma_thresh.local,
                                          config->tag.rndv.am_thresh.local),
                                  config->tag.rndv.rma_thresh.local,
                                  config->tag.rndv.am_thresh.local);
        ucp_ep_config_print_proto(stream, "tag_send_sync",
                                  config->tag.eager.max_short,
                                  config->tag.eager.sync_zcopy_thresh[0],
                                  config->tag.rndv.rma_thresh.remote,
                                  config->tag.rndv.am_thresh.remote);
    }

    if (context->config.features & UCP_FEATURE_STREAM) {
        ucp_ep_config_print_proto(stream, "stream_send",
                                  config->am.max_short,
                                  config->am.zcopy_thresh[0],
                                  /* disable rndv */
                                  SIZE_MAX, SIZE_MAX);
    }

    if (context->config.features & UCP_FEATURE_AM) {
        ucp_ep_config_print_proto(stream, "am_send",
                                  config->am_u.max_eager_short.memtype_on,
                                  config->am.zcopy_thresh[0],
                                  config->rndv.rma_thresh.remote,
                                  config->rndv.am_thresh.remote);
    }

    if (context->config.features & UCP_FEATURE_RMA) {
        for (lane = 0; lane < config->key.num_lanes; ++lane) {
            if (ucp_ep_config_get_multi_lane_prio(config->key.rma_lanes,
                                                  lane) == -1) {
                continue;
            }
            ucp_ep_config_print_rma_proto(stream, "put", lane,
                                          config->rma[lane].max_put_short,
                                          config->rma[lane].put_zcopy_thresh);
            ucp_ep_config_print_rma_proto(stream, "get", lane,
                                          config->rma[lane].max_get_short,
                                          config->rma[lane].get_zcopy_thresh);
        }
    }

    if (context->config.features &
        (UCP_FEATURE_TAG | UCP_FEATURE_RMA | UCP_FEATURE_AM)) {
        fprintf(stream, "#\n");
        fprintf(stream, "# %23s: mds ", "rma_bw");
        ucs_for_each_bit(md_index, config->key.rma_bw_md_map) {
            fprintf(stream, "[%d] ", md_index);
        }
    }

    if (context->config.features & (UCP_FEATURE_TAG | UCP_FEATURE_AM)) {
        fprintf(stream, "rndv_rkey_size %zu\n", config->rndv.rkey_size);
    }
}

static void ucp_ep_print_info_internal(ucp_ep_h ep, const char *name,
                                       FILE *stream)
{
    ucp_worker_h worker     = ep->worker;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    ucp_rsc_index_t aux_rsc_index;
    ucp_lane_index_t wireup_msg_lane;
    ucs_string_buffer_t strb;
    uct_ep_h wireup_ep;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP endpoint %s\n", name);
    fprintf(stream, "#\n");
    fprintf(stream, "#               peer: %s\n", ucp_ep_peer_name(ep));

    /* if there is a wireup lane, set aux_rsc_index to the stub ep resource */
    aux_rsc_index   = UCP_NULL_RESOURCE;
    wireup_msg_lane = config->key.wireup_msg_lane;
    if (wireup_msg_lane != UCP_NULL_LANE) {
        wireup_ep   = ep->uct_eps[wireup_msg_lane];
        if (ucp_wireup_ep_test(wireup_ep)) {
            aux_rsc_index = ucp_wireup_ep_get_aux_rsc_index(wireup_ep);
        }
    }

    ucp_ep_config_print(stream, worker, ep, NULL, aux_rsc_index);
    fprintf(stream, "#\n");

    if (worker->context->config.ext.proto_enable) {
        ucs_string_buffer_init(&strb);
        ucp_proto_select_info(worker, ep->cfg_index, UCP_WORKER_CFG_INDEX_NULL,
                              &config->proto_select, &strb);
        ucs_string_buffer_dump(&strb, "# ", stream);
        ucs_string_buffer_cleanup(&strb);
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
}

void ucp_ep_print_info(ucp_ep_h ep, FILE *stream)
{
    return ucp_ep_print_info_internal(ep, "", stream);
}

void ucp_worker_mem_type_eps_print_info(ucp_worker_h worker, FILE *stream)
{
    ucs_memory_type_t mem_type;
    ucp_ep_h ep;

    ucs_memory_type_for_each(mem_type) {
        UCS_STRING_BUFFER_ONSTACK(strb, 128);

        ep = worker->mem_type_ep[mem_type];
        if (ep == NULL) {
            continue;
        }

        ucs_string_buffer_appendf(&strb, "for %s",
                                  ucs_memory_type_descs[mem_type]);
        ucp_ep_print_info_internal(ep, ucs_string_buffer_cstr(&strb), stream);
    }
}

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const ucs_linear_func_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth)
{
    double zcopy_thresh;
    double bcopy_bw = context->config.ext.bcopy_bw;

    zcopy_thresh = (iovcnt * reg_cost->c) /
                   ((1.0 / bcopy_bw) - (1.0 / bandwidth) - (iovcnt * reg_cost->m));

    if (zcopy_thresh < 0.0) {
        return SIZE_MAX;
    }

    return zcopy_thresh;
}

ucp_wireup_ep_t* ucp_ep_get_cm_wireup_ep(ucp_ep_h ep)
{
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    if (ep->cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        return NULL;
    }

    lane = ucp_ep_get_cm_lane(ep);
    if (lane == UCP_NULL_LANE) {
        return NULL;
    }

    uct_ep = ep->uct_eps[lane];
    return (uct_ep != NULL) ? ucp_wireup_ep(uct_ep) : NULL;
}

uct_ep_h ucp_ep_get_cm_uct_ep(ucp_ep_h ep)
{
    ucp_lane_index_t lane;
    ucp_wireup_ep_t *wireup_ep;

    lane = ucp_ep_get_cm_lane(ep);
    if (lane == UCP_NULL_LANE) {
        return NULL;
    }

    if (ep->uct_eps[lane] == NULL) {
        return NULL;
    }

    wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    return (wireup_ep == NULL) ? ep->uct_eps[lane] : wireup_ep->super.uct_ep;
}

int ucp_ep_is_cm_local_connected(ucp_ep_h ep)
{
    return (ucp_ep_get_cm_uct_ep(ep) != NULL) &&
           (ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
}

int ucp_ep_is_local_connected(ucp_ep_h ep)
{
    int is_local_connected = !!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
    ucp_wireup_ep_t *wireup_ep;
    ucp_lane_index_t i;

    if (ucp_ep_has_cm_lane(ep)) {
        /* For CM case need to check all wireup lanes because transport lanes
         * can be not connected yet. */
        for (i = 0; is_local_connected && (i < ucp_ep_num_lanes(ep)); ++i) {
            wireup_ep          = ucp_wireup_ep(ep->uct_eps[i]);
            is_local_connected = (wireup_ep == NULL) ||
                                 (wireup_ep->flags &
                                  UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED);
        }
    }

    return is_local_connected;
}

void ucp_ep_get_tl_bitmap(ucp_ep_h ep, ucp_tl_bitmap_t *tl_bitmap)
{
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_idx;

    UCS_BITMAP_CLEAR(tl_bitmap);
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (lane == ucp_ep_get_cm_lane(ep)) {
            continue;
        }

        rsc_idx = ucp_ep_get_rsc_index(ep, lane);
        if (rsc_idx == UCP_NULL_RESOURCE) {
            continue;
        }

        UCS_BITMAP_SET(*tl_bitmap, rsc_idx);
    }
}

void ucp_ep_get_lane_info_str(ucp_ep_h ucp_ep, ucp_lane_index_t lane,
                              ucs_string_buffer_t *lane_info_strb)
{
    ucp_rsc_index_t rsc_index;
    uct_tl_resource_desc_t *tl_rsc;

    if (lane == UCP_NULL_LANE) {
        ucs_string_buffer_appendf(lane_info_strb, "NULL lane");
    } else if (lane == ucp_ep_get_cm_lane(ucp_ep)) {
        ucs_string_buffer_appendf(lane_info_strb, "CM lane");
    } else {
        rsc_index = ucp_ep_get_rsc_index(ucp_ep, lane);
        tl_rsc    = &ucp_ep->worker->context->tl_rscs[rsc_index].tl_rsc;

        ucs_string_buffer_appendf(lane_info_strb,
                                  UCT_TL_RESOURCE_DESC_FMT,
                                  UCT_TL_RESOURCE_DESC_ARG(tl_rsc));
    }
}

void ucp_ep_invoke_err_cb(ucp_ep_h ep, ucs_status_t status)
{
    ucs_assert(ucp_ep_ext_control(ep)->err_cb != NULL);

    /* Do not invoke error handler if the EP has been closed by user, or error
     * callback already called */
    if ((ep->flags & (UCP_EP_FLAG_CLOSED | UCP_EP_FLAG_ERR_HANDLER_INVOKED))) {
        return;
    }

    ucs_assert(ep->flags & UCP_EP_FLAG_USED);
    ucs_debug("ep %p: calling user error callback %p with arg %p and status %s",
              ep, ucp_ep_ext_control(ep)->err_cb, ucp_ep_ext_gen(ep)->user_data,
              ucs_status_string(status));
    ucp_ep_update_flags(ep, UCP_EP_FLAG_ERR_HANDLER_INVOKED, 0);
    ucp_ep_ext_control(ep)->err_cb(ucp_ep_ext_gen(ep)->user_data, ep, status);
}

int ucp_ep_is_am_keepalive(ucp_ep_h ep, ucp_rsc_index_t rsc_index, int is_p2p)
{
    return /* Not a CM lane */
            (rsc_index != UCP_NULL_RESOURCE) &&
            /* Have a remote endpoint ID to send in the keepalive active message */
            (ep->flags & UCP_EP_FLAG_REMOTE_ID) &&
            /* Transport is not connected as point-to-point */
            !is_p2p &&
            /* Transport supports active messages */
            (ucp_worker_iface(ep->worker, rsc_index)->flags &
             UCT_IFACE_FLAG_AM_BCOPY);
}

ucs_status_t ucp_ep_do_uct_ep_am_keepalive(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                                           ucp_rsc_index_t rsc_idx)
{
    ucp_tl_bitmap_t tl_bitmap = UCS_BITMAP_ZERO;
    ucs_status_t status;
    ssize_t packed_len;
    struct iovec wireup_msg_iov[2];
    ucp_wireup_msg_t wireup_msg;

    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_FAILED));

    UCS_BITMAP_SET(tl_bitmap, rsc_idx);

    status = ucp_wireup_msg_prepare(ucp_ep, UCP_WIREUP_MSG_EP_CHECK,
                                    &tl_bitmap, NULL, &wireup_msg,
                                    &wireup_msg_iov[1].iov_base,
                                    &wireup_msg_iov[1].iov_len);
    if (status != UCS_OK) {
        return status;
    }

    wireup_msg_iov[0].iov_base = &wireup_msg;
    wireup_msg_iov[0].iov_len  = sizeof(wireup_msg);

    packed_len = uct_ep_am_bcopy(uct_ep, UCP_AM_ID_WIREUP,
                                 ucp_wireup_msg_pack, wireup_msg_iov,
                                 UCT_SEND_FLAG_PEER_CHECK);
    ucs_free(wireup_msg_iov[1].iov_base);
    return (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
}

static void ucp_ep_req_purge_send(ucp_request_t *req, ucs_status_t status)
{
    ucs_assertv(UCS_STATUS_IS_ERR(status), "req %p: status %s", req,
                ucs_status_string(status));

    if ((ucp_ep_config(req->send.ep)->key.err_mode !=
         UCP_ERR_HANDLING_MODE_NONE) &&
        (req->flags & UCP_REQUEST_FLAG_RKEY_INUSE) &&
        !(req->flags & UCP_REQUEST_FLAG_USER_MEMH)) {
        ucp_request_dt_invalidate(req, status);
        return;
    }

    ucp_request_complete_and_dereg_send(req, status);
}

void ucp_ep_req_purge(ucp_ep_h ucp_ep, ucp_request_t *req,
                      ucs_status_t status, int recursive)
{
    ucp_trace_req(req, "purged with status %s (%d) on ep %p",
                  ucs_status_string(status), status, ucp_ep);

    /* RNDV GET/PUT Zcopy operations shouldn't be handled here, because they
     * don't allocate local request ID, so they are not added on a list of
     * tracked operations */
    ucs_assert((req->send.uct.func != ucp_rndv_progress_rma_get_zcopy) &&
               (req->send.uct.func != ucp_rndv_progress_rma_put_zcopy));

    /* Only send operations could have request ID allocated */
    if (!(req->flags &
          (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG))) {
        ucp_send_request_id_release(req);
    }

    if (req->flags & (UCP_REQUEST_FLAG_SEND_AM | UCP_REQUEST_FLAG_SEND_TAG)) {
        ucs_assert(!(req->flags & UCP_REQUEST_FLAG_SUPER_VALID));
        ucs_assert(req->send.ep == ucp_ep);
        ucp_ep_req_purge_send(req, status);
    } else if (req->flags & UCP_REQUEST_FLAG_RECV_AM) {
        ucs_assert(!(req->flags & UCP_REQUEST_FLAG_SUPER_VALID));
        ucs_assert(recursive); /* Mustn't be directly contained in an EP list
                                * of tracking requests */
        ucp_request_recv_buffer_dereg(req);
        ucp_request_complete_am_recv(req, status);
    } else if (req->flags & UCP_REQUEST_FLAG_RECV_TAG) {
        ucs_assert(!(req->flags & UCP_REQUEST_FLAG_SUPER_VALID));
        ucs_assert(recursive); /* Mustn't be directly contained in an EP list
                                * of tracking requests */
        ucp_request_recv_buffer_dereg(req);
        ucp_request_complete_tag_recv(req, status);
    } else if (req->flags & UCP_REQUEST_FLAG_RNDV_FRAG) {
        ucs_assert(req->flags & UCP_REQUEST_FLAG_SUPER_VALID);
        ucs_assert(req->send.ep == ucp_ep);
        ucs_assert(recursive); /* Mustn't be directly contained in an EP list
                                * of tracking requests */

        /* It means that purging started from a request responsible for sending
         * RTR, so a request is responsible for copying data from staging buffer
         * and it uses a receive part of a request */
        req->super_req->recv.remaining -= req->recv.length;
        if (req->super_req->recv.remaining == 0) {
            ucp_ep_req_purge(ucp_ep, ucp_request_get_super(req), status, 1);
        }

        ucp_request_put(req);
    } else if ((req->send.uct.func == ucp_rma_sw_proto.progress_get) ||
               (req->send.uct.func == ucp_amo_sw_proto.progress_fetch)) {
        /* Currently we don't support UCP EP request purging for proto mode */
        ucs_assert(!ucp_ep->worker->context->config.ext.proto_enable);
        ucs_assert(req->send.ep == ucp_ep);

        ucp_request_send_buffer_dereg(req);
        ucp_request_complete_send(req, status);
        ucp_ep_rma_remote_request_completed(ucp_ep);
    } else {
        /* SW RMA/PUT and AMO/Post operations don't allocate local request ID
         * and don't need to be tracked, since they complete UCP request upon
         * sending all data to a peer. Receiving RMA/CMPL and AMO/REP packets
         * complete flush requests */
        ucs_assert((req->send.uct.func != ucp_rma_sw_proto.progress_put) &&
                   (req->send.uct.func != ucp_amo_sw_proto.progress_post));
        ucs_assert(req->send.ep == ucp_ep);

        ucp_ep_req_purge(ucp_ep, ucp_request_get_super(req), status, 1);
        ucp_request_put(req);
    }
}

void ucp_ep_reqs_purge(ucp_ep_h ucp_ep, ucs_status_t status)
{
    ucs_hlist_head_t *proto_reqs = &ucp_ep_ext_gen(ucp_ep)->proto_reqs;
    ucp_ep_flush_state_t *flush_state;
    ucp_request_t *req;

    while (!ucs_hlist_is_empty(proto_reqs)) {
        req = ucs_hlist_head_elem(proto_reqs, ucp_request_t, send.list);
        if (ucp_ep->worker->context->config.ext.proto_enable) {
            ucp_proto_request_abort(req, status);
        } else {
            ucp_ep_req_purge(ucp_ep, req, status, 0);
        }
    }

    if (/* Flush state is already valid (i.e. EP doesn't exist on matching
         * context) and not invalidated yet */
        !(ucp_ep->flags & UCP_EP_FLAG_ON_MATCH_CTX)) {
        flush_state = ucp_ep_flush_state(ucp_ep);

        /* Adjust 'comp_sn' value to a value stored in 'send_sn' by emulating
         * remote completion of RMA operations because those uncompleted
         * uncompleted operations won't be completed anymore. This could be only
         * SW RMA/PUT or AMO/Post operations, because SW RMA/GET or AMO/Fetch
         * operations should already complete flush operations which are waiting
         * for completion packets */
        while (UCS_CIRCULAR_COMPARE32(flush_state->cmpl_sn, <,
                                      flush_state->send_sn)) {
            ucp_ep_rma_remote_request_completed(ucp_ep);
        }
    }
}

ucs_status_t ucp_ep_query_sockaddr(ucp_ep_h ep, ucp_ep_attr_t *attr)
{
    uct_ep_h uct_cm_ep = ucp_ep_get_cm_uct_ep(ep);
    uct_ep_attr_t uct_cm_ep_attr;
    ucs_status_t status;

    if ((uct_cm_ep == NULL) || ucp_is_uct_ep_failed(uct_cm_ep)) {
        ucs_debug("ep %p: no cm", ep);
        return UCS_ERR_NOT_CONNECTED;
    }

    memset(&uct_cm_ep_attr, 0, sizeof(uct_ep_attr_t));

    if (attr->field_mask & UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR) {
        uct_cm_ep_attr.field_mask |= UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR;
    }

    if (attr->field_mask & UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR) {
        uct_cm_ep_attr.field_mask |= UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR;
    }

    status = uct_ep_query(uct_cm_ep, &uct_cm_ep_attr);
    if (status != UCS_OK) {
        return status;
    }

    if (attr->field_mask & UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR) {
        status = ucs_sockaddr_copy((struct sockaddr*)&attr->local_sockaddr,
                                   (struct sockaddr*)&uct_cm_ep_attr.local_address);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (attr->field_mask & UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR) {
        status = ucs_sockaddr_copy((struct sockaddr*)&attr->remote_sockaddr,
                                   (struct sockaddr*)&uct_cm_ep_attr.remote_address);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t ucp_ep_query(ucp_ep_h ep, ucp_ep_attr_t *attr)
{
    if (attr->field_mask & UCP_EP_ATTR_FIELD_NAME) {
#if ENABLE_DEBUG_DATA
        ucs_strncpy_safe(attr->name, ep->name, UCP_ENTITY_NAME_MAX);
#else
        ucs_snprintf_zero(attr->name, UCP_ENTITY_NAME_MAX, "%p", ep);
#endif
    }

    if (attr->field_mask &
        (UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR)) {
        return ucp_ep_query_sockaddr(ep, attr);
    }

    return UCS_OK;
}
