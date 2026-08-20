/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_ep_failover.h"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/wireup/wireup.h>
#include <ucp/proto/proto_failover.h>
#include <ucp/wireup/wireup_ep.h>
#include <uct/api/v2/uct_v2.h>
#include <uct/base/uct_iface.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/ptr_arith.h>

enum ucp_ep_failover_lane_flags {
    UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN          = UCS_BIT(0),
    UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED         = UCS_BIT(1),
    UCP_EP_FAILOVER_LANE_FLAG_PENDING_EXTRACTED = UCS_BIT(2)
};


enum ucp_ep_failover_flags {
    UCP_EP_FAILOVER_FLAG_FLUSH_STARTED = UCS_BIT(0),
    UCP_EP_FAILOVER_FLAG_FLUSH_DONE    = UCS_BIT(1),
    UCP_EP_FAILOVER_FLAG_ABORTED       = UCS_BIT(2)
};


/* Per failed-lane context used while failover to alternate lanes is in progress. */
typedef struct ucp_ep_failover_lane_ctx {
    ucp_ep_failover_ctx_t             *ctx;
    ucp_ep_h                         ep;
    uct_ep_h                         uct_ep;
    ucp_lane_index_t                 lane;
    ucp_rsc_index_t                  rsc_index;
    void                             *rx_token;
    uint8_t                          rx_token_length;
    unsigned                         flags;
    ucs_status_t                     status;
    ucp_ep_failover_lane_done_cb_t   done_cb;
    ucp_ep_failover_lane_failed_cb_t failed_cb;
    void                             *done_arg;
    /* Temporary queue of extracted ops until they are posted as pending
     * requests; empty while replay is in flight (tracked by undelivered_count). */
    ucs_queue_head_t                 replay_queue;
    unsigned                         undelivered_count;
} ucp_ep_failover_lane_ctx_t;


typedef struct {
    ucp_ep_failover_lane_ctx_t *lane;
    ucs_status_t               status;
} ucp_ep_failover_extract_arg_t;


struct ucp_ep_failover_ctx {
    ucp_lane_map_t             lane_map;
    uint64_t                   request_id;
    unsigned                   flags;
    ucs_status_t               status;
    ucp_request_t              *super_req;
    ucs_queue_head_t           pending_queue;
    ucp_ep_failover_lane_ctx_t lanes[UCP_MAX_LANES];
};


static void ucp_ep_failover_schedule(ucp_ep_h ep);
static unsigned ucp_ep_failover_progress_cb(void *arg);
static ucs_status_t ucp_ep_failover_flush_start(ucp_ep_h ep);

static int ucp_ep_failover_iface_token_supported(uct_iface_h iface)
{
    uct_iface_attr_v2_t attr;
    ucs_status_t status;

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH;
    status          = uct_iface_query_v2(iface, &attr);
    if (status != UCS_OK) {
        return 0;
    }

    return (attr.cap.flags & UCT_IFACE_FLAG_V2_QUERY_TOKEN) &&
           (attr.tx_token_length > 0) && (attr.tx_token_length <= UINT8_MAX);
}


static int ucp_ep_failover_lane_token_supported(ucp_ep_h ep, uct_ep_h uct_ep,
                                                ucp_lane_index_t lane)
{
    if ((uct_ep == NULL) || ucp_wireup_ep_test(uct_ep) ||
        (ucp_ep_get_rsc_index(ep, lane) == UCP_NULL_RESOURCE)) {
        return 0;
    }

    return ucp_ep_failover_iface_token_supported(uct_ep->iface);
}


ucs_status_t ucp_ep_failover_enable_lanes(ucp_ep_h ep)
{
    ucp_wireup_ep_t *wireup_ep;
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;
    ucs_status_t status;

    if ((ep->ext == NULL) ||
        !ucp_ep_err_mode_eq(ep, UCP_ERR_HANDLING_MODE_FAILOVER) ||
        (ucp_ep_config(ep)->key.dst_version <
         UCP_WIREUP_LANE_STATE_MIN_VERSION)) {
        return UCS_OK;
    }

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = ucp_ep_get_lane(ep, lane);
        if ((uct_ep != NULL) && ucp_wireup_ep_test(uct_ep)) {
            wireup_ep = ucp_wireup_ep(uct_ep);
            if (!ucp_wireup_ep_has_next_ep(wireup_ep)) {
                continue;
            }

            uct_ep = wireup_ep->super.uct_ep;
        }

        if (!ucp_ep_failover_lane_token_supported(ep, uct_ep, lane)) {
            ucs_debug("ep %p: lane %u uct_ep %p does not support failover "
                      "tokens",
                      ep, lane, uct_ep);
            continue;
        }

        status = uct_ep_failover_enable(uct_ep);
        if (status != UCS_OK) {
            ucs_debug("ep %p: failed to enable lane %u uct_ep %p for "
                      "failover: %s",
                      ep, lane, uct_ep, ucs_status_string(status));
            return status;
        }

        ucs_debug("ep %p: enabled lane %u uct_ep %p for failover extraction",
                  ep, lane, uct_ep);
    }

    return UCS_OK;
}

static void ucp_ep_failover_replay_purge(ucp_ep_failover_lane_ctx_t *lane,
                                         ucs_status_t status)
{
    ucp_proto_failover_replay_op_t *op;

    while (!ucs_queue_is_empty(&lane->replay_queue)) {
        op = ucs_queue_pull_elem_non_empty(&lane->replay_queue,
                                           ucp_proto_failover_replay_op_t,
                                           queue);
        ucp_proto_failover_replay_op_destroy(op, status);
        ucs_assert(lane->undelivered_count > 0);
        --lane->undelivered_count;
    }
}


static void ucp_ep_failover_pending_purge(ucp_ep_failover_ctx_t *ctx,
                                          ucs_status_t status)
{
    uct_pending_req_t *uct_req;

    ucs_queue_for_each_extract(uct_req, &ctx->pending_queue, priv, 1) {
        ucp_ep_err_pending_purge(uct_req, UCS_STATUS_PTR(status));
    }
}


static void ucp_ep_failover_pending_extract(ucp_ep_failover_lane_ctx_t *lane)
{
    ucs_assert(lane->uct_ep != NULL);
    ucs_assert(lane->flags & UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED);
    ucs_assert(!(lane->flags & UCP_EP_FAILOVER_LANE_FLAG_PENDING_EXTRACTED));

    uct_ep_pending_purge(lane->uct_ep, ucp_request_purge_enqueue_cb,
                         &lane->ctx->pending_queue);
    lane->flags |= UCP_EP_FAILOVER_LANE_FLAG_PENDING_EXTRACTED;
}


static void ucp_ep_failover_destroy_uct_ep(ucp_ep_failover_lane_ctx_t *lane)
{
    if (lane->uct_ep == NULL) {
        return;
    }

    ucp_ep_unprogress_uct_ep(lane->ep, lane->uct_ep, lane->rsc_index);
    uct_ep_destroy(lane->uct_ep);
    lane->uct_ep = NULL;
}


static void
ucp_ep_failover_extract_cb(const uct_ep_op_info_t *op_info, void *arg)
{
    ucp_ep_failover_extract_arg_t *extract_arg = arg;
    ucp_proto_failover_replay_op_t *op;
    ucs_status_t status;

    if (extract_arg->status != UCS_OK) {
        status = extract_arg->status;
        goto err;
    }

    ucs_debug("ft psn ucp extracted ep %p failed_lane %u operation %d "
              "field_mask 0x%" PRIx64 " comp %p",
              extract_arg->lane->ep, extract_arg->lane->lane,
              (int)op_info->operation, op_info->field_mask,
              (op_info->field_mask & UCT_EP_OP_INFO_FIELD_COMP) ?
                      op_info->comp : NULL);

    status = ucp_proto_failover_replay_op_create(op_info, &op);
    if (status == UCS_ERR_UNSUPPORTED) {
        /* Only AM short/bcopy WQEs are re-posted. RMA and zcopy operations own
         * their user buffers and are recovered by restarting the owning UCP
         * request from its completion callback, so drop the extracted copy and
         * keep the extraction successful. */
        ucs_debug("ep %p: lane %u ignoring extracted op %d, not replayable",
                  extract_arg->lane->ep, extract_arg->lane->lane,
                  (int)op_info->operation);
        status = UCS_ERR_CANCELED;
        goto err;
    } else if (status != UCS_OK) {
        ucs_debug("ep %p: failed to save extracted failover op %d: %s",
                  extract_arg->lane->ep, (int)op_info->operation,
                  ucs_status_string(status));
        extract_arg->status = status;
        goto err;
    }

    /* Park the op on the lane until extract finishes successfully; posting
     * starts only after outstanding_purge returns OK. */
    ucs_queue_push(&extract_arg->lane->replay_queue, &op->queue);
    ++extract_arg->lane->undelivered_count;
    ucs_debug("ft psn ucp replay queued ep %p failed_lane %u replay_op %p "
              "operation %d undelivered %u",
              extract_arg->lane->ep, extract_arg->lane->lane, op,
              (int)op_info->operation,
              extract_arg->lane->undelivered_count);
    return;

err:
    if ((op_info->field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op_info->comp != NULL)) {
        uct_invoke_completion(op_info->comp, status);
    }
}


void ucp_ep_failover_init(ucp_ep_h ep)
{
    if (ep->ext == NULL) {
        return;
    }

    ep->ext->failover.progress_scheduled = 0;
    ep->ext->failover.ctx                = NULL;
}


uct_ep_h ucp_ep_failover_get_uct_ep(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_ep_failover_ctx_t *ctx;
    uct_ep_h uct_ep;

    if (ep->ext == NULL) {
        if (lane >= UCP_MAX_FAST_PATH_LANES) {
            return NULL;
        }

        uct_ep = ucp_ep_get_fast_lane(ep, lane);
    } else {
        ctx = ep->ext->failover.ctx;
        if ((ctx != NULL) && (ctx->lane_map & UCS_BIT(lane)) &&
            (ctx->lanes[lane].uct_ep != NULL)) {
            return ctx->lanes[lane].uct_ep;
        }

        uct_ep = ucp_ep_get_lane(ep, lane);
    }

    if ((uct_ep != NULL) && ucp_wireup_ep_test(uct_ep)) {
        return NULL;
    }

    return uct_ep;
}


int ucp_ep_failover_is_uct_ep(ucp_ep_h ep, ucp_lane_index_t lane,
                              uct_ep_h uct_ep)
{
    ucp_ep_failover_ctx_t *ctx;

    if ((ep->ext == NULL) || (lane >= UCP_MAX_LANES)) {
        return 0;
    }

    ctx = ep->ext->failover.ctx;
    return (ctx != NULL) && (ctx->lane_map & UCS_BIT(lane)) &&
           (ctx->lanes[lane].uct_ep == uct_ep);
}


/* Release the context after its last armed lane is gone. An EP flush posted for
 * replayed operations owns super_req, so in that case the flush completion
 * callback releases the context instead. */
static void ucp_ep_failover_ctx_release(ucp_ep_h ep, ucp_ep_failover_ctx_t *ctx)
{
    ucs_assert(ctx->lane_map == 0);
    ucs_assert(ucs_queue_is_empty(&ctx->pending_queue));

    if ((ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_STARTED) &&
        !(ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_DONE)) {
        return;
    }

    ep->ext->failover.ctx = NULL;
    ucp_request_put(ctx->super_req);
    ucs_free(ctx);
}


static int ucp_ep_failover_lane_complete(ucp_ep_failover_ctx_t *ctx,
                                         ucp_lane_index_t lane_index,
                                         ucs_status_t status)
{
    ucp_ep_failover_lane_ctx_t *lane       = &ctx->lanes[lane_index];
    ucp_ep_h ep                            = lane->ep;
    ucp_worker_h worker                    = ep->worker;
    ucp_ep_failover_lane_done_cb_t done_cb = lane->done_cb;
    void *done_arg                         = lane->done_arg;
    int failover_done;

    ucs_debug("ep %p: completed failover for lane %u status %s", ep, lane_index,
              ucs_status_string(status));

    ucp_ep_failover_destroy_uct_ep(lane);
    ucs_assert(lane->undelivered_count == 0);
    ucs_free(lane->rx_token);

    ctx->lane_map &= ~UCS_BIT(lane_index);
    memset(lane, 0, sizeof(*lane));
    failover_done = (ctx->lane_map == 0);
    if (failover_done) {
        ucs_assert(ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_DONE);
        ucp_ep_failover_ctx_release(ep, ctx);
    }

    ucp_worker_flush_ops_count_add(worker, -1);
    if (done_cb != NULL) {
        done_cb(NULL, status, done_arg);
    }

    ucp_ep_refcount_remove(ep, discard);
    return failover_done;
}


static void ucp_ep_failover_lane_close(ucp_ep_h ep,
                                       ucp_lane_index_t lane_index,
                                       ucs_status_t discard_status)
{
    ucp_ep_failover_ctx_t *ctx = ep->ext->failover.ctx;
    ucp_worker_h worker        = ep->worker;
    ucp_ep_failover_lane_ctx_t *lane;
    ucp_ep_failover_lane_done_cb_t done_cb;
    ucp_ep_failover_lane_failed_cb_t failed_cb;
    ucp_rsc_index_t rsc_index;
    uct_ep_h uct_ep;
    void *done_arg;

    if ((ctx == NULL) || !(ctx->lane_map & UCS_BIT(lane_index))) {
        return;
    }

    lane      = &ctx->lanes[lane_index];
    uct_ep    = lane->uct_ep;
    rsc_index = lane->rsc_index;
    done_cb   = lane->done_cb;
    failed_cb = lane->failed_cb;
    done_arg  = lane->done_arg;

    ucs_debug("ep %p: closing failover lane %u status %s", ep, lane_index,
              ucs_status_string(discard_status));

    ucp_ep_failover_replay_purge(lane, discard_status);
    ucs_free(lane->rx_token);
    memset(lane, 0, sizeof(*lane));
    ctx->lane_map &= ~UCS_BIT(lane_index);

    if (failed_cb != NULL) {
        failed_cb(discard_status, done_arg);
    }

    /* The lane owns in-flight WQEs whose completions are deferred, so it cannot
     * be discarded through a flush - the flush would never complete. Release
     * the pending requests and close the endpoint instead. Extraction already
     * destroyed the endpoint of an extracted lane. */
    if (uct_ep != NULL) {
        uct_ep_pending_purge(uct_ep, ucp_ep_err_pending_purge,
                             UCS_STATUS_PTR(discard_status));
        ucp_ep_unprogress_uct_ep(ep, uct_ep, rsc_index);
        uct_ep_destroy(uct_ep);
    }

    if (ctx->lane_map == 0) {
        ucp_ep_failover_pending_purge(ctx, discard_status);
        ucp_ep_failover_ctx_release(ep, ctx);
    }

    if (done_cb != NULL) {
        done_cb(NULL, discard_status, done_arg);
    }

    ucp_worker_flush_ops_count_add(worker, -1);
    ucp_ep_refcount_remove(ep, discard);
}


ucs_status_t
ucp_ep_failover_add_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                          uct_ep_h *uct_eps, ucp_ep_failover_lane_done_cb_t cb,
                          ucp_ep_failover_lane_failed_cb_t failed_cb, void *arg,
                          ucp_lane_map_t *failover_lanes_p)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    *failover_lanes_p = 0;
    ucs_assert(ep->ext != NULL);
    ucs_assert(ucp_ep_err_mode_eq(ep, UCP_ERR_HANDLING_MODE_FAILOVER));
    ucs_assert(lane_map != 0);

    if (ucp_ep_config(ep)->key.dst_version <
        UCP_WIREUP_LANE_STATE_MIN_VERSION) {
        return UCS_ERR_UNSUPPORTED;
    }

    ctx = ep->ext->failover.ctx;
    ucs_for_each_bit(lane, lane_map) {
        uct_ep = uct_eps[lane];
        if (!ucp_ep_failover_lane_token_supported(ep, uct_ep, lane) ||
            ((ctx != NULL) && (ctx->lane_map & UCS_BIT(lane)))) {
            return UCS_ERR_UNSUPPORTED;
        }
    }

    if (ctx == NULL) {
        ctx = ucs_calloc(1, sizeof(*ctx), "ep_failover_ctx");
        if (ctx == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        ctx->super_req = ucp_request_get(ep->worker);
        if (ctx->super_req == NULL) {
            ucs_free(ctx);
            return UCS_ERR_NO_MEMORY;
        }

        ctx->status            = UCS_OK;
        ctx->super_req->status = UCS_INPROGRESS;
        ctx->super_req->flags  = 0;
        ctx->super_req->user_data = ctx;
        ctx->super_req->send.ep   = ep;
        ucs_queue_head_init(&ctx->pending_queue);
        ep->ext->failover.ctx = ctx;
    }

    ctx->request_id = 0;
    ucs_for_each_bit(lane, lane_map) {
        uct_ep = uct_eps[lane];
        lane_ctx            = &ctx->lanes[lane];
        lane_ctx->ctx       = ctx;
        lane_ctx->ep        = ep;
        lane_ctx->uct_ep    = uct_ep;
        lane_ctx->lane      = lane;
        lane_ctx->rsc_index = ucp_ep_get_rsc_index(ep, lane);
        lane_ctx->status    = UCS_OK;
        lane_ctx->done_cb   = cb;
        lane_ctx->failed_cb = failed_cb;
        lane_ctx->done_arg  = arg;
        ucs_queue_head_init(&lane_ctx->replay_queue);
        lane_ctx->undelivered_count = 0;

        /* Own in-flight WQEs for extract; safe if the lane was already
         * invalidated without DEFER by the error-injection path. */
        {
            uct_ep_invalidate_params_t inv_params = {
                .field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS,
                .flags      = UCT_EP_INVALIDATE_FLAG_DEFER_COMPLETIONS
            };
            ucs_status_t inv_status = uct_ep_invalidate(uct_ep, &inv_params);
            if ((inv_status != UCS_OK) &&
                (inv_status != UCS_ERR_UNSUPPORTED)) {
                ucs_debug("ep %p: lane %u defer-completions invalidate: %s",
                          ep, lane, ucs_status_string(inv_status));
            }
        }

        ucp_ep_refcount_add(ep, discard);
        ucp_worker_flush_ops_count_add(ep->worker, +1);

        ucs_trace("ep %p: lane %u failover extraction armed", ep, lane);
        ctx->lane_map     |= UCS_BIT(lane);
        *failover_lanes_p |= UCS_BIT(lane);
    }

    ucs_debug("ep %p: started failover for lanes 0x%" PRIx64, ep,
              (uint64_t)*failover_lanes_p);

    return UCS_OK;
}


void ucp_ep_failover_cancel_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;

    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL)) {
        return;
    }

    ctx           = ep->ext->failover.ctx;
    ctx->request_id = 0;
    ucs_debug("ep %p: canceling failover for lanes 0x%" PRIx64, ep,
              (uint64_t)(lane_map & ctx->lane_map));
    ucs_for_each_bit(lane, lane_map & ctx->lane_map) {
        lane_ctx = &ctx->lanes[lane];
        /* Completions are deferred on these endpoints, so they must be closed
         * rather than handed back to a CANCEL flush discard path. */
        ucp_ep_failover_replay_purge(lane_ctx, UCS_ERR_CANCELED);
        ucs_free(lane_ctx->rx_token);
        ucp_ep_failover_destroy_uct_ep(lane_ctx);
        memset(lane_ctx, 0, sizeof(*lane_ctx));
        ctx->lane_map &= ~UCS_BIT(lane);

        ucp_worker_flush_ops_count_add(ep->worker, -1);
        ucp_ep_refcount_remove(ep, discard);
    }

    if (ctx->lane_map == 0) {
        ucp_ep_failover_pending_purge(ctx, UCS_ERR_CANCELED);
        ucp_ep_failover_ctx_release(ep, ctx);
    }
}


static ucp_lane_map_t
ucp_ep_failover_pending_rx_lanes_internal(const ucp_ep_failover_ctx_t *ctx)
{
    ucp_lane_map_t lane_map = 0;
    ucp_lane_index_t lane;

    ucs_for_each_bit(lane, ctx->lane_map) {
        if (!(ctx->lanes[lane].flags & UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN)) {
            lane_map |= UCS_BIT(lane);
        }
    }

    return lane_map;
}


ucp_lane_map_t ucp_ep_failover_pending_rx_lanes(ucp_ep_h ep)
{
    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL)) {
        return 0;
    }

    return ucp_ep_failover_pending_rx_lanes_internal(ep->ext->failover.ctx);
}


void ucp_ep_failover_set_request_id(ucp_ep_h ep, uint64_t request_id)
{
    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL)) {
        return;
    }

    ep->ext->failover.ctx->request_id = request_id;
}


uint64_t ucp_ep_failover_get_request_id(ucp_ep_h ep)
{
    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL)) {
        return 0;
    }

    return ep->ext->failover.ctx->request_id;
}


ucs_status_t
ucp_ep_failover_apply_rx_tokens(ucp_ep_h ep, uint64_t request_id,
                                ucp_lane_map_t lane_map,
                                const uint8_t *token_lengths,
                                const void *tokens)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    void *rx_tokens[UCP_MAX_LANES]          = {NULL};
    uint8_t rx_token_lengths[UCP_MAX_LANES] = {0};
    ucs_status_t fallback_status[UCP_MAX_LANES];
    ucp_lane_map_t expected_lanes;
    ucp_lane_map_t apply_lanes;
    ucp_lane_map_t fallback_lanes = 0;
    ucp_lane_index_t lane;
    unsigned token_index = 0;
    size_t token_offset  = 0;
    uint64_t token_value;

    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL) ||
        (lane_map == 0) || (token_lengths == NULL)) {
        return UCS_OK;
    }

    ctx            = ep->ext->failover.ctx;
    expected_lanes = ucp_ep_failover_pending_rx_lanes_internal(ctx);
    apply_lanes    = lane_map & expected_lanes;
    if (apply_lanes == 0) {
        return UCS_OK;
    }

    if ((request_id != 0) && (ctx->request_id != 0) &&
        (request_id != ctx->request_id)) {
        ucs_debug("ep %p: ignoring ADDR RX tokens id 0x%" PRIx64
                  ", expected 0x%" PRIx64,
                  ep, request_id, ctx->request_id);
        return UCS_OK;
    }

    if (request_id != 0) {
        ctx->request_id = request_id;
    }

    ucs_debug("ep %p: applying ADDR RX tokens id 0x%" PRIx64
              " for lanes 0x%" PRIx64,
              ep, request_id, (uint64_t)apply_lanes);

    /* Lengths/tokens are ordered by the full lane_map from the wire message. */
    ucs_for_each_bit(lane, lane_map) {
        if (!(apply_lanes & UCS_BIT(lane))) {
            token_offset += token_lengths[token_index++];
            continue;
        }

        rx_token_lengths[lane] = token_lengths[token_index];
        if (rx_token_lengths[lane] == 0) {
            fallback_lanes       |= UCS_BIT(lane);
            fallback_status[lane] = UCS_ERR_UNSUPPORTED;
        } else {
            rx_tokens[lane] = ucs_malloc(rx_token_lengths[lane],
                                         "ep_failover_rx_token");
            if (rx_tokens[lane] == NULL) {
                fallback_lanes       |= UCS_BIT(lane);
                fallback_status[lane] = UCS_ERR_NO_MEMORY;
            } else if (tokens != NULL) {
                memcpy(rx_tokens[lane],
                       UCS_PTR_BYTE_OFFSET(tokens, token_offset),
                       rx_token_lengths[lane]);
            }
        }

        token_value = 0;
        if ((tokens != NULL) && (rx_token_lengths[lane] != 0)) {
            memcpy(&token_value, UCS_PTR_BYTE_OFFSET(tokens, token_offset),
                   ucs_min(rx_token_lengths[lane], sizeof(token_value)));
        }

        ucs_debug("ft psn ucp apply rx token ep %p id 0x%" PRIx64
                  " lane %u length %u raw 0x%" PRIx64 " fallback %d",
                  ep, request_id, lane, rx_token_lengths[lane], token_value,
                  !!(fallback_lanes & UCS_BIT(lane)));
        token_offset += token_lengths[token_index];
        ++token_index;
    }

    ucs_for_each_bit(lane, apply_lanes & ~fallback_lanes) {
        lane_ctx = &ctx->lanes[lane];
        ucs_free(lane_ctx->rx_token);
        lane_ctx->rx_token        = rx_tokens[lane];
        lane_ctx->rx_token_length = rx_token_lengths[lane];
        lane_ctx->flags          |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
        rx_tokens[lane]           = NULL;
    }

    ucs_for_each_bit(lane, fallback_lanes) {
        ucp_ep_failover_lane_close(ep, lane, fallback_status[lane]);
    }

    if (ep->ext->failover.ctx != NULL) {
        ucp_ep_failover_schedule(ep);
    }
    return UCS_OK;
}


static int
ucp_ep_failover_progress_remove_filter(const ucs_callbackq_elem_t *elem,
                                       void *arg)
{
    return (elem->cb == ucp_ep_failover_progress_cb) && (elem->arg == arg);
}


void ucp_ep_failover_cleanup(ucp_ep_h ep)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_lane_index_t lane;

    if (ep->ext == NULL) {
        return;
    }

    ucs_callbackq_remove_oneshot(&ep->worker->uct->progress_q, ep,
                                 ucp_ep_failover_progress_remove_filter, ep);
    ep->ext->failover.progress_scheduled = 0;

    ctx = ep->ext->failover.ctx;
    if (ctx != NULL) {
        ucs_for_each_bit(lane, ctx->lane_map) {
            ucp_ep_failover_replay_purge(&ctx->lanes[lane], UCS_ERR_CANCELED);
            ucs_free(ctx->lanes[lane].rx_token);
        }

        ucp_ep_failover_pending_purge(ctx, UCS_ERR_CANCELED);
        ucp_request_put(ctx->super_req);
        ucs_free(ctx);
        ep->ext->failover.ctx = NULL;
    }
}


static ucs_status_t
ucp_ep_failover_start_extracted_ops(ucp_ep_failover_lane_ctx_t *lane)
{
    ucp_proto_failover_replay_op_t *op;
    ucs_status_t status;
    unsigned remaining;

    remaining = lane->undelivered_count;
    while (!ucs_queue_is_empty(&lane->replay_queue)) {
        /* Pull before start: ucp_request_send may complete synchronously and
         * free the op (via replay_finish), which would UAF if it were still
         * the queue head when progress re-enters this function. */
        op = ucs_queue_pull_elem_non_empty(&lane->replay_queue,
                                           ucp_proto_failover_replay_op_t,
                                           queue);
        status = ucp_proto_failover_replay_op_start(lane->ep, lane->lane,
                                                    lane->ctx->super_req, op);
        if (status == UCS_ERR_NO_RESOURCE) {
            /* No usable lane yet (e.g. wireup proxy not installed). Retry from
             * the failover progress callback. */
            ucs_queue_push_head(&lane->replay_queue, &op->queue);
            ucp_ep_failover_schedule(lane->ep);
            return UCS_OK;
        } else if (status != UCS_OK) {
            ucp_proto_failover_replay_op_destroy(op, status);
            ucs_assert(lane->undelivered_count > 0);
            --lane->undelivered_count;
            ucp_ep_failover_replay_purge(lane, status);
            return status;
        }

        --remaining;
    }

    ucs_assert(remaining == 0 ||
               (remaining == lane->undelivered_count)); /* sync completions */
    return UCS_OK;
}


static ucs_status_t ucp_ep_failover_lanes_replay(ucp_ep_h ep)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (ep->ext->failover.ctx == NULL) {
        return UCS_OK;
    }

    ctx = ep->ext->failover.ctx;
    ucs_for_each_bit(lane, ctx->lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (!(lane_ctx->flags & UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED) ||
            (lane_ctx->status != UCS_OK)) {
            continue;
        }

        if (!ucs_queue_is_empty(&lane_ctx->replay_queue)) {
            status = ucp_ep_failover_start_extracted_ops(lane_ctx);
            if (status != UCS_OK) {
                return status;
            }
        }
    }

    /* Extracted ops are posted as pending-capable requests. Wait until they
     * complete, then flush and release the original UCT pending queue. */
    return ucp_ep_failover_flush_start(ep);
}


static ucs_status_t
ucp_ep_failover_extract_lane(ucp_ep_failover_lane_ctx_t *lane)
{
    uct_ep_outstanding_purge_params_t params;
    ucp_ep_failover_extract_arg_t extract_arg;
    ucs_status_t status;

    ucs_assertv(lane->rx_token_length > 0,
                "ep %p lane %u: rx token required for extract", lane->ep,
                lane->lane);
    ucs_assert(lane->rx_token != NULL);

    extract_arg.lane   = lane;
    extract_arg.status = UCS_OK;

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                        UCT_EP_OUTSTANDING_FIELD_CB |
                        UCT_EP_OUTSTANDING_FIELD_ARG;
    params.rx_token   = lane->rx_token;
    params.cb         = ucp_ep_failover_extract_cb;
    params.arg        = &extract_arg;

    status = uct_ep_outstanding_purge(lane->uct_ep, &params);
    if (status != UCS_OK) {
        ucs_debug("ep %p: lane %u outstanding extract failed: %s", lane->ep,
                  lane->lane, ucs_status_string(status));
        ucp_ep_failover_replay_purge(lane, status);
        ucp_ep_failover_lane_close(lane->ep, lane->lane, status);
        return status;
    }

    if (extract_arg.status != UCS_OK) {
        ucs_debug("ep %p: lane %u outstanding extract callback failed: %s",
                  lane->ep, lane->lane, ucs_status_string(extract_arg.status));
        ucp_ep_failover_replay_purge(lane, extract_arg.status);
        ucp_ep_failover_lane_close(lane->ep, lane->lane, extract_arg.status);
        return extract_arg.status;
    }

    lane->status = UCS_OK;
    lane->flags |= UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED;

    ucs_debug("ep %p: extracted %u outstanding operations from lane %u",
              lane->ep, lane->undelivered_count, lane->lane);

    /* Pending requests have no WQE/MSN and remain owned by the old UCT EP until
     * hardware outstanding extraction succeeds. Append them logically after
     * the extracted WQEs so replay preserves the original posting order. */
    ucp_ep_failover_pending_extract(lane);

    /* Extract transferred all user operation ownership. Destroying the old EP
     * moves its QP to ERR and lets the regular asynchronous QP GC wait for the
     * last WQE while replay proceeds on live lanes. */
    ucp_ep_failover_destroy_uct_ep(lane);

    /* Defer posting until lanes_replay: recovery may still be installing
     * wireup proxies, and capability-based lane selection needs them. */
    return UCS_OK;
}


void ucp_ep_failover_replay_completed(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                                      ucs_status_t status)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane;

    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL)) {
        return;
    }

    ctx = ep->ext->failover.ctx;
    if (!(ctx->lane_map & UCS_BIT(failed_lane))) {
        return;
    }

    lane = &ctx->lanes[failed_lane];
    ucs_assert(lane->undelivered_count > 0);
    --lane->undelivered_count;
    if ((status != UCS_OK) && (lane->status == UCS_OK)) {
        lane->status = status;
    }

    if ((ctx->flags & UCP_EP_FAILOVER_FLAG_ABORTED) &&
        (lane->undelivered_count == 0)) {
        ucp_ep_failover_lane_close(ep, failed_lane, ctx->status);
        return;
    }

    ucp_ep_failover_schedule(ep);
}


static void ucp_ep_failover_abort_all(ucp_ep_h ep, ucs_status_t status);


void ucp_ep_failover_abort(ucp_ep_h ep, ucs_status_t status)
{
    ucp_ep_failover_abort_all(ep, status);
}


static void ucp_ep_failover_abort_all(ucp_ep_h ep, ucs_status_t status)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane;

    ucs_assert(status != UCS_OK);
    ctx = ep->ext->failover.ctx;
    if (ctx == NULL) {
        return;
    }

    ctx->request_id = 0;
    ctx->status     = status;
    ctx->flags     |= UCP_EP_FAILOVER_FLAG_ABORTED;

    lane_map = ctx->lane_map;
    ucs_debug("ep %p: aborting failover for lanes 0x%" PRIx64 " status %s", ep,
              (uint64_t)lane_map, ucs_status_string(status));

    /* Release the lanes here rather than from the failover progress: an armed
     * lane cannot be handed to a flush-based discard, and waiting for the replay
     * flush would keep the worker flush operations pinned if that flush cannot
     * complete. Lanes with replayed operations still in flight are closed from
     * ucp_ep_failover_replay_completed(). */
    ucs_for_each_bit(lane, lane_map) {
        lane_ctx         = &ctx->lanes[lane];
        lane_ctx->status = status;
        ucp_ep_failover_replay_purge(lane_ctx, status);
        if (lane_ctx->undelivered_count == 0) {
            ucp_ep_failover_lane_close(ep, lane, status);
        }
    }
}


static ucs_status_t ucp_ep_failover_lanes_extract(ucp_ep_h ep)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (ep->ext->failover.ctx == NULL) {
        return UCS_OK;
    }

    ctx = ep->ext->failover.ctx;
    ucs_for_each_bit(lane, ctx->lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (!(lane_ctx->flags & UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN)) {
            continue;
        }

        if (!(lane_ctx->flags & UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED)) {
            status = ucp_ep_failover_extract_lane(lane_ctx);
            if (status != UCS_OK) {
                return status;
            }
        }
    }

    return UCS_OK;
}


static void
ucp_ep_failover_flush_complete(ucp_ep_failover_ctx_t *ctx, ucs_status_t status)
{
    ucp_ep_h ep = ctx->super_req->send.ep;

    ucs_assert(ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_STARTED);
    ucs_assert(!(ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_DONE));

    ctx->flags |= UCP_EP_FAILOVER_FLAG_FLUSH_DONE;
    if (ctx->flags & UCP_EP_FAILOVER_FLAG_ABORTED) {
        /* Failover was aborted while this flush was in flight and the flush
         * owned super_req until now. Lanes which still have replayed operations
         * in flight are released when those complete. */
        if (ctx->lane_map == 0) {
            ucp_ep_failover_ctx_release(ep, ctx);
        }

        return;
    }

    ctx->status = status;
    if (status == UCS_OK) {
        ucp_wireup_replay_pending_requests(ep, &ctx->pending_queue);
    } else {
        ucp_ep_failover_pending_purge(ctx, status);
    }

    ucp_ep_failover_schedule(ep);
}


static void ucp_ep_failover_flush_cb(ucp_request_t *req)
{
    ucp_request_t *super_req   = ucp_request_get_super(req);
    ucp_ep_failover_ctx_t *ctx = super_req->user_data;

    ucp_ep_failover_flush_complete(ctx, req->status);
    ucp_request_put(req);
}


static int ucp_ep_failover_all_wqes_reposted(ucp_ep_failover_ctx_t *ctx)
{
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;

    ucs_for_each_bit(lane, ctx->lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (!(lane_ctx->flags & UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED) ||
            (lane_ctx->undelivered_count != 0)) {
            return 0;
        }
    }

    return 1;
}


static ucs_status_t ucp_ep_failover_flush_start(ucp_ep_h ep)
{
    ucp_ep_failover_ctx_t *ctx = ep->ext->failover.ctx;
    ucs_status_ptr_t req;
    ucs_status_t status;

    if ((ctx == NULL) || (ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_STARTED) ||
        !ucp_ep_failover_all_wqes_reposted(ctx)) {
        return UCS_OK;
    }

    ctx->flags |= UCP_EP_FAILOVER_FLAG_FLUSH_STARTED;
    req = ucp_ep_flush_internal(ep, UCP_REQUEST_FLAG_RELEASED,
                                &ucp_request_null_param, ctx->super_req,
                                ucp_ep_failover_flush_cb, "failover repost",
                                0);
    if (UCS_PTR_IS_PTR(req)) {
        return UCS_OK;
    }

    status = UCS_PTR_STATUS(req);
    ucp_ep_failover_flush_complete(ctx, status);
    return status;
}


static int ucp_ep_failover_lanes_complete(ucp_ep_h ep)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane;
    ucs_status_t status;

    ctx = ep->ext->failover.ctx;
    if ((ctx == NULL)) {
        return 0;
    }

    if (!(ctx->flags & UCP_EP_FAILOVER_FLAG_FLUSH_DONE)) {
        return 0;
    }

    lane_map = ctx->lane_map;
    ucs_for_each_bit(lane, lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (!ucs_test_all_flags(
                    lane_ctx->flags,
                    UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN |
                            UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED |
                            UCP_EP_FAILOVER_LANE_FLAG_PENDING_EXTRACTED)) {
            continue;
        }

        if (lane_ctx->undelivered_count != 0) {
            continue;
        }

        status = (ctx->status == UCS_OK) ? lane_ctx->status : ctx->status;
        if (ucp_ep_failover_lane_complete(ctx, lane, status)) {
            return 1;
        }
    }

    return 0;
}


static unsigned ucp_ep_failover_progress_cb(void *arg)
{
    ucp_ep_h ep         = arg;
    ucp_worker_h worker = ep->worker;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);
    ep->ext->failover.progress_scheduled = 0;

    if ((ep->ext->failover.ctx != NULL) &&
        (ep->ext->failover.ctx->flags & UCP_EP_FAILOVER_FLAG_ABORTED)) {
        /* Aborted lanes are released by ucp_ep_failover_abort_all() and by the
         * completion of their replayed operations. */
        goto out;
    }

    status = ucp_ep_failover_lanes_extract(ep);
    if (status != UCS_OK) {
        ucp_ep_failover_abort_all(ep, status);
        goto out;
    }

    status = ucp_ep_failover_lanes_replay(ep);
    if (status != UCS_OK) {
        ucp_ep_failover_abort_all(ep, status);
        goto out;
    }

    ucp_ep_failover_lanes_complete(ep);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}


static void ucp_ep_failover_schedule(ucp_ep_h ep)
{
    if ((ep->ext == NULL) || ep->ext->failover.progress_scheduled) {
        return;
    }

    ep->ext->failover.progress_scheduled = 1;
    ucs_callbackq_add_oneshot(&ep->worker->uct->progress_q, ep,
                              ucp_ep_failover_progress_cb, ep);
}


ucp_lane_map_t ucp_ep_failover_test_pending_rx_lane_map(ucp_ep_h ep)
{
    return ucp_ep_failover_pending_rx_lanes(ep);
}
