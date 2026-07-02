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
#include <ucp/core/ucp_worker.h>
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


/* Per failed-lane context used while failover to alternate lanes is in progress. */
typedef struct ucp_ep_failover_lane_ctx {
    ucp_ep_h                       ep;
    uct_ep_h                       uct_ep;
    ucp_lane_index_t               lane;
    ucp_rsc_index_t                rsc_index;
    void                           *rx_token;
    uint8_t                        rx_token_length;
    unsigned                       flags;
    ucs_status_t                   status;
    ucp_ep_failover_lane_done_cb_t done_cb;
    ucp_ep_failover_lane_failed_cb_t failed_cb;
    void                           *done_arg;
    /* Copied undelivered WQEs precede the extracted unposted requests. */
    ucs_queue_head_t               replay_queue;
    unsigned                       undelivered_count;
} ucp_ep_failover_lane_ctx_t;


typedef struct {
    ucp_ep_failover_lane_ctx_t *lane;
    ucs_status_t               status;
} ucp_ep_failover_extract_arg_t;


struct ucp_ep_failover_ctx {
    ucp_lane_map_t             lane_map;
    ucp_ep_failover_lane_ctx_t lanes[UCP_MAX_LANES];
};


static void ucp_ep_failover_schedule(ucp_ep_h ep);
static unsigned ucp_ep_failover_progress_cb(void *arg);

static int ucp_ep_failover_lane_token_supported(ucp_ep_h ep, uct_ep_h uct_ep,
                                                ucp_lane_index_t lane)
{
    uct_iface_attr_v2_t attr;
    ucs_status_t status;

    if ((uct_ep == NULL) || ucp_wireup_ep_test(uct_ep) ||
        (ucp_ep_get_rsc_index(ep, lane) == UCP_NULL_RESOURCE)) {
        return 0;
    }

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS;
    status          = uct_iface_query_v2(uct_ep->iface, &attr);
    if (status != UCS_OK) {
        return 0;
    }

    return !!(attr.cap.flags & UCT_IFACE_FLAG_V2_QUERY_TOKEN);
}


void ucp_ep_failover_arm_lane(ucp_ep_h ep, ucp_lane_index_t lane,
                              uct_ep_h uct_ep)
{
    if ((ep->ext == NULL) ||
        !ucp_ep_err_mode_eq(ep, UCP_ERR_HANDLING_MODE_FAILOVER) ||
        (ucp_ep_config(ep)->key.dst_version <
         UCP_WIREUP_LANE_STATE_MIN_VERSION) ||
        (uct_ep == NULL) || ucp_wireup_ep_test(uct_ep)) {
        return;
    }

    if (!ucp_ep_failover_lane_token_supported(ep, uct_ep, lane)) {
        return;
    }

    if (uct_ep_failover_arm(uct_ep) != UCS_OK) {
        ucs_debug("ep %p: lane %u uct_ep %p does not support failover arm", ep,
                  lane, uct_ep);
        return;
    }

    ucs_trace("ep %p: lane %u uct_ep %p armed for outstanding extract", ep,
              lane, uct_ep);
}

static void ucp_ep_failover_replay_purge(ucp_ep_failover_lane_ctx_t *lane,
                                         ucs_status_t status)
{
    ucp_proto_failover_replay_op_t *op;
    uct_pending_req_t *uct_req;

    while (lane->undelivered_count > 0) {
        op = ucs_queue_pull_elem_non_empty(&lane->replay_queue,
                                           ucp_proto_failover_replay_op_t,
                                           queue);
        ucp_proto_failover_replay_op_destroy(op);
        --lane->undelivered_count;
    }

    ucs_queue_for_each_extract(uct_req, &lane->replay_queue, priv, 1) {
        ucp_ep_err_pending_purge(uct_req, UCS_STATUS_PTR(status));
    }
}


static void ucp_ep_failover_pending_extract(ucp_ep_failover_lane_ctx_t *lane)
{
    ucs_assert(lane->uct_ep != NULL);
    ucs_assert(lane->flags & UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED);
    ucs_assert(!(lane->flags & UCP_EP_FAILOVER_LANE_FLAG_PENDING_EXTRACTED));

    uct_ep_pending_purge(lane->uct_ep, ucp_request_purge_enqueue_cb,
                         &lane->replay_queue);
    lane->flags |= UCP_EP_FAILOVER_LANE_FLAG_PENDING_EXTRACTED;
}


static void
ucp_ep_failover_destroy_uct_ep(ucp_ep_failover_lane_ctx_t *lane)
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

    status = ucp_proto_failover_replay_op_create(op_info, &op);
    if (status != UCS_OK) {
        ucs_debug("ep %p: failed to save extracted failover op %d: %s",
                  extract_arg->lane->ep, (int)op_info->operation,
                  ucs_status_string(status));
        extract_arg->status = status;
        return;
    }

    ucs_queue_push(&extract_arg->lane->replay_queue, &op->queue);
    ++extract_arg->lane->undelivered_count;
}


void ucp_ep_failover_init(ucp_ep_h ep)
{
    if (ep->ext == NULL) {
        return;
    }

    ep->ext->failover.query_lane_map     = 0;
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

    ucs_trace("ep %p: complete failover for lane %u status %s", ep, lane_index,
              ucs_status_string(status));

    ucp_ep_failover_destroy_uct_ep(lane);
    ucs_assert(ucs_queue_is_empty(&lane->replay_queue));
    ucs_assert(lane->undelivered_count == 0);
    ucs_free(lane->rx_token);

    ctx->lane_map &= ~UCS_BIT(lane_index);
    memset(lane, 0, sizeof(*lane));
    failover_done = (ctx->lane_map == 0);
    if (failover_done) {
        ep->ext->failover.ctx = NULL;
        ucs_free(ctx);
    }

    ucp_worker_flush_ops_count_add(worker, -1);
    if (done_cb != NULL) {
        done_cb(NULL, status, done_arg);
    }

    ucp_ep_refcount_remove(ep, discard);
    return failover_done;
}


static void ucp_ep_failover_lane_fallback_discard(ucp_ep_h ep,
                                                  ucp_lane_index_t lane_index,
                                                  ucs_status_t discard_status)
{
    ucp_ep_failover_ctx_t *ctx = ep->ext->failover.ctx;
    ucp_ep_failover_lane_ctx_t *lane;
    ucp_ep_failover_lane_done_cb_t done_cb;
    ucp_ep_failover_lane_failed_cb_t failed_cb;
    ucp_rsc_index_t rsc_index;
    uct_ep_h uct_ep;
    void *done_arg;
    ucs_status_t status;

    if ((ctx == NULL) || !(ctx->lane_map & UCS_BIT(lane_index))) {
        return;
    }

    lane      = &ctx->lanes[lane_index];
    uct_ep    = lane->uct_ep;
    rsc_index = lane->rsc_index;
    done_cb   = lane->done_cb;
    failed_cb = lane->failed_cb;
    done_arg  = lane->done_arg;

    ucs_trace("ep %p: fallback discard for failover lane %u status %s", ep,
              lane_index, ucs_status_string(discard_status));

    ucp_ep_failover_replay_purge(lane, discard_status);
    ucs_free(lane->rx_token);
    memset(lane, 0, sizeof(*lane));
    ctx->lane_map &= ~UCS_BIT(lane_index);
    if (ctx->lane_map == 0) {
        ep->ext->failover.ctx = NULL;
        ucs_free(ctx);
    }

    if (failed_cb != NULL) {
        failed_cb(discard_status, done_arg);
    }

    status = ucp_worker_discard_uct_ep(ep, uct_ep, rsc_index,
                                       UCT_FLUSH_FLAG_CANCEL,
                                       ucp_ep_err_pending_purge,
                                       UCS_STATUS_PTR(discard_status), done_cb,
                                       done_arg);
    if ((status != UCS_OK) && (status != UCS_INPROGRESS)) {
        ucs_debug("ep %p: failed to discard failover lane %u uct_ep %p: %s", ep,
                  lane_index, uct_ep, ucs_status_string(status));
        ucp_ep_unprogress_uct_ep(ep, uct_ep, rsc_index);
        uct_ep_destroy(uct_ep);
        if (done_cb != NULL) {
            done_cb(NULL, discard_status, done_arg);
        }
    }

    ucp_worker_flush_ops_count_add(ep->worker, -1);
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
    if ((ep->ext == NULL) ||
        !ucp_ep_err_mode_eq(ep, UCP_ERR_HANDLING_MODE_FAILOVER) ||
        (ucp_ep_config(ep)->key.dst_version <
         UCP_WIREUP_LANE_STATE_MIN_VERSION)) {
        return UCS_ERR_UNSUPPORTED;
    }

    ctx = ep->ext->failover.ctx;
    if (ctx == NULL) {
        ctx = ucs_calloc(1, sizeof(*ctx), "ep_failover_ctx");
        if (ctx == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        ep->ext->failover.ctx = ctx;
    }

    ucs_for_each_bit(lane, lane_map) {
        uct_ep = uct_eps[lane];
        if (!ucp_ep_failover_lane_token_supported(ep, uct_ep, lane) ||
            (ctx->lane_map & UCS_BIT(lane))) {
            continue;
        }

        lane_ctx                    = &ctx->lanes[lane];
        lane_ctx->ep                = ep;
        lane_ctx->uct_ep            = uct_ep;
        lane_ctx->lane              = lane;
        lane_ctx->rsc_index         = ucp_ep_get_rsc_index(ep, lane);
        lane_ctx->status            = UCS_OK;
        lane_ctx->done_cb           = cb;
        lane_ctx->failed_cb         = failed_cb;
        lane_ctx->done_arg          = arg;
        ucs_queue_head_init(&lane_ctx->replay_queue);
        lane_ctx->undelivered_count = 0;

        ucp_ep_refcount_add(ep, discard);
        ucp_worker_flush_ops_count_add(ep->worker, +1);

        ucs_trace("ep %p: lane %u failover extraction armed", ep, lane);
        ctx->lane_map     |= UCS_BIT(lane);
        *failover_lanes_p |= UCS_BIT(lane);
    }

    if (ctx->lane_map == 0) {
        ep->ext->failover.ctx = NULL;
        ucs_free(ctx);
    }

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

    ctx                               = ep->ext->failover.ctx;
    ep->ext->failover.query_lane_map &= ~lane_map;
    ucs_for_each_bit(lane, lane_map & ctx->lane_map) {
        lane_ctx = &ctx->lanes[lane];
        ucp_ep_failover_replay_purge(lane_ctx, UCS_ERR_CANCELED);
        ucs_free(lane_ctx->rx_token);
        memset(lane_ctx, 0, sizeof(*lane_ctx));
        ctx->lane_map &= ~UCS_BIT(lane);

        ucp_worker_flush_ops_count_add(ep->worker, -1);
        ucp_ep_refcount_remove(ep, discard);
    }

    if (ctx->lane_map == 0) {
        ep->ext->failover.ctx = NULL;
        ucs_free(ctx);
    }
}


ucs_status_t
ucp_ep_failover_lanes_schedule(ucp_ep_h ep, ucp_lane_map_t lane_map)
{
    if ((ep->ext == NULL) || (lane_map == 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ep->ext->failover.query_lane_map |= lane_map;
    ucp_ep_failover_schedule(ep);
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
    ep->ext->failover.query_lane_map     = 0;

    ctx = ep->ext->failover.ctx;
    if (ctx != NULL) {
        ucs_for_each_bit(lane, ctx->lane_map) {
            ucp_ep_failover_replay_purge(&ctx->lanes[lane], UCS_ERR_CANCELED);
            ucs_free(ctx->lanes[lane].rx_token);
        }

        ucs_free(ctx);
        ep->ext->failover.ctx = NULL;
    }
}


ucs_status_t
ucp_ep_failover_on_lane_state(ucp_ep_h ep,
                              const ucp_wireup_lane_state_t *lane_state)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    const uint8_t *token_lengths;
    const void *tokens;
    ucp_lane_map_t missing_lanes;
    ucp_lane_index_t lane;
    unsigned token_index = 0;
    size_t token_offset  = 0;
    unsigned num_tokens;

    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL) ||
        !ucp_ep_err_mode_eq(ep, UCP_ERR_HANDLING_MODE_FAILOVER)) {
        return UCS_OK;
    }

    num_tokens    = ucp_wireup_lane_state_num_tokens(lane_state);
    ctx           = ep->ext->failover.ctx;
    token_lengths = ucp_wireup_lane_state_token_lengths(lane_state);
    tokens = UCS_PTR_BYTE_OFFSET(lane_state, sizeof(*lane_state) + num_tokens);

    ucs_for_each_bit(lane, lane_state->lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (!(ctx->lane_map & UCS_BIT(lane))) {
            token_offset += token_lengths[token_index];
            ++token_index;
            continue;
        }

        ucs_free(lane_ctx->rx_token);
        lane_ctx->rx_token        = NULL;
        lane_ctx->rx_token_length = token_lengths[token_index];
        if (lane_ctx->rx_token_length == 0) {
            ucs_debug("ep %p: lane %u missing rx token in lane_state reply", ep,
                      lane);
            ucp_ep_failover_lane_fallback_discard(ep, lane,
                                                  UCS_ERR_UNSUPPORTED);
            token_offset += token_lengths[token_index];
            ++token_index;
            continue;
        }

        lane_ctx->rx_token = ucs_malloc(lane_ctx->rx_token_length,
                                        "ep_failover_rx_token");
        if (lane_ctx->rx_token == NULL) {
            ucp_ep_failover_lane_fallback_discard(ep, lane, UCS_ERR_NO_MEMORY);
            token_offset += token_lengths[token_index];
            ++token_index;
            continue;
        }

        memcpy(lane_ctx->rx_token,
               UCS_PTR_BYTE_OFFSET(tokens, token_offset),
               lane_ctx->rx_token_length);
        lane_ctx->flags |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
        token_offset    += token_lengths[token_index];
        ++token_index;
    }

    missing_lanes = ctx->lane_map & ~lane_state->lane_map;
    ucs_for_each_bit(lane, missing_lanes) {
        ucs_trace("ep %p: lane %u missing in lane_state reply", ep, lane);
        ucp_ep_failover_lane_fallback_discard(ep, lane, UCS_ERR_NOT_CONNECTED);
    }

    if (ep->ext->failover.ctx != NULL) {
        ucp_ep_failover_schedule(ep);
    }
    return UCS_OK;
}


static ucs_status_t
ucp_ep_failover_extract_lane(ucp_ep_failover_lane_ctx_t *lane)
{
    uct_ep_outstanding_extract_params_t params;
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
                        UCT_EP_OUTSTANDING_FIELD_FLAGS;
    params.rx_token   = lane->rx_token;
    params.cb         = ucp_ep_failover_extract_cb;
    params.arg        = &extract_arg;
    params.flags      = UCT_EP_OUTSTANDING_FLAG_COMPLETE_DELIVERED;

    status = uct_ep_outstanding_extract(lane->uct_ep, &params);
    if (status != UCS_OK) {
        ucs_debug("ep %p: lane %u outstanding extract failed: %s", lane->ep,
                  lane->lane, ucs_status_string(status));
        ucp_ep_failover_lane_fallback_discard(lane->ep, lane->lane, status);
        return status;
    }

    if (extract_arg.status != UCS_OK) {
        ucs_debug("ep %p: lane %u outstanding extract callback failed: %s",
                  lane->ep, lane->lane,
                  ucs_status_string(extract_arg.status));
        ucp_ep_failover_lane_fallback_discard(lane->ep, lane->lane,
                                              extract_arg.status);
        return extract_arg.status;
    }

    lane->status = UCS_OK;
    lane->flags |= UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED;

    /* Pending requests have no WQE/MSN and remain owned by the old UCT EP until
     * hardware outstanding extraction succeeds. Append them logically after
     * the extracted WQEs so replay preserves the original posting order. */
    ucp_ep_failover_pending_extract(lane);

    /* Extract transferred all user operation ownership. Destroying the old EP
     * moves its QP to ERR and lets the regular asynchronous QP GC wait for the
     * last WQE while replay proceeds on live lanes. */
    ucp_ep_failover_destroy_uct_ep(lane);

    return UCS_OK;
}


static void ucp_ep_failover_abort_all(ucp_ep_h ep, ucs_status_t status)
{
    ucp_lane_map_t fallback_lanes = 0;
    int complete_lanes            = 0;
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane;

    ucs_assert(status != UCS_OK);
    ep->ext->failover.query_lane_map = 0;

    ctx = ep->ext->failover.ctx;
    if (ctx == NULL) {
        return;
    }

    lane_map = ctx->lane_map;
    ucs_for_each_bit(lane, lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (lane_ctx->flags & UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED) {
            lane_ctx->status = status;
            lane_ctx->flags  |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
            ucp_ep_failover_replay_purge(lane_ctx, status);
            complete_lanes = 1;
        } else {
            fallback_lanes |= UCS_BIT(lane);
        }
    }

    ucs_for_each_bit(lane, fallback_lanes) {
        ucp_ep_failover_lane_fallback_discard(ep, lane, status);
    }

    if (complete_lanes) {
        ucp_ep_failover_schedule(ep);
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


static ucs_status_t ucp_ep_failover_replay_lane(ucp_ep_failover_lane_ctx_t *lane)
{
    ucp_proto_failover_replay_op_t *op;
    ucs_status_t status;

    while (lane->undelivered_count > 0) {
        op     = ucs_queue_head_elem_non_empty(&lane->replay_queue,
                                               ucp_proto_failover_replay_op_t,
                                               queue);
        status = ucp_proto_failover_replay_op_progress(lane->ep, lane->lane,
                                                       op);
        if (status == UCS_ERR_NO_RESOURCE) {
            ucp_ep_failover_schedule(lane->ep);
            return UCS_OK;
        } else if (status != UCS_OK) {
            ucs_debug("ep %p: lane %u failed to replay extracted op %d: %s",
                      lane->ep, lane->lane, (int)op->info.operation,
                      ucs_status_string(status));
            lane->status = status;
            ucp_ep_failover_replay_purge(lane, status);
            return status;
        }

        op = ucs_queue_pull_elem_non_empty(&lane->replay_queue,
                                           ucp_proto_failover_replay_op_t,
                                           queue);
        ucp_proto_failover_replay_op_destroy(op);
        --lane->undelivered_count;
    }

    if (!ucs_queue_is_empty(&lane->replay_queue)) {
        ucp_wireup_replay_pending_requests(lane->ep, &lane->replay_queue);
    }

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

        status = ucp_ep_failover_replay_lane(lane_ctx);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}


static int ucp_ep_failover_queries_done(ucp_ep_h ep)
{
    return ep->ext->failover.query_lane_map == 0;
}


static int ucp_ep_failover_lanes_complete(ucp_ep_h ep)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if ((ep->ext->failover.ctx == NULL) || !ucp_ep_failover_queries_done(ep)) {
        return 0;
    }

    ctx      = ep->ext->failover.ctx;
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

        if (!ucs_queue_is_empty(&lane_ctx->replay_queue)) {
            continue;
        }

        status = lane_ctx->status;
        if (ucp_ep_failover_lane_complete(ctx, lane, status)) {
            return 1;
        }
    }

    return 0;
}


static ucs_status_t ucp_ep_failover_query_progress(ucp_ep_h ep)
{
    ucp_lane_map_t lane_map;
    ucs_status_t status;

    if (ep->ext->failover.query_lane_map == 0) {
        return UCS_OK;
    }

    lane_map = ep->ext->failover.query_lane_map;
    status   = ucp_wireup_send_query_lane_state(ep, lane_map);
    if (status == UCS_OK) {
        ep->ext->failover.query_lane_map &= ~lane_map;
    } else if (status == UCS_ERR_NO_RESOURCE) {
        ucp_ep_failover_schedule(ep);
    } else {
        ucs_debug("ep %p: failed to query failover lane state for lanes 0x%lx: "
                  "%s",
                  ep, lane_map, ucs_status_string(status));
    }

    return status;
}


static unsigned ucp_ep_failover_progress_cb(void *arg)
{
    ucp_ep_h ep         = arg;
    ucp_worker_h worker = ep->worker;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);
    ep->ext->failover.progress_scheduled = 0;

    status = ucp_ep_failover_query_progress(ep);
    if ((status != UCS_OK) && (status != UCS_ERR_NO_RESOURCE)) {
        ucp_ep_failover_abort_all(ep, status);
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


ucp_lane_map_t ucp_ep_failover_test_query_lane_map(ucp_ep_h ep)
{
    return (ep->ext == NULL) ? 0 : ep->ext->failover.query_lane_map;
}


ucs_status_t
ucp_ep_failover_test_validate_lane_state(ucp_ep_h ep,
                                         const ucp_wireup_lane_state_t *state,
                                         size_t length)
{
    return ucp_wireup_lane_state_validate(ep, state, length);
}
