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
#include <ucp/wireup/wireup_ep.h>
#include <uct/base/uct_iface.h>
#include <ucs/sys/ptr_arith.h>


typedef struct {
    ucp_ep_h         ep;
    ucp_lane_index_t lane;
    ucs_status_t     status;
} ucp_ep_failover_extract_arg_t;


enum ucp_ep_failover_lane_flags {
    UCP_EP_FAILOVER_LANE_FLAG_DRAINED   = UCS_BIT(0),
    UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN  = UCS_BIT(1),
    UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED = UCS_BIT(2)
};


/* Per failed-lane context used while failover to alternate lanes is in progress. */
typedef struct {
    ucp_ep_h                       ep;
    uct_ep_h                       uct_ep;
    ucp_lane_index_t               lane;
    ucp_rsc_index_t                rsc_index;
    void                           *rx_token;
    uint8_t                        rx_token_length;
    unsigned                       flags;
    ucs_status_t                   discard_status;
    ucp_ep_failover_lane_done_cb_t done_cb;
    void                           *done_arg;
} ucp_ep_failover_lane_ctx_t;


struct ucp_ep_failover_ctx {
    ucp_lane_map_t             lane_map;
    ucp_ep_failover_lane_ctx_t lanes[UCP_MAX_LANES];
};


static void ucp_ep_failover_schedule(ucp_ep_h ep);
static unsigned ucp_ep_failover_progress_cb(void *arg);

static void
ucp_ep_failover_extract_cb(const uct_ep_op_info_t *op_info, void *arg)
{
    ucp_ep_failover_extract_arg_t *extract_arg = arg;

    ucs_debug("ep %p: unsupported extracted failover op %d", extract_arg->ep,
              (int)op_info->operation);
    extract_arg->status = UCS_ERR_UNSUPPORTED;
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

    if (ep->ext != NULL) {
        ctx = ep->ext->failover.ctx;
        if ((ctx != NULL) && (ctx->lane_map & UCS_BIT(lane)) &&
            (ctx->lanes[lane].uct_ep != NULL)) {
            return ctx->lanes[lane].uct_ep;
        }
    }

    uct_ep = ucp_ep_get_lane(ep, lane);
    if ((uct_ep != NULL) && ucp_wireup_ep_test(uct_ep)) {
        return NULL;
    }

    return uct_ep;
}


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


static int ucp_ep_failover_lane_complete(ucp_ep_failover_ctx_t *ctx,
                                         ucp_lane_index_t lane_index,
                                         ucs_status_t status)
{
    ucp_ep_failover_lane_ctx_t *lane       = &ctx->lanes[lane_index];
    ucp_ep_h ep                            = lane->ep;
    ucp_worker_h worker                    = ep->worker;
    uct_ep_h uct_ep                        = lane->uct_ep;
    ucp_rsc_index_t rsc_index              = lane->rsc_index;
    ucp_ep_failover_lane_done_cb_t done_cb = lane->done_cb;
    void *done_arg                         = lane->done_arg;
    int failover_done;

    ucs_trace("ep %p: complete failover for lane %u status %s", ep, lane_index,
              ucs_status_string(status));

    ucp_ep_unprogress_uct_ep(ep, uct_ep, rsc_index);
    uct_ep_destroy(uct_ep);
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


ucs_status_t
ucp_ep_failover_add_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                          uct_ep_h *uct_eps, ucs_status_t discard_status,
                          ucp_ep_failover_lane_done_cb_t cb, void *arg,
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

        lane_ctx                 = &ctx->lanes[lane];
        lane_ctx->ep             = ep;
        lane_ctx->uct_ep         = uct_ep;
        lane_ctx->lane           = lane;
        lane_ctx->rsc_index      = ucp_ep_get_rsc_index(ep, lane);
        lane_ctx->discard_status = discard_status;
        lane_ctx->done_cb        = cb;
        lane_ctx->done_arg       = arg;

        ucp_ep_refcount_add(ep, discard);
        ucp_worker_flush_ops_count_add(ep->worker, +1);

        ucs_trace("ep %p: lane %u failover extraction armed", ep, lane);
        lane_ctx->flags   |= UCP_EP_FAILOVER_LANE_FLAG_DRAINED;
        ctx->lane_map     |= UCS_BIT(lane);
        *failover_lanes_p |= UCS_BIT(lane);
    }

    if (ctx->lane_map == 0) {
        ep->ext->failover.ctx = NULL;
        ucs_free(ctx);
    }

    if (*failover_lanes_p != 0) {
        ucp_ep_failover_schedule(ep);
    }

    return UCS_OK;
}


void ucp_ep_failover_abort_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                                 ucs_status_t status)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;

    if ((ep->ext == NULL) || (ep->ext->failover.ctx == NULL)) {
        return;
    }

    ctx = ep->ext->failover.ctx;
    ucs_for_each_bit(lane, lane_map & ctx->lane_map) {
        lane_ctx                 = &ctx->lanes[lane];
        lane_ctx->discard_status = status;
        lane_ctx->flags         |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
    }

    ucp_ep_failover_schedule(ep);
}


ucs_status_t ucp_ep_failover_query_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map)
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
        if (lane_ctx->rx_token_length > 0) {
            lane_ctx->rx_token = ucs_malloc(lane_ctx->rx_token_length,
                                            "ep_failover_rx_token");
            if (lane_ctx->rx_token == NULL) {
                lane_ctx->rx_token_length = 0;
                lane_ctx->discard_status  = UCS_ERR_NO_MEMORY;
            } else {
                memcpy(lane_ctx->rx_token,
                       UCS_PTR_BYTE_OFFSET(tokens, token_offset),
                       lane_ctx->rx_token_length);
            }
        }

        lane_ctx->flags |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
        token_offset    += token_lengths[token_index];
        ++token_index;
    }

    missing_lanes = ctx->lane_map & ~lane_state->lane_map;
    ucs_for_each_bit(lane, missing_lanes) {
        ucs_trace("ep %p: lane %u missing in lane_state reply", ep, lane);
        lane_ctx                  = &ctx->lanes[lane];
        lane_ctx->rx_token_length = 0;
        lane_ctx->flags          |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
    }

    ucp_ep_failover_schedule(ep);
    return UCS_OK;
}


static ucs_status_t
ucp_ep_failover_extract_lane(ucp_ep_failover_lane_ctx_t *lane)
{
    uct_ep_outstanding_extract_params_t params;
    ucp_ep_failover_extract_arg_t extract_arg;
    ucs_status_t status;

    if (lane->rx_token_length == 0) {
        lane->flags |= UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED;
        return UCS_OK;
    }

    extract_arg.ep          = lane->ep;
    extract_arg.lane        = lane->lane;
    extract_arg.status      = UCS_OK;

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
        lane->discard_status = status;
    } else if (extract_arg.status != UCS_OK) {
        lane->discard_status = extract_arg.status;
    }

    lane->flags |= UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED;
    return status;
}


static void ucp_ep_failover_abort_all(ucp_ep_h ep, ucs_status_t status)
{
    ucp_ep_failover_ctx_t *ctx;
    ucp_ep_failover_lane_ctx_t *lane_ctx;
    ucp_lane_index_t lane;

    ucs_assert(status != UCS_OK);
    ep->ext->failover.query_lane_map = 0;

    ctx = ep->ext->failover.ctx;
    if (ctx == NULL) {
        return;
    }

    ucs_for_each_bit(lane, ctx->lane_map) {
        lane_ctx                 = &ctx->lanes[lane];
        lane_ctx->discard_status = status;
        lane_ctx->flags         |= UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN;
    }

    ucp_ep_failover_schedule(ep);
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
        if (!ucs_test_all_flags(lane_ctx->flags,
                                UCP_EP_FAILOVER_LANE_FLAG_DRAINED |
                                        UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN)) {
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


static int ucp_ep_failover_query_drained(ucp_ep_h ep)
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

    if ((ep->ext->failover.ctx == NULL) ||
        !ucp_ep_failover_query_drained(ep)) {
        return 0;
    }

    ctx      = ep->ext->failover.ctx;
    lane_map = ctx->lane_map;
    ucs_for_each_bit(lane, lane_map) {
        lane_ctx = &ctx->lanes[lane];
        if (!ucs_test_all_flags(lane_ctx->flags,
                                UCP_EP_FAILOVER_LANE_FLAG_DRAINED |
                                        UCP_EP_FAILOVER_LANE_FLAG_RX_TOKEN |
                                        UCP_EP_FAILOVER_LANE_FLAG_EXTRACTED)) {
            continue;
        }

        status = lane_ctx->discard_status;
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
