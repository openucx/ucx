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
#include <uct/api/v2/uct_v2.h>
#include <inttypes.h>


static unsigned ucp_ep_failover_progress_cb(void *arg);


static int ucp_ep_failover_is_token_supported(uct_ep_h uct_ep)
{
    uct_iface_attr_v2_t attr;
    ucs_status_t status;

    if ((uct_ep == NULL) || ucp_wireup_ep_test(uct_ep)) {
        return 0;
    }

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH;
    status          = uct_iface_query_v2(uct_ep->iface, &attr);
    if (status != UCS_OK) {
        return 0;
    }

    return (attr.cap.flags & UCT_IFACE_FLAG_V2_QUERY_TOKEN) &&
           (attr.tx_token_length > 0) && (attr.tx_token_length <= UINT8_MAX);
}

int ucp_ep_failover_in_progress(ucp_ep_h ep)
{
    const ucp_ep_recovery_arg_t *rec;

    ucs_assert(ep->ext != NULL);
    rec = ep->ext->recovery_arg;
    return (rec != NULL) && (rec->failover.lane_map != 0);
}

static void ucp_ep_failover_lane_close(ucp_ep_h ep,
                                       ucp_lane_index_t lane_index,
                                       ucs_status_t discard_status)
{
    ucp_ep_recovery_arg_t *arg = ep->ext->recovery_arg;
    ucp_worker_h worker        = ep->worker;
    ucp_ep_failover_lane_t *lane;
    ucp_send_nbx_callback_t done_cb;
    ucp_rsc_index_t rsc_index;
    uct_ep_h uct_ep;
    void *done_arg;

    ucs_assert(arg != NULL);
    ucs_assert(arg->failover.lane_map & UCS_BIT(lane_index));
    ucs_assert(discard_status != UCS_OK);

    lane      = &arg->failover.lanes[lane_index];
    uct_ep    = lane->uct_ep;
    rsc_index = lane->rsc_index;
    done_cb   = lane->done_cb;
    done_arg  = lane->done_arg;

    ucs_assert(uct_ep != NULL);
    ucs_debug("ep %p: closing failover lane %u status %s", ep, lane_index,
              ucs_status_string(discard_status));

    memset(lane, 0, sizeof(*lane));
    arg->failover.lane_map &= ~UCS_BIT(lane_index);
    if (arg->failover.lane_map == 0) {
        arg->failover.status = UCS_OK;
    }

    uct_ep_pending_purge(uct_ep, ucp_ep_err_pending_purge,
                         UCS_STATUS_PTR(discard_status));
    ucp_ep_unprogress_uct_ep(ep, uct_ep, rsc_index);
    uct_ep_destroy(uct_ep);

    if (done_cb != NULL) {
        done_cb(NULL, discard_status, done_arg);
    }

    ucp_worker_flush_ops_count_add(worker, -1);
    ucp_ep_refcount_remove(ep, discard);
}

static int
ucp_ep_failover_progress_remove_filter(const ucs_callbackq_elem_t *elem,
                                       void *arg)
{
    return (elem->cb == ucp_ep_failover_progress_cb) && (elem->arg == arg);
}

static unsigned ucp_ep_failover_progress_cb(void *arg)
{
    ucp_ep_h ep         = arg;
    ucp_worker_h worker = ep->worker;
    ucp_ep_recovery_arg_t *rec;

    UCS_ASYNC_BLOCK(&worker->async);
    rec = ep->ext->recovery_arg;
    if ((rec != NULL) && (rec->failover.lane_map != 0)) {
        ucp_ep_failover_abort(ep, rec->failover.status);
        if (!(ep->flags & UCP_EP_FLAG_FAILED)) {
            ucp_ep_recovery_arm(ep);
        }

    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

ucs_status_t
ucp_ep_failover_add_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                          uct_ep_h *uct_eps, ucp_send_nbx_callback_t cb,
                          void *arg)
{
    uct_ep_invalidate_params_t inv_params = {0};
    ucp_ep_recovery_arg_t *rec;
    ucp_ep_failover_lane_t *lane_ctx;
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;
    ucs_status_t inv_status;

    ucs_assert(ep->ext != NULL);
    ucs_assert(ucp_ep_err_mode_eq(ep, UCP_ERR_HANDLING_MODE_FAILOVER));
    ucs_assert(lane_map != 0);

    rec = ep->ext->recovery_arg;
    ucs_for_each_bit(lane, lane_map) {
        uct_ep = uct_eps[lane];
        if ((ucp_ep_get_rsc_index(ep, lane) == UCP_NULL_RESOURCE) ||
            !ucp_ep_failover_is_token_supported(uct_ep)) {
            return UCS_ERR_UNSUPPORTED;
        }

        ucs_assert((rec == NULL) ||
                   !(rec->failover.lane_map & UCS_BIT(lane)));
    }

    if (rec == NULL) {
        rec = ucs_calloc(1, sizeof(*rec), "ucp_ep_recovery_arg");
        if (rec == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        ep->ext->recovery_arg = rec;
    }

    ucs_for_each_bit(lane, lane_map) {
        uct_ep              = uct_eps[lane];
        lane_ctx            = &rec->failover.lanes[lane];
        lane_ctx->uct_ep    = uct_ep;
        lane_ctx->rsc_index = ucp_ep_get_rsc_index(ep, lane);
        lane_ctx->done_cb   = cb;
        lane_ctx->done_arg  = arg;

        ucp_ep_refcount_add(ep, discard);
        ucp_worker_flush_ops_count_add(ep->worker, +1);

        ucs_trace("ep %p: lane %u failover armed", ep, lane);
        rec->failover.lane_map |= UCS_BIT(lane);

        /* Invalidate so an unaware peer also gets an error CQE.
         * Already-ERR QPs may fail modify_qp; that is not fatal. */
        inv_status = uct_ep_invalidate(uct_ep, &inv_params);
        if ((inv_status != UCS_OK) && (inv_status != UCS_ERR_UNSUPPORTED)) {
            ucs_debug("ep %p: lane %u invalidate: %s", ep, lane,
                      ucs_status_string(inv_status));
        }
    }

    ucs_debug("ep %p: started failover for lanes 0x%" PRIx64, ep,
              (uint64_t)lane_map);

    return UCS_OK;
}

void ucp_ep_failover_cleanup(ucp_ep_h ep)
{
    ucs_assert(ep->ext != NULL);
    ucs_assert((ep->ext->recovery_arg == NULL) ||
               (ep->ext->recovery_arg->failover.lane_map == 0));

    ucs_callbackq_remove_oneshot(&ep->worker->uct->progress_q, ep,
                                 ucp_ep_failover_progress_remove_filter, ep);
}

void ucp_ep_failover_abort(ucp_ep_h ep, ucs_status_t status)
{
    ucp_ep_recovery_arg_t *rec;
    ucp_lane_map_t lane_map;
    ucp_lane_index_t lane;

    ucs_assert(status != UCS_OK);
    ucs_assert(ep->ext != NULL);

    rec = ep->ext->recovery_arg;
    if ((rec == NULL) || (rec->failover.lane_map == 0)) {
        return;
    }

    rec->failover.status = status;
    lane_map             = rec->failover.lane_map;

    ucs_debug("ep %p: aborting failover for lanes 0x%" PRIx64 " status %s", ep,
              (uint64_t)lane_map, ucs_status_string(status));

    ucs_for_each_bit(lane, lane_map) {
        ucp_ep_failover_lane_close(ep, lane, status);
    }
}

void ucp_ep_failover_schedule_abort(ucp_ep_h ep, ucs_status_t status)
{
    ucp_ep_recovery_arg_t *rec;

    ucs_assert(status != UCS_OK);
    ucs_assert(ep->ext != NULL);

    rec = ep->ext->recovery_arg;
    ucs_assert(rec != NULL);
    ucs_assert(rec->failover.lane_map != 0);

    if (rec->failover.status != UCS_OK) {
        rec->failover.status = status;
        return;
    }

    rec->failover.status = status;
    ucs_callbackq_add_oneshot(&ep->worker->uct->progress_q, ep,
                              ucp_ep_failover_progress_cb, ep);
}
