/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_EP_H
#define UCT_RC_EP_H

#include "rc_iface.h"

#include <uct/api/uct.h>


enum {
    UCT_RC_EP_STAT_QP_FULL,
    UCT_RC_EP_STAT_SINGAL,
    UCT_RC_EP_STAT_LAST
};


/*
 * Macro to generate functions for AMO completions.
 */
#define UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(_num_bits, _is_be) \
    uct_rc_ep_atomic_handler_##_num_bits##_be##_is_be

/*
 * Check for send resources
 */
#define UCT_RC_CHECK_RES(_iface, _ep) \
    if (!uct_rc_iface_have_tx_cqe_avail(_iface)) { \
        UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_RC_IFACE_STAT_NO_CQE, 1); \
        UCT_TL_IFACE_STAT_TX_NO_RES(&(_iface)->super.super); \
        return UCS_ERR_NO_RESOURCE; \
    } \
    if ((_ep)->available <= 0) { \
        UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_RC_EP_STAT_QP_FULL, 1); \
        UCT_TL_IFACE_STAT_TX_NO_RES(&(_iface)->super.super); \
        return UCS_ERR_NO_RESOURCE; \
    }


struct uct_rc_ep {
    uct_base_ep_t       super;
    struct ibv_qp       *qp;
    ucs_queue_head_t    outstanding;
    uint16_t            unsignaled;
    int16_t             available;
    uint8_t             sl;
    uint8_t             path_bits;
    ucs_list_link_t     list;
    ucs_arbiter_group_t arb_group;
    UCS_STATS_NODE_DECLARE(stats);
};
UCS_CLASS_DECLARE(uct_rc_ep_t, uct_rc_iface_t*);


ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, struct sockaddr *addr);

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, const struct sockaddr *addr);

void uct_rc_ep_am_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                              void *data, size_t length, size_t valid_length,
                              char *buffer, size_t max);

void uct_rc_ep_get_bcopy_handler(uct_rc_iface_send_op_t *op);

void uct_rc_ep_get_bcopy_handler_no_completion(uct_rc_iface_send_op_t *op);

void uct_rc_ep_send_completion_proxy_handler(uct_rc_iface_send_op_t *op);

ucs_status_t uct_rc_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n);

void uct_rc_ep_pending_purge(uct_ep_h ep, uct_pending_callback_t cb);

ucs_arbiter_cb_result_t uct_rc_ep_process_pending(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg);

void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(32, 0)(uct_rc_iface_send_op_t *op);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(32, 1)(uct_rc_iface_send_op_t *op);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(64, 0)(uct_rc_iface_send_op_t *op);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(64, 1)(uct_rc_iface_send_op_t *op);

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_add_send_op(uct_rc_ep_t *ep, uct_rc_iface_send_op_t *op, uint16_t sn)
{
    ucs_assert(op != NULL);
    op->sn = sn;
    ucs_queue_push(&ep->outstanding, &op->queue);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_add_send_comp(uct_rc_iface_t *iface, uct_rc_ep_t *ep,
                        uct_completion_t *comp, uint16_t sn)
{
    uct_rc_iface_send_op_t *op;

    if (comp == NULL) {
        return;
    }

    op            = uct_rc_iface_get_send_op(iface);
    op->handler   = uct_rc_ep_send_completion_proxy_handler;
    op->user_comp = comp;
    uct_rc_ep_add_send_op(ep, op, sn);
}

static inline void uct_rc_ep_process_tx_completion(uct_rc_iface_t *iface,
                                                   uct_rc_ep_t *ep, uint16_t sn)
{
    uct_rc_iface_send_op_t *op;

    ucs_queue_for_each_extract(op, &ep->outstanding, queue,
                               UCS_CIRCULAR_COMPARE16(op->sn, <=, sn)) {
        op->handler(op);
    }

    ucs_arbiter_group_schedule(&iface->tx.arbiter, &ep->arb_group);
    ucs_arbiter_dispatch(&iface->tx.arbiter, 1, uct_rc_ep_process_pending, NULL);
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_iface_tx_moderation(uct_rc_iface_t *iface, uct_rc_ep_t *ep, uint8_t flag)
{
    return (ep->unsignaled >= iface->config.tx_moderation) ? flag : 0;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_tx_posted(uct_rc_ep_t *ep, int signaled)
{
    uct_rc_iface_t *iface;
    if (signaled) {
        iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);
        ucs_assert(uct_rc_iface_have_tx_cqe_avail(iface));
        ep->unsignaled = 0;
        --iface->tx.cq_available;
        UCS_STATS_UPDATE_COUNTER(ep->stats, UCT_RC_EP_STAT_SINGAL, 1);
    } else {
        ++ep->unsignaled;
    }
}

#endif
