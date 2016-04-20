/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_EP_H
#define UCT_RC_EP_H

#include "rc_iface.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_instr.h>


enum {
    UCT_RC_EP_STAT_QP_FULL,
    UCT_RC_EP_STAT_SINGAL,
    UCT_RC_EP_STAT_LAST
};

/*
 * Auxillary AM ID bits used by FC protocol.
 */
enum {

    /* Soft Credit Request: indicates that peer needs to piggy-back credits
     * grant to counter AM (if any). Can be bundled with
     * UCT_RC_EP_FC_FLAG_GRANT  */
    UCT_RC_EP_FC_FLAG_SOFT_REQ  = UCS_BIT(UCT_AM_ID_BITS),

    /* Hard Credit Request: indicates that wnd is close to be exhausted.
     * The peer must send separate AM with credit grant as soon as it
     * receives AM  with this bit set. Can be bundled with
     * UCT_RC_EP_FC_FLAG_GRANT */
    UCT_RC_EP_FC_FLAG_HARD_REQ  = UCS_BIT((UCT_AM_ID_BITS) + 1),

    /* Credit Grant: ep should update its FC wnd as soon as it receives AM with
     * this bit set. Can be bundled with either soft or hard request bits */
    UCT_RC_EP_FC_FLAG_GRANT     = UCS_BIT((UCT_AM_ID_BITS) + 2),

    /* Special FC AM with Credit Grant: Just an empty message indicating
     * credit grant. Can't be bundled with any other FC flag (as it consumes
     * all 3 FC bits). */
    UCT_RC_EP_FC_PURE_GRANT     = (UCT_RC_EP_FC_FLAG_HARD_REQ |
                                   UCT_RC_EP_FC_FLAG_SOFT_REQ |
                                   UCT_RC_EP_FC_FLAG_GRANT)
};

/*
 * FC protocol header mask
 */
#define UCT_RC_EP_FC_MASK UCT_RC_EP_FC_PURE_GRANT

/*
 * Macro to generate functions for AMO completions.
 */
#define UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(_num_bits, _is_be) \
    uct_rc_ep_atomic_handler_##_num_bits##_be##_is_be

/*
 * Check for send resources
 */
#define UCT_RC_CHECK_CQE(_iface) \
    if (!uct_rc_iface_have_tx_cqe_avail(_iface)) { \
        UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_RC_IFACE_STAT_NO_CQE, 1); \
        UCT_TL_IFACE_STAT_TX_NO_RES(&(_iface)->super.super); \
        return UCS_ERR_NO_RESOURCE; \
    } 

#define UCT_RC_CHECK_RES(_iface, _ep) \
    UCT_RC_CHECK_CQE(_iface) \
    if (uct_rc_txqp_available(&(_ep)->txqp) <= 0) { \
        UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCT_RC_EP_STAT_QP_FULL, 1); \
        UCT_TL_IFACE_STAT_TX_NO_RES(&(_iface)->super.super); \
        return UCS_ERR_NO_RESOURCE; \
    }

/*
 * check for FC credits and add FC protocol bits (if any)
 */
#define UCT_RC_CHECK_FC_WND(_iface, _ep, _am_id) \
    do { \
        if (ucs_unlikely((_ep)->fc_wnd <= (_iface)->config.fc_soft_thresh)) { \
            if ((_ep)->fc_wnd <= 0) { \
                return UCS_ERR_NO_RESOURCE; \
            } \
            (_am_id) |= uct_rc_ep_fc_req_moderation(_iface, _ep); \
        } \
        (_am_id) |= uct_rc_ep_get_fc_hdr((_ep)->flags); /* take grant bit */ \
    } while (0)

#define UCT_RC_UPDATE_FC_WND(_ep) \
    do { \
        (_ep)->fc_wnd--; \
        (_ep)->flags = 0; \
    } while (0)


/* this is a common type for all rc and dc transports */
typedef struct uct_rc_txqp {
    struct ibv_qp       *qp;
    ucs_queue_head_t    outstanding;
    uint16_t            unsignaled;
    int16_t             available;
    /* Not more than fc_wnd active messages can be sent w/o acknowledgment */
    int16_t             fc_wnd;
    /* used only for FC protocol at this point (3 higher bits) */
    uint8_t             flags;
} uct_rc_txqp_t; 


struct uct_rc_ep {
    uct_base_ep_t       super;
    uct_rc_txqp_t       txqp;
    uint8_t             sl;
    uint8_t             path_bits;
    ucs_list_link_t     list;
    ucs_arbiter_group_t arb_group;
    uct_pending_req_t   fc_grant_req;
    UCS_STATS_NODE_DECLARE(stats);
};
UCS_CLASS_DECLARE(uct_rc_ep_t, uct_rc_iface_t*);


typedef struct uct_rc_ep_address {
    uct_ib_uint24_t  qp_num;
} UCS_S_PACKED uct_rc_ep_address_t;


ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, const uct_device_addr_t *dev_addr,
                                     const uct_ep_addr_t *ep_addr);

void uct_rc_ep_reset_qp(uct_rc_ep_t *ep);

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

ucs_status_t uct_rc_ep_fc_grant(uct_pending_req_t *self);

void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(32, 0)(uct_rc_iface_send_op_t *op);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(32, 1)(uct_rc_iface_send_op_t *op);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(64, 0)(uct_rc_iface_send_op_t *op);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(64, 1)(uct_rc_iface_send_op_t *op);

ucs_status_t uct_rc_txqp_init(uct_rc_txqp_t *txqp, uct_rc_iface_t *iface,
                              int qp_type, struct ibv_qp_cap *cap);
void uct_rc_txqp_cleanup(uct_rc_txqp_t *txqp);

static inline int16_t uct_rc_txqp_available(uct_rc_txqp_t *txqp) 
{
    return txqp->available;
}

static inline void uct_rc_txqp_available_add(uct_rc_txqp_t *txqp, int16_t val) 
{
    txqp->available += val;
}

static inline void uct_rc_txqp_available_set(uct_rc_txqp_t *txqp, int16_t val) 
{
    txqp->available = val;
}

static inline uint16_t uct_rc_txqp_unsignaled(uct_rc_txqp_t *txqp) 
{
    return txqp->unsignaled;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_add_send_op(uct_rc_txqp_t *txqp, uct_rc_iface_send_op_t *op, uint16_t sn)
{

    /* NOTE: We insert the descriptor with the sequence number after the post,
     * because when polling completions, we get the number of completions (rather
     * than completion zero-based index).
     */
    ucs_assert(op != NULL);
    op->sn = sn;
    ucs_queue_push(&txqp->outstanding, &op->queue);
    UCT_IB_INSTRUMENT_RECORD_SEND_OP(op);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_add_send_comp(uct_rc_iface_t *iface, uct_rc_txqp_t *txqp,
                          uct_completion_t *comp, uint16_t sn)
{
    uct_rc_iface_send_op_t *op;

    if (comp == NULL) {
        return;
    }

    op            = uct_rc_iface_get_send_op(iface);
    op->handler   = uct_rc_ep_send_completion_proxy_handler;
    op->user_comp = comp;
    uct_rc_txqp_add_send_op(txqp, op, sn);
}

static inline void 
uct_rc_txqp_completion(uct_rc_txqp_t *txqp, uint16_t sn) 
{
    uct_rc_iface_send_op_t *op;

    ucs_queue_for_each_extract(op, &txqp->outstanding, queue,
                               UCS_CIRCULAR_COMPARE16(op->sn, <=, sn)) {
        op->handler(op);
    }
    UCT_IB_INSTRUMENT_RECORD_SEND_OP(op);
}

static inline void uct_rc_ep_process_tx_completion(uct_rc_iface_t *iface,
                                                   uct_rc_ep_t *ep, uint16_t sn)
{
    uct_rc_txqp_completion(&ep->txqp, sn);
    ucs_arbiter_group_schedule(&iface->tx.arbiter, &ep->arb_group);
    ucs_arbiter_dispatch(&iface->tx.arbiter, 1, uct_rc_ep_process_pending, NULL);
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_iface_tx_moderation(uct_rc_iface_t *iface, uct_rc_txqp_t *txqp, uint8_t flag)
{
    return (txqp->unsignaled >= iface->config.tx_moderation) ? flag : 0;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_posted(uct_rc_txqp_t *txqp, uct_rc_iface_t *iface, uint16_t res_count, int signaled)
{
    if (signaled) {
        ucs_assert(uct_rc_iface_have_tx_cqe_avail(iface));
        txqp->unsignaled = 0;
        --iface->tx.cq_available;
    } else {
        txqp->unsignaled++;
    }
    txqp->available -= res_count;
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_ep_get_fc_hdr(uint8_t id)
{
    return id & UCT_RC_EP_FC_MASK;
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_ep_fc_req_moderation(uct_rc_iface_t *iface, uct_rc_ep_t *ep)
{
    return (ep->fc_wnd == iface->config.fc_hard_thresh) ?
            UCT_RC_EP_FC_FLAG_HARD_REQ :
           (ep->fc_wnd == iface->config.fc_soft_thresh) ?
            UCT_RC_EP_FC_FLAG_SOFT_REQ : 0;
}

#endif
