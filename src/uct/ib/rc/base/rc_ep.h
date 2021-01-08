/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_EP_H
#define UCT_RC_EP_H

#include "rc_iface.h"

#include <uct/api/uct.h>
#include <ucs/debug/debug.h>


#define RC_UNSIGNALED_INF UINT16_MAX

enum {
    UCT_RC_FC_STAT_NO_CRED,
    UCT_RC_FC_STAT_TX_GRANT,
    UCT_RC_FC_STAT_TX_PURE_GRANT,
    UCT_RC_FC_STAT_TX_SOFT_REQ,
    UCT_RC_FC_STAT_TX_HARD_REQ,
    UCT_RC_FC_STAT_RX_GRANT,
    UCT_RC_FC_STAT_RX_PURE_GRANT,
    UCT_RC_FC_STAT_RX_SOFT_REQ,
    UCT_RC_FC_STAT_RX_HARD_REQ,
    UCT_RC_FC_STAT_FC_WND,
    UCT_RC_FC_STAT_LAST
};

enum {
    UCT_RC_TXQP_STAT_QP_FULL,
    UCT_RC_TXQP_STAT_SIGNAL,
    UCT_RC_TXQP_STAT_LAST
};

/*
 * Auxillary AM ID bits used by FC protocol.
 */
enum {
    /* Keepalive Request scheduled: indicates that keepalive request
     * is scheduled in pending queue and no more keepalive actions
     * are needed */
    UCT_RC_EP_FLAG_KEEPALIVE_PENDING   = UCS_BIT(0),

    /* EP is connected to peer */
    UCT_RC_EP_FLAG_CONNECTED           = UCS_BIT(1),

    /* Flush cancel was executed on EP */
    UCT_RC_EP_FLAG_FLUSH_CANCEL        = UCS_BIT(2),

    /* Error handler already called or flush(CANCEL) disabled it */
    UCT_RC_EP_FLAG_ERR_HANDLER_INVOKED = UCS_BIT(3),

    /* Soft Credit Request: indicates that peer needs to piggy-back credits
     * grant to counter AM (if any). Can be bundled with
     * UCT_RC_EP_FLAG_FC_GRANT  */
    UCT_RC_EP_FLAG_FC_SOFT_REQ         = UCS_BIT(UCT_AM_ID_BITS),

    /* Hard Credit Request: indicates that wnd is close to be exhausted.
     * The peer must send separate AM with credit grant as soon as it
     * receives AM  with this bit set. Can be bundled with
     * UCT_RC_EP_FLAG_FC_GRANT */
    UCT_RC_EP_FLAG_FC_HARD_REQ         = UCS_BIT((UCT_AM_ID_BITS) + 1),

    /* Credit Grant: ep should update its FC wnd as soon as it receives AM with
     * this bit set. Can be bundled with either soft or hard request bits */
    UCT_RC_EP_FLAG_FC_GRANT            = UCS_BIT((UCT_AM_ID_BITS) + 2),

    /* Special FC AM with Credit Grant: Just an empty message indicating
     * credit grant. Can't be bundled with any other FC flag (as it consumes
     * all 3 FC bits). */
    UCT_RC_EP_FC_PURE_GRANT            = (UCT_RC_EP_FLAG_FC_HARD_REQ |
                                          UCT_RC_EP_FLAG_FC_SOFT_REQ |
                                          UCT_RC_EP_FLAG_FC_GRANT)
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
#define UCT_RC_CHECK_CQE_RET(_iface, _ep, _ret) \
    /* tx_moderation == 0 for TLs which don't support it */ \
    if (ucs_unlikely((_iface)->tx.cq_available <= \
        (signed)(_iface)->config.tx_moderation)) { \
        if (!uct_rc_iface_have_tx_cqe_avail(_iface)) { \
            UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_RC_IFACE_STAT_NO_CQE, 1); \
            UCS_STATS_UPDATE_COUNTER((_ep)->super.stats, UCT_EP_STAT_NO_RES, 1); \
            return _ret; \
        } \
        (_ep)->txqp.unsignaled = RC_UNSIGNALED_INF; \
    }

#define UCT_RC_CHECK_TXQP_RET(_iface, _ep, _ret) \
    if (uct_rc_txqp_available(&(_ep)->txqp) <= 0) { \
        UCS_STATS_UPDATE_COUNTER((_ep)->txqp.stats, UCT_RC_TXQP_STAT_QP_FULL, 1); \
        UCS_STATS_UPDATE_COUNTER((_ep)->super.stats, UCT_EP_STAT_NO_RES, 1); \
        return _ret; \
    }

#define UCT_RC_CHECK_NUM_RDMA_READ_RET(_iface, _ret) \
    if (ucs_unlikely((_iface)->tx.reads_available <= 0)) { \
        UCS_STATS_UPDATE_COUNTER((_iface)->stats, \
                                 UCT_RC_IFACE_STAT_NO_READS, 1); \
        return _ret; \
    }

#define UCT_RC_RDMA_READ_POSTED(_iface, _length) \
    { \
        ucs_assert((_iface)->tx.reads_available > 0); \
        (_iface)->tx.reads_available -= (_length); \
    }

#define UCT_RC_CHECK_TX_CQ_RES(_iface, _ep) \
    UCT_RC_CHECK_CQE_RET(_iface, _ep, UCS_ERR_NO_RESOURCE) \
    UCT_RC_CHECK_TXQP_RET(_iface, _ep, UCS_ERR_NO_RESOURCE)

/**
 * All operations are not allowed if no RDMA_READ credits. Otherwise operations
 * ordering can be broken. If some AM sends added to the pending queue after
 * RDMA_READ operation, it may be stuck there until RDMA_READ credits arrive,
 * therefore need to block even AM sends, until all resources are available.
 */
#define UCT_RC_CHECK_RES(_iface, _ep) \
    ucs_assert((_ep)->flags & UCT_RC_EP_FLAG_CONNECTED); \
    UCT_RC_CHECK_TX_CQ_RES(_iface, _ep) \
    UCT_RC_CHECK_NUM_RDMA_READ_RET(_iface, UCS_ERR_NO_RESOURCE)

/*
 * check for FC credits and add FC protocol bits (if any)
 */
#define UCT_RC_CHECK_FC_WND(_fc, _stats)\
    if ((_fc)->fc_wnd <= 0) { \
        UCS_STATS_UPDATE_COUNTER((_fc)->stats, UCT_RC_FC_STAT_NO_CRED, 1); \
        UCS_STATS_UPDATE_COUNTER(_stats, UCT_EP_STAT_NO_RES, 1); \
        return UCS_ERR_NO_RESOURCE; \
    } \


#define UCT_RC_UPDATE_FC_WND(_iface, _fc) \
    { \
        /* For performance reasons, prefer to update fc_wnd unconditionally */ \
        (_fc)->fc_wnd--; \
        \
        if ((_iface)->config.fc_enabled) { \
            UCS_STATS_SET_COUNTER((_fc)->stats, UCT_RC_FC_STAT_FC_WND, \
                                  (_fc)->fc_wnd); \
        } \
    }

#define UCT_RC_CHECK_FC(_iface, _ep, _am_id) \
    { \
        if (ucs_unlikely((_ep)->fc.fc_wnd <= (_iface)->config.fc_soft_thresh)) { \
            if ((_iface)->config.fc_enabled) { \
                UCT_RC_CHECK_FC_WND(&(_ep)->fc, (_ep)->super.stats); \
                (_am_id) |= uct_rc_fc_req_moderation(&(_ep)->fc, _iface); \
            } else { \
                /* Set fc_wnd to max, to send as much as possible without checks */ \
                (_ep)->fc.fc_wnd = INT16_MAX; \
            } \
        } \
        (_am_id) |= uct_rc_fc_get_fc_hdr((_ep)->flags); /* take grant bit */ \
    }

#define UCT_RC_UPDATE_FC(_iface, _ep, _fc_hdr) \
    { \
        if ((_fc_hdr) & UCT_RC_EP_FLAG_FC_GRANT) { \
            UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, UCT_RC_FC_STAT_TX_GRANT, 1); \
        } \
        if ((_fc_hdr) & UCT_RC_EP_FLAG_FC_SOFT_REQ) { \
            UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, UCT_RC_FC_STAT_TX_SOFT_REQ, 1); \
        } else if ((_fc_hdr) & UCT_RC_EP_FLAG_FC_HARD_REQ) { \
            UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, UCT_RC_FC_STAT_TX_HARD_REQ, 1); \
        } \
        \
        (_ep)->flags &= ~UCT_RC_EP_FC_MASK; \
        \
        UCT_RC_UPDATE_FC_WND(_iface, &(_ep)->fc) \
    }

#define UCT_RC_CHECK_RES_AND_FC(_iface, _ep, _am_id) \
    UCT_RC_CHECK_RES(_iface, _ep) \
    UCT_RC_CHECK_FC(_iface, _ep, _am_id) \
    uct_rc_iface_check_pending(_iface, &(_ep)->arb_group);

/* this is a common type for all rc and dc transports */
struct uct_rc_txqp {
    ucs_queue_head_t    outstanding;
    /* RC_UNSIGNALED_INF value forces signaled in moderation logic when
     * CQ credits are close to zero (less tx_moderation value) */
    uint16_t            unsignaled;
    int16_t             available;
    UCS_STATS_NODE_DECLARE(stats)
};

typedef struct uct_rc_fc {
    /* Not more than fc_wnd active messages can be sent w/o acknowledgment */
    int16_t             fc_wnd;
    /* used only for FC protocol at this point (3 higher bits) */
    UCS_STATS_NODE_DECLARE(stats)
} uct_rc_fc_t;

struct uct_rc_ep {
    uct_base_ep_t       super;
    uct_rc_txqp_t       txqp;
    ucs_list_link_t     list;
    ucs_arbiter_group_t arb_group;
    uct_rc_fc_t         fc;
    uint16_t            atomic_mr_offset;
    uint8_t             path_index;
    uint8_t             flags;
};


/* EP QP TX cleanup context */
typedef struct {
    uct_ib_async_event_wait_t super;      /* LAST_WQE event callback */
    ucs_list_link_t           list;       /* entry in interface ep_gc_list */
    uct_rc_iface_t            *iface;     /* interface */
} uct_rc_ep_cleanup_ctx_t;


UCS_CLASS_DECLARE(uct_rc_ep_t, uct_rc_iface_t*, uint32_t, const uct_ep_params_t*);


void uct_rc_ep_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                           void *data, size_t length, size_t valid_length,
                           char *buffer, size_t max);

void uct_rc_ep_send_op_set_iov(uct_rc_iface_send_op_t *op, const uct_iov_t *iov,
                               size_t iovcnt);

void uct_rc_ep_get_bcopy_handler(uct_rc_iface_send_op_t *op, const void *resp);

void uct_rc_ep_get_bcopy_handler_no_completion(uct_rc_iface_send_op_t *op,
                                               const void *resp);

void uct_rc_ep_get_zcopy_completion_handler(uct_rc_iface_send_op_t *op,
                                            const void *resp);

void uct_rc_ep_send_op_completion_handler(uct_rc_iface_send_op_t *op,
                                          const void *resp);

void uct_rc_ep_flush_op_completion_handler(uct_rc_iface_send_op_t *op,
                                           const void *resp);

ucs_status_t uct_rc_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n,
                                   unsigned flags);

ucs_arbiter_cb_result_t uct_rc_ep_arbiter_purge_cb(ucs_arbiter_t *arbiter,
                                                   ucs_arbiter_group_t *group,
                                                   ucs_arbiter_elem_t *elem,
                                                   void *arg);

void uct_rc_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                             void*arg);

ucs_arbiter_cb_result_t uct_rc_ep_process_pending(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_group_t *group,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg);

ucs_status_t uct_rc_fc_init(uct_rc_fc_t *fc, int16_t winsize
                            UCS_STATS_ARG(ucs_stats_node_t* stats_parent));
void uct_rc_fc_cleanup(uct_rc_fc_t *fc);

ucs_status_t uct_rc_ep_fc_grant(uct_pending_req_t *self);

void uct_rc_txqp_purge_outstanding_op(uct_rc_iface_t *iface,
                                      uct_rc_iface_send_op_t *op,
                                      ucs_status_t status, int warn);

void uct_rc_txqp_purge_outstanding(uct_rc_iface_t *iface, uct_rc_txqp_t *txqp,
                                   ucs_status_t status, uint16_t sn, int warn);

ucs_status_t uct_rc_ep_flush(uct_rc_ep_t *ep, int16_t max_available,
                             unsigned flags);

ucs_status_t
uct_rc_ep_check(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp);

void uct_rc_ep_cleanup_qp(uct_rc_iface_t *iface, uct_rc_ep_t *ep,
                          uct_rc_ep_cleanup_ctx_t *cleanup_ctx, uint32_t qp_num);

void uct_rc_ep_cleanup_qp_done(uct_rc_ep_cleanup_ctx_t *cleanup_ctx,
                               uint32_t qp_num);

void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(32, 0)(uct_rc_iface_send_op_t *op,
                                                   const void *resp);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(32, 1)(uct_rc_iface_send_op_t *op,
                                                   const void *resp);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(64, 0)(uct_rc_iface_send_op_t *op,
                                                   const void *resp);
void UCT_RC_DEFINE_ATOMIC_HANDLER_FUNC_NAME(64, 1)(uct_rc_iface_send_op_t *op,
                                                   const void *resp);

ucs_status_t uct_rc_txqp_init(uct_rc_txqp_t *txqp, uct_rc_iface_t *iface,
                              uint32_t qp_num
                              UCS_STATS_ARG(ucs_stats_node_t* stats_parent));
void uct_rc_txqp_cleanup(uct_rc_iface_t *iface, uct_rc_txqp_t *txqp);

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

static UCS_F_ALWAYS_INLINE
int uct_rc_fc_has_resources(uct_rc_iface_t *iface, uct_rc_fc_t *fc)
{
    /* When FC is disabled, fc_wnd may still become 0 because it's decremented
     * unconditionally (for performance reasons) */
    return (fc->fc_wnd > 0) || !iface->config.fc_enabled;
}

static UCS_F_ALWAYS_INLINE int uct_rc_ep_has_tx_resources(uct_rc_ep_t *ep)
{
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_iface_t);

    return (uct_rc_txqp_available(&ep->txqp) > 0) &&
           uct_rc_fc_has_resources(iface, &ep->fc);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_add_send_op(uct_rc_txqp_t *txqp, uct_rc_iface_send_op_t *op)
{

    /* NOTE: We insert the descriptor with the sequence number after the post,
     * because when polling completions, we get the number of completions (rather
     * than completion zero-based index).
     */
    ucs_assert(op != NULL);
    ucs_assertv(!(op->flags & UCT_RC_IFACE_SEND_OP_FLAG_INUSE), "op=%p", op);
    op->flags |= UCT_RC_IFACE_SEND_OP_FLAG_INUSE;
    ucs_queue_push(&txqp->outstanding, &op->queue);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_add_send_op_sn(uct_rc_txqp_t *txqp, uct_rc_iface_send_op_t *op, uint16_t sn)
{
    ucs_trace_poll("txqp %p add send op %p sn %d handler %s", txqp, op, sn,
                   ucs_debug_get_symbol_name((void*)op->handler));
    op->sn = sn;
    uct_rc_txqp_add_send_op(txqp, op);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_add_send_comp(uct_rc_iface_t *iface, uct_rc_txqp_t *txqp,
                          uct_rc_send_handler_t handler, uct_completion_t *comp,
                          uint16_t sn, uint16_t flags, const uct_iov_t *iov,
                          size_t iovcnt, size_t length)
{
    uct_rc_iface_send_op_t *op;

    if (comp == NULL) {
        return;
    }

    op            = uct_rc_iface_get_send_op(iface);
    op->handler   = handler;
    op->user_comp = comp;
    op->flags    |= flags;
    op->length    = length;
    if (op->flags & UCT_RC_IFACE_SEND_OP_FLAG_IOV) {
        /* coverity[dead_error_line] */
        uct_rc_ep_send_op_set_iov(op, iov, iovcnt);
    }
    uct_rc_txqp_add_send_op_sn(txqp, op, sn);
}

static inline void
uct_rc_ep_init_send_op(uct_rc_iface_send_op_t *op, unsigned flags,
                            uct_completion_t *comp,
                            uct_rc_send_handler_t handler)
{
    op->flags     = flags;
    op->user_comp = comp;
    op->handler   = handler;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_txqp_add_flush_comp(uct_rc_iface_t *iface, uct_base_ep_t *ep,
                           uct_rc_txqp_t *txqp, uct_completion_t *comp,
                           uint16_t sn)
{
    uct_rc_iface_send_op_t *op;

    if (comp != NULL) {
        op = (uct_rc_iface_send_op_t*)ucs_mpool_get(&iface->tx.send_op_mp);
        if (ucs_unlikely(op == NULL)) {
            ucs_error("Failed to allocate flush completion");
            return UCS_ERR_NO_MEMORY;
        }

        uct_rc_ep_init_send_op(op, 0, comp, uct_rc_ep_flush_op_completion_handler);
        op->iface = iface;
        uct_rc_txqp_add_send_op_sn(txqp, op, sn);
    }
    UCT_TL_EP_STAT_FLUSH_WAIT(ep);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_completion_op(uct_rc_iface_send_op_t *op, const void *resp)
{
    ucs_trace_poll("complete op %p sn %d handler %s", op, op->sn,
                   ucs_debug_get_symbol_name((void*)op->handler));
    ucs_assert(op->flags & UCT_RC_IFACE_SEND_OP_FLAG_INUSE);
    op->flags &= ~(UCT_RC_IFACE_SEND_OP_FLAG_INUSE |
                   UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY);
    op->handler(op, resp);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_completion_desc(uct_rc_txqp_t *txqp, uint16_t sn)
{
    uct_rc_iface_send_op_t *op;

    ucs_trace_poll("txqp %p complete ops up to sn %d", txqp, sn);
    ucs_queue_for_each_extract(op, &txqp->outstanding, queue,
                               UCS_CIRCULAR_COMPARE16(op->sn, <=, sn)) {
        uct_rc_txqp_completion_op(op, ucs_derived_of(op, uct_rc_iface_send_desc_t) + 1);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_completion_inl_resp(uct_rc_txqp_t *txqp, const void *resp, uint16_t sn)
{
    uct_rc_iface_send_op_t *op;

    ucs_trace_poll("txqp %p complete ops up to sn %d", txqp, sn);
    ucs_queue_for_each_extract(op, &txqp->outstanding, queue,
                               UCS_CIRCULAR_COMPARE16(op->sn, <=, sn)) {
        ucs_assert(!(op->flags & UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY));
        uct_rc_txqp_completion_op(op, resp);
    }
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_iface_tx_moderation(uct_rc_iface_t *iface, uct_rc_txqp_t *txqp, uint8_t flag)
{
    return (txqp->unsignaled >= iface->config.tx_moderation) ? flag : 0;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_txqp_posted(uct_rc_txqp_t *txqp, uct_rc_iface_t *iface, uint16_t res_count,
                   int signaled)
{
    if (signaled) {
        ucs_assert(uct_rc_iface_have_tx_cqe_avail(iface));
        txqp->unsignaled = 0;
        UCS_STATS_UPDATE_COUNTER(txqp->stats, UCT_RC_TXQP_STAT_SIGNAL, 1);
    } else {
        ucs_assert(txqp->unsignaled != RC_UNSIGNALED_INF);
        ++txqp->unsignaled;
    }

    /* reserve cq credits for every posted operation,
     * in case it would complete with error */
    iface->tx.cq_available -= res_count;
    txqp->available -= res_count;
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_fc_get_fc_hdr(uint8_t id)
{
    return id & UCT_RC_EP_FC_MASK;
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_fc_req_moderation(uct_rc_fc_t *fc, uct_rc_iface_t *iface)
{
    return (fc->fc_wnd == iface->config.fc_hard_thresh) ?
            UCT_RC_EP_FLAG_FC_HARD_REQ :
           (fc->fc_wnd == iface->config.fc_soft_thresh) ?
            UCT_RC_EP_FLAG_FC_SOFT_REQ : 0;
}

static UCS_F_ALWAYS_INLINE int
uct_rc_ep_fm(uct_rc_iface_t *iface, uct_ib_fence_info_t* fi, int flag)
{
    int fence;

    /* a call to iface_fence increases beat, so if endpoint beat is not in
     * sync with iface beat it means the endpoint did not post any WQE with
     * fence flag yet */
    fence          = (fi->fence_beat != iface->tx.fi.fence_beat) ? flag : 0;
    fi->fence_beat = iface->tx.fi.fence_beat;
    return fence;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_ep_fence(uct_ep_h tl_ep, uct_ib_fence_info_t* fi, int fence)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_iface_t);

    /* in case if fence is requested and enabled by configuration
     * we need to schedule fence for next RDMA operation */
    if (fence && (iface->config.fence_mode != UCT_RC_FENCE_MODE_NONE)) {
        fi->fence_beat = iface->tx.fi.fence_beat - 1;
    }

    UCT_TL_EP_STAT_FENCE(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_fence_put(uct_rc_iface_t *iface, uct_ib_fence_info_t *fi,
                    uct_rkey_t *rkey, uint64_t *addr, uint16_t offset)
{
    if (uct_rc_ep_fm(iface, fi, 1)) {
        *rkey = uct_ib_resolve_atomic_rkey(*rkey, offset, addr);
    } else {
        *rkey = uct_ib_md_direct_rkey(*rkey);
    }
}

#endif
