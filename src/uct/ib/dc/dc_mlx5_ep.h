/**
* Copyright (C) Mellanox Technologies Ltd. 2016-2020.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_EP_H
#define UCT_DC_EP_H

#include <uct/api/uct.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/sys/compiler_def.h>

#include "dc_mlx5.h"

#define UCT_DC_MLX5_EP_NO_DCI ((uint8_t)-1)


enum uct_dc_mlx5_ep_flags {
    UCT_DC_MLX5_EP_FLAG_TX_WAIT           = UCS_BIT(0), /* ep is in the tx_wait state. See
                                                           description of the dcs+quota dci
                                                           selection policy above */
    UCT_DC_MLX5_EP_FLAG_GRH               = UCS_BIT(1), /* ep has GRH address. Used by
                                                           dc_mlx5 endpoint */
    UCT_DC_MLX5_EP_FLAG_VALID             = UCS_BIT(2), /* ep is a valid endpoint */
    /* Indicates that FC grant has been requested, but is not received yet.
     * Flush will not complete until an outgoing grant request is acked.
     * It is needed to avoid the following cases:
     * 1) Grant arrives for the recently deleted ep.
     * 2) QP resources are available, but there are some pending requests. */
    UCT_DC_MLX5_EP_FLAG_FC_WAIT_FOR_GRANT = UCS_BIT(3),
    /* Keepalive Request scheduled: indicates that keepalive request
     * is scheduled in outstanding queue and no more keepalive actions
     * are needed */
    UCT_DC_MLX5_EP_FLAG_KEEPALIVE_POSTED  = UCS_BIT(4)
};


struct uct_dc_mlx5_ep {
    /*
     * per value of 'flags':
     * INVALID   - 'list' is added to iface->tx.gc_list.
     * Otherwise - 'super' and 'arb_group' are used.
     */
    union {
        struct {
            uct_base_ep_t         super;
            ucs_arbiter_group_t   arb_group;
        };
        ucs_list_link_t           list;
    };

    uint8_t                       dci;
    uint8_t                       flags;
    uint16_t                      atomic_mr_offset;
    uct_rc_fc_t                   fc;
    uct_ib_mlx5_base_av_t         av;
};

typedef struct {
    uct_dc_mlx5_ep_t                    super;
    struct mlx5_grh_av                  grh_av;
} uct_dc_mlx5_grh_ep_t;

typedef struct {
    uct_pending_req_priv_arb_t arb;
    uct_dc_mlx5_ep_t           *ep;
} uct_dc_mlx5_pending_req_priv_t;


UCS_CLASS_DECLARE(uct_dc_mlx5_ep_t, uct_dc_mlx5_iface_t *, const uct_dc_mlx5_iface_addr_t *,
                  uct_ib_mlx5_base_av_t *);

UCS_CLASS_DECLARE(uct_dc_mlx5_grh_ep_t, uct_dc_mlx5_iface_t *,
                  const uct_dc_mlx5_iface_addr_t *,
                  uct_ib_mlx5_base_av_t *, struct mlx5_grh_av *);


ucs_status_t uct_dc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *payload,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey);

ssize_t uct_dc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_dc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_get_bcopy(uct_ep_h tl_ep,
                                      uct_unpack_callback_t unpack_cb,
                                      void *arg, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     const void *buffer, unsigned length);

ssize_t uct_dc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                unsigned flags);

ucs_status_t uct_dc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const uct_iov_t *iov,
                                     size_t iovcnt, unsigned flags,
                                     uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                          uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_dc_mlx5_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                          uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_dc_mlx5_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                           uint64_t value, uint64_t *result,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                           uint32_t value, uint32_t *result,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp);

#if IBV_HW_TM
ucs_status_t uct_dc_mlx5_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                            const void *data, size_t length);

ssize_t uct_dc_mlx5_ep_tag_eager_bcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                       uint64_t imm,
                                       uct_pack_callback_t pack_cb,
                                       void *arg, unsigned flags);

ucs_status_t uct_dc_mlx5_ep_tag_eager_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                            uint64_t imm, const uct_iov_t *iov,
                                            size_t iovcnt, unsigned flags,
                                            uct_completion_t *comp);

ucs_status_ptr_t uct_dc_mlx5_ep_tag_rndv_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                               const void *header,
                                               unsigned header_length,
                                               const uct_iov_t *iov,
                                               size_t iovcnt, unsigned flags,
                                               uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_tag_rndv_request(uct_ep_h tl_ep, uct_tag_t tag,
                                             const void* header,
                                             unsigned header_length,
                                             unsigned flags);

ucs_status_t uct_dc_mlx5_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                              uct_tag_t tag,
                                              uct_tag_t tag_mask,
                                              const uct_iov_t *iov,
                                              size_t iovcnt,
                                              uct_tag_context_t *ctx);

ucs_status_t uct_dc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                               uct_tag_context_t *ctx,
                                               int force);
#endif

ucs_status_t uct_dc_mlx5_ep_fence(uct_ep_h tl_ep, unsigned flags);

ucs_status_t uct_dc_mlx5_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                    uct_rc_pending_req_t *req);

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_pending_wait(ucs_arbiter_t *arbiter,
                                      ucs_arbiter_group_t *group,
                                      ucs_arbiter_elem_t *elem,
                                      void *arg);

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_dcs_pending_tx(ucs_arbiter_t *arbiter,
                                        ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem,
                                        void *arg);

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_rand_pending_tx(ucs_arbiter_t *arbiter,
                                         ucs_arbiter_group_t *group,
                                         ucs_arbiter_elem_t *elem,
                                         void *arg);

ucs_status_t uct_dc_mlx5_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *r,
                                        unsigned flags);
void uct_dc_mlx5_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb, void *arg);

void uct_dc_mlx5_ep_pending_common(uct_dc_mlx5_iface_t *iface,
                                   uct_dc_mlx5_ep_t *ep, uct_pending_req_t *r,
                                   unsigned flags, int push_to_head);

void uct_dc_mlx5_ep_cleanup(uct_ep_h tl_ep, ucs_class_t *cls);

void uct_dc_mlx5_ep_release(uct_dc_mlx5_ep_t *ep);

ucs_status_t
uct_dc_mlx5_ep_check(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp);


static UCS_F_ALWAYS_INLINE uct_dc_mlx5_pending_req_priv_t *
uct_dc_mlx5_pending_req_priv(uct_pending_req_t *req)
{
    return (uct_dc_mlx5_pending_req_priv_t *)&(req)->priv;
}

static UCS_F_ALWAYS_INLINE int uct_dc_mlx5_iface_is_dci_rand(uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.policy == UCT_DC_TX_POLICY_RAND;
}

static UCS_F_ALWAYS_INLINE ucs_arbiter_group_t*
uct_dc_mlx5_ep_rand_arb_group(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    ucs_assert(uct_dc_mlx5_iface_is_dci_rand(iface) &&
               (ep->dci != UCT_DC_MLX5_EP_NO_DCI));
    /* If DCI random policy is used, DCI is always assigned to EP */
    return &iface->tx.dcis[ep->dci].arb_group;
}

static UCS_F_ALWAYS_INLINE ucs_arbiter_group_t*
uct_dc_mlx5_ep_arb_group(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    return (uct_dc_mlx5_iface_is_dci_rand(iface)) ?
            uct_dc_mlx5_ep_rand_arb_group(iface, ep) : &ep->arb_group;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dci_sched_tx(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface),
                                   uct_dc_mlx5_ep_rand_arb_group(iface, ep));
    } else if (uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci)) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface),
                                   &ep->arb_group);
    }
}

static UCS_F_ALWAYS_INLINE uct_dc_mlx5_ep_t *
uct_dc_mlx5_ep_from_dci(uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    /* Can be used with dcs* policies only, with rand policy every dci may
     * be used by many eps */
    ucs_assert(!uct_dc_mlx5_iface_is_dci_rand(iface));
    return iface->tx.dcis[dci].ep;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_ep_clear_fc_grant_flag(uct_dc_mlx5_iface_t *iface,
                                   uct_dc_mlx5_ep_t *ep)
{
    ucs_assert((ep->flags & UCT_DC_MLX5_EP_FLAG_FC_WAIT_FOR_GRANT) &&
               iface->tx.fc_grants);
    ep->flags &= ~UCT_DC_MLX5_EP_FLAG_FC_WAIT_FOR_GRANT;
    --iface->tx.fc_grants;
}

void uct_dc_mlx5_ep_handle_failure(uct_dc_mlx5_ep_t *ep, void *arg,
                                   ucs_status_t status);

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_ep_basic_init(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    ucs_arbiter_group_init(&ep->arb_group);

    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        /* coverity[dont_call] */
        ep->dci = rand_r(&iface->tx.rand_seed) % iface->tx.ndci;
    } else {
        ep->dci = UCT_DC_MLX5_EP_NO_DCI;
    }

    /* valid = 1, global = 0, tx_wait = 0 */
    ep->flags = UCT_DC_MLX5_EP_FLAG_VALID;

    return uct_rc_fc_init(&ep->fc, iface->super.super.config.fc_wnd_size
                          UCS_STATS_ARG(ep->super.stats));
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_dci_can_alloc(uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.stack_top < iface->tx.ndci;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_progress_pending(uct_dc_mlx5_iface_t *iface)
{
    do {
        /**
         * Pending op on the tx_waitq can complete with the UCS_OK
         * status without actually sending anything on the dci.
         * In this case pending ops on the waitq may never be
         * scheduled.
         *
         * So we keep progressing pending while dci_waitq is not
         * empty and it is possible to allocate a dci.
         * NOTE: in case of rand dci allocation policy, dci_waitq is always
         * empty.
         */
        if (uct_dc_mlx5_iface_dci_can_alloc(iface) &&
            !uct_dc_mlx5_iface_is_dci_rand(iface)) {
            ucs_arbiter_dispatch(uct_dc_mlx5_iface_dci_waitq(iface), 1,
                                 uct_dc_mlx5_iface_dci_do_pending_wait, NULL);
        }
        ucs_arbiter_dispatch(uct_dc_mlx5_iface_tx_waitq(iface), 1,
                             iface->tx.pend_cb, NULL);

    } while (ucs_unlikely(!ucs_arbiter_is_empty(uct_dc_mlx5_iface_dci_waitq(iface)) &&
                           uct_dc_mlx5_iface_dci_can_alloc(iface)));
}

static inline int uct_dc_mlx5_iface_dci_ep_can_send(uct_dc_mlx5_ep_t *ep)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);
    return (!(ep->flags & UCT_DC_MLX5_EP_FLAG_TX_WAIT)) &&
           uct_rc_fc_has_resources(&iface->super.super, &ep->fc) &&
           uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci);
}

static UCS_F_ALWAYS_INLINE
void uct_dc_mlx5_iface_schedule_dci_alloc(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    /* If FC window is empty the group will be scheduled when
     * grant is received */
    if (uct_rc_fc_has_resources(&iface->super.super, &ep->fc)) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_dci_waitq(iface), &ep->arb_group);
    }
}

static UCS_F_ALWAYS_INLINE void
 uct_dc_mlx5_iface_dci_release(uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    iface->tx.stack_top--;
    iface->tx.dcis_stack[iface->tx.stack_top] = dci;
#if UCS_ENABLE_ASSERT
    iface->tx.dcis[dci].flags = 0;
#endif
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dci_put(uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    uct_dc_mlx5_ep_t *ep;

    if (uct_dc_mlx5_iface_is_dci_rand(iface) ||
        ucs_unlikely(dci >= iface->tx.ndci)) {
        return;
    }

    ep = uct_dc_mlx5_ep_from_dci(iface, dci);

    ucs_assert(iface->tx.stack_top > 0);

    if (ucs_unlikely(ep == NULL)) {
        if (!uct_dc_mlx5_iface_dci_has_outstanding(iface, dci)) {
            uct_dc_mlx5_iface_dci_release(iface, dci);
        }
        return;
    }

    if (uct_dc_mlx5_iface_dci_has_outstanding(iface, dci)) {
        if (iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) {
            /* in tx_wait state:
             * -  if there are no eps are waiting for dci allocation
             *    ep goes back to normal state
             */
            if (ep->flags & UCT_DC_MLX5_EP_FLAG_TX_WAIT) {
                if (!ucs_arbiter_is_empty(uct_dc_mlx5_iface_dci_waitq(iface))) {
                    return;
                }
                ep->flags &= ~UCT_DC_MLX5_EP_FLAG_TX_WAIT;
            }
        }
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface), &ep->arb_group);
        return;
    }

    uct_dc_mlx5_iface_dci_release(iface, dci);

    ucs_assert(uct_dc_mlx5_ep_from_dci(iface, dci)->dci != UCT_DC_MLX5_EP_NO_DCI);
    ep->dci    = UCT_DC_MLX5_EP_NO_DCI;
    ep->flags &= ~UCT_DC_MLX5_EP_FLAG_TX_WAIT;
    iface->tx.dcis[dci].ep = NULL;

    /* it is possible that dci is released while ep still has scheduled pending ops.
     * move the group to the 'wait for dci alloc' state
     */
    ucs_arbiter_group_desched(uct_dc_mlx5_iface_tx_waitq(iface), &ep->arb_group);
    uct_dc_mlx5_iface_schedule_dci_alloc(iface, ep);
}

static inline void uct_dc_mlx5_iface_dci_alloc(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    /* take a first available dci from stack.
     * There is no need to check txqp because
     * dci must have resources to transmit.
     */
    ucs_assert(!uct_dc_mlx5_iface_is_dci_rand(iface));
    ep->dci = iface->tx.dcis_stack[iface->tx.stack_top];
    ucs_assert(ep->dci < iface->tx.ndci);
    ucs_assert(uct_dc_mlx5_ep_from_dci(iface, ep->dci) == NULL);
    ucs_assert(iface->tx.dcis[ep->dci].flags == 0);
    iface->tx.dcis[ep->dci].ep = ep;
    iface->tx.stack_top++;
}

static inline void uct_dc_mlx5_iface_dci_free(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    uint8_t dci;

    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        return;
    }

    dci = ep->dci;

    ucs_assert(dci != UCT_DC_MLX5_EP_NO_DCI);
    ucs_assert(iface->tx.stack_top > 0);

    if (uct_dc_mlx5_iface_dci_has_outstanding(iface, dci)) {
        return;
    }

    uct_dc_mlx5_iface_dci_release(iface, dci);

    iface->tx.dcis[dci].ep = NULL;
    ep->dci                = UCT_DC_MLX5_EP_NO_DCI;
    ep->flags             &= ~UCT_DC_MLX5_EP_FLAG_TX_WAIT;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_iface_dci_get(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    uct_rc_txqp_t *txqp;
    int16_t available;

    ucs_assert(!iface->super.super.config.tx_moderation);

    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        if (uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci)) {
            return UCS_OK;
        } else {
            UCS_STATS_UPDATE_COUNTER(iface->tx.dcis[ep->dci].txqp.stats,
                                     UCT_RC_TXQP_STAT_QP_FULL, 1);
            goto out_no_res;
        }
    }

    if (ep->dci != UCT_DC_MLX5_EP_NO_DCI) {
        /* dci is already assigned - keep using it */
        if ((iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) &&
            (ep->flags & UCT_DC_MLX5_EP_FLAG_TX_WAIT)) {
            goto out_no_res;
        }

        /* if dci has sent more than quota, and there are eps waiting for dci
         * allocation ep goes into tx_wait state.
         */
        txqp      = &iface->tx.dcis[ep->dci].txqp;
        available = uct_rc_txqp_available(txqp);
        if ((iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) &&
            (available <= iface->tx.available_quota) &&
            !ucs_arbiter_is_empty(uct_dc_mlx5_iface_dci_waitq(iface)))
        {
            ep->flags |= UCT_DC_MLX5_EP_FLAG_TX_WAIT;
            goto out_no_res;
        }

        if (available <= 0) {
            UCS_STATS_UPDATE_COUNTER(txqp->stats, UCT_RC_TXQP_STAT_QP_FULL, 1);
            goto out_no_res;
        }

        return UCS_OK;
    }

    /* Do not alloc dci if no TX desc resources,
     * otherwise this dci may never be released. */
    if (uct_dc_mlx5_iface_dci_can_alloc(iface) &&
        uct_dc_mlx5_iface_has_tx_resources(iface)) {
        uct_dc_mlx5_iface_dci_alloc(iface, ep);
        return UCS_OK;
    }

out_no_res:
    /* we will have to wait until someone releases dci */
    UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
    return UCS_ERR_NO_RESOURCE;
}

static UCS_F_ALWAYS_INLINE int uct_dc_mlx5_ep_fc_wait_for_grant(uct_dc_mlx5_ep_t *ep)
{
    return ep->flags & UCT_DC_MLX5_EP_FLAG_FC_WAIT_FOR_GRANT;
}

ucs_status_t uct_dc_mlx5_ep_check_fc(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep);

static inline struct mlx5_grh_av *uct_dc_mlx5_ep_get_grh(uct_dc_mlx5_ep_t *ep)
{
   return (ep->flags & UCT_DC_MLX5_EP_FLAG_GRH) ?
          &(ucs_derived_of(ep, uct_dc_mlx5_grh_ep_t)->grh_av) : NULL;
}


#define UCT_DC_MLX5_TXQP_DECL(_txqp, _txwq) \
    uct_rc_txqp_t UCS_V_UNUSED *_txqp; \
    uct_ib_mlx5_txwq_t UCS_V_UNUSED *_txwq;


#define UCT_DC_MLX5_CHECK_DCI_RES(_iface, _ep) \
    { \
        ucs_status_t _status = uct_dc_mlx5_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(_status != UCS_OK)) { \
            return _status; \
        } \
    }


#define UCT_DC_CHECK_RES_PTR(_iface, _ep) \
    { \
        UCT_RC_CHECK_NUM_RDMA_READ_RET(&(_iface)->super.super, \
                                       UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE)) \
        { \
            ucs_status_t status = \
                uct_dc_mlx5_iface_dci_get(_iface, _ep); \
            if (ucs_unlikely(status != UCS_OK)) { \
                return UCS_STATUS_PTR(status); \
            } \
        } \
    }


/**
 * All operations are not allowed if no RDMA_READ credits. Otherwise operations
 * ordering can be broken. If some AM sends added to the pending queue after
 * RDMA_READ operation, it may be stuck there until RDMA_READ credits arrive,
 * therefore need to block even AM sends, until all resources are available.
 */
#define UCT_DC_MLX5_CHECK_RES(_iface, _ep) \
    { \
        UCT_RC_CHECK_NUM_RDMA_READ_RET(&(_iface)->super.super, \
                                       UCS_ERR_NO_RESOURCE) \
        UCT_DC_MLX5_CHECK_DCI_RES(_iface, _ep) \
    }


/* First, check whether we have FC window. If hard threshold is reached, credit
 * request will be sent by "fc_ctrl" as a separate message. TX resources
 * are checked after FC, because fc credits request may consume latest
 * available TX resources. */
#define UCT_DC_CHECK_RES_AND_FC(_iface, _ep) \
    { \
        if (ucs_unlikely((_ep)->fc.fc_wnd <= \
                         (_iface)->super.super.config.fc_hard_thresh)) { \
            ucs_status_t _status = uct_dc_mlx5_ep_check_fc(_iface, _ep); \
            if (ucs_unlikely(_status != UCS_OK)) { \
                if (((_ep)->dci != UCT_DC_MLX5_EP_NO_DCI) && \
                    !uct_dc_mlx5_iface_is_dci_rand(_iface)) { \
                    ucs_assertv_always(uct_dc_mlx5_iface_dci_has_outstanding(_iface, \
                                                                             (_ep)->dci), \
                                       "iface (%p) ep (%p) dci leak detected: dci=%d", \
                                       _iface, _ep, (_ep)->dci); \
                } \
                return _status; \
            } \
        } \
        UCT_DC_MLX5_CHECK_RES(_iface, _ep) \
        if (!uct_dc_mlx5_iface_is_dci_rand(_iface)) { \
            uct_rc_iface_check_pending(&(_iface)->super.super, \
                                       &(_ep)->arb_group); \
        } \
    }


#endif
