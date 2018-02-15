/**
* Copyright (C) Mellanox Technologies Ltd. 2016-2017.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_EP_H
#define UCT_DC_EP_H

#include <uct/api/uct.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/sys/compiler_def.h>

#include "dc_iface.h"

enum {
    /* Indicates that FC grant has been requested, but is not received yet.
     * Flush will not complete until an outgoing grant request is acked.
     * It is needed to avoid the case when grant arrives for the recently
     * deleted ep. */
    UCT_DC_EP_FC_FLAG_WAIT_FOR_GRANT = UCS_BIT(0)
};

struct uct_dc_ep {
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
};

UCS_CLASS_DECLARE(uct_dc_ep_t, uct_dc_iface_t *, const uct_dc_iface_addr_t *);

ucs_arbiter_cb_result_t
uct_dc_iface_dci_do_pending_wait(ucs_arbiter_t *arbiter,
                                 ucs_arbiter_elem_t *elem,
                                 void *arg);

ucs_arbiter_cb_result_t
uct_dc_iface_dci_do_pending_tx(ucs_arbiter_t *arbiter,
                               ucs_arbiter_elem_t *elem,
                               void *arg);

ucs_status_t uct_dc_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *r);
void uct_dc_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb, void *arg);

void uct_dc_ep_cleanup(uct_ep_h tl_ep, ucs_class_t *cls);

void uct_dc_ep_release(uct_dc_ep_t *ep);

static inline void uct_dc_iface_dci_sched_tx(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    /* TODO: other policies have to add group always */
    if (uct_dc_iface_dci_has_tx_resources(iface, ep->dci)) {
        ucs_arbiter_group_schedule(uct_dc_iface_tx_waitq(iface), &ep->arb_group);
    }
}

/**
 * dci policies:
 * - fixed: all eps always use same dci no matter what
 * - dcs:
 *    - ep uses already assigned dci or
 *    - free dci is assigned in LIFO (stack) order or
 *    - ep has not resources to transmit
 *    - on FULL completion (once there are no outstanding ops)
 *      dci is pushed to the stack of free dcis
 *    it is possible that ep will never release its dci:
 *      ep send, gets some completion, sends more, repeat
 * - dcs + quota:
 *    - same as dcs with following addition:
 *    - if dci can not tx, and there are eps waiting for dci
 *      allocation ep goes into tx_wait state
 *    - in tx_wait state:
 *          - ep can not transmit while there are eps
 *            waiting for dci allocation. This will break
 *            starvation.
 *          - if there are no eps that are waiting for dci allocation
 *            ep goes back to normal state
 *
 * Not implemented policies:
 *
 * - hash:
 *    - dci is allocated to ep by some hash function
 *      for example dlid % ndci
 *
 * - random
 *    - dci is choosen by random() % ndci
 *    - ep keeps using dci as long as it has oustanding sends
 */

enum uct_dc_ep_flags {
    UCT_DC_EP_FLAG_TX_WAIT  = UCS_BIT(0), /* ep is in the tx_wait state. See
                                             description of the dcs+quota dci
                                             selection policy above */
    UCT_DC_EP_FLAG_GRH      = UCS_BIT(1), /* ep has GRH address. Used by 
                                             dc_mlx5 endpoint */
    UCT_DC_EP_FLAG_VALID    = UCS_BIT(2)  /* ep is a valid endpoint */
};


#define UCT_DC_EP_NO_DCI ((uint8_t)-1)

ucs_status_t uct_dc_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp);

#define uct_dc_iface_dci_put       uct_dc_iface_dci_put_dcs
#define uct_dc_iface_dci_get       uct_dc_iface_dci_get_dcs
#define uct_dc_iface_dci_can_alloc uct_dc_iface_dci_can_alloc_dcs
#define uct_dc_iface_dci_alloc     uct_dc_iface_dci_alloc_dcs
#define uct_dc_iface_dci_free      uct_dc_iface_dci_free_dcs

static UCS_F_ALWAYS_INLINE ucs_status_t uct_dc_ep_basic_init(uct_dc_iface_t *iface,
                                                             uct_dc_ep_t *ep)
{
    ucs_arbiter_group_init(&ep->arb_group);
    ep->dci   = UCT_DC_EP_NO_DCI;
    /* valid = 1, global = 0, tx_wait = 0 */
    ep->flags = UCT_DC_EP_FLAG_VALID;

    return uct_rc_fc_init(&ep->fc, iface->super.config.fc_wnd_size
                          UCS_STATS_ARG(ep->super.stats));
}

static UCS_F_ALWAYS_INLINE int
uct_dc_iface_dci_can_alloc_dcs(uct_dc_iface_t *iface)
{
    return iface->tx.stack_top < iface->tx.ndci;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_iface_progress_pending(uct_dc_iface_t *iface)
{
    do {
        /**
         * Pending op on the tx_waitq can complete with the UCS_OK
         * status without actually sending anything on the dci.
         * In this case pending ops on the waitq may never be
         * scdeduled.
         *
         * So we keep progressing pending while dci_waitq is not
         * empty and it is possible to allocate a dci.
         */
        if (uct_dc_iface_dci_can_alloc(iface)) {
            ucs_arbiter_dispatch(uct_dc_iface_dci_waitq(iface), 1,
                                 uct_dc_iface_dci_do_pending_wait, NULL);
        }
        ucs_arbiter_dispatch(uct_dc_iface_tx_waitq(iface), 1,
                             uct_dc_iface_dci_do_pending_tx, NULL);

    } while (ucs_unlikely(!ucs_arbiter_is_empty(uct_dc_iface_dci_waitq(iface)) &&
                           uct_dc_iface_dci_can_alloc_dcs(iface)));
}

static inline int uct_dc_iface_dci_ep_can_send(uct_dc_ep_t *ep)
{
    uct_dc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_iface_t);
    return (!(ep->flags & UCT_DC_EP_FLAG_TX_WAIT)) &&
           uct_rc_fc_has_resources(&iface->super, &ep->fc) &&
           uct_dc_iface_dci_has_tx_resources(iface, ep->dci);
}

static UCS_F_ALWAYS_INLINE
void uct_dc_iface_schedule_dci_alloc(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    /* If FC window is empty the group will be scheduled when
     * grant is received */
    if (uct_rc_fc_has_resources(&iface->super, &ep->fc)) {
        ucs_arbiter_group_schedule(uct_dc_iface_dci_waitq(iface), &ep->arb_group);
    }
}

static inline void uct_dc_iface_dci_put_dcs(uct_dc_iface_t *iface, uint8_t dci)
{
    uct_dc_ep_t *ep = iface->tx.dcis[dci].ep;

    ucs_assert(iface->tx.stack_top > 0);

    if (uct_dc_iface_dci_has_outstanding(iface, dci)) {
        if (ep == NULL) {
            /* The EP was destroyed after flush cancel */
            ucs_assert(ucs_test_all_flags(iface->tx.dcis[dci].flags,
                                          (UCT_DC_DCI_FLAG_EP_CANCELED |
                                           UCT_DC_DCI_FLAG_EP_DESTROYED)));
            return;
        }
        if (iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) {
            /* in tx_wait state:
             * -  if there are no eps are waiting for dci allocation
             *    ep goes back to normal state
             */
            if (ep->flags & UCT_DC_EP_FLAG_TX_WAIT) {
                if (!ucs_arbiter_is_empty(uct_dc_iface_dci_waitq(iface))) {
                    return;
                }
                ep->flags &= ~UCT_DC_EP_FLAG_TX_WAIT;
            }
        }
        ucs_arbiter_group_schedule(uct_dc_iface_tx_waitq(iface), &ep->arb_group);
        return;
    }
    iface->tx.stack_top--;
    iface->tx.dcis_stack[iface->tx.stack_top] = dci;

    if (ucs_unlikely(ep == NULL)) {
        return;
    }

    ucs_assert(iface->tx.dcis[dci].ep->dci != UCT_DC_EP_NO_DCI);
    ep->dci    = UCT_DC_EP_NO_DCI;
    ep->flags &= ~UCT_DC_EP_FLAG_TX_WAIT;
    iface->tx.dcis[dci].ep = NULL;
#if ENABLE_ASSERT
    iface->tx.dcis[dci].flags = 0;
#endif
    /* it is possible that dci is released while ep still has scheduled pending ops.
     * move the group to the 'wait for dci alloc' state
     */
    ucs_arbiter_group_desched(uct_dc_iface_tx_waitq(iface), &ep->arb_group);
    uct_dc_iface_schedule_dci_alloc(iface, ep);
}

static inline ucs_status_t
uct_dc_iface_check_txqp(uct_dc_iface_t *iface, uct_dc_ep_t *ep, uct_rc_txqp_t *txqp)
{
    UCT_RC_CHECK_TXQP(&iface->super, ep, txqp);
    return UCS_OK;
}

static inline void uct_dc_iface_dci_alloc_dcs(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    /* take a first available dci from stack.
     * There is no need to check txqp because
     * dci must have resources to transmit.
     */
    ep->dci = iface->tx.dcis_stack[iface->tx.stack_top];
    ucs_assert(ep->dci < iface->tx.ndci);
    ucs_assert(iface->tx.dcis[ep->dci].ep == NULL);
    ucs_assert(iface->tx.dcis[ep->dci].flags == 0);
    iface->tx.dcis[ep->dci].ep = ep;
    iface->tx.stack_top++;
}

static inline void uct_dc_iface_dci_free_dcs(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    uint8_t dci = ep->dci;

    ucs_assert(dci != UCT_DC_EP_NO_DCI);
    ucs_assert(iface->tx.stack_top > 0);

    if (uct_dc_iface_dci_has_outstanding(iface, dci)) {
        return;
    }

    iface->tx.stack_top--;
    iface->tx.dcis_stack[iface->tx.stack_top] = dci;
    iface->tx.dcis[dci].ep                    = NULL;
#if ENABLE_ASSERT
    iface->tx.dcis[ep->dci].flags             = 0;
#endif

    ep->dci    = UCT_DC_EP_NO_DCI;
    ep->flags &= ~UCT_DC_EP_FLAG_TX_WAIT;
}

static inline ucs_status_t uct_dc_iface_dci_get_dcs(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    uct_rc_txqp_t *txqp;
    int16_t available;

    if (ep->dci != UCT_DC_EP_NO_DCI) {
        /* dci is already assigned - keep using it */
        if ((iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) &&
            (ep->flags & UCT_DC_EP_FLAG_TX_WAIT)) {
            UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
            return UCS_ERR_NO_RESOURCE;
        }

        /* if dci has sent more than quota, and there are eps waiting for dci
         * allocation ep goes into tx_wait state.
         */
        txqp      = &iface->tx.dcis[ep->dci].txqp;
        available = uct_rc_txqp_available(txqp);
        if ((iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) &&
            (available <= iface->tx.available_quota) &&
            !ucs_arbiter_is_empty(uct_dc_iface_dci_waitq(iface)))
        {
            ep->flags |= UCT_DC_EP_FLAG_TX_WAIT;
            UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
            return UCS_ERR_NO_RESOURCE;
        }

        if (available <= 0) {
            UCS_STATS_UPDATE_COUNTER(txqp->stats, UCT_RC_TXQP_STAT_QP_FULL, 1);
            UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
            return UCS_ERR_NO_RESOURCE;
        }

        return UCS_OK;
    }

    if (uct_dc_iface_dci_can_alloc_dcs(iface)) {
        uct_dc_iface_dci_alloc_dcs(iface, ep);
        return UCS_OK;
    }

    /* we will have to wait until someone releases dci */
    UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
    return UCS_ERR_NO_RESOURCE;
}

static UCS_F_ALWAYS_INLINE int uct_dc_ep_fc_wait_for_grant(uct_dc_ep_t *ep)
{
    return ep->fc.flags & UCT_DC_EP_FC_FLAG_WAIT_FOR_GRANT;
}

ucs_status_t uct_dc_ep_check_fc(uct_dc_iface_t *iface, uct_dc_ep_t *ep);


#define UCT_DC_CHECK_RES(_iface, _ep) \
    { \
        ucs_status_t status; \
        status = uct_dc_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return status; \
        } \
        UCT_RC_CHECK_CQE(&(_iface)->super, _ep, \
                         &(_iface)->tx.dcis[(_ep)->dci].txqp); \
    }


#define UCT_DC_CHECK_RES_PTR(_iface, _ep) \
    { \
        ucs_status_t status; \
        status = uct_dc_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return UCS_STATUS_PTR(status); \
        } \
        UCT_RC_CHECK_CQE_RET(&(_iface)->super, _ep, \
                             &(_iface)->tx.dcis[(_ep)->dci].txqp, \
                             UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE)); \
    }


/* First, check whether we have FC window. If hard threshold is reached, credit
 * request will be sent by "fc_ctrl" as a separate message. TX resources
 * are checked after FC, because fc credits request may consume latest
 * available TX resources. */
#define UCT_DC_CHECK_RES_AND_FC(_iface, _ep) \
    { \
        if (ucs_unlikely((_ep)->fc.fc_wnd <= \
                         (_iface)->super.config.fc_hard_thresh)) { \
            ucs_status_t status = uct_dc_ep_check_fc(_iface, _ep); \
            if (ucs_unlikely(status != UCS_OK)) { \
                if ((_ep)->dci != UCT_DC_EP_NO_DCI) { \
                    ucs_assertv_always(uct_dc_iface_dci_has_outstanding(_iface, (_ep)->dci), \
                                       "iface (%p) ep (%p) dci leak detected: dci=%d", \
                                       _iface, _ep, (_ep)->dci); \
                } \
                return status; \
            } \
        } \
        UCT_DC_CHECK_RES(_iface, _ep) \
    }


#endif
