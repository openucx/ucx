/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_EP_H
#define UCT_DC_EP_H

#include <uct/api/uct.h>
#include <ucs/datastruct/arbiter.h>

#include "dc_iface.h"

struct uct_dc_ep {
    uct_base_ep_t         super;
    ucs_arbiter_group_t   arb_group;
    uint8_t               dci;
    uint8_t               state;
    uint16_t              umr_offset;
    uint8_t               path_bits;
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
 *          - if there are no eps are waiting for dci allocation
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

enum uct_dc_ep_state {
    UCT_DC_EP_TX_OK,
    UCT_DC_EP_TX_WAIT          
};

#define UCT_DC_EP_NO_DCI ((uint8_t)-1)

ucs_status_t uct_dc_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp);

#define uct_dc_iface_dci_put       uct_dc_iface_dci_put_dcs
#define uct_dc_iface_dci_get       uct_dc_iface_dci_get_dcs
#define uct_dc_iface_dci_can_alloc uct_dc_iface_dci_can_alloc_dcs
#define uct_dc_iface_dci_alloc     uct_dc_iface_dci_alloc_dcs

static inline int uct_dc_iface_dci_can_alloc_dcs(uct_dc_iface_t *iface)
{
    return iface->tx.stack_top < iface->tx.ndci;
}

static inline int uct_dc_iface_dci_ep_can_send(uct_dc_ep_t *ep)
{
    uct_dc_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_iface_t);
    return (ep->state != UCT_DC_EP_TX_WAIT) &&
           uct_dc_iface_dci_has_tx_resources(iface, ep->dci);
}

static inline void uct_dc_iface_dci_put_dcs(uct_dc_iface_t *iface, uint8_t dci)
{
    uct_dc_ep_t *ep = iface->tx.dcis[dci].ep;

    ucs_assert(iface->tx.stack_top > 0);

    if (uct_rc_txqp_available(&iface->tx.dcis[dci].txqp) < (int16_t)iface->super.config.tx_qp_len) {
        if (ucs_unlikely(ep == NULL)) {
            /* ep was destroyed while holding dci */
            return;
        }
        if (iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) {
            /* in tx_wait state: 
             * -  if there are no eps are waiting for dci allocation
             *    ep goes back to normal state 
             */
            if (ep->state == UCT_DC_EP_TX_WAIT) {
                if (!ucs_arbiter_is_empty(uct_dc_iface_dci_waitq(iface))) {
                    return;
                }
                ep->state = UCT_DC_EP_TX_OK;
            }
        }
        ucs_arbiter_group_schedule(uct_dc_iface_tx_waitq(iface), &ep->arb_group); 
        return;
    }
    iface->tx.stack_top--;
    iface->tx.dcis_stack[iface->tx.stack_top] = dci;

    if (ucs_unlikely(ep == NULL)) {
        /* ep was destroyed while holding dci */
        return;
    }
            
    ucs_assert(iface->tx.dcis[dci].ep->dci != UCT_DC_EP_NO_DCI);
    ep->dci   = UCT_DC_EP_NO_DCI;
    ep->state = UCT_DC_EP_TX_OK;
    iface->tx.dcis[dci].ep = NULL;

    /* it is possible that dci is released while ep still has scheduled pending ops.
     * move the group to the 'wait for dci alloc' state
     */
    ucs_arbiter_group_desched(uct_dc_iface_tx_waitq(iface), &ep->arb_group); 
    ucs_arbiter_group_schedule(uct_dc_iface_dci_waitq(iface), &ep->arb_group);
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
    iface->tx.dcis[ep->dci].ep = ep;
    iface->tx.stack_top++;
}

static inline ucs_status_t uct_dc_iface_dci_get_dcs(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    ucs_status_t status;
    
    if (ep->dci != UCT_DC_EP_NO_DCI) {
        /* dci is already assigned - keep using it */
        if ((iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) &&
            (ep->state == UCT_DC_EP_TX_WAIT)) {
            UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
            return UCS_ERR_NO_RESOURCE;
        }
        status = uct_dc_iface_check_txqp(iface, ep, &iface->tx.dcis[ep->dci].txqp); 
        /* if dci can not tx, and there are eps waiting for dci 
         * allocation ep goes into tx_wait state
         */
        if ((iface->tx.policy == UCT_DC_TX_POLICY_DCS_QUOTA) &&
            (status != UCS_OK) && 
            !ucs_arbiter_is_empty(uct_dc_iface_dci_waitq(iface))) {
            ep->state = UCT_DC_EP_TX_WAIT;
        }
        return status;
    }
    if (uct_dc_iface_dci_can_alloc_dcs(iface)) {
        uct_dc_iface_dci_alloc_dcs(iface, ep);
        return UCS_OK;
    }
    /* we will have to wait until someone releases dci */
    UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
    return UCS_ERR_NO_RESOURCE;
}


#define UCT_DC_CHECK_RES(_iface, _ep) \
    { \
        ucs_status_t status; \
        UCT_RC_CHECK_CQE(&(_iface)->super, _ep); \
        status = uct_dc_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return status; \
        } \
    }
#endif
