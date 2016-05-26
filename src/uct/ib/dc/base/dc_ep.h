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
    ucs_arbiter_group_t   pending_group;
    uint8_t               dci;
};

UCS_CLASS_DECLARE(uct_dc_ep_t, uct_dc_iface_t *);


/**
 * dci policies:
 * - fixed: always use same dci no matter what
 * - dcs:
 *    - ep uses already assigned dci or
 *    - free dci is assigned in LIFO (stack) order or
 *    - ep has not resources to transmit
 *    - on FULL completion (once there are no outstanding ops)
 *      dci is pushed to the stack of free dcis
 *    TDOO: it is possible that ep will never release its dci:
 *      ep send, gets some completion, sends more, repeat
 */
#define UCT_DC_EP_NO_DCI 0xff

#define uct_dc_iface_dci_put uct_dc_iface_dci_put_dcs
#define uct_dc_iface_dci_get uct_dc_iface_dci_get_dcs

static inline void uct_dc_iface_dci_put_dcs(uct_dc_iface_t *iface, uint8_t dci)
{

    if (uct_rc_txqp_available(&iface->tx.dcis[dci].txqp) < (int16_t)iface->super.config.tx_qp_len) {
        return;
    }

    ucs_assert(iface->tx.stack_top > 0);
    ucs_assert(iface->tx.dcis[dci].ep != NULL);
    ucs_assert(iface->tx.dcis[dci].ep->dci != UCT_DC_EP_NO_DCI);

    iface->tx.stack_top--;
    iface->tx.dcis_stack[iface->tx.stack_top] = dci;
    iface->tx.dcis[dci].ep->dci = UCT_DC_EP_NO_DCI;
    iface->tx.dcis[dci].ep = NULL;
}

static inline ucs_status_t uct_dc_iface_dci_get_dcs(uct_dc_iface_t *iface, uct_dc_ep_t *ep)
{
    
    if (ep->dci != UCT_DC_EP_NO_DCI) {
        /* dci is already assigned - keep using it */
        UCT_RC_CHECK_TXQP(&iface->super, &iface->tx.dcis[ep->dci].txqp); 
        return UCS_OK;
    }
    if (iface->tx.stack_top < iface->tx.ndci) {
        /* take a first available dci from stack. 
         * There is no need to check txqp because
         * dci must have resources to transmit.
         */
        ep->dci = iface->tx.dcis_stack[iface->tx.stack_top];
        ucs_assert((ep->dci >= 0) && (ep->dci < iface->tx.ndci));
        ucs_assert(iface->tx.dcis[ep->dci].ep == NULL);
        iface->tx.dcis[ep->dci].ep = ep;
        iface->tx.stack_top++;
        return UCS_OK;
    }
    /* we will have to wait until someone releases dci */
    return UCS_ERR_NO_RESOURCE;
}


#define UCT_DC_CHECK_RES(_iface, _ep) \
    { \
        ucs_status_t status; \
        UCT_RC_CHECK_CQE(&(_iface)->super); \
        status = uct_dc_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return status; \
        } \
    }
#endif
