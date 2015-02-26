/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_EP_H
#define UCT_RC_EP_H

#include "rc_iface.h"

#include <uct/api/uct.h>

/*
 * Macro to generate functions for AMO completions.
 */
#define UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(_num_bits, _is_be) \
    uct_rc_ep_atomic_completion_##_num_bits##_be##_is_be


struct uct_rc_ep_addr {
    uct_ep_addr_t     super;
    uint32_t          qp_num;
};


struct uct_rc_ep {
    uct_ep_t            super;
    struct ibv_qp       *qp;
    ucs_callbackq_t     comp;
    unsigned            unsignaled;
    uint8_t             sl;
    uint8_t             path_bits;
};


ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                     uct_ep_addr_t *tl_ep_addr);

void uct_rc_ep_am_packet_dump(void *data, size_t length, size_t valid_length,
                              char *buffer, size_t max);

void uct_rc_ep_get_bcopy_completion(ucs_callback_t *self);

void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(32, 0)(ucs_callback_t *self);
void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(32, 1)(ucs_callback_t *self);
void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(64, 0)(ucs_callback_t *self);
void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(64, 1)(ucs_callback_t *self);

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_add_user_completion(uct_rc_ep_t* ep, uct_completion_t* comp, uint16_t sn)
{
    ucs_callbackq_elem_t* cbq;

    if (comp == NULL) {
        return;
    }

    cbq = ucs_derived_of(&comp->super, ucs_callbackq_elem_t);
    cbq->sn = sn;
    ucs_callbackq_push(&ep->comp, cbq);
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_iface_tx_moderation(uct_rc_iface_t* iface, uct_rc_ep_t* ep, uint8_t flag)
{
    return (ep->unsignaled >= iface->config.tx_moderation) ? flag : 0;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_tx_posted(uct_rc_ep_t *ep, int signaled)
{
    uct_rc_iface_t *iface;
    if (signaled) {
        iface = ucs_derived_of(ep->super.iface, uct_rc_iface_t);
        ucs_assert(uct_rc_iface_have_tx_cqe_avail(iface));
        ep->unsignaled = 0;
        --iface->tx.cq_available;
    } else {
        ++ep->unsignaled;
    }
}

#endif
