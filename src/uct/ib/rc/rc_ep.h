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


enum {
    UCT_RC_EP_STAT_QP_FULL,
    UCT_RC_EP_STAT_SINGAL,
    UCT_RC_EP_STAT_LAST
};


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
    uct_base_ep_t       super;
    struct ibv_qp       *qp;
    ucs_queue_head_t    comp;
    unsigned            unsignaled;
    uint8_t             sl;
    uint8_t             path_bits;
    UCS_STATS_NODE_DECLARE(stats);
};
UCS_CLASS_DECLARE(uct_rc_ep_t, uct_rc_iface_t*);


ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, const uct_iface_addr_t *tl_iface_addr,
                                     const uct_ep_addr_t *tl_ep_addr);

void uct_rc_ep_am_packet_dump(void *data, size_t length, size_t valid_length,
                              char *buffer, size_t max);

void uct_rc_ep_get_bcopy_completion(uct_completion_t *self);

void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(32, 0)(uct_completion_t *self);
void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(32, 1)(uct_completion_t *self);
void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(64, 0)(uct_completion_t *self);
void UCT_RC_DEFINE_ATOMIC_COMPLETION_FUNC_NAME(64, 1)(uct_completion_t *self);

static UCS_F_ALWAYS_INLINE void
uct_rc_ep_add_user_completion(uct_rc_ep_t* ep, uct_completion_t* comp, uint16_t sn)
{
    uct_rc_completion_t* rc_comp;

    if (comp == NULL) {
        return;
    }

    rc_comp = ucs_derived_of(comp, uct_rc_completion_t);
    rc_comp->sn = sn;
    ucs_queue_push(&ep->comp, &rc_comp->queue);
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
