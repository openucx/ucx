/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_EP_H
#define UCT_RC_EP_H

#include "rc_def.h"

#include <uct/api/uct.h>
#include <ucs/datastruct/callbackq.h>


struct uct_rc_ep_addr {
    uct_ep_addr_t     super;
    uint32_t          qp_num;
};


struct uct_rc_ep {
    uct_ep_t            super;
    struct ibv_qp       *qp;
    struct {
        ucs_callbackq_t comp;
        unsigned        unsignaled;
    } tx;
};


/**
 * RC network header.
 */
typedef struct uct_rc_hdr {
    uint8_t           am_id;  /* Active message ID */
} UCS_S_PACKED uct_rc_hdr_t;


/*
 * Short active message header (active message header is always 64 bit).
 */
typedef struct uct_rc_am_short_hdr {
    uct_rc_hdr_t      rc_hdr;
    uint64_t          am_hdr;
} UCS_S_PACKED uct_rc_am_short_hdr_t;


ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                     uct_ep_addr_t *tl_ep_addr);

static inline void
uct_rc_ep_add_user_completion(uct_rc_ep_t* ep, uct_completion_t* comp, uint16_t sn)
{
    ucs_callbackq_elem_t* cbq;

    if (comp == NULL) {
        return;
    }

    cbq = ucs_derived_of(&comp->super, ucs_callbackq_elem_t);
    cbq->sn = sn;
    ucs_callbackq_push(&ep->tx.comp, cbq);
}


#endif
