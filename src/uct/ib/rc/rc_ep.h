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


struct uct_rc_ep_addr {
    uct_ep_addr_t     super;
    uint32_t          qp_num;
};


struct uct_rc_ep {
    uct_ep_t          super;
    struct ibv_qp     *qp;
    unsigned          qp_num;
    uct_rc_ep_t       *next;
};


ucs_status_t uct_rc_ep_init(uct_rc_ep_t *ep);

void uct_rc_ep_cleanup(uct_rc_ep_t *ep);

ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                     uct_ep_addr_t *tl_ep_addr);

#endif
