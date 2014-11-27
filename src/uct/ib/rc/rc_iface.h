/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_IFACE_H
#define UCT_RC_IFACE_H

#include "rc_ep.h"

#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/sglib_wrapper.h>


struct uct_rc_iface {
    uct_ib_iface_t           super;

    struct {
        unsigned             outstanding;
    } tx;

    uct_rc_ep_t              *eps[UCT_RC_QP_HASH_SIZE];

};

typedef struct uct_rc_iface_config {
    uct_ib_iface_config_t    super;
} uct_rc_iface_config_t;


extern ucs_config_field_t uct_rc_iface_config_table[];

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

uct_rc_ep_t *uct_rc_iface_lookup_ep(uct_rc_iface_t *iface, unsigned qp_num);

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);
void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface, uct_req_h *req_p,
                                uct_completion_cb_t cb);

#endif
