/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_IFACE_H
#define UCT_RC_IFACE_H

#include <uct/ib/base/ib_iface.h>


typedef struct uct_rc_iface {
    uct_ib_iface_t      super;
} uct_rc_iface_t;


ucs_status_t uct_rc_iface_open(uct_context_h context, const char *hw_name,
                               uct_iface_h *iface_p);

void uct_rc_iface_close(uct_iface_h tl_iface);

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_rc_iface_flush(uct_iface_h iface, uct_req_h *req_p,
                                uct_completion_cb_t cb);

ucs_status_t uct_rc_iface_flush(uct_iface_h iface, uct_req_h *req_p,
                                uct_completion_cb_t cb);

#endif
