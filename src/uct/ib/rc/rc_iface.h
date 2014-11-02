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

void uct_rc_iface_close(uct_iface_h iface);


#endif
