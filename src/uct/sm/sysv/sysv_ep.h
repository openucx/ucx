/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_SYSV_EP_H
#define UCT_SYSV_EP_H

#include <uct/sm/base/sm_ep.h>

typedef struct uct_sysv_ep {
    uct_sm_ep_t      super; /* point to sm_bae */
    struct uct_sysv_ep *next;
} uct_sysv_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_sysv_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_sysv_ep_t, uct_ep_t);

#endif
