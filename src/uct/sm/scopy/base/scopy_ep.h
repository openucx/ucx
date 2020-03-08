/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SCOPY_EP_H
#define UCT_SCOPY_EP_H

#include "scopy_iface.h"

#include <uct/base/uct_iface.h>
#include <uct/sm/base/sm_ep.h>


typedef struct uct_scopy_ep {
    uct_base_ep_t                   super;
} uct_scopy_ep_t;


UCS_CLASS_DECLARE(uct_scopy_ep_t, const uct_ep_params_t *);

#endif
