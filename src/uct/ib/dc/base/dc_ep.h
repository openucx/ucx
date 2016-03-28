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
};

UCS_CLASS_DECLARE(uct_dc_ep_t, uct_dc_iface_t *)

#endif
