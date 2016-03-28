/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_ep.h"
#include "dc_iface.h"


UCS_CLASS_INIT_FUNC(uct_dc_ep_t, uct_dc_iface_t *iface)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super);
    ucs_arbiter_group_init(&self->pending_group);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_ep_t)
{
    ucs_arbiter_group_cleanup(&self->pending_group);
}

UCS_CLASS_DEFINE(uct_dc_ep_t, uct_base_ep_t);
