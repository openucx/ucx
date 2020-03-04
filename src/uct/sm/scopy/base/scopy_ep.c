/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "scopy_ep.h"


UCS_CLASS_INIT_FUNC(uct_scopy_ep_t, const uct_ep_params_t *params)
{
    uct_scopy_iface_t *iface = ucs_derived_of(params->iface, uct_scopy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_scopy_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_scopy_ep_t, uct_base_ep_t)
