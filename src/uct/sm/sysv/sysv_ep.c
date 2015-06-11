/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "sysv_ep.h"

static UCS_CLASS_INIT_FUNC(uct_sysv_ep_t, uct_iface_t *tl_iface,
                           const struct sockaddr *addr)
{
    /* point to dsm */
    uct_dsm_iface_t *iface = ucs_derived_of(tl_iface, uct_dsm_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_dsm_ep_t, iface)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sysv_ep_t)
{
    /* No op */
}

/* point to dsm */
UCS_CLASS_DEFINE(uct_sysv_ep_t, uct_dsm_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_sysv_ep_t, uct_ep_t, uct_iface_t*,
                          const struct sockaddr *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sysv_ep_t, uct_ep_t);
