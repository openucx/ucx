/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_ib_plugin.h"

ucs_status_t __attribute__((weak))
uct_ib_plugin_iface_query(uct_iface_h iface, uct_iface_attr_v2_t *iface_attr)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t __attribute__((weak))
uct_ib_plugin_ep_query(uct_ep_h ep, uct_ep_attr_t *ep_attr)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t __attribute__((weak)) uct_ib_plugin_outstanding_extract(
        uct_ep_h ep, const uct_ep_outstanding_extract_params_t *params)
{
    return UCS_ERR_UNSUPPORTED;
}
