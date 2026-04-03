/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_ib_plugin.h"

uint64_t __attribute__((weak)) uct_ib_plugin_iface_flags(void)
{
    return 0;
}

ucs_status_t __attribute__((weak))
uct_ib_plugin_qp_query(const uct_ib_plugin_qp_query_params_t *params,
                       uct_ib_plugin_qp_query_attr_t *attr)
{
    return UCS_ERR_UNSUPPORTED;
}
