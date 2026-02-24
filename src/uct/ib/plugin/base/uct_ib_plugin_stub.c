/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "../uct_ib_plugin.h"

#include <ucs/type/status.h>

/* Static plugin info for stub implementation */
static const uct_ib_plugin_info_t stub_plugin_info = {
    .name        = "UCX IB Plugin (stub)",
    .version     = "N/A",
    .description = "Plugin not available"
};

/* Use weak symbols so plugin library can override these functions */
ucs_status_t __attribute__((weak)) ucx_plugin_init(void)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t __attribute__((weak)) ucx_plugin_query(uint64_t *capability_flags)
{
    if (capability_flags == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    
    *capability_flags = UCT_IB_PLUGIN_CAP_NONE;
    return UCS_ERR_UNSUPPORTED;
}

void __attribute__((weak)) ucx_plugin_cleanup(void)
{
    /* No-op: nothing to clean up when plugin is not available */
}

const uct_ib_plugin_info_t* __attribute__((weak)) ucx_plugin_get_info(void)
{
    return &stub_plugin_info;
}

ucs_status_t __attribute__((weak)) ucx_plugin_hello(void)
{
    return UCS_ERR_UNSUPPORTED;
}
