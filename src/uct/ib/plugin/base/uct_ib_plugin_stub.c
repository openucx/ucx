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

/* Use weak symbols so plugin library can override these functions */
ucs_status_t __attribute__((weak)) ucx_plugin_init(void)
{
    return UCS_ERR_UNSUPPORTED;
}

void __attribute__((weak)) ucx_plugin_cleanup(void)
{
    /* No-op: nothing to clean up when plugin is not available */
}

const uct_ib_plugin_info_t* __attribute__((weak)) ucx_plugin_get_info(void)
{
    return NULL;
}
