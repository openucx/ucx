/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "uct_ib_plugin.h"

uint64_t __attribute__((weak)) uct_ib_plugin_iface_flags(void)
{
    return 0;
}

ucs_status_t __attribute__((weak)) uct_ib_plugin_query_token(
    const uct_ib_plugin_qp_ctx_t *qp_ctx, uct_ep_attr_t *ep_attr)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}
