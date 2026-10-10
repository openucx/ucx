/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_repost.h"

#include <ucs/sys/compiler_def.h>


/* Stand-in until the repost protocol is available. */
ucs_status_t
ucp_proto_repost_submit(ucp_ep_h UCS_V_UNUSED ep,
                        const uct_ep_op_info_t *UCS_V_UNUSED op_info)
{
    return UCS_ERR_UNREACHABLE;
}
