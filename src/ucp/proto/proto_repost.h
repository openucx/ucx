/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_REPOST_H_
#define UCP_PROTO_REPOST_H_

#include <ucp/core/ucp_types.h>
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>


/* Repost undelivered operation @a op_info. */
ucs_status_t ucp_proto_repost_submit(ucp_ep_h ep,
                                     const uct_ep_op_info_t *op_info);

#endif
