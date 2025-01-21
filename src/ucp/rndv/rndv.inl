/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

#ifndef UCP_RNDV_RNDV_INL_
#define UCP_RNDV_RNDV_INL_

#include "proto_rndv.h"

#include <ucp/core/ucp_worker.h>


static UCS_F_ALWAYS_INLINE int
ucp_rndv_rts_is_am(const ucp_rndv_rts_hdr_t *rts_hdr)
{
    return rts_hdr->opcode == UCP_RNDV_RTS_AM;
}

static UCS_F_ALWAYS_INLINE int
ucp_rndv_rts_is_tag(const ucp_rndv_rts_hdr_t *rts_hdr)
{
    return rts_hdr->opcode == UCP_RNDV_RTS_TAG_OK;
}

static UCS_F_ALWAYS_INLINE void
ucp_rndv_receive_start(ucp_worker_h worker, ucp_request_t *rreq,
                       const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                       const void *rkey_buf, size_t rkey_length)
{
    if (worker->context->config.ext.proto_enable) {
        ucp_proto_rndv_receive_start(worker, rreq, rndv_rts_hdr, rkey_buf,
                                     rkey_length);
    } else {
        ucp_rndv_receive(worker, rreq, rndv_rts_hdr, rkey_buf);
    }
}

#endif
