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
ucs_status_t
ucp_proto_repost_submit(ucp_ep_h ep, const uct_ep_op_info_t *op_info);

/* Initialize replay batch metadata; chunks are allocated on first use. */
ucs_status_t ucp_proto_repost_init_mpool(ucp_worker_h worker);

/* Post a native PUT/GET batch with its replay identity. NULL indices denotes
 * consecutive elements starting at the iterator offset; otherwise count is
 * at most UCP_MAX_IOV and indices is copied. */
ucs_status_t ucp_proto_repost_sgl_zcopy(
        ucp_request_t *origin, uct_ep_h ep, uct_ep_operation_t operation,
        void *const *buffers, const size_t *lengths, uct_mem_h const *memhs,
        const uint64_t *remote_addrs, uct_rkey_t const *rkeys, size_t count,
        const size_t *indices);

#endif
