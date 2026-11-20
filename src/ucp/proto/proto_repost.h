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

/* Keep logical element identity until a native SGL batch is completed.
 * NULL indices denotes consecutive elements starting at the iterator offset;
 * otherwise count is at most UCP_MAX_IOV and indices is copied. */
uct_completion_t *ucp_proto_repost_sgl_completion(ucp_request_t *origin,
                                                  const size_t *indices,
                                                  size_t count);

/* Post a native PUT batch with its replay identity. */
ucs_status_t
ucp_proto_repost_sgl_put(ucp_request_t *origin, uct_ep_h ep,
                         void *const *buffers, const size_t *lengths,
                         uct_mem_h const *memhs, const uint64_t *remote_addrs,
                         uct_rkey_t const *rkeys, size_t count,
                         const size_t *indices);

/* Release an unposted batch, including NO_RESOURCE; do not complete origin. */
void ucp_proto_repost_sgl_release(uct_completion_t *comp);

#endif
