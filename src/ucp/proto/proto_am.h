/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_AM_H_
#define UCP_PROTO_AM_H_

#include <ucp/core/ucp_types.h>
#include <ucs/sys/compiler.h>


/**
 * Header segment for a transaction
 */
typedef struct {
    uint64_t                  ep_id;
    uint64_t                  req_id;
} UCS_S_PACKED ucp_request_hdr_t;


/**
 * Header for transaction acknowledgment
 */
typedef struct {
    uint64_t                  req_id;
    ucs_status_t              status;
} UCS_S_PACKED ucp_reply_hdr_t;


/**
 * Header for transaction with offset
 */
typedef struct {
    uint64_t req_id;
    size_t   offset;
} UCS_S_PACKED ucp_request_data_hdr_t;


ucs_status_t
ucp_do_am_single(uct_pending_req_t *self, uint8_t am_id,
                 uct_pack_callback_t pack_cb, ssize_t max_packed_size);

ucs_status_t ucp_proto_progress_am_single(uct_pending_req_t *self);

void ucp_proto_am_zcopy_completion(uct_completion_t *self);

void ucp_proto_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

#endif
