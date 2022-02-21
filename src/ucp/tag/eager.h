/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_EAGER_H_
#define UCP_TAG_EAGER_H_

#include "tag_match.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.h>
#include <ucp/dt/dt.inl>


/* Convenience macros for setting eager protocols descriptions  */
#define UCP_PROTO_EAGER_BCOPY_DESC \
    "eager " UCP_PROTO_COPY_IN_DESC " " UCP_PROTO_COPY_OUT_DESC
#define UCP_PROTO_EAGER_ZCOPY_DESC \
    "eager " UCP_PROTO_ZCOPY_DESC " " UCP_PROTO_COPY_OUT_DESC

/*
 * EAGER_ONLY, EAGER_MIDDLE
 */
typedef struct {
    ucp_tag_hdr_t             super;
} UCS_S_PACKED ucp_eager_hdr_t;


/*
 * EAGER_FIRST
 */
typedef struct {
    ucp_eager_hdr_t           super;
    size_t                    total_len;
    uint64_t                  msg_id;
} UCS_S_PACKED ucp_eager_first_hdr_t;


/*
 * EAGER_MIDDLE
 */
typedef struct {
    uint64_t                  msg_id;
    size_t                    offset;
} UCS_S_PACKED ucp_eager_middle_hdr_t;


/*
 * EAGER_SYNC_ONLY
 */
typedef struct {
    ucp_eager_hdr_t           super;
    ucp_request_hdr_t         req;
} UCS_S_PACKED ucp_eager_sync_hdr_t;


/*
 * EAGER_SYNC_FIRST
 */
typedef struct {
    ucp_eager_first_hdr_t     super;
    ucp_request_hdr_t         req;
} UCS_S_PACKED ucp_eager_sync_first_hdr_t;


extern const ucp_request_send_proto_t ucp_tag_eager_proto;
extern const ucp_request_send_proto_t ucp_tag_eager_sync_proto;

void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, void *hdr, uint16_t recv_flags);

void ucp_tag_eager_sync_completion(ucp_request_t *req, uint32_t flag,
                                   ucs_status_t status);

void ucp_proto_eager_sync_ack_handler(ucp_worker_h worker,
                                      const ucp_reply_hdr_t *rep_hdr);

void ucp_tag_eager_zcopy_completion(uct_completion_t *self);

void ucp_tag_eager_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

void ucp_tag_eager_sync_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

void ucp_tag_eager_sync_zcopy_completion(uct_completion_t *self);

static UCS_F_ALWAYS_INLINE int
ucp_proto_eager_check_op_id(const ucp_proto_init_params_t *init_params,
                            ucp_operation_id_t op_id, int offload_enabled)
{
    return (init_params->select_param->op_id == op_id) &&
           (offload_enabled ==
            ucp_ep_config_key_has_tag_lane(init_params->ep_config_key));
}

#endif
