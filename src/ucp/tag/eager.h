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


/*
 * EAGER_ONLY, EAGER_MIDDLE
 */
typedef struct {
    ucp_tag_hdr_t             super;
    uint64_t                  ep_id;
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
    uint64_t                  ep_id;
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

static UCS_F_ALWAYS_INLINE void
ucp_add_uct_iov_elem(uct_iov_t *iov, void *buffer, size_t length,
                     uct_mem_h memh, size_t *iov_cnt)
{
    iov[*iov_cnt].buffer = buffer;
    iov[*iov_cnt].length = length;
    iov[*iov_cnt].count  = 1;
    iov[*iov_cnt].stride = 0;
    iov[*iov_cnt].memh   = memh;
    ++(*iov_cnt);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_send_am_short_iov(ucp_ep_h ep, const void *buffer, size_t length,
                          ucp_tag_t tag)
{
    size_t iov_cnt      = 0ul;
    ucp_eager_hdr_t hdr = { .super.tag = tag,
                            .ep_id     = ucp_ep_remote_id(ep) };
    uct_iov_t iov[2];

    ucp_add_uct_iov_elem(iov, &hdr, sizeof(hdr), UCT_MEM_HANDLE_NULL, &iov_cnt);
    ucp_add_uct_iov_elem(iov, (void*)buffer, length, UCT_MEM_HANDLE_NULL,
                         &iov_cnt);
    return uct_ep_am_short_iov(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_EAGER_ONLY,
                               iov, iov_cnt);
}

#endif
