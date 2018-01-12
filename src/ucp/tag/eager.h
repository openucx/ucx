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
#include <ucp/proto/proto.h>


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


extern const ucp_proto_t ucp_tag_eager_proto;
extern const ucp_proto_t ucp_tag_eager_sync_proto;

void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, void *hdr, uint16_t flags);

void ucp_tag_eager_sync_completion(ucp_request_t *req, uint16_t flag,
                                   ucs_status_t status);

void ucp_tag_eager_zcopy_completion(uct_completion_t *self, ucs_status_t status);

void ucp_tag_eager_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

void ucp_tag_eager_sync_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

void ucp_tag_eager_sync_zcopy_completion(uct_completion_t *self, ucs_status_t status);

static inline ucs_status_t ucp_tag_send_eager_short(ucp_ep_t *ep, ucp_tag_t tag,
                                                    const void *buffer, size_t length)
{
    if (ep->flags & UCP_EP_FLAG_TAG_OFFLOAD_ENABLED) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uct_tag_t));
        return uct_ep_tag_eager_short(ucp_ep_get_tag_uct_ep(ep), tag, buffer, length);
    } else {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(ucp_eager_hdr_t));
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uint64_t));
        return uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_EAGER_ONLY, tag,
                               buffer, length);
    }
}

#endif
