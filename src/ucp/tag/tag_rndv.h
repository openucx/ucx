/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_RNDV_H_
#define UCP_TAG_RNDV_H_

#include <ucp/rndv/rndv.h>
#include <ucp/tag/tag_match.h>
#include <ucp/core/ucp_request.h>


#define ucp_tag_hdr_from_rts(_rts) \
    ({ \
        UCS_STATIC_ASSERT(sizeof((_rts)->hdr) == sizeof(ucp_tag_hdr_t)); \
        ((ucp_tag_hdr_t*)&(_rts)->hdr); \
    })


ucs_status_t
ucp_tag_send_start_rndv(ucp_request_t *req, const ucp_request_param_t *param);

void ucp_tag_rndv_matched(ucp_worker_h worker, ucp_request_t *req,
                          const ucp_rndv_rts_hdr_t *rts_hdr, size_t hdr_length);

ucs_status_t ucp_tag_rndv_process_rts(ucp_worker_h worker,
                                      ucp_rndv_rts_hdr_t *rts_hdr,
                                      size_t length, unsigned tl_flags);

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg);

ucs_status_t ucp_proto_progress_tag_rndv_rts(uct_pending_req_t *self);


static UCS_F_ALWAYS_INLINE ucp_rndv_rts_hdr_t *
ucp_tag_rndv_rts_from_rdesc(ucp_recv_desc_t *rdesc)
{
    ucs_assert(rdesc->payload_offset == sizeof(ucp_rndv_rts_hdr_t));

    return (ucp_rndv_rts_hdr_t*)(rdesc + 1);
}

#endif
