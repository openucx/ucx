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


/*
 * TAG API Rendezvous RTS header
 */
typedef struct {
    ucp_rndv_rts_hdr_t        super;
    ucp_tag_hdr_t             tag;
    /* packed rkeys follows */
} UCS_S_PACKED ucp_tag_rndv_rts_hdr_t;


ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *req);

void ucp_tag_rndv_matched(ucp_worker_h worker, ucp_request_t *req,
                          const ucp_tag_rndv_rts_hdr_t *rndv_rts_hdr);

ucs_status_t ucp_tag_rndv_process_rts(ucp_worker_h worker,
                                      ucp_rndv_rts_hdr_t *rts_hdr,
                                      size_t length, unsigned tl_flags);

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg);

/* In case of RNDV, there is a tag(ucp_tag_t) right after rdesc and before
 * the TAG RTS header (ucp_tag_rndv_rts_hdr_t). It is needed to unify all
 * TAG API protocol headers and make search in unexpected queue fast.
 */
static UCS_F_ALWAYS_INLINE ucp_tag_rndv_rts_hdr_t*
ucp_tag_rndv_rts_from_rdesc(ucp_recv_desc_t *rdesc)
{
    ucs_assert(rdesc->payload_offset ==
               (sizeof(ucp_tag_rndv_rts_hdr_t) + sizeof(ucp_tag_t)));

    return UCS_PTR_BYTE_OFFSET(rdesc + 1, sizeof(ucp_tag_t));
}

#endif
