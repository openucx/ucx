/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RNDV_H_
#define UCP_RNDV_H_

#include <ucp/api/ucp.h>
#include <ucp/tag/tag_match.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_ep.inl>


enum ucp_rndv_rts_flags {
    UCP_RNDV_RTS_FLAG_TAG = UCS_BIT(0)
};


/*
 * Rendezvous RTS
 */
typedef struct {
    ucp_tag_hdr_t             tag;      /* Tag used by TAG API calls */
    ucp_request_hdr_t         sreq;     /* send request on the rndv initiator side */
    uint64_t                  address;  /* holds the address of the data buffer on the sender's side */
    size_t                    size;     /* size of the data for sending */
    uint16_t                  flags;    /* rndv proto flags, as defined by
                                           ucp_rndv_rts_flags */
} UCS_S_PACKED ucp_rndv_rts_hdr_t;


/*
 * Rendezvous RTR
 */
typedef struct {
    uint64_t                  sreq_id;  /* request ID on the rndv initiator side - sender */
    uint64_t                  rreq_id;  /* request ID on the rndv receiver side */
    uint64_t                  address;  /* holds the address of the data buffer on the receiver's side */
    size_t                    size;     /* size of the data to receive */
    size_t                    offset;   /* offset of the data in the recv buffer */
    /* packed rkeys follow */
} UCS_S_PACKED ucp_rndv_rtr_hdr_t;


/*
 * RNDV_DATA
 */
typedef struct {
    uint64_t                  rreq_id; /* request ID on the rndv receiver side */
    size_t                    offset;
} UCS_S_PACKED ucp_rndv_data_hdr_t;


ucs_status_t ucp_rndv_progress_rma_get_zcopy(uct_pending_req_t *self);

ucs_status_t ucp_rndv_progress_rma_put_zcopy(uct_pending_req_t *self);

size_t ucp_rndv_rts_pack(ucp_request_t *sreq, ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                         uint16_t flags);

ucs_status_t ucp_rndv_reg_send_buffer(ucp_request_t *sreq);

void ucp_rndv_receive(ucp_worker_h worker, ucp_request_t *rreq,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr);

static UCS_F_ALWAYS_INLINE int
ucp_rndv_is_get_zcopy(ucp_request_t *req, ucp_context_h context)
{
    return ((context->config.ext.rndv_mode == UCP_RNDV_MODE_GET_ZCOPY) ||
            ((context->config.ext.rndv_mode == UCP_RNDV_MODE_AUTO) &&
             (!UCP_MEM_IS_CUDA(req->send.mem_type) ||
              (req->send.length < context->config.ext.rndv_pipeline_send_thresh))));
}

#endif
