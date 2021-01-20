/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RNDV_H_
#define UCP_RNDV_H_

#include <ucp/core/ucp_types.h>
#include <ucp/proto/proto_am.h>
#include <ucs/datastruct/ptr_map.h>


enum ucp_rndv_rts_flags {
    UCP_RNDV_RTS_FLAG_TAG = UCS_BIT(0),
    UCP_RNDV_RTS_FLAG_AM  = UCS_BIT(1)
};


/*
 * Rendezvous RTS
 */
typedef struct {
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
                         size_t rndv_rts_hdr_size, uint16_t flags);

ucs_status_t ucp_rndv_reg_send_buffer(ucp_request_t *sreq);

void ucp_rndv_receive(ucp_worker_h worker, ucp_request_t *rreq,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                      const void *rkey_buf);

void ucp_rndv_req_send_ats(ucp_request_t *rndv_req, ucp_request_t *rreq,
                           ucs_ptr_map_key_t remote_req_id, ucs_status_t status);

ucs_status_t ucp_rndv_rts_handle_status_from_pending(ucp_request_t *sreq,
                                                     ucs_status_t status);

#endif
