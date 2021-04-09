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


typedef enum {
    /* RNDV TAG operation with status UCS_OK (kept for wire compatibility with
     * the previous UCP versions) */
    UCP_RNDV_RTS_TAG_OK       = UCS_OK,
    /* RNDV TAG operation with status UCS_ERR_CANCELED (kept for wire
     * compatibility with the previous UCP versions) */
    UCP_RNDV_RTS_TAG_CANCELED = (uint8_t)UCS_ERR_CANCELED,
    /* RNDV AM oepration */
    UCP_RNDV_RTS_AM           = 1
} UCS_S_PACKED ucp_rndv_rts_opcode_t;


/*
 * Rendezvous RTS
 */
typedef struct {
    /* Protocol-specific header */
    uint64_t          hdr;
    /* Send request on the rndv initiator side */
    ucp_request_hdr_t sreq;
    /* Holds the address of the data buffer on the sender's side */
    uint64_t          address;
    /* Size of the data for sending */
    size_t            size;
    /* RNDV proto opcode */
    uint8_t           opcode;
    /*
     * 1. Packed rkeys follow
     * 2. AM only: User header follows, if am->header_length is not 0
     */
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
 * Rendezvous data
 */
typedef struct {
    uint64_t                  rreq_id; /* request ID on the rndv receiver side */
    size_t                    offset;
} UCS_S_PACKED ucp_rndv_data_hdr_t;


ucs_status_t ucp_rndv_send_rts(ucp_request_t *sreq, uct_pack_callback_t pack_cb,
                               size_t rts_body_size);

void ucp_rndv_req_send_ack(ucp_request_t *ack_req, ucp_request_t *req,
                           ucs_ptr_map_key_t remote_req_id, ucs_status_t status,
                           ucp_am_id_t am_id, const char *ack_str);

ucs_status_t ucp_rndv_progress_rma_get_zcopy(uct_pending_req_t *self);

ucs_status_t ucp_rndv_progress_rma_put_zcopy(uct_pending_req_t *self);

size_t ucp_rndv_rts_pack(ucp_request_t *sreq, ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                         ucp_rndv_rts_opcode_t opcode);

ucs_status_t ucp_rndv_reg_send_buffer(ucp_request_t *sreq);

void ucp_rndv_receive(ucp_worker_h worker, ucp_request_t *rreq,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr,
                      const void *rkey_buf);

ucs_status_t ucp_rndv_rts_handle_status_from_pending(ucp_request_t *sreq,
                                                     ucs_status_t status);

static UCS_F_ALWAYS_INLINE int
ucp_rndv_rts_is_am(const ucp_rndv_rts_hdr_t *rts_hdr)
{
    return rts_hdr->opcode == UCP_RNDV_RTS_AM;
}

static UCS_F_ALWAYS_INLINE int
ucp_rndv_rts_is_tag(const ucp_rndv_rts_hdr_t *rts_hdr)
{
    return (rts_hdr->opcode == UCP_RNDV_RTS_TAG_OK) ||
           (rts_hdr->opcode == UCP_RNDV_RTS_TAG_CANCELED);
}

#endif
