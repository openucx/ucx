/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_RNDV_H_
#define UCP_TAG_RNDV_H_

#include "tag_match.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_ep.inl>


/*
 * Rendezvous RTS
 */
typedef struct {
    ucp_tag_hdr_t             super;
    ucp_request_hdr_t         sreq;     /* send request on the rndv initiator side */
    uint64_t                  address;  /* holds the address of the data buffer on the sender's side */
    size_t                    size;     /* size of the data for sending */
    /* packed rkeys follow */
} UCS_S_PACKED ucp_rndv_rts_hdr_t;

/*
 * Rendezvous RTR
 */
typedef struct {
    uintptr_t                 sreq_ptr; /* request on the rndv initiator side - sender */
    uintptr_t                 rreq_ptr; /* request on the rndv receiver side */
    uint64_t                  address;  /* holds the address of the data buffer on the receiver's side */
    size_t                    size;     /* size of the data to receive */
    size_t                    offset;   /* offset of the data in the recv buffer */
    /* packed rkeys follow */
} UCS_S_PACKED ucp_rndv_rtr_hdr_t;

/*
 * RNDV_DATA
 */
typedef struct {
    uintptr_t                 rreq_ptr; /* request on the rndv receiver side */
    size_t                    offset;
} UCS_S_PACKED ucp_rndv_data_hdr_t;


ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *req);

void ucp_rndv_matched(ucp_worker_h worker, ucp_request_t *req,
                      const ucp_rndv_rts_hdr_t *rndv_rts_hdr);

ucs_status_t ucp_rndv_progress_rma_get_zcopy(uct_pending_req_t *self);

ucs_status_t ucp_rndv_progress_rma_put_zcopy(uct_pending_req_t *self);

ucs_status_t ucp_rndv_process_rts(void *arg, void *data, size_t length,
                                  unsigned tl_flags);

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg);

ucs_status_t ucp_tag_rndv_reg_send_buffer(ucp_request_t *sreq);

static UCS_F_ALWAYS_INLINE int ucp_rndv_is_get_zcopy(ucs_memory_type_t mem_type,
                                                     ucp_rndv_mode_t rndv_mode)
{
    return ((rndv_mode == UCP_RNDV_MODE_GET_ZCOPY) ||
            ((rndv_mode == UCP_RNDV_MODE_AUTO) &&
             (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type) ||
              UCP_MEM_IS_ROCM(mem_type))));
}

#endif
