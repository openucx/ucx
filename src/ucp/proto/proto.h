/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_H_
#define UCP_PROTO_H_

#include <ucp/core/ucp_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <ucp/dt/dt.h>


/**
 * Header segment for a transaction
 */
typedef struct {
    uint64_t                  sender_uuid;
    uintptr_t                 reqptr;
} UCS_S_PACKED ucp_request_hdr_t;


/**
 * Header for transaction acknowledgment
 */
typedef struct {
    uint64_t                  reqptr;
    ucs_status_t              status;
} UCS_S_PACKED ucp_reply_hdr_t;


/**
 * Defines functions for a protocol, on all possible data types.
 */
struct ucp_proto {
    uct_pending_callback_t     contig_short;     /**< Progress short data */
    uct_pending_callback_t     bcopy_single;     /**< Progress bcopy single fragment */
    uct_pending_callback_t     bcopy_multi;      /**< Progress bcopy multi-fragment */
    uct_pending_callback_t     zcopy_single;     /**< Progress zcopy single fragment */
    uct_pending_callback_t     zcopy_multi;      /**< Progress zcopy multi-fragment */
    uct_completion_callback_t  zcopy_completion; /**< Callback for UCT zcopy completion */
    size_t                     only_hdr_size;    /**< Header size for single / short */
    size_t                     first_hdr_size;   /**< Header size for first of multi */
    size_t                     mid_hdr_size;     /**< Header size for rest of multi */
};


ucs_status_t ucp_proto_progress_am_bcopy_single(uct_pending_req_t *self);


/*
 * Make sure the remote worker would be able to send replies to our endpoint.
 * Should be used before sending a message which requires a reply.
 */
static inline void ucp_ep_connect_remote(ucp_ep_h ep)
{
    if (ucs_unlikely(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED))) {
        ucp_wireup_send_request(ep);
    }
}

size_t ucp_dt_stride_copy_uct(uct_iov_t *iov, size_t *iovcnt, size_t max_dst_iov,
                              ucp_dt_state_t *state, const ucp_dt_iov_t *src_iov,
                              ucp_datatype_t datatype, size_t length_max);

size_t ucp_dt_iov_copy_uct(uct_iov_t *iov, size_t *iovcnt, size_t max_dst_iov,
                           ucp_dt_state_t *state, const ucp_dt_iov_t *src_iov,
                           ucp_datatype_t datatype, size_t length_max);

ucs_status_t ucp_dt_reusable_create(uct_ep_h ep, void *buffer, size_t length,
                                    ucp_datatype_t datatype, ucp_dt_state_t *state);

ucs_status_t ucp_dt_reusable_update(uct_ep_h ep, void *buffer, size_t length,
                                    ucp_datatype_t datatype, ucp_dt_state_t *state);

#endif
