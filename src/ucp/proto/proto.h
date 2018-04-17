/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_H_
#define UCP_PROTO_H_

#include <ucp/core/ucp_ep.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>


/**
 * Header segment for a transaction
 */
typedef struct {
    uintptr_t                 ep_ptr;
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

void ucp_proto_am_zcopy_completion(uct_completion_t *self, ucs_status_t status);

void ucp_proto_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

#endif
