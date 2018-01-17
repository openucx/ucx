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

void ucp_proto_am_zcopy_completion(uct_completion_t *self, ucs_status_t status);

void ucp_proto_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

/*
 * Make sure the remote worker would be able to send replies to our endpoint.
 * Should be used before sending a message which requires a reply.
 */
static inline void ucp_ep_connect_remote(ucp_ep_h ep)
{
    if (ucs_unlikely(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED))) {
        ucs_assert(ep->flags & UCP_EP_FLAG_DEST_UUID_PEER);
        ucp_wireup_send_request(ep, ep->dest_uuid);
    }
}


#endif
