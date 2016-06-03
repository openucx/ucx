/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_H_
#define UCP_PROTO_H_

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_request.h>
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
    uint64_t                  reqptr;
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
typedef struct ucp_proto {
    uct_pending_callback_t     contig_short;           /* Progress short data */
    uct_pending_callback_t     contig_bcopy_single;    /* Progress bcopy single fragment */
    uct_pending_callback_t     contig_bcopy_multi;     /* Progress bcopy multi-fragment */
    uct_pending_callback_t     contig_zcopy_single;    /* Progress zcopy single fragment */
    uct_pending_callback_t     contig_zcopy_multi;     /* Progress zcopy multi-fragment */
    uct_completion_callback_t  contig_zcopy_completion;/* Callback for UCT zcopy completion */
    uct_pending_callback_t     generic_single;         /* Progress bcopy single fragment, generic dt */
    uct_pending_callback_t     generic_multi;          /* Progress bcopy multi-fragment, generic dt */
    size_t                     only_hdr_size;          /* Header size for single / short */
    size_t                     first_hdr_size;         /* Header size for first of multi */
    size_t                     mid_hdr_size;           /* Header size for rest of multi */
} ucp_proto_t;


ucs_status_t ucp_proto_progress_am_bcopy_single(uct_pending_req_t *self);


/*
 * Make sure the remote worker would be able to send replies to our endpoint.
 * Should be used before a sending a message which requires a reply.
 */
static inline void ucp_ep_connect_remote(ucp_ep_h ep)
{
    if (ucs_unlikely(!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_SENT))) {
        ucp_wireup_send_request(ep);
    }
}


#endif
