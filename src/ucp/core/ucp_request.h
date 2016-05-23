/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_REQUEST_H_
#define UCP_REQUEST_H_

#include "ucp_context.h"

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucp/wireup/wireup.h>


/**
 * Request flags
 */
enum {
    UCP_REQUEST_FLAG_COMPLETED            = UCS_BIT(0),
    UCP_REQUEST_FLAG_RELEASED             = UCS_BIT(1),
    UCP_REQUEST_FLAG_BLOCKING             = UCS_BIT(2),
    UCP_REQUEST_FLAG_EXPECTED             = UCS_BIT(3),
    UCP_REQUEST_FLAG_LOCAL_COMPLETED      = UCS_BIT(4),
    UCP_REQUEST_FLAG_REMOTE_COMPLETED     = UCS_BIT(5),
};


/**
 * Receive descriptor flags.
 */
enum {
    UCP_RECV_DESC_FLAG_FIRST = UCS_BIT(0),
    UCP_RECV_DESC_FLAG_LAST  = UCS_BIT(1),
    UCP_RECV_DESC_FLAG_EAGER = UCS_BIT(2),
    UCP_RECV_DESC_FLAG_SYNC  = UCS_BIT(3),
    UCP_RECV_DESC_FLAG_RNDV  = UCS_BIT(4),
};


/*
 * - Mark the request as completed
 * - If it has a callback - call it.
 * - Otherwise - the request might be released - if so, return it to mpool.
 */
#define ucp_request_complete(_req, _cb, ...) \
    { \
        ucs_trace_data("completing request %p (%p) flags 0x%x", (_req), \
                       (_req) + 1, (_req)->flags); \
        (_cb)((_req) + 1, ## __VA_ARGS__); \
        if (((_req)->flags |= UCP_REQUEST_FLAG_COMPLETED) & UCP_REQUEST_FLAG_RELEASED) { \
            ucs_mpool_put(_req); \
        } \
    }

typedef struct ucp_send_state {
    size_t                        offset;
    union {
        struct {
            uct_mem_h             memh;
        } contig;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_frag_state_t;


/**
 * Request in progress.
 */
struct ucp_request {
    uint16_t                      flags;   /* Request flags */

    union {
        ucp_send_callback_t       send;
        ucp_tag_recv_callback_t   tag_recv;
    } cb;

    union {
        struct {
            ucp_ep_h              ep;
            const void            *buffer;  /* Send buffer */
            ucp_datatype_t        datatype; /* Send type */
            uct_mem_h             mem;

            union {
                ucp_tag_t         tag;      /* Tagged send */

                ucp_wireup_msg_t  wireup;

                struct {
                    uint64_t      remote_addr; /* remote address */
                    ucp_rkey_h    rkey;        /* Rkey */
                } rma;

                struct {
                    uintptr_t     remote_request;
                    uint8_t       am_id;
                    ucs_status_t  status;
                } proto;

                struct {
                    uct_pending_req_t *req;
                    ucp_stub_ep_t*    stub_ep;
                } proxy;
            };

            size_t                length;   /* Total length, in bytes */
            ucp_frag_state_t      state;
            uct_pending_req_t     uct;      /* Pending request */
            uct_completion_t      uct_comp;
        } send;

        struct {
            ucs_queue_elem_t      queue;    /* Expected queue element */
            void                  *buffer;  /* Buffer to receive data to */
            size_t                count;    /* Receive count */
            ucp_datatype_t        datatype; /* Receive type */
            ucp_tag_t             tag;      /* Expected tag */
            ucp_tag_t             tag_mask; /* Expected tag mask */
            ucp_tag_recv_info_t   info;     /* Completion info to fill */
            ucp_frag_state_t      state;
        } recv;
    };
};


/**
 * Unexpected receive descriptor.
 */
typedef struct ucp_recv_desc {
    ucs_queue_elem_t              queue;    /* Queue element */
    size_t                        length;   /* Received length */
    uint16_t                      hdr_len;  /* Header size */
    uint16_t                      flags;    /* Flags */
} ucp_recv_desc_t;


extern ucs_mpool_ops_t ucp_request_mpool_ops;

#endif
