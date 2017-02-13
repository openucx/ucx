/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
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
    UCP_REQUEST_FLAG_CALLBACK             = UCS_BIT(6),
    UCP_REQUEST_FLAG_RECV                 = UCS_BIT(7),
    UCP_REQUEST_FLAG_SYNC                 = UCS_BIT(8),
    UCP_REQUEST_FLAG_RNDV                 = UCS_BIT(9),

#if ENABLE_ASSERT
    UCP_REQUEST_DEBUG_FLAG_EXTERNAL       = UCS_BIT(15)
#else
    UCP_REQUEST_DEBUG_FLAG_EXTERNAL       = 0
#endif
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


/* Callback for UCP requests */
typedef void (*ucp_request_callback_t)(ucp_request_t *req);


typedef struct ucp_frag_state {
    size_t                        offset;  /* Total offset in overall payload. */
    union {
        struct {
            uct_mem_h             memh;
        } contig;
        struct {
            size_t                iov_offset;     /* Offset in the IOV item */
            size_t                iovcnt_offset;  /* The IOV item to start copy */
            size_t                iovcnt;         /* Number of IOV buffers */
            uct_mem_h             *memh;          /* Pointer to IOV memh[iovcnt] */
        } iov;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_frag_state_t;


/**
 * Request in progress.
 */
struct ucp_request {
    ucs_status_t                  status;  /* Operation status */
    uint16_t                      flags;   /* Request flags */

    union {
        struct {
            ucp_ep_h              ep;
            const void            *buffer;  /* Send buffer */
            ucp_datatype_t        datatype; /* Send type */
            size_t                length;   /* Total length, in bytes */
            ucp_send_callback_t   cb;       /* Completion callback */

            union {
                ucp_tag_t         tag;      /* Tagged send */
                ucp_wireup_msg_t  wireup;

                struct {
                    uint64_t      remote_addr; /* Remote address */
                    ucp_rkey_h    rkey;     /* Remote memory key */
                } rma;

                struct {
                    uintptr_t     remote_request; /* pointer to the send request on receiver side */
                    uint8_t       am_id;
                    ucs_status_t  status;
                    uintptr_t     rreq_ptr; /* receive request ptr on the recv side */
                } proto;

                struct {
                    uct_pending_req_t *req;
                    ucp_stub_ep_t*    stub_ep;
                } proxy;

                struct {
                    uint64_t      remote_address; /* address of the sender's data buffer */
                    uintptr_t     remote_request; /* pointer to the sender's send request */
                    uct_rkey_bundle_t rkey_bundle;
                    ucp_request_t *rreq;    /* receive request on the recv side */
                } rndv_get;

                struct {
                    ucp_request_callback_t    flushed_cb;/* Called when flushed */
                    ucs_callbackq_slow_elem_t cbq_elem;  /* Slow-path callback */
                    uint8_t                   cbq_elem_on;
                    ucp_lane_map_t            lanes;     /* Which lanes need to be flushed */
                } flush;
                struct {
                    uint64_t              remote_addr; /* Remote address */
                    ucp_atomic_fetch_op_t op; /* Requested AMO */
                    ucp_rkey_h            rkey;     /* Remote memory key */
                    uint64_t              value;
                    void                  *result;
                } amo;
            };

            ucp_lane_index_t      lane;     /* Lane on which this request is being sent */
            ucp_frag_state_t      state;    /* Position in the send buffer */
            uct_pending_req_t     uct;      /* UCT pending request */
            uct_completion_t      uct_comp; /* UCT completion */
        } send;

        struct {
            ucs_queue_elem_t      queue;    /* Expected queue element */
            void                  *buffer;  /* Buffer to receive data to */
            ucp_datatype_t        datatype; /* Receive type */
            size_t                length;   /* Total length, in bytes */
            ucp_tag_t             tag;      /* Expected tag */
            ucp_tag_t             tag_mask; /* Expected tag mask */
            ucp_tag_recv_callback_t cb;     /* Completion callback */
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

/**
 * Start sending a request.
 *
 * @param [in]  req   Request to start.
 *
 * @return UCS_OK - completed (callback will not be called)
 *         UCS_INPROGRESS - started but not completed
 *         other error - failure
 */
ucs_status_t ucp_request_start_send(ucp_request_t *req);

int ucp_request_pending_add(ucp_request_t *req, ucs_status_t *req_status);

void ucp_request_release_pending_send(uct_pending_req_t *self, void *arg);

ucs_status_t ucp_request_send_buffer_reg(ucp_request_t *req, ucp_lane_index_t lane);

void ucp_request_send_buffer_dereg(ucp_request_t *req, ucp_lane_index_t lane);

#endif
