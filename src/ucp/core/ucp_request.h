/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_REQUEST_H_
#define UCP_REQUEST_H_

#include "ucp_context.h"
#include "ucp_mm.h"

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucp/dt/dt.h>
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
    UCP_REQUEST_FLAG_OFFLOADED            = UCS_BIT(10),
    UCP_REQUEST_FLAG_BLOCK_OFFLOAD        = UCS_BIT(11),

#if ENABLE_ASSERT
    UCP_REQUEST_DEBUG_FLAG_EXTERNAL       = UCS_BIT(15)
#else
    UCP_REQUEST_DEBUG_FLAG_EXTERNAL       = 0
#endif
};


/**
 * Protocols enumerator to work with send request state
 */
enum {
    UCP_REQUEST_SEND_PROTO_BCOPY_AM = 0,
    UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
    UCP_REQUEST_SEND_PROTO_RNDV_GET,
    UCP_REQUEST_SEND_PROTO_RMA
};

/**
 * Receive descriptor flags.
 */
enum {
    UCP_RECV_DESC_FLAG_FIRST    = UCS_BIT(0),
    UCP_RECV_DESC_FLAG_LAST     = UCS_BIT(1),
    UCP_RECV_DESC_FLAG_EAGER    = UCS_BIT(2),
    UCP_RECV_DESC_FLAG_SYNC     = UCS_BIT(3),
    UCP_RECV_DESC_FLAG_RNDV     = UCS_BIT(4),
    UCP_RECV_DESC_FLAG_UCT_DESC = UCS_BIT(5),
    UCP_RECV_DESC_FLAG_OFFLOAD  = UCS_BIT(6)
};


/**
 * Receive descriptor list pointers
 */
enum {
    UCP_RDESC_HASH_LIST = 0,
    UCP_RDESC_ALL_LIST  = 1
};


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
                    uintptr_t     rreq_ptr;    /* receive request ptr on the recv side */
                    uint64_t      sender_uuid; /* Sender uuid, which is sent back in sync ack */
                    ucp_tag_t     sender_tag;  /* Sender tag, which is sent back in sync ack */
                } proto;

                struct {
                    uct_pending_req_t    *req;
                    ucp_wireup_ep_t      *wireup_ep;
                } proxy;

                struct {
                    uint64_t      remote_address; /* address of the sender's data buffer */
                    uintptr_t     remote_request; /* pointer to the sender's send request */
                    uct_rkey_bundle_t rkey_bundle;
                    ucp_request_t *rreq;    /* receive request on the recv side */
                } rndv_get;

                struct {
                    ucp_request_callback_t flushed_cb;/* Called when flushed */
                    uct_worker_cb_id_t     prog_id;   /* Progress callback ID */
                    ucp_lane_map_t         lanes;     /* Which lanes need to be flushed */
                    unsigned               uct_flags; /* Flags to pass to
                                                            @ref uct_ep_flush */
                } flush;

                struct {
                    uct_worker_cb_id_t        prog_id;/* Slow-path callback */
                } disconnect;

                struct {
                    uint64_t              remote_addr; /* Remote address */
                    ucp_atomic_fetch_op_t op; /* Requested AMO */
                    ucp_rkey_h            rkey;     /* Remote memory key */
                    uint64_t              value;
                    void                  *result;
                } amo;

                struct {
                    ucs_queue_elem_t  queue;     /* Elem in outgoing ssend reqs queue */
                    ucp_tag_t         ssend_tag; /* Tag in offload sync send */
                    void              *rndv_op;  /* Handler of issued rndv send. Need to cancel
                                                    the operation if it is completed by SW. */
                 } tag_offload;

            };

            /* This structure holds all mutable fields, and everything else
             * except common send/recv fields 'status' and 'flags' is
             * immutable
             * TODO: rework RMA case where length is used instead of dt.offset */
            struct {
                ucp_dt_state_t    dt;       /* Position in the send buffer */
                uct_completion_t  uct_comp; /* UCT completion */
            } state;

            ucp_lane_index_t      lane;     /* Lane on which this request is being sent */
            ucp_rsc_index_t       reg_rsc;  /* Resource on which memory is registered */
            uct_pending_req_t     uct;      /* UCT pending request */
        } send;

        struct {
            ucs_queue_elem_t      queue;    /* Expected queue element */
            void                  *buffer;  /* Buffer to receive data to */
            ucp_datatype_t        datatype; /* Receive type */
            size_t                length;   /* Total length, in bytes */
            ucp_dt_state_t        state;
            ucp_worker_t          *worker;
            ucp_mem_desc_t        *rdesc;
            uct_tag_context_t     uct_ctx;  /* Transport offload context */

            union {
                struct {
                    ucp_tag_t               tag;      /* Expected tag */
                    ucp_tag_t               tag_mask; /* Expected tag mask */
                    uint64_t                sn;       /* Tag match sequence */
                    ucp_tag_recv_callback_t cb;       /* Completion callback */
                    ucp_tag_recv_info_t     info;     /* Completion info to fill */
                } tag;

                struct {
                    ucp_stream_recv_callback_t cb;     /* Completion callback */
                    size_t                     count;  /* Completion info to fill */
                } stream;
            };
        } recv;

        struct {
            ucp_worker_h          worker;   /* Worker to flush */
            ucp_send_callback_t   cb;       /* Completion callback */
            uct_worker_cb_id_t    prog_id;  /* Progress callback ID */
        } flush_worker;
    };
};


/**
 * Unexpected receive descriptor.
 */
struct ucp_recv_desc {
    union {
        ucs_list_link_t           tag_list[2];  /* Hash list TAG-element */
        ucs_queue_elem_t          stream_queue; /* Queue STREAM-element */
    };
    size_t                        length;   /* Received length */
    uint16_t                      hdr_len;  /* Header size */
    uint16_t                      flags;    /* Flags */
};


extern ucs_mpool_ops_t ucp_request_mpool_ops;


int ucp_request_pending_add(ucp_request_t *req, ucs_status_t *req_status);

ucs_status_t ucp_request_send_buffer_reg(ucp_request_t *req, ucp_lane_index_t lane);

void ucp_request_send_buffer_dereg(ucp_request_t *req, ucp_lane_index_t lane);

ucs_status_t ucp_request_memory_reg(ucp_context_t *context, ucp_rsc_index_t rsc_index,
                                    void *buffer, size_t length,
                                    ucp_datatype_t datatype, ucp_dt_state_t *state);

void ucp_request_memory_dereg(ucp_context_t *context, ucp_rsc_index_t rsc_index,
                              ucp_datatype_t datatype, ucp_dt_state_t *state);

ucs_status_t ucp_request_send_start(ucp_request_t *req, ssize_t max_short,
                                    size_t zcopy_thresh, size_t multi_thresh,
                                    size_t rndv_thresh, const ucp_proto_t *proto);

#endif
