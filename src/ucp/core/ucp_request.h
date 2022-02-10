/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_REQUEST_H_
#define UCP_REQUEST_H_

#include "ucp_context.h"
#include "ucp_mm.h"

#include <ucp/api/ucp.h>
#include <ucp/dt/datatype_iter.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/debug/assert.h>
#include <ucp/dt/dt.h>
#include <ucp/rma/rma.h>
#include <ucp/wireup/wireup.h>
#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_worker.h>


#define ucp_trace_req(_sreq, _message, ...) \
    ucs_trace_req("req %p: " _message, (_sreq), ## __VA_ARGS__)


/**
 * Request flags
 */
enum {
    UCP_REQUEST_FLAG_COMPLETED             = UCS_BIT(0),
    UCP_REQUEST_FLAG_RELEASED              = UCS_BIT(1),
    UCP_REQUEST_FLAG_PROTO_SEND            = UCS_BIT(2),
    UCP_REQUEST_FLAG_USER_MEMH             = UCS_BIT(3),
    UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED  = UCS_BIT(4),
    UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED = UCS_BIT(5),
    UCP_REQUEST_FLAG_CALLBACK              = UCS_BIT(6),
    UCP_REQUEST_FLAG_PROTO_INITIALIZED     = UCS_BIT(7),
    UCP_REQUEST_FLAG_SYNC                  = UCS_BIT(8),
    UCP_REQUEST_FLAG_OFFLOADED             = UCS_BIT(10),
    UCP_REQUEST_FLAG_BLOCK_OFFLOAD         = UCS_BIT(11),
    UCP_REQUEST_FLAG_STREAM_RECV_WAITALL   = UCS_BIT(12),
    UCP_REQUEST_FLAG_SEND_AM               = UCS_BIT(13),
    UCP_REQUEST_FLAG_SEND_TAG              = UCS_BIT(14),
    UCP_REQUEST_FLAG_RNDV_FRAG             = UCS_BIT(15),
    UCP_REQUEST_FLAG_RECV_AM               = UCS_BIT(16),
    UCP_REQUEST_FLAG_RECV_TAG              = UCS_BIT(17),
    UCP_REQUEST_FLAG_RKEY_INUSE            = UCS_BIT(18),
#if UCS_ENABLE_ASSERT
    UCP_REQUEST_FLAG_STREAM_RECV           = UCS_BIT(19),
    UCP_REQUEST_DEBUG_FLAG_EXTERNAL        = UCS_BIT(20),
    UCP_REQUEST_FLAG_SUPER_VALID           = UCS_BIT(21)
#else
    UCP_REQUEST_FLAG_STREAM_RECV           = 0,
    UCP_REQUEST_DEBUG_FLAG_EXTERNAL        = 0,
    UCP_REQUEST_FLAG_SUPER_VALID           = 0
#endif
};


/**
 * Protocols enumerator to work with send request state
 */
enum {
    UCP_REQUEST_SEND_PROTO_BCOPY_AM = 0,
    UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
    UCP_REQUEST_SEND_PROTO_RNDV_GET,
    UCP_REQUEST_SEND_PROTO_RNDV_PUT,
    UCP_REQUEST_SEND_PROTO_RMA
};


/**
 * Receive descriptor flags.
 */
enum {
    UCP_RECV_DESC_FLAG_UCT_DESC         = UCS_BIT(0), /* Descriptor allocated by UCT */
    UCP_RECV_DESC_FLAG_EAGER            = UCS_BIT(1), /* Eager tag message */
    UCP_RECV_DESC_FLAG_EAGER_ONLY       = UCS_BIT(2), /* Eager tag message with single fragment */
    UCP_RECV_DESC_FLAG_EAGER_SYNC       = UCS_BIT(3), /* Eager tag message which requires reply */
    UCP_RECV_DESC_FLAG_EAGER_OFFLOAD    = UCS_BIT(4), /* Eager tag from offload */
    UCP_RECV_DESC_FLAG_EAGER_LAST       = UCS_BIT(5), /* Last fragment of eager tag message.
                                                         Used by tag offload protocol. */
    UCP_RECV_DESC_FLAG_RNDV             = UCS_BIT(6), /* Rendezvous request */
    UCP_RECV_DESC_FLAG_RECV_STARTED     = UCS_BIT(7), /* Used in two different flows:
                                                         1) AM: receive operation on this
                                                            descriptor was initiated by
                                                            ucp_am_recv_data_nbx
                                                         2) TAG offload eager: multi fragment
                                                            eager message is being received, but
                                                            not all fragments received yet. Once
                                                            all fragments arrive, this flag is cleared.
                                                            Note: it is set for the first fragment only. */
    UCP_RECV_DESC_FLAG_MALLOC           = UCS_BIT(8), /* Descriptor was allocated with malloc
                                                         and must be freed, not returned to the
                                                         memory pool or UCT */
    UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS = UCS_BIT(9), /* Descriptor should not be released,
                                                         because UCT AM callback is still in
                                                         the call stack and descriptor is not
                                                         initialized yet. */
    UCP_RECV_DESC_FLAG_RELEASED         = UCS_BIT(10) /* Indicates that the descriptor was
                                                         released and cannot be used. */
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
    /* Operation status */
    ucs_status_t      status;
    /* Request flags */
    uint32_t          flags;
    /* Local request ID taken from PTR MAP */
    ucs_ptr_map_key_t id;

    union {
        void                      *user_data; /* Completion user data */
        ucp_request_t             *super_req; /* Super request that is used
                                                 by protocols */
    };

    union {

        /* "send" part - used for tag_send, am_send, stream_send, put, get, and atomic
         * operations */
        struct {
            ucp_ep_h                   ep;
            union {
                void                   *buffer; /* Send buffer */
                ucp_request_callback_t flushed_cb; /* Called when flushed */
            };
            ucp_datatype_t          datatype; /* Send type */
            size_t                  length; /* Total length, in bytes */
            ucp_send_nbx_callback_t cb; /* Completion callback */
            ucs_hlist_link_t        list; /* Element in the per-EP list of UCP
                                             flush/proto requests */

            const ucp_proto_config_t *proto_config; /* Selected protocol for the request */

            /* This structure holds all mutable fields, and everything else
             * except common send/recv fields 'status' and 'flags' is immutable
             * TODO: rework RMA case where length is used instead of dt.offset */
            struct {
                union {
                    ucp_datatype_iter_t  dt_iter;  /* Send buffer state */
                    ucp_dt_state_t       dt;       /* Position in the send buffer */
                };
                union {
                    /* UCT completion, used by flush and zero-copy operations */
                    uct_completion_t uct_comp;

                    /* Used by rndv/rtr to track received data size
                     * Used by rndv/ppln to track completed fragments
                     * Used by rkey_ptr to track copied data size
                     */
                    size_t           completed_size;
                };
            } state;

            union {
                ucp_wireup_msg_t  wireup;

                struct {
                    /* Used to identify matching parts of a large message */
                    uint64_t message_id;
                    union {
                        ucp_tag_t tag;

                        struct {
                            union {
                                /* Can be union, because once header is packed to
                                 * reg_desc, it is not accessed anymore. */
                                void           *header;
                                ucp_mem_desc_t *reg_desc; /* pointer to pre-registered buffer,
                                                             used for sending header with
                                                             zcopy protocol */
                            };
                            uint32_t       header_length;
                            uint16_t       am_id;
                            uint16_t       flags;
                        } am;
                    };
                } msg_proto;

                struct {
                    uint64_t   remote_addr; /* Remote address */
                    ucp_rkey_h rkey; /* Remote memory key */
                } rma;

                struct {
                    /* Remote request ID received from a peer */
                    ucs_ptr_map_key_t      remote_req_id;
                    uint8_t                am_id;
                    ucs_status_t           status;
                    ucp_tag_t              sender_tag; /* Sender tag, which is
                                                          sent back in sync ack */
                    ucp_request_callback_t comp_cb;    /* Called to complete the
                                                          request */
                } proto;

                struct {
                    uct_pending_req_t    *req;
                    ucp_wireup_ep_t      *wireup_ep;
                } proxy;

                struct {
                    /* Remote request ID to acknowledge */
                    ucs_ptr_map_key_t remote_req_id;

                    /* Remote buffer address for get/put operation */
                    uint64_t          remote_address;

                    /* Key for remote buffer operation */
                    ucp_rkey_h        rkey;

                    union {
                        /* Descriptor for staging rendezvous data */
                        ucp_mem_desc_t *mdesc;

                        /* Pointer for access to remote memory */
                        void           *rkey_ptr_addr;
                    };

                    union {
                        /* Used by "old" rendezvous protocols, in rndv.c */
                        struct {
                            /* Actual lanes map */
                            ucp_lane_map_t lanes_map_all;

                            /* Actual lanes count */
                            uint8_t        lanes_count;

                            /* Remote key index map */
                            uint8_t        rkey_index[UCP_MAX_LANES];
                        };

                        /* Used by "new" rendezvous protocols, in proto_rndv.c */
                        struct {
                            /* Data start offset of this request */
                            size_t offset;

                            union {
                                /* Used by rndv/put and rndv/put/frag */
                                struct {
                                    /* Which lanes need to flush (0 in fence mode) */
                                    ucp_lane_map_t flush_map;

                                    /* Which lanes need to send atp */
                                    ucp_lane_map_t atp_map;
                                } put;

                                /* Used by rndv/send/ppln and rndv/recv/ppln */
                                struct {
                                    /* Size to send in ack message */
                                    size_t ack_data_size;
                                } ppln;

                                /* Used by rndv/rkey_ptr */
                                struct {
                                    /* Element in queue for segmented RKEY ptr */
                                    ucs_queue_elem_t  queue_elem;
                                } rkey_ptr;
                            };
                        };
                    };
                } rndv;

                struct {
                    /* Remote request ID received from a peer */
                    ucs_ptr_map_key_t remote_req_id;
                } rndv_data;

                struct {
                    /* Element in queue for segmented RKEY ptr */
                    ucs_queue_elem_t  queue_elem;
                    /* Remote request ID received from a peer */
                    ucs_ptr_map_key_t remote_req_id;
                    /* Key for remote send buffer */
                    ucp_rkey_h        rkey;
                } rkey_ptr;

                struct {
                    /* The length of the data that should be fetched from sender
                     * side */
                    size_t            length;
                    /* Offset in the receiver's buffer */
                    size_t            offset;
                } rndv_rtr;

                struct {
                    unsigned           uct_flags; /* Flags to pass to @ref uct_ep_flush */
                    uct_worker_cb_id_t prog_id; /* Progress callback ID */
                    uint32_t           cmpl_sn; /* Sequence number of the remote completion
                                                   this request is waiting for */
                    uint8_t            sw_started;
                    uint8_t            sw_done;
                    uint8_t            num_lanes; /* How many lanes are being flushed */
                    ucp_lane_map_t     started_lanes; /* Which lanes need were flushed */
                } flush;

                struct {
                    ucp_worker_h       worker;
                } invalidate;

                struct {
                    /* UCT EP that should be flushed and destroyed */
                    uct_ep_h           uct_ep;
                    /* Flags that should be passed into @ref uct_ep_flush */
                    unsigned           ep_flush_flags;
                    /* Progress ID, if it's UCS_CALLBACKQ_ID_NULL, no operations
                     * are in-progress */
                    uct_worker_cb_id_t cb_id;
                    /* Index of UCT EP to be flushed and destroyed */
                    ucp_rsc_index_t    rsc_index;
                } discard_uct_ep;

                struct {
                    uint64_t              remote_addr; /* Remote address */
                    ucp_rkey_h            rkey;        /* Remote memory key */
                    uint64_t              value;       /* Atomic argument */
                    uct_atomic_op_t       uct_op;      /* Requested UCT AMO */
                } amo;

                struct {
                    ucs_queue_elem_t  queue;     /* Elem in outgoing ssend reqs queue */
                    ucp_tag_t         ssend_tag; /* Tag in offload sync send */
                    void              *rndv_op;  /* Handler of issued rndv send. Need to cancel
                                                    the operation if it is completed by SW. */
                } tag_offload;

                struct {
                    /* Remote request ID received from a peer */
                    ucs_ptr_map_key_t remote_req_id;
                } get_reply;

                struct {
                    /* Remote request ID received from a peer */
                    ucs_ptr_map_key_t  remote_req_id;
                    /* Atomic reply data */
                    ucp_atomic_reply_t data;
                } atomic_reply;
            };

            union {
                ucp_lane_index_t  am_bw_index;     /* AM BW lane index */
                ucp_lane_index_t  multi_lane_idx;  /* Index of the lane with multi-send */
                ucp_lane_map_t    lanes_map_avail; /* Used lanes map */
            };
            uint8_t               mem_type;        /* Memory type, values are
                                                    * ucs_memory_type_t */
            ucp_lane_index_t      pending_lane;    /* Lane on which request was moved
                                                    * to pending state */
            ucp_lane_index_t      lane;            /* Lane on which this request is being sent */
            uint8_t               proto_stage;     /* Protocol current stage */
            uct_pending_req_t     uct;             /* UCT pending request */
        } send;

        /* "receive" part - used for tag_recv, am_recv and stream_recv operations */
        struct {
            ucs_queue_elem_t      queue;    /* Expected queue element */
            void                  *buffer;  /* Buffer to receive data to */
            ucp_datatype_t        datatype; /* Receive type */
            size_t                length;   /* Total length, in bytes */
            ucs_memory_type_t     mem_type; /* Memory type */
            ucp_dt_state_t        state;
            ucp_worker_t          *worker;
            uct_tag_context_t     uct_ctx;  /* Transport offload context */
            union {
                /* How much more data to be received */
                ssize_t   remaining;

                /* Offset in recv buffer for multi fragment tag offload flow */
                size_t    offset;

                /* User-defined memory handle supplied to ucp_[tag|am)_recv_nbx,
                   valid if UCP_REQUEST_FLAG_USER_MEMH is set */
                ucp_mem_h user_memh;
            };

            /* Remote request ID received from a peer */
            ucs_ptr_map_key_t     remote_req_id;

#if ENABLE_DEBUG_DATA
            /* For rendezvous receive with new protocols: selected protocol for
               fetching remote data */
            const ucp_proto_config_t *proto_rndv_config;
#endif

            union {
                struct {
                    ucp_tag_t                   tag;        /* Expected tag */
                    ucp_tag_t                   tag_mask;   /* Expected tag mask */
                    uint64_t                    sn;         /* Tag match sequence */
                    ucp_tag_recv_nbx_callback_t cb;         /* Completion callback */
                    ucp_tag_recv_info_t         info;       /* Completion info to fill */

                    /* Can use union, because rdesc is used in expected flow,
                     * while non_contig_buf is used in unexpected flow only. */
                    union {
                        ucp_mem_desc_t      *rdesc;   /* Offload bounce buffer */
                        void                *non_contig_buf; /* Used for assembling
                                                                multi-fragment
                                                                non-contig unexpected
                                                                message in tag offload flow. */
                    };
                    ucp_worker_iface_t      *wiface;    /* Cached iface this request
                                                           is received on. Used in
                                                           tag offload expected callbacks*/
                } tag;

                struct {
                    size_t                  offset;   /* offset in recv buffer */
                } frag;

                struct {
                    ucp_stream_recv_nbx_callback_t cb;     /* Completion callback */
                    size_t                         offset; /* Receive data offset */
                    size_t                         length; /* Completion info to fill */
                } stream;

                 struct {
                    ucp_am_recv_data_nbx_callback_t cb;    /* Completion callback */
                    ucp_recv_desc_t                 *desc; /* RTS desc */
                } am;
            };
        } recv;

        struct {
            ucp_worker_h            worker;     /* Worker to flush */
            ucp_send_nbx_callback_t cb;         /* Completion callback */
            uct_worker_cb_id_t      prog_id;    /* Progress callback ID */
            int                     comp_count; /* Countdown to request completion */
            ucp_ep_ext_gen_t        *next_ep;   /* Next endpoint to flush */
        } flush_worker;
    };
};


/**
 * Unexpected receive descriptor. If it is initialized in the headroom of UCT
 * descriptor, the layout looks like the following:
 *
 *
 * headroom                                    data
 * |-------------------------------------------|-------------------------|
 * | unused | ucp_recv_desc |      priv_length |                         |
 * |        |               |                  |                         |
 * |-------------------------------------------|-------------------------|
 *
 * Some protocols (i. e. tag offload) may need some space right before the
 * incoming data to add specific headers needed for further message processing.
 * Note: priv_length value should be in [0, UCP_WORKER_HEADROOM_PRIV_SIZE] range.
 */
struct ucp_recv_desc {
    union {
        ucs_list_link_t     tag_list[2];     /* Hash list TAG-element */
        ucs_queue_elem_t    stream_queue;    /* Queue STREAM-element */
        ucs_queue_elem_t    tag_frag_queue;  /* Tag fragments queue */
        ucp_am_first_desc_t am_first;        /* AM first fragment data needed
                                                for assembling the message */
        ucs_queue_elem_t    am_mid_queue;    /* AM middle fragments queue */
    };
    uint32_t                length;          /* Received length */
    uint32_t                payload_offset;  /* Offset from end of the
                                                descriptor to AM data */
    uint16_t                flags;           /* Flags */
    int16_t                 release_desc_offset; /* Offset which needs to be
                                                    substructed from rdesc when
                                                    releasing it back to UCT or
                                                    AM memory pool or freeing it
                                                    in case of assembled
                                                    multi-fragment active message */
#if ENABLE_DEBUG_DATA
    const char              *name;           /* Object name, debug only */
#endif
};


/**
 * Defines protocol functions for ucp_request_send_start() function.
 * TODO will be removed when switching to new protocols implementation.
 */
struct ucp_request_send_proto {
    uct_pending_callback_t     contig_short;     /**< Progress short data */
    uct_pending_callback_t     bcopy_single;     /**< Progress bcopy single fragment */
    uct_pending_callback_t     bcopy_multi;      /**< Progress bcopy multi-fragment */
    uct_pending_callback_t     zcopy_single;     /**< Progress zcopy single fragment */
    uct_pending_callback_t     zcopy_multi;      /**< Progress zcopy multi-fragment */
    uct_completion_callback_t  zcopy_completion; /**< Callback for UCT zcopy completion */
    size_t                     only_hdr_size;    /**< Header size for single / short */
};


extern ucs_mpool_ops_t ucp_request_mpool_ops;
extern ucs_mpool_ops_t ucp_rndv_get_mpool_ops;
extern const ucp_request_param_t ucp_request_null_param;


int ucp_request_pending_add(ucp_request_t *req);

ucs_status_t
ucp_request_memory_reg(ucp_context_t *context, ucp_md_map_t md_map,
                       void *buffer, size_t length, ucp_datatype_t datatype,
                       ucp_dt_state_t *state, ucs_memory_type_t mem_type,
                       ucp_request_t *req, unsigned uct_flags);

void ucp_request_memory_dereg(ucp_context_t *context, ucp_datatype_t datatype,
                              ucp_dt_state_t *state, ucp_request_t *req);

void ucp_request_dt_invalidate(ucp_request_t *req, ucs_status_t status);

ucs_status_t ucp_request_send_start(ucp_request_t *req, ssize_t max_short,
                                    size_t zcopy_thresh, size_t zcopy_max,
                                    size_t dt_count, size_t priv_iov_count,
                                    size_t length,
                                    const ucp_ep_msg_config_t *msg_config,
                                    const ucp_request_send_proto_t *proto,
                                    const ucp_request_param_t *param);

/* Fast-forward to data end */
void ucp_request_send_state_ff(ucp_request_t *req, ucs_status_t status);

ucs_status_t ucp_request_recv_msg_truncated(ucp_request_t *req, size_t length,
                                            size_t offset);

void ucp_request_purge_enqueue_cb(uct_pending_req_t *self, void *arg);

ucs_status_t ucp_request_progress_wrapper(uct_pending_req_t *self);

#endif
