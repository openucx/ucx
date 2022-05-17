/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_types.h"

#include <ucp/proto/lane_type.h>
#include <ucp/proto/proto_select.h>
#include <ucp/wireup/ep_match.h>
#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/ptr_map.h>
#include <ucs/datastruct/strided_alloc.h>
#include <ucs/debug/assert.h>
#include <ucs/stats/stats.h>


#define UCP_MAX_IOV                16UL


/* Endpoint flags type */
#if ENABLE_DEBUG_DATA || UCS_ENABLE_ASSERT
typedef uint32_t                   ucp_ep_flags_t;
#else
typedef uint16_t                   ucp_ep_flags_t;
#endif

#if UCS_ENABLE_ASSERT
#define UCP_EP_ASSERT_COUNTER_INC(_counter) \
    do { \
        ucs_assert(*(_counter) < UINT_MAX); \
        ++(*(_counter)); \
    } while (0)

#define UCP_EP_ASSERT_COUNTER_DEC(_counter) \
    do { \
        ucs_assert(*(_counter) > 0); \
        --(*(_counter)); \
    } while (0)
#else
#define UCP_EP_ASSERT_COUNTER_INC(_counter)
#define UCP_EP_ASSERT_COUNTER_DEC(_counter)
#endif


#define ucp_ep_refcount_add(_ep, _type) \
({ \
    ucs_assertv((_ep)->refcount < UINT8_MAX, "ep=%p", _ep); \
    ++(_ep)->refcount; \
    UCP_EP_ASSERT_COUNTER_INC(&(_ep)->refcounts._type); \
})

/* Return 1 if the endpoint was destroyed, 0 if not */
#define ucp_ep_refcount_remove(_ep, _type) \
({ \
    int __ret = 0; \
    \
    UCP_EP_ASSERT_COUNTER_DEC(&(_ep)->refcounts._type); \
    ucs_assertv((_ep)->refcount > 0, "ep=%p", _ep); \
    if (--(_ep)->refcount == 0) { \
        ucp_ep_destroy_base(_ep); \
        __ret = 1; \
    } \
    \
    (__ret); \
})

#define ucp_ep_refcount_field_assert(_ep, _refcount_field, _cmp, _val) \
    ucs_assertv((_ep)->_refcount_field _cmp (_val), "ep=%p: %s=%u vs %u", \
                (_ep), UCS_PP_MAKE_STRING(_refcount_field), \
                (_ep)->_refcount_field, _val);

#define ucp_ep_refcount_assert(_ep, _type_refcount, _cmp, _val) \
    ucp_ep_refcount_field_assert(_ep, refcounts._type_refcount, _cmp, _val)


#define UCP_SA_DATA_HEADER_VERSION_SHIFT 5


/**
 * Endpoint flags
 */
enum {
    UCP_EP_FLAG_LOCAL_CONNECTED        = UCS_BIT(0), /* All local endpoints are connected,
                                                        for CM case - local address was packed,
                                                        UCT did not report errors during
                                                        connection establishment protocol
                                                        and disconnect not called yet */
    UCP_EP_FLAG_REMOTE_CONNECTED       = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_QUEUED     = UCS_BIT(2), /* Connection request was queued */
    UCP_EP_FLAG_FAILED                 = UCS_BIT(3), /* EP is in failed state */
    UCP_EP_FLAG_USED                   = UCS_BIT(4), /* EP is in use by the user */
    UCP_EP_FLAG_STREAM_HAS_DATA        = UCS_BIT(5), /* EP has data in the ext.stream.match_q */
    UCP_EP_FLAG_ON_MATCH_CTX           = UCS_BIT(6), /* EP is on match queue */
    UCP_EP_FLAG_REMOTE_ID              = UCS_BIT(7), /* remote ID is valid */
    UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED = UCS_BIT(9), /* Pre-Connection request was queued */
    UCP_EP_FLAG_CLOSED                 = UCS_BIT(10),/* EP was closed */
    /* 11 bit is vacant for a flag */
    UCP_EP_FLAG_ERR_HANDLER_INVOKED    = UCS_BIT(12),/* error handler was called */
    UCP_EP_FLAG_INTERNAL               = UCS_BIT(13),/* the internal EP which holds
                                                        temporary wireup configuration or
                                                        mem-type EP */
    UCP_EP_FLAG_INDIRECT_ID            = UCS_BIT(14),/* protocols on this endpoint will send
                                                        indirect endpoint id instead of pointer,
                                                        can be replaced with looking at local ID */

    /* DEBUG bits */
    UCP_EP_FLAG_CONNECT_REQ_SENT       = UCS_BIT(16),/* DEBUG: Connection request was sent */
    UCP_EP_FLAG_CONNECT_REP_SENT       = UCS_BIT(17),/* DEBUG: Connection reply was sent */
    UCP_EP_FLAG_CONNECT_ACK_SENT       = UCS_BIT(18),/* DEBUG: Connection ACK was sent */
    UCP_EP_FLAG_CONNECT_REQ_IGNORED    = UCS_BIT(19),/* DEBUG: Connection request was ignored */
    UCP_EP_FLAG_CONNECT_PRE_REQ_SENT   = UCS_BIT(20),/* DEBUG: Connection pre-request was sent */
    UCP_EP_FLAG_FLUSH_STATE_VALID      = UCS_BIT(21),/* DEBUG: flush_state is valid */
    UCP_EP_FLAG_DISCONNECTED_CM_LANE   = UCS_BIT(22),/* DEBUG: CM lane was disconnected, i.e.
                                                        @uct_ep_disconnect was called for CM EP */
    UCP_EP_FLAG_CLIENT_CONNECT_CB      = UCS_BIT(23),/* DEBUG: Client connect callback invoked */
    UCP_EP_FLAG_SERVER_NOTIFY_CB       = UCS_BIT(24),/* DEBUG: Server notify callback invoked */
    UCP_EP_FLAG_DISCONNECT_CB_CALLED   = UCS_BIT(25),/* DEBUG: Got disconnect notification */
    UCP_EP_FLAG_CONNECT_WAIT_PRE_REQ   = UCS_BIT(26) /* DEBUG: Connection pre-request needs to be
                                                        received from a peer */
};


/**
 * UCP endpoint statistics counters
 */
enum {
    UCP_EP_STAT_TAG_TX_EAGER,
    UCP_EP_STAT_TAG_TX_EAGER_SYNC,
    UCP_EP_STAT_TAG_TX_RNDV,
    UCP_EP_STAT_LAST
};


/**
 * Endpoint init flags
 */
enum {
    UCP_EP_INIT_FLAG_MEM_TYPE          = UCS_BIT(0),  /**< Endpoint for local mem type transfers */
    UCP_EP_INIT_CREATE_AM_LANE         = UCS_BIT(1),  /**< Endpoint requires an AM lane */
    UCP_EP_INIT_CM_WIREUP_CLIENT       = UCS_BIT(2),  /**< Endpoint wireup protocol is based on CM,
                                                           client side */
    UCP_EP_INIT_CM_WIREUP_SERVER       = UCS_BIT(3),  /**< Endpoint wireup protocol is based on CM,
                                                           server side */
    UCP_EP_INIT_ERR_MODE_PEER_FAILURE  = UCS_BIT(4),  /**< Endpoint requires an
                                                           @ref UCP_ERR_HANDLING_MODE_PEER */
    UCP_EP_INIT_CM_PHASE               = UCS_BIT(5),  /**< Endpoint connection to a peer is on
                                                           CM phase */
    UCP_EP_INIT_FLAG_INTERNAL          = UCS_BIT(6),  /**< Endpoint for internal usage
                                                           (e.g. memtype, reply on keepalive) */
    UCP_EP_INIT_CONNECT_TO_IFACE_ONLY  = UCS_BIT(7),  /**< Select transports which
                                                           support CONNECT_TO_IFACE
                                                           mode only */
    UCP_EP_INIT_CREATE_AM_LANE_ONLY    = UCS_BIT(8),  /**< Endpoint requires an AM lane only */
    UCP_EP_INIT_KA_FROM_EXIST_LANES    = UCS_BIT(9),  /**< Use only existing lanes to create
                                                           keepalive lane */
    UCP_EP_INIT_ALLOW_AM_AUX_TL        = UCS_BIT(10)  /**< Endpoint allows selecting of auxiliary
                                                           transports for AM lane */
};


#define UCP_EP_STAT_TAG_OP(_ep, _op) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCP_EP_STAT_TAG_TX_##_op, 1);


typedef struct ucp_ep_config_key_lane {
    ucp_rsc_index_t      rsc_index; /* Resource index */
    ucp_md_index_t       dst_md_index; /* Destination memory domain index */
    ucs_sys_device_t     dst_sys_dev; /* Destination system device */
    uint8_t              path_index; /* Device path index */
    ucp_lane_type_mask_t lane_types; /* Which types of operations this lane
                                        was selected for */
    size_t               seg_size; /* Maximal fragment size which can be
                                      received by the peer */
} ucp_ep_config_key_lane_t;


/*
 * Endpoint configuration key.
 * This is filled by to the transport selection logic, according to the local
 * resources and set of remote addresses.
 */
struct ucp_ep_config_key {

    ucp_lane_index_t         num_lanes;       /* Number of active lanes */
    ucp_ep_config_key_lane_t lanes[UCP_MAX_LANES]; /* Active lanes */

    ucp_lane_index_t         am_lane;         /* Lane for AM (can be NULL) */
    ucp_lane_index_t         tag_lane;        /* Lane for tag matching offload (can be NULL) */
    ucp_lane_index_t         wireup_msg_lane; /* Lane for wireup messages (can be NULL) */
    ucp_lane_index_t         cm_lane;         /* Lane for holding a CM connection (can be NULL) */
    ucp_lane_index_t         keepalive_lane;  /* Lane for checking a connection state (can be NULL) */

    /* Lanes for remote memory access, sorted by priority, highest first */
    ucp_lane_index_t         rma_lanes[UCP_MAX_LANES];

    /* Lanes for high-bw memory access, sorted by priority, highest first */
    ucp_lane_index_t         rma_bw_lanes[UCP_MAX_LANES];

    /* Lane for obtaining remote memory pointer */
    ucp_lane_index_t         rkey_ptr_lane;

    /* Lanes for atomic operations, sorted by priority, highest first */
    ucp_lane_index_t         amo_lanes[UCP_MAX_LANES];

    /* Lanes for high-bw active messages, sorted by priority, highest first */
    ucp_lane_index_t         am_bw_lanes[UCP_MAX_LANES];

    /* Local memory domains to send remote keys for in high-bw rma protocols
     * NOTE: potentially it can be different than what is imposed by rma_bw_lanes,
     * since these are the MDs used by remote side for accessing our memory. */
    ucp_md_map_t             rma_bw_md_map;

    /* Bitmap of remote mds which are reachable from this endpoint (with any set
     * of transports which could be selected in the future).
     */
    ucp_md_map_t             reachable_md_map;

    /* Array with popcount(reachable_md_map) elements, each entry holds the local
     * component index to be used for unpacking remote key from each set bit in
     * reachable_md_map */
    ucp_rsc_index_t          *dst_md_cmpts;

    /* Error handling mode */
    ucp_err_handling_mode_t  err_mode;
};


/*
 * Configuration for RMA protocols
 */
typedef struct ucp_ep_rma_config {
    ssize_t                max_put_short;    /* Maximal payload of put short */
    size_t                 max_put_bcopy;    /* Maximal total size of put_bcopy */
    size_t                 max_put_zcopy;
    ssize_t                max_get_short;    /* Maximal payload of get short */
    size_t                 max_get_bcopy;    /* Maximal total size of get_bcopy */
    size_t                 max_get_zcopy;
    size_t                 put_zcopy_thresh;
    size_t                 get_zcopy_thresh;
} ucp_ep_rma_config_t;


/*
 * Configuration for AM and tag offload protocols
 */
typedef struct ucp_ep_msg_config {
        ssize_t            max_short;
        size_t             max_bcopy;
        size_t             max_zcopy;
        size_t             max_hdr;
        size_t             max_iov;

        /* zero-copy threshold for operations which do not have to wait for remote side */
        size_t             zcopy_thresh[UCP_MAX_IOV];

        /* zero-copy threshold for mem type buffers */
        size_t             mem_type_zcopy_thresh[UCS_MEMORY_TYPE_LAST];

        /* zero-copy threshold for operations which anyways have to wait for remote side */
        size_t             sync_zcopy_thresh[UCP_MAX_IOV];
        uint8_t            zcopy_auto_thresh; /* if != 0 the zcopy enabled */
} ucp_ep_msg_config_t;


/*
 * Thresholds with and without non-host memory
 */
typedef struct ucp_memtype_thresh {
        ssize_t            memtype_on;
        ssize_t            memtype_off;
} ucp_memtype_thresh_t;


/*
 * Rendezvous thresholds
 */
typedef struct ucp_rndv_thresh {
    /* threshold calculated assuming faster remote completion */
    size_t            remote;
    /* threshold calculated assuming faster local completion, for instance
     * when UCP_OP_ATTR_FLAG_FAST_CMP flag is provided to send operation
     * parameters */
    size_t            local;
} ucp_rndv_thresh_t;


/*
 * Rendezvous Zcopy configuration
 */
typedef struct ucp_rndv_zcopy {
    /* Maximal total size of Zcopy operation */
    size_t           max;
    /* Minimal size of Zcopy operation */
    size_t           min;
    /* Can the message which are > maximal size be split to the segments which are
     * >= minimal size */
    int              split;
    /* Lanes for Zcopy operation */
    ucp_lane_index_t lanes[UCP_MAX_LANES];
    /* BW based scale factor for zcopy lanes */
    double           scale[UCP_MAX_LANES];
} ucp_ep_rndv_zcopy_config_t;


/*
 * Element in ep peer memory hash. The element represents remote peer shared
 * memory segement. Having it hashed helps to avoid expensive rkey unpacking
 * and md registration procedures. Unpacking is expensive, because for shared
 * memory segments it assumes attach/mmap calls. Registration is needed for
 * better performance of CPU<->GPU memory transfers and is typically quite
 * expensive on memtype ep mds, such as cuda copy.
 */
typedef struct {
    /* Unpacked rkey with the only MD supporting RKEY_PTR */
    ucp_rkey_h       rkey;
    /* Size of the buffer corresponding to the unpacked rkey */
    size_t           size;
    /* MD index corresponding to memtype ep */
    ucp_md_index_t   md_index;
    /* Memory handle holding registration of the remote buffer on memtype
     * ep MD */
    uct_mem_h        uct_memh;
} ucp_ep_peer_mem_data_t;


KHASH_DECLARE(ucp_ep_peer_mem_hash, uint64_t, ucp_ep_peer_mem_data_t);


struct ucp_ep_config {

    /* A key which uniquely defines the configuration, and all other fields of
     * configuration (in the current worker) and defined only by it.
     */
    ucp_ep_config_key_t     key;

    /* Bitmap of which lanes are p2p; affects the behavior of connection
     * establishment protocols.
     */
    ucp_lane_map_t          p2p_lanes;

    /* Flags which has to be used @ref uct_md_mkey_pack_v2 */
    unsigned                uct_rkey_pack_flags;

    /* Configuration for each lane that provides RMA */
    ucp_ep_rma_config_t     rma[UCP_MAX_LANES];

    /* Threshold for switching from put_short to put_bcopy */
    size_t                  bcopy_thresh;

    /* Configuration for AM lane */
    ucp_ep_msg_config_t     am;

    /* MD index of each lane */
    ucp_md_index_t          md_index[UCP_MAX_LANES];

    struct {
        /* RNDV GET Zcopy configuration */
        ucp_ep_rndv_zcopy_config_t get_zcopy;
        /* RNDV PUT Zcopy configuration */
        ucp_ep_rndv_zcopy_config_t put_zcopy;
        /* Threshold for switching from eager to RMA based rendezvous */
        ucp_rndv_thresh_t          rma_thresh;
        /* Threshold for switching from eager to AM based rendezvous */
        ucp_rndv_thresh_t          am_thresh;
        /* Total size of packed rkey, according to high-bw md_map */
        size_t                     rkey_size;
        /* Remote memory domains which support rkey_ptr */
        ucp_md_map_t               rkey_ptr_dst_mds;
    } rndv;

    struct {
        /* Protocols used for tag matching operations
         * (can be AM based or tag offload). */
        const ucp_request_send_proto_t   *proto;
        const ucp_request_send_proto_t   *sync_proto;

        /* Lane used for tag matching operations. */
        ucp_lane_index_t     lane;

        /* Maximal size for eager short. */
        ucp_memtype_thresh_t max_eager_short;

        /* Configuration of the lane used for eager protocols
         * (can be AM or tag offload). */
        ucp_ep_msg_config_t  eager;

        /* Threshold for switching from eager to rendezvous. Can be different
         * from AM thresholds if tag offload is enabled and tag offload lane is
         * not the same as AM lane. */
        struct {
            ucp_rndv_thresh_t    rma_thresh;
            ucp_rndv_thresh_t    am_thresh;
        } rndv;

        struct {
            /* Maximal size for eager short. */
            ucp_memtype_thresh_t max_eager_short;

            /* Maximal iov count for RNDV offload */
            size_t          max_rndv_iov;
            /* Maximal total size for RNDV offload */
            size_t          max_rndv_zcopy;
        } offload;
    } tag;

    struct {
        /* Protocols used for stream operations
         * (currently it's only AM based). */
        const ucp_request_send_proto_t   *proto;
    } stream;

    struct {
        /* Protocols used for am operations */
        const ucp_request_send_proto_t   *proto;
        const ucp_request_send_proto_t   *reply_proto;

        /* Maximal size for eager short */
        ucp_memtype_thresh_t             max_eager_short;

        /* Maximal size for eager short with reply protocol */
        ucp_memtype_thresh_t             max_reply_eager_short;
    } am_u;

    /* Protocol selection data */
    ucp_proto_select_t            proto_select;

    /* Bitmap of preregistration for am_bw lanes */
    ucp_md_map_t                  am_bw_prereg_md_map;
};


/**
 * Protocol layer endpoint, represents a connection to a remote worker
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */

    uint8_t                       refcount;      /* Reference counter: 0 - it is
                                                    allowed to destroy EP */
    ucp_worker_cfg_index_t        cfg_index;     /* Configuration index */
    ucp_ep_match_conn_sn_t        conn_sn;       /* Sequence number for remote connection */
    ucp_lane_index_t              am_lane;       /* Cached value */
    ucp_ep_flags_t                flags;         /* Endpoint flags */

    /* TODO allocate ep dynamically according to number of lanes */
    uct_ep_h                      uct_eps[UCP_MAX_LANES]; /* Transports for every lane */

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_ADDRESS_NAME_MAX];
    /* Endpoint name for tracing and analysis */
    char                          name[UCP_ENTITY_NAME_MAX];
#endif

#if UCS_ENABLE_ASSERT
    struct {
        /* How many times the EP create was done */
        unsigned                      create;
        /* How many Worker flush operations are in-progress where the EP is the
         * next EP for flushing */
        unsigned                      flush;
        /* How many UCT EP discarding operations are in-progress scheduled for
         * the EP */
        unsigned                      discard;
    } refcounts;
#endif

    UCS_STATS_NODE_DECLARE(stats)

} ucp_ep_t;


/**
 * Status of protocol-level remote completions
 */
typedef struct {
    ucs_hlist_head_t reqs; /* Queue of flush requests which
                              are waiting for remote completion */
    uint32_t         send_sn; /* Sequence number of sent operations */
    uint32_t         cmpl_sn; /* Sequence number of completions */
} ucp_ep_flush_state_t;


/**
 * Endpoint extension for control data path
 */
typedef struct {
    ucp_rsc_index_t               cm_idx; /* CM index */
    ucs_ptr_map_key_t             local_ep_id; /* Local EP ID */
    ucs_ptr_map_key_t             remote_ep_id; /* Remote EP ID */
    ucp_err_handler_cb_t          err_cb; /* Error handler */
    ucp_request_t                 *close_req; /* Close protocol request */
#if UCS_ENABLE_ASSERT
    ucs_time_t                    ka_last_round; /* Time of last KA round done */
#endif
    khash_t(ucp_ep_peer_mem_hash) *peer_mem; /* Hash of remote memory segments
                                                used by 2-stage ppln rndv proto */

} ucp_ep_ext_control_t;


/**
 * Endpoint extension for generic non fast-path data
 */
typedef struct {
    void                          *user_data;    /* User data associated with ep */
    ucs_list_link_t               ep_list;       /* List entry in worker's all eps list */
    /* Endpoint match context and remote completion status are mutually exclusive,
     * since remote completions are counted only after the endpoint is already
     * matched to a remote peer.
     */
    union {
        ucp_ep_match_elem_t       ep_match;      /* Matching with remote endpoints */
        ucp_ep_flush_state_t      flush_state;   /* Remote completion status */
    };
    ucp_ep_ext_control_t          *control_ext;  /* Control data path extension */
    /* List of requests which are waiting for remote completion */
    ucs_hlist_head_t              proto_reqs;
} ucp_ep_ext_gen_t;


/**
 * Endpoint extension for specific protocols
 */
typedef struct {
    struct {
        ucs_list_link_t           ready_list;    /* List entry in worker's EP list */
        ucs_queue_head_t          match_q;       /* Queue of receive data or requests,
                                                    depends on UCP_EP_FLAG_STREAM_HAS_DATA */
    } stream;

    struct {
        ucs_list_link_t           started_ams;
        ucs_queue_head_t          mid_rdesc_q; /* queue of middle fragments, which
                                                  arrived before the first one */
    } am;
} ucp_ep_ext_proto_t;


enum {
    UCP_WIREUP_SA_DATA_CM_ADDR   = UCS_BIT(1)  /* Sockaddr client data contains address
                                                  for CM based wireup: there is only
                                                  iface and ep address of transport
                                                  lanes, remote device address is
                                                  provided by CM and has to be added to
                                                  unpacked UCP address locally. */
};


/* Sockaddr data flags that are packed to the header field in
 * ucp_wireup_sockaddr_data_base_t structure.
 */
enum {
    /* Indicates support of UCP_ERR_HANDLING_MODE_PEER error mode. */
    UCP_SA_DATA_FLAG_ERR_MODE_PEER = UCS_BIT(0)
};


/* Basic sockaddr data. Version 1 uses some additional fields which are not
 * really needed and removed in version 2.
 */
typedef struct ucp_wireup_sockaddr_data_base {
    uint64_t                  ep_id; /**< Endpoint ID */

    /* This field has different meaning for sa_data v1 and other versions:
     * v1:           it is error handling mode
     * v2 and newer: it is sa_data header with the following format:
     *   +---+-----+
     *   | 3 |  5  |
     *   +---+-----+
     *     v    |
     * version  |
     *          v
     *        flags
     *
     * It is safe to keep version in 3 MSB, because it will always be zeros
     * (i.e. UCP_OBJECT_VERSION_V1) in sa_data v1 (err_mode value is small).
     */
    uint8_t                   header;
    /* packed worker address (or sa_data v1) follows */
} UCS_S_PACKED ucp_wireup_sockaddr_data_base_t;


typedef struct ucp_wireup_sockaddr_data_v1 {
    ucp_wireup_sockaddr_data_base_t super;
    uint8_t                         addr_mode; /**< The attached address format
                                                    defined by
                                                    UCP_WIREUP_SA_DATA_xx */
    uint8_t                         dev_index; /**< Device address index used to
                                                    build remote address in
                                                    UCP_WIREUP_SA_DATA_CM_ADDR
                                                    mode */
    /* packed worker address follows */
} UCS_S_PACKED ucp_wireup_sockaddr_data_v1_t;


typedef struct ucp_conn_request {
    ucp_listener_h              listener;
    uct_listener_h              uct_listener;
    uct_conn_request_h          uct_req;
    ucp_rsc_index_t             cm_idx;
    char                        dev_name[UCT_DEVICE_NAME_MAX];
    uct_device_addr_t           *remote_dev_addr;
    struct sockaddr_storage     client_address;
    ucp_ep_h                    ep; /* valid only if request is handled internally */
    /* sa_data and packed worker address follow */
} ucp_conn_request_t;


int ucp_is_uct_ep_failed(uct_ep_h uct_ep);

void ucp_ep_config_key_reset(ucp_ep_config_key_t *key);

void ucp_ep_config_cm_lane_info_str(ucp_worker_h worker,
                                    const ucp_ep_config_key_t *key,
                                    ucp_lane_index_t lane,
                                    ucp_rsc_index_t cm_index,
                                    ucs_string_buffer_t *buf);

void ucp_ep_config_lane_info_str(ucp_worker_h worker,
                                 const ucp_ep_config_key_t *key,
                                 const unsigned *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 ucs_string_buffer_t *buf);

ucs_status_t ucp_ep_create_base(ucp_worker_h worker, unsigned ep_init_flags,
                                const char *peer_name, const char *message,
                                ucp_ep_h *ep_p);

void ucp_ep_destroy_base(ucp_ep_h ep);

void ucp_ep_delete(ucp_ep_h ep);

void ucp_ep_flush_state_reset(ucp_ep_h ep);

void ucp_ep_flush_state_invalidate(ucp_ep_h ep);

void ucp_ep_release_id(ucp_ep_h ep);

ucs_status_t
ucp_ep_config_err_mode_check_mismatch(ucp_ep_h ep,
                                      ucp_err_handling_mode_t err_mode);

ucs_status_t ucp_ep_init_create_wireup(ucp_ep_h ep, unsigned ep_init_flags,
                                       ucp_wireup_ep_t **wireup_ep);

ucs_status_t
ucp_ep_create_to_worker_addr(ucp_worker_h worker,
                             const ucp_tl_bitmap_t *local_tl_bitmap,
                             const ucp_unpacked_address_t *remote_address,
                             unsigned ep_init_flags, const char *message,
                             ucp_ep_h *ep_p);

ucs_status_t ucp_ep_create_server_accept(ucp_worker_h worker,
                                         const ucp_conn_request_h conn_request,
                                         ucp_ep_h *ep_p);

ucs_status_ptr_t ucp_ep_flush_internal(ucp_ep_h ep, unsigned req_flags,
                                       const ucp_request_param_t *param,
                                       ucp_request_t *worker_req,
                                       ucp_request_callback_t flushed_cb,
                                       const char *debug_name);

void ucp_ep_config_key_set_err_mode(ucp_ep_config_key_t *key,
                                    unsigned ep_init_flags);

void ucp_ep_err_pending_purge(uct_pending_req_t *self, void *arg);

void ucp_destroyed_ep_pending_purge(uct_pending_req_t *self, void *arg);

void ucp_ep_disconnected(ucp_ep_h ep, int force);

void ucp_ep_destroy_internal(ucp_ep_h ep);

ucs_status_t
ucp_ep_set_failed(ucp_ep_h ucp_ep, ucp_lane_index_t lane, ucs_status_t status);

void ucp_ep_set_failed_schedule(ucp_ep_h ucp_ep, ucp_lane_index_t lane,
                                ucs_status_t status);

void ucp_ep_unprogress_uct_ep(ucp_ep_h ep, uct_ep_h uct_ep,
                              ucp_rsc_index_t rsc_index);

void ucp_ep_cleanup_lanes(ucp_ep_h ep);

ucs_status_t ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config,
                                const ucp_ep_config_key_t *key);

void ucp_ep_config_cleanup(ucp_worker_h worker, ucp_ep_config_t *config);

int ucp_ep_config_lane_is_peer_match(const ucp_ep_config_key_t *key1,
                                     ucp_lane_index_t lane1,
                                     const ucp_ep_config_key_t *key2,
                                     ucp_lane_index_t lane2);

void ucp_ep_config_lanes_intersect(const ucp_ep_config_key_t *key1,
                                   const ucp_rsc_index_t *dst_rsc_indices1,
                                   const ucp_ep_config_key_t *key2,
                                   const ucp_rsc_index_t *dst_rsc_indices2,
                                   ucp_lane_index_t *lane_map);

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2);

int ucp_ep_config_get_multi_lane_prio(const ucp_lane_index_t *lanes,
                                      ucp_lane_index_t lane);

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const ucs_linear_func_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth);

ucs_status_t ucp_worker_mem_type_eps_create(ucp_worker_h worker);

void ucp_worker_mem_type_eps_destroy(ucp_worker_h worker);

void ucp_worker_mem_type_eps_print_info(ucp_worker_h worker,
                                              FILE *stream);

ucp_wireup_ep_t * ucp_ep_get_cm_wireup_ep(ucp_ep_h ep);

void ucp_ep_get_tl_bitmap(ucp_ep_h ep, ucp_tl_bitmap_t *tl_bitmap);

uct_ep_h ucp_ep_get_cm_uct_ep(ucp_ep_h ep);

int ucp_ep_is_cm_local_connected(ucp_ep_h ep);

int ucp_ep_is_local_connected(ucp_ep_h ep);

unsigned ucp_ep_local_disconnect_progress(void *arg);

size_t ucp_ep_tag_offload_min_rndv_thresh(ucp_ep_config_t *config);

void ucp_ep_config_rndv_zcopy_commit(ucp_lane_index_t lanes_count,
                                     ucp_ep_rndv_zcopy_config_t *rndv_zcopy);

void ucp_ep_get_lane_info_str(ucp_ep_h ucp_ep, ucp_lane_index_t lane,
                              ucs_string_buffer_t *lane_info_strb);

void ucp_ep_config_rndv_zcopy_commit(ucp_lane_index_t lanes_count,
                                     ucp_ep_rndv_zcopy_config_t *rndv_zcopy);

void ucp_ep_invoke_err_cb(ucp_ep_h ep, ucs_status_t status);

ucs_status_t ucp_ep_flush_progress_pending(uct_pending_req_t *self);

void ucp_ep_flush_completion(uct_completion_t *self);

void ucp_ep_flush_request_ff(ucp_request_t *req, ucs_status_t status);

void
ucp_ep_purge_lanes(ucp_ep_h ep, uct_pending_purge_callback_t purge_cb,
                   void *purge_arg);

void ucp_ep_register_disconnect_progress(ucp_request_t *req);

ucp_lane_index_t ucp_ep_lookup_lane(ucp_ep_h ucp_ep, uct_ep_h uct_ep);

void ucp_ep_peer_mem_destroy(ucp_context_h context,
                             ucp_ep_peer_mem_data_t *data);

ucp_ep_peer_mem_data_t*
ucp_ep_peer_mem_get(ucp_context_h context, ucp_ep_h ep, uint64_t address,
                    size_t size, void *rkey_buf, ucp_md_index_t md_index);

/**
 * @brief Indicates AM-based keepalive necessity.
 * 
 * @param [in] ep      UCP endpoint to check.
 * @param [in] rsc_idx Resource index to check.
 * @param [in] is_p2p  Flag that indicates whether UCT EP was created as p2p
 *                     (i.e. CONNECT_TO_EP) or not.
 *
 * @return Whether AM-based keepalive is required or not.
 */
int ucp_ep_is_am_keepalive(ucp_ep_h ep, ucp_rsc_index_t rsc_idx, int is_p2p);

/**
 * @brief Do AM-based keepalive operation for a specific UCT EP.
 *
 * @param [in] ucp_ep  UCP Endpoint object to operate keepalive.
 * @param [in] uct_ep  UCT Endpoint object to do keepalive on.
 * @param [in] rsc_idx Resource index to check.
 *
 * @return Status of keepalive operation.
 */
ucs_status_t ucp_ep_do_uct_ep_am_keepalive(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                                           ucp_rsc_index_t rsc_idx);


/**
 * @brief Purge the protocol request scheduled on a given UCP endpoint.
 *
 * @param [in]     ucp_ep           Endpoint object on which the request should
 *                                  be purged.
 * @param [in]     req              The request to purge.
 * @param [in]     status           Completion status.
 * @param [in]     recursive        Indicates if the function was called from
 *                                  the @ref ucp_ep_req_purge recursively.
 */
void ucp_ep_req_purge(ucp_ep_h ucp_ep, ucp_request_t *req,
                      ucs_status_t status, int recursive);


/**
 * @brief Purge flush and protocol requests scheduled on a given UCP endpoint.
 *
 * @param [in]     ucp_ep           Endpoint object on which requests should be
 *                                  purged.
 * @param [in]     status           Completion status.
 */
void ucp_ep_reqs_purge(ucp_ep_h ucp_ep, ucs_status_t status);


/**
 * @brief Query local and/or remote socket address of endpoint @a ucp_ep.
 *
 * @param [in]     ucp_ep           Endpoint object to query.
 * @param [inout]  attr             Filled with attributes containing socket
 *                                  address of the endpoint.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_ep_query_sockaddr(ucp_ep_h ucp_ep, ucp_ep_attr_t *attr);

#endif
