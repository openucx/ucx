/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_types.h"

#include <uct/api/uct.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/stats/stats.h>
#include <limits.h>
#include <ucs/datastruct/strided_alloc.h>

#define UCP_MAX_IOV                16UL


/* Configuration */
typedef uint16_t                   ucp_ep_cfg_index_t;


/* Endpoint flags type */
typedef uint32_t                   ucp_ep_flags_t;

        /**
 * Endpoint flags
 */
enum {
    UCP_EP_FLAG_LOCAL_CONNECTED        = UCS_BIT(0), /* All local endpoints are connected */
    UCP_EP_FLAG_REMOTE_CONNECTED       = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_QUEUED     = UCS_BIT(2), /* Connection request was queued */
    UCP_EP_FLAG_FAILED                 = UCS_BIT(3), /* EP is in failed state */
    UCP_EP_FLAG_HIDDEN                 = UCS_BIT(4), /* EP is not user visible */
    UCP_EP_FLAG_STREAM_HAS_DATA        = UCS_BIT(5), /* EP has data in the ext.stream.match_q */
    UCP_EP_FLAG_ON_MATCH_CTX           = UCS_BIT(6), /* EP is on match queue */
    UCP_EP_FLAG_DEST_EP                = UCS_BIT(7), /* dest_ep_ptr is valid */
    UCP_EP_FLAG_LISTENER               = UCS_BIT(8), /* EP holds pointer to a listener
                                                        (on server side due to receiving partial
                                                        worker address from the client) */
    UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED = UCS_BIT(9), /* Pre-Connection request was queued */
    UCP_EP_FLAG_FIN_REQ_QUEUED         = UCS_BIT(10),  /* Fin request was queued */
    UCP_EP_FLAG_FIN_REQ_COMPLETED      = UCS_BIT(11),  /* All local endpoints are disconnected */
    UCP_EP_FLAG_FIN_MSG_RECVD          = UCS_BIT(12), /* Remote endpoints are disconnected */
    /* Full event mask to destroy endpoint after FIN protocol */
    UCP_EP_MASK_FIN_DONE               = UCP_EP_FLAG_FIN_REQ_COMPLETED |
                                         UCP_EP_FLAG_FIN_MSG_RECVD     |
                                         UCP_EP_FLAG_HIDDEN,

    /* DEBUG bits */
    UCP_EP_FLAG_CONNECT_REQ_SENT       = UCS_BIT(16), /* DEBUG: Connection request was sent */
    UCP_EP_FLAG_CONNECT_REP_SENT       = UCS_BIT(17), /* DEBUG: Connection reply was sent */
    UCP_EP_FLAG_CONNECT_ACK_SENT       = UCS_BIT(18), /* DEBUG: Connection ACK was sent */
    UCP_EP_FLAG_CONNECT_REQ_IGNORED    = UCS_BIT(19), /* DEBUG: Connection request was ignored */
    UCP_EP_FLAG_CONNECT_PRE_REQ_SENT   = UCS_BIT(20), /* DEBUG: Connection pre-request was sent */
    UCP_EP_FLAG_SOCKADDR_PARTIAL_ADDR  = UCS_BIT(21)  /* DEBUG: Partial worker address was sent
                                                                to the remote peer when starting
                                                                connection establishment on this EP */
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
    UCP_EP_CREATE_AM_LANE              = UCS_BIT(1)   /**< Endpoint requires an AM lane */
};


#define UCP_EP_STAT_TAG_OP(_ep, _op) \
    UCS_STATS_UPDATE_COUNTER((_ep)->stats, UCP_EP_STAT_TAG_TX_##_op, 1);


/*
 * Endpoint configuration key.
 * This is filled by to the transport selection logic, according to the local
 * resources and set of remote addresses.
 */
typedef struct ucp_ep_config_key {

    ucp_lane_index_t       num_lanes;    /* Number of active lanes */

    struct {
        ucp_rsc_index_t    rsc_index;    /* Resource index */
        ucp_lane_index_t   proxy_lane;   /* UCP_NULL_LANE - no proxy
                                            otherwise - in which lane the real
                                            transport endpoint is stored */
        ucp_md_index_t     dst_md_index; /* Destination memory domain index */
    } lanes[UCP_MAX_LANES];

    ucp_lane_index_t       am_lane;      /* Lane for AM (can be NULL) */
    ucp_lane_index_t       tag_lane;     /* Lane for tag matching offload (can be NULL) */
    ucp_lane_index_t       wireup_lane;  /* Lane for wireup messages (can be NULL) */

    /* Lanes for remote memory access, sorted by priority, highest first */
    ucp_lane_index_t       rma_lanes[UCP_MAX_LANES];

    /* Lanes for high-bw memory access, sorted by priority, highest first */
    ucp_lane_index_t       rma_bw_lanes[UCP_MAX_LANES];

    /* Lanes for atomic operations, sorted by priority, highest first */
    ucp_lane_index_t       amo_lanes[UCP_MAX_LANES];

    /* Lanes for high-bw active messages, sorted by priority, highest first */
    ucp_lane_index_t       am_bw_lanes[UCP_MAX_LANES];

    /* Local memory domains to send remote keys for in high-bw rma protocols
     * NOTE: potentially it can be different than what is imposed by rma_bw_lanes,
     * since these are the MDs used by remote side for accessing our memory. */
    ucp_md_map_t           rma_bw_md_map;

    /* Bitmap of remote mds which are reachable from this endpoint (with any set
     * of transports which could be selected in the future).
     */
    ucp_md_map_t           reachable_md_map;

    /* Error handling mode */
    ucp_err_handling_mode_t    err_mode;
    ucs_status_t               status;
} ucp_ep_config_key_t;


/*
 * Configuration for RMA protocols
 */
typedef struct ucp_ep_rma_config {
    size_t                 max_put_short;    /* Maximal payload of put short */
    size_t                 max_put_bcopy;    /* Maximal total size of put_bcopy */
    size_t                 max_put_zcopy;
    size_t                 max_get_short;    /* Maximal payload of get short */
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
        size_t             max_iov;

        /* zero-copy threshold for operations which do not have to wait for remote side */
        size_t             zcopy_thresh[UCP_MAX_IOV];

        /* zero-copy threshold for mem type buffers */
        size_t             mem_type_zcopy_thresh[UCT_MD_MEM_TYPE_LAST];

        /* zero-copy threshold for operations which anyways have to wait for remote side */
        size_t             sync_zcopy_thresh[UCP_MAX_IOV];
        uint8_t            zcopy_auto_thresh; /* if != 0 the zcopy enabled */
} ucp_ep_msg_config_t;


typedef struct ucp_ep_config {

    /* A key which uniquely defines the configuration, and all other fields of
     * configuration (in the current worker) and defined only by it.
     */
    ucp_ep_config_key_t     key;

    /* Bitmap of which lanes are p2p; affects the behavior of connection
     * establishment protocols.
     */
    ucp_lane_map_t          p2p_lanes;

    /* Configuration for each lane that provides RMA */
    ucp_ep_rma_config_t     rma[UCP_MAX_LANES];

    /* Threshold for switching from put_short to put_bcopy */
    size_t                  bcopy_thresh;

    /* Configuration for AM lane */
    ucp_ep_msg_config_t     am;

    /* MD index of each lane */
    ucp_md_index_t          md_index[UCP_MAX_LANES];

    struct {
        /* Protocols used for tag matching operations
         * (can be AM based or tag offload). */
        const ucp_proto_t   *proto;
        const ucp_proto_t   *sync_proto;

        /* Lane used for tag matching operations. */
        ucp_lane_index_t    lane;

        /* Configuration of the lane used for eager protocols
         * (can be AM or tag offload). */
        ucp_ep_msg_config_t eager;

        struct {
            /* Maximal total size of rndv_get_zcopy */
            size_t          max_get_zcopy;
            /* Maximal total size of rndv_put_zcopy */
            size_t          max_put_zcopy;
            /* Threshold for switching from eager to RMA based rendezvous */
            size_t          rma_thresh;
            /* Threshold for switching from eager to AM based rendezvous */
            size_t          am_thresh;
            /* Total size of packed rkey, according to high-bw md_map */
            size_t          rkey_size;
        } rndv;

        /* special thresholds for the ucp_tag_send_nbr() */
        struct {
            /* Threshold for switching from eager to RMA based rendezvous */
            size_t          rma_thresh;
            /* Threshold for switching from eager to AM based rendezvous */
            size_t          am_thresh;
        } rndv_send_nbr;

        struct {
            /* Maximal size for eager short */
            ssize_t         max_eager_short;
            /* Maximal iov count for RNDV offload */
            size_t          max_rndv_iov;
            /* Maximal total size for RNDV offload */
            size_t          max_rndv_zcopy;
        } offload;
    } tag;

    struct {
        /* Protocols used for stream operations
         * (currently it's only AM based). */
        const ucp_proto_t   *proto;
    } stream;
} ucp_ep_config_t;


/**
 * Protocol layer endpoint, represents a connection to a remote worker
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */

    ucp_ep_flags_t                flags;         /* Endpoint flags */
    ucp_ep_cfg_index_t            cfg_index;     /* Configuration index */
    ucp_lane_index_t              am_lane;       /* Cached value */

    /* TODO allocate ep dynamically according to number of lanes */
    uct_ep_h                      uct_eps[UCP_MAX_LANES]; /* Transports for every lane */

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_NAME_MAX];
#endif

    UCS_STATS_NODE_DECLARE(stats);

} ucp_ep_t;


/*
 * Endpoint extension for generic non fast-path data
 */
typedef struct {
    union {
        void                      *user_data;    /* User data associated with ep */
        ucp_listener_h            listener;      /* Listener that may be associated with ep */
    };
    ucp_err_handler_cb_t          err_cb;        /* Error handler */
    ucs_list_link_t               ep_list;       /* List entry in worker's all eps list */

    /* matching with remote endpoints */
    struct {
        uint64_t                  dest_uuid;     /* Destination worker UUID */
        ucs_list_link_t           list;          /* List entry into endpoint
                                                    matching structure */
    } ep_match;
} ucp_ep_ext_gen_t;


/*
 * Endpoint extension for specific protocols
 */
typedef struct {
    struct {
        ucp_ep_conn_sn_t              conn_sn;       /* Sequence number for remote connection */
        uintptr_t                     dest_ep_ptr;   /* Remote EP pointer */
    } conn;

    struct {
        ucs_list_link_t           ready_list;        /* List entry in worker's EP list */
        ucs_queue_head_t          match_q;           /* Queue of receive data or requests,
                                                        depends on UCP_EP_FLAG_STREAM_HAS_DATA */
    } stream;
} ucp_ep_ext_proto_t;

void ucp_ep_config_key_reset(ucp_ep_config_key_t *key);

void ucp_ep_config_lane_info_str(ucp_context_h context,
                                 const ucp_ep_config_key_t *key,
                                 const uint8_t *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 char *buf, size_t max);

ucs_status_t ucp_ep_new(ucp_worker_h worker, const char *peer_name,
                        const char *message, ucp_ep_h *ep_p);

void ucp_ep_delete(ucp_ep_h ep);

ucs_status_t ucp_ep_init_create_wireup(ucp_ep_h ep,
                                       const ucp_ep_params_t *params,
                                       ucp_wireup_ep_t **wireup_ep);

ucs_status_t ucp_ep_create_to_worker_addr(ucp_worker_h worker,
                                          const ucp_ep_params_t *params,
                                          const ucp_unpacked_address_t *remote_address,
                                          unsigned ep_init_flags,
                                          const char *message, ucp_ep_h *ep_p);

ucs_status_ptr_t ucp_ep_flush_internal(ucp_ep_h ep, unsigned uct_flags,
                                       ucp_send_callback_t req_cb,
                                       unsigned req_flags,
                                       ucp_request_callback_t flushed_cb);

ucs_status_t ucp_ep_create_sockaddr_aux(ucp_worker_h worker,
                                        const ucp_ep_params_t *params,
                                        const ucp_unpacked_address_t *remote_address,
                                        ucp_ep_h *ep_p);

void ucp_ep_config_key_set_params(ucp_ep_config_key_t *key,
                                  const ucp_ep_params_t *params);

void ucp_ep_err_pending_purge(uct_pending_req_t *self, void *arg);

void ucp_ep_disconnected(ucp_ep_h ep, int force);

void ucp_ep_destroy_internal(ucp_ep_h ep);

void ucp_ep_cleanup_lanes(ucp_ep_h ep);

int ucp_ep_is_sockaddr_stub(ucp_ep_h ep);

void ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config);

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2);

int ucp_ep_config_get_multi_lane_prio(const ucp_lane_index_t *lanes,
                                      ucp_lane_index_t lane);

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const uct_linear_growth_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth);

ucs_status_t ucp_worker_create_mem_type_endpoints(ucp_worker_h worker);

#endif
