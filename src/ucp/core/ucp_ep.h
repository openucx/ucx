/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_types.h"

#include <ucp/wireup/ep_match.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/queue.h>
#include <ucs/stats/stats.h>
#include <ucs/datastruct/strided_alloc.h>
#include <ucs/debug/assert.h>


#define UCP_MAX_IOV                16UL


/* Configuration */
typedef uint16_t                   ucp_ep_cfg_index_t;


/* Endpoint flags type */
#if ENABLE_DEBUG_DATA || UCS_ENABLE_ASSERT
typedef uint32_t                   ucp_ep_flags_t;
#else
typedef uint16_t                   ucp_ep_flags_t;
#endif


/**
 * Endpoint flags
 */
enum {
    UCP_EP_FLAG_LOCAL_CONNECTED        = UCS_BIT(0), /* All local endpoints are connected */
    UCP_EP_FLAG_REMOTE_CONNECTED       = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_QUEUED     = UCS_BIT(2), /* Connection request was queued */
    UCP_EP_FLAG_FAILED                 = UCS_BIT(3), /* EP is in failed state */
    UCP_EP_FLAG_USED                   = UCS_BIT(4), /* EP is in use by the user */
    UCP_EP_FLAG_STREAM_HAS_DATA        = UCS_BIT(5), /* EP has data in the ext.stream.match_q */
    UCP_EP_FLAG_ON_MATCH_CTX           = UCS_BIT(6), /* EP is on match queue */
    UCP_EP_FLAG_DEST_EP                = UCS_BIT(7), /* dest_ep_ptr is valid */
    UCP_EP_FLAG_LISTENER               = UCS_BIT(8), /* EP holds pointer to a listener
                                                        (on server side due to receiving partial
                                                        worker address from the client) */
    UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED = UCS_BIT(9), /* Pre-Connection request was queued */
    UCP_EP_FLAG_CLOSED                 = UCS_BIT(10),/* EP was closed */

    /* DEBUG bits */
    UCP_EP_FLAG_CONNECT_REQ_SENT       = UCS_BIT(16),/* DEBUG: Connection request was sent */
    UCP_EP_FLAG_CONNECT_REP_SENT       = UCS_BIT(17),/* DEBUG: Connection reply was sent */
    UCP_EP_FLAG_CONNECT_ACK_SENT       = UCS_BIT(18),/* DEBUG: Connection ACK was sent */
    UCP_EP_FLAG_CONNECT_REQ_IGNORED    = UCS_BIT(19),/* DEBUG: Connection request was ignored */
    UCP_EP_FLAG_CONNECT_PRE_REQ_SENT   = UCS_BIT(20),/* DEBUG: Connection pre-request was sent */
    UCP_EP_FLAG_SOCKADDR_PARTIAL_ADDR  = UCS_BIT(21),/* DEBUG: Partial worker address was sent
                                                               to the remote peer when starting
                                                               connection establishment on this EP */
    UCP_EP_FLAG_FLUSH_STATE_VALID      = UCS_BIT(22),/* DEBUG: flush_state is valid */
    UCP_EP_FLAG_CLOSE_REQ_VALID        = UCS_BIT(23) /* DEBUG: close protocol is started and
                                                               close_req is valid */
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
    UCP_EP_INIT_ERR_MODE_PEER_FAILURE  = UCS_BIT(4)   /**< Endpoint requires an
                                                           @ref UCP_ERR_HANDLING_MODE_PEER */
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
    ucp_lane_index_t       cm_lane;      /* Lane for holding a CM connection */

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

    /* Array with popcount(reachable_md_map) elements, each entry holds the local
     * component index to be used for unpacking remote key from each set bit in
     * reachable_md_map */
    ucp_rsc_index_t        *dst_md_cmpts;

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

        /* Maximal size for eager short. */
        ucp_memtype_thresh_t max_eager_short;

        /* Configuration of the lane used for eager protocols
         * (can be AM or tag offload). */
        ucp_ep_msg_config_t eager;

        struct {
            /* Maximal total size of rndv_get_zcopy */
            size_t          max_get_zcopy;
            /* Minimal size of rndv_get_zcopy */
            size_t          min_get_zcopy;
            /* Maximal total size of rndv_put_zcopy */
            size_t          max_put_zcopy;
            /* Minimal size of rndv_put_zcopy */
            size_t          min_put_zcopy;
            /* Threshold for switching from eager to RMA based rendezvous */
            size_t          rma_thresh;
            /* Threshold for switching from eager to AM based rendezvous */
            size_t          am_thresh;
            /* Total size of packed rkey, according to high-bw md_map */
            size_t          rkey_size;
            /* BW based scale factor */
            double          scale[UCP_MAX_LANES];
        } rndv;

        /* special thresholds for the ucp_tag_send_nbr() */
        struct {
            /* Threshold for switching from eager to RMA based rendezvous */
            size_t          rma_thresh;
            /* Threshold for switching from eager to AM based rendezvous */
            size_t          am_thresh;
        } rndv_send_nbr;

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
        const ucp_proto_t   *proto;
    } stream;
    
    struct {
        /* Protocols used for am operations */
        const ucp_proto_t *proto;
        const ucp_proto_t *reply_proto;
    } am_u;

} ucp_ep_config_t;


/**
 * Protocol layer endpoint, represents a connection to a remote worker
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */

    ucp_ep_cfg_index_t            cfg_index;     /* Configuration index */
    ucp_ep_conn_sn_t              conn_sn;       /* Sequence number for remote connection */
    ucp_lane_index_t              am_lane;       /* Cached value */
    ucp_ep_flags_t                flags;         /* Endpoint flags */

    /* TODO allocate ep dynamically according to number of lanes */
    uct_ep_h                      uct_eps[UCP_MAX_LANES]; /* Transports for every lane */

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_NAME_MAX];
#endif

    UCS_STATS_NODE_DECLARE(stats)

} ucp_ep_t;


/**
 * Status of protocol-level remote completions
 */
typedef struct {
    ucs_queue_head_t              reqs;         /* Queue of flush requests which
                                                   are waiting for remote completion */
    uint32_t                      send_sn;      /* Sequence number of sent operations */
    uint32_t                      cmpl_sn;      /* Sequence number of completions */
} ucp_ep_flush_state_t;


/**
 * Status of protocol-level remote completions
 */
typedef struct {
    ucp_request_t             *req;             /* Flush request which is
                                                   used in close protocol */
} ucp_ep_close_proto_req_t;

/*
 * Endpoint extension for generic non fast-path data
 */
typedef struct {
    uintptr_t                     dest_ep_ptr;   /* Remote EP pointer */
    void                          *user_data;    /* User data associated with ep */
    ucs_list_link_t               ep_list;       /* List entry in worker's all eps list */
    ucp_err_handler_cb_t          err_cb;        /* Error handler */

    /* Endpoint match context and remote completion status are mutually exclusive,
     * since remote completions are counted only after the endpoint is already
     * matched to a remote peer.
     */
    union {
        ucp_ep_match_t            ep_match;      /* Matching with remote endpoints */
        ucp_ep_flush_state_t      flush_state;   /* Remove completion status */
        ucp_listener_h            listener;      /* Listener that may be associated with ep */
        ucp_ep_close_proto_req_t  close_req;     /* Close protocol request */
    };
} ucp_ep_ext_gen_t;


/*
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
    } am;
} ucp_ep_ext_proto_t;


enum {
    UCP_WIREUP_SA_DATA_FULL_ADDR = 0,   /* Sockaddr client data contains full
                                           address. */
    UCP_WIREUP_SA_DATA_PARTIAL_ADDR,    /* Sockaddr client data contains partial
                                           address, wireup protocol requires
                                           extra MSGs. */
    UCP_WIREUP_SA_DATA_CM_ADDR          /* Sockaddr client data contains address
                                           for CM based wireup: there is only
                                           iface and ep address of transport
                                           lanes, remote device address is
                                           provided by CM and has to be added to
                                           unpacked UCP address locally. */
};


typedef struct ucp_wireup_sockaddr_data {
    uintptr_t                 ep_ptr;        /**< Endpoint pointer */
    ucp_err_handling_mode_t   err_mode;      /**< Error handling mode */
    uint8_t                   addr_mode;     /**< The attached address format
                                                  defined by
                                                  UCP_WIREUP_SOCKADDR_CD_xx */
    /* packed worker address follows */
} UCS_S_PACKED ucp_wireup_sockaddr_data_t;


typedef struct ucp_conn_request {
    ucp_listener_h              listener;
    union {
        uct_listener_h          listener;
        uct_iface_h             iface;
    } uct;
    uct_conn_request_h          uct_req;
    char                        dev_name[UCT_DEVICE_NAME_MAX];
    uct_device_addr_t           *remote_dev_addr;
    ucp_wireup_sockaddr_data_t  sa_data;
    /* packed worker address follows */
} ucp_conn_request_t;


void ucp_ep_config_key_reset(ucp_ep_config_key_t *key);

void ucp_ep_config_lane_info_str(ucp_context_h context,
                                 const ucp_ep_config_key_t *key,
                                 const unsigned *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 char *buf, size_t max);

ucs_status_t ucp_ep_new(ucp_worker_h worker, const char *peer_name,
                        const char *message, ucp_ep_h *ep_p);

void ucp_ep_delete(ucp_ep_h ep);

ucs_status_t ucp_ep_init_create_wireup(ucp_ep_h ep, unsigned ep_init_flags,
                                       ucp_wireup_ep_t **wireup_ep);

ucs_status_t ucp_ep_create_to_worker_addr(ucp_worker_h worker,
                                          const ucp_unpacked_address_t *remote_address,
                                          unsigned ep_init_flags,
                                          const char *message, ucp_ep_h *ep_p);

ucs_status_t ucp_ep_create_server_accept(ucp_worker_h worker,
                                         const ucp_conn_request_h conn_request,
                                         ucp_ep_h *ep_p);

ucs_status_ptr_t ucp_ep_flush_internal(ucp_ep_h ep, unsigned uct_flags,
                                       ucp_send_callback_t req_cb,
                                       unsigned req_flags,
                                       ucp_request_t *worker_req,
                                       ucp_request_callback_t flushed_cb,
                                       const char *debug_name);

ucs_status_t
ucp_ep_create_sockaddr_aux(ucp_worker_h worker, unsigned ep_init_flags,
                           const ucp_unpacked_address_t *remote_address,
                           ucp_ep_h *ep_p);

void ucp_ep_config_key_set_err_mode(ucp_ep_config_key_t *key,
                                    unsigned ep_init_flags);

void ucp_ep_err_pending_purge(uct_pending_req_t *self, void *arg);

void ucp_ep_disconnected(ucp_ep_h ep, int force);

void ucp_ep_destroy_internal(ucp_ep_h ep);

void ucp_ep_cleanup_lanes(ucp_ep_h ep);

int ucp_ep_is_sockaddr_stub(ucp_ep_h ep);

ucs_status_t ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config,
                                const ucp_ep_config_key_t *key);

void ucp_ep_config_cleanup(ucp_worker_h worker, ucp_ep_config_t *config);

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2);

int ucp_ep_config_get_multi_lane_prio(const ucp_lane_index_t *lanes,
                                      ucp_lane_index_t lane);

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const uct_linear_growth_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth);

ucs_status_t ucp_worker_create_mem_type_endpoints(ucp_worker_h worker);

ucp_wireup_ep_t * ucp_ep_get_cm_wireup_ep(ucp_ep_h ep);

uint64_t ucp_ep_get_tl_bitmap(ucp_ep_h ep);

uct_ep_h ucp_ep_get_cm_uct_ep(ucp_ep_h ep);

unsigned ucp_ep_do_disconnect(void *arg);

#endif
