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

#define UCP_MAX_IOV                16UL


/* Configuration */
typedef uint16_t                   ucp_ep_cfg_index_t;


/**
 * Endpoint flags
 */
enum {
    UCP_EP_FLAG_LOCAL_CONNECTED     = UCS_BIT(0), /* All local endpoints are connected */
    UCP_EP_FLAG_REMOTE_CONNECTED    = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_QUEUED  = UCS_BIT(2), /* Connection request was queued */
    UCP_EP_FLAG_TAG_OFFLOAD_ENABLED = UCS_BIT(3), /* Endpoint uses tl offload for tag matching */
    UCP_EP_FLAG_FAILED              = UCS_BIT(4), /* EP is in failed state */

    UCP_EP_FLAG_CONNECT_REQ_SENT    = UCS_BIT(5), /* DEBUG: Connection request was sent */
    UCP_EP_FLAG_CONNECT_REP_SENT    = UCS_BIT(6), /* DEBUG: Connection reply was sent */
    UCP_EP_FLAG_CONNECT_ACK_SENT    = UCS_BIT(7)  /* DEBUG: Connection ACK was sent */
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
    ucp_lane_index_t       rndv_lane;    /* Lane for zcopy Rendezvous (can be NULL) */
    ucp_lane_index_t       tag_lane;     /* Lane for tag matching offload (can be NULL) */
    ucp_lane_index_t       wireup_lane;  /* Lane for wireup messages (can be NULL) */

    /* Lanes for remote memory access, sorted by priority, highest first */
    ucp_lane_index_t       rma_lanes[UCP_MAX_LANES];

    /* Lanes for atomic operations, sorted by priority, highest first */
    ucp_lane_index_t       amo_lanes[UCP_MAX_LANES];

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
            /* Threshold for switching from eager to RMA based rendezvous */
            size_t          rma_thresh;
            /* Threshold for switching from eager to AM based rendezvous */
            size_t          am_thresh;
        } rndv;

        struct {
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
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */

    ucp_ep_cfg_index_t            cfg_index;     /* Configuration index */
    ucp_lane_index_t              am_lane;       /* Cached value */
    uint8_t                       flags;         /* Endpoint flags */

    uint64_t                      dest_uuid;     /* Destination worker uuid */
    void                          *user_data;    /* user data associated with
                                                    the endpoint */

    ucs_queue_head_t              stream_data;  /* Queue of receive descriptors
                                                   with data */
    UCS_STATS_NODE_DECLARE(stats);

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_NAME_MAX];
#endif

    /* TODO allocate ep dynamically according to number of lanes */
    uct_ep_h                      uct_eps[UCP_MAX_LANES]; /* Transports for every lane */

} ucp_ep_t;


void ucp_ep_config_key_reset(ucp_ep_config_key_t *key);

void ucp_ep_config_lane_info_str(ucp_context_h context,
                                 const ucp_ep_config_key_t *key,
                                 const uint8_t *addr_indices,
                                 ucp_lane_index_t lane,
                                 ucp_rsc_index_t aux_rsc_index,
                                 char *buf, size_t max);

ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *peer_name, const char *message,
                        ucp_ep_h *ep_p);

ucs_status_ptr_t ucp_ep_flush_internal(ucp_ep_h ep, unsigned uct_flags,
                                       ucp_send_callback_t req_cb,
                                       unsigned req_flags,
                                       ucp_request_callback_t flushed_cb);

ucs_status_t ucp_ep_create_stub(ucp_worker_h worker, uint64_t dest_uuid,
                                const char *message, ucp_ep_h *ep_p);

void ucp_ep_err_pending_purge(uct_pending_req_t *self, void *arg);

void ucp_ep_destroy_internal(ucp_ep_h ep, const char *message);

int ucp_ep_is_stub(ucp_ep_h ep);

void ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config);

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2);

int ucp_ep_config_get_rma_prio(const ucp_lane_index_t *lanes,
                               ucp_lane_index_t lane);

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const uct_linear_growth_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth);

#endif
