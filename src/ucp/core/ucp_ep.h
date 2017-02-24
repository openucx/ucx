/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_context.h"

#include <uct/api/uct.h>
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
    UCP_EP_FLAG_LOCAL_CONNECTED  = UCS_BIT(0), /* All local endpoints are connected */
    UCP_EP_FLAG_REMOTE_CONNECTED = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_SENT = UCS_BIT(2), /* Connection request was sent */
    UCP_EP_FLAG_CONNECT_REP_SENT = UCS_BIT(3), /* Debug: Connection reply was sent */
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


/* Lanes configuration.
 * Every lane is a UCT endpoint to the same remote worker.
 */
typedef struct ucp_ep_config_key {
    /* Lookup for rma and amo lanes:
     * Every group of UCP_MD_INDEX_BITS consecutive bits in the map (in total
     * UCP_MAX_LANES such groups) is a bitmap. All bits in it are zero, except
     * the md_index-th bit of that lane. If the lane is unused, all bits are zero.
     * For example, the bitmap '00000100' means the lane remote md_index is 2.
     * It allows to quickly lookup
     */
    ucp_md_lane_map_t      rma_lane_map;

    /* AMO lanes point to another indirect lookup array */
    ucp_md_lane_map_t      amo_lane_map;
    ucp_lane_index_t       amo_lanes[UCP_MAX_LANES];

    /* Bitmap of remote mds which are reachable from this endpoint (with any set
     * of transports which could be selected in the future)
     */
    ucp_md_map_t           reachable_md_map;

    ucp_lane_index_t       am_lane;             /* Lane for AM (can be NULL) */
    ucp_lane_index_t       rndv_lane;           /* Lane for zcopy Rendezvous (can be NULL) */
    ucp_lane_index_t       wireup_msg_lane;     /* Lane for wireup messages (can be NULL) */
    ucp_rsc_index_t        lanes[UCP_MAX_LANES];/* Resource index for every lane */
    ucp_lane_index_t       num_lanes;           /* Number of lanes */
} ucp_ep_config_key_t;


/**
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


typedef struct ucp_ep_config {

    /* A key which uniquely defines the configuration, and all other fields of
     * configuration (in the current worker) and defined only by it.
     */
    ucp_ep_config_key_t    key;

    /* Bitmap of which lanes are p2p; affects the behavior of connection
     * establishment protocols.
     */
    ucp_lane_map_t         p2p_lanes;

    /* Limits for active-message based protocols */
    struct {
        ssize_t                max_eager_short;  /* Maximal payload of eager short */
        ssize_t                max_short;        /* Maximal payload of am short */
        size_t                 max_bcopy;        /* Maximal total size of am_bcopy */
        size_t                 max_zcopy;        /* Maximal total size of am_zcopy */
        size_t                 max_iovcnt;       /* Maximal size of iovcnt */
        /* zero-copy threshold for operations which do not have to wait for remote side */
        size_t                 zcopy_thresh[UCP_MAX_IOV];
        /* zero-copy threshold for operations which anyways have to wait for remote side */
        size_t                 sync_zcopy_thresh[UCP_MAX_IOV];
        uint8_t                zcopy_auto_thresh; /* if != 0 the zcopy enabled */
    } am;

    /* Configuration for each lane that provides RMA */
    ucp_ep_rma_config_t        rma[UCP_MAX_LANES];
    /* Threshold for switching from put_short to put_bcopy */
    size_t                     bcopy_thresh;

    struct {
        /* Maximal total size of rndv_get_zcopy */
        size_t                 max_get_zcopy;
        /* Threshold for switching from eager to RMA based rendezvous */
        size_t                 rma_thresh;
        /* Threshold for switching from eager to AM based rendezvous */
        size_t                 am_thresh;
    } rndv;
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

    UCS_STATS_NODE_DECLARE(stats);

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_NAME_MAX];
#endif

    /* TODO allocate ep dynamically according to number of lanes */
    uct_ep_h                      uct_eps[UCP_MAX_LANES]; /* Transports for every lane */

} ucp_ep_t;


ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *peer_name, const char *message,
                        ucp_ep_h *ep_p);

ucs_status_t ucp_ep_create_stub(ucp_worker_h worker, uint64_t dest_uuid,
                                const char *message, ucp_ep_h *ep_p);

void ucp_ep_destroy_internal(ucp_ep_h ep, const char *message);

int ucp_ep_is_stub(ucp_ep_h ep);

void ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config);

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2);

ucp_md_map_t ucp_ep_config_get_rma_md_map(const ucp_ep_config_key_t *key,
                                          ucp_lane_index_t lane);

ucp_md_map_t ucp_ep_config_get_amo_md_map(const ucp_ep_config_key_t *key,
                                          ucp_lane_index_t lane);

size_t ucp_ep_config_get_zcopy_auto_thresh(size_t iovcnt,
                                           const uct_linear_growth_t *reg_cost,
                                           const ucp_context_h context,
                                           double bandwidth);

#endif
