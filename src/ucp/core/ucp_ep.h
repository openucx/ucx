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
#include <ucs/debug/log.h>
#include <limits.h>


/**
 * Endpoint flags
 */
enum {
    UCP_EP_FLAG_LOCAL_CONNECTED  = UCS_BIT(0), /* All local endpoints are connected */
    UCP_EP_FLAG_REMOTE_CONNECTED = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_SENT = UCS_BIT(2), /* Connection request was sent */
};


/* Lanes configuration.
 * Every lane is a UCT endpoint to the same remote worker.
 */
typedef struct ucp_ep_config_key {
    /* Lookup for rma and amo lanes:
     * Every group of UCP_PD_INDEX_BITS consecutive bits in the map (in total
     * UCP_MAX_LANES such groups) is a bitmap. All bits in it are zero, except
     * the pd_index-th bit of that lane. If the lane is unused, all bits are zero.
     * For example, the bitmap '00000100' means the lane remote pd_index is 2.
     * It allows to quickly lookup
     */
    ucp_pd_lane_map_t      rma_lane_map;
    ucp_pd_lane_map_t      amo_lane_map;

    /* Bitmap of remote pds which are reachable from this endpoint (with any set
     * of transports which could be selected in the future)
     */
    ucp_pd_map_t           reachable_pd_map;

    ucp_lane_index_t       am_lane;             /* Lane for AM (can be NULL) */
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
    size_t                 max_get_bcopy;    /* Maximal total size of get_bcopy */
} ucp_ep_rma_config_t;


typedef struct ucp_ep_config {

    /* A key which uniquely defines the configuration, and all other fields of
     * configuration (in the current worker) and defined only by it.
     */
    ucp_ep_config_key_t    key;

    /* Limits for active-message based protocols */
    size_t                 max_eager_short;  /* Maximal payload of eager short */
    size_t                 max_am_short;     /* Maximal payload of am short */
    size_t                 max_am_bcopy;     /* Maximal total size of am_bcopy */
    size_t                 max_am_zcopy;     /* Maximal total size of am_zcopy */

    ucp_ep_rma_config_t    rma[UCP_MAX_LANES];

    /* Threshold for switching from put_short to put_bcopy */
    size_t                 bcopy_thresh;

    /* Threshold for switching from eager to rendezvous */
    size_t                 rndv_thresh;

    /* threshold for switching from eager-sync to rendezvous */
    size_t                 sync_rndv_thresh;

    /* zero-copy threshold for operations which do not have to wait for remote side */
    size_t                 zcopy_thresh;

    /* zero-copy threshold for operations which anyways have to wait for remote side */
    size_t                 sync_zcopy_thresh;

} ucp_ep_config_t;


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */

    uint16_t                      cfg_index;     /* Configuration index */
    ucp_lane_index_t              am_lane;       /* Cached value */
    uint8_t                       flags;         /* Endpoint flags */

    uint64_t                      dest_uuid;     /* Destination worker uuid */
    ucp_ep_h                      next;          /* Next in hash table linked list */

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

int ucp_ep_is_stub(ucp_ep_h ep);

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep);

ucs_status_t ucp_ep_add_pending_uct(ucp_ep_h ep, uct_ep_h uct_ep,
                                    uct_pending_req_t *req);

void ucp_ep_add_pending(ucp_ep_h ep, uct_ep_h uct_ep, ucp_request_t *req,
                        int progress);

ucs_status_t ucp_ep_pending_req_release(uct_pending_req_t *self);

void ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config);

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2);

#endif
