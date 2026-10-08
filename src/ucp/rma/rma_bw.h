/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_BW_H_
#define UCP_RMA_BW_H_

#include <ucp/core/ucp_request.h>

#define UCP_RMA_BW_MAX_FRAGS       128
#define UCP_RMA_BW_MAX_ACTIVE      4
#define UCP_RMA_BW_SAMPLE_INTERVAL 1.0
#define UCP_RMA_BW_STALE_INTERVAL  30.0
#define UCP_RMA_BW_MIN_LANE_LENGTH (128 * UCS_KBYTE)

typedef enum {
    UCP_RMA_BW_PUT,
    UCP_RMA_BW_GET,
    UCP_RMA_BW_DIR_LAST
} ucp_rma_bw_dir_t;

typedef enum {
    UCP_RMA_BW_POST_NONE,
    UCP_RMA_BW_POST_TRACK,
    UCP_RMA_BW_POST_SAMPLE
} ucp_rma_bw_post_mode_t;

typedef enum {
    UCP_RMA_BW_REJECT_NONE,
    UCP_RMA_BW_REJECT_INCOMPLETE,
    UCP_RMA_BW_REJECT_UNSATURATED,
    UCP_RMA_BW_REJECT_GENERATION,
    UCP_RMA_BW_REJECT_RATE_LIMIT,
    UCP_RMA_BW_REJECT_WARMUP,
    UCP_RMA_BW_REJECT_WINDOW
} ucp_rma_bw_reject_t;

typedef struct {
    uct_completion_t    comp;
    ucp_rma_bw_sample_t *sample;
    ucs_time_t          post_time;
    uint8_t             lane_idx;
} ucp_rma_bw_frag_t;

typedef struct {
    size_t     bytes;
    ucs_time_t first_post;
    ucs_time_t last_comp;
    uint64_t   posted_total;
    unsigned   num_frags;
    unsigned   num_async;
    unsigned   pending;
} ucp_rma_bw_lane_t;

typedef struct {
    double           nominal;
    double           smoothed;
    ucs_time_t       last_update;
    ucs_time_t       marker_time;
    uint64_t         posted_total;
    uint64_t         marker_posted;
    unsigned         samples;
    ucp_lane_index_t lane_id;
} ucp_rma_bw_lane_estimate_t;

struct ucp_rma_bw_estimator {
    uint64_t                   epoch;
    uint32_t                   generation;
    ucp_lane_index_t           num_lanes;
    ucp_rma_bw_lane_estimate_t lanes[];
};

typedef struct ucp_rma_bw_ep_state {
    uint32_t               generation;
    ucp_rma_bw_estimator_t *dirs[UCP_RMA_BW_DIR_LAST];
} ucp_rma_bw_ep_state_t;

struct ucp_rma_bw_sample {
    ucp_request_t     *req;
    unsigned          num_frags;
    unsigned          pending;
    unsigned          invalid;
    unsigned          concurrent;
    ucp_lane_map_t    active_lanes;
    uint64_t          epoch;
    uint32_t          generation;
    ucp_lane_index_t  num_lanes;
    ucp_rma_bw_dir_t  dir;
    ucp_lane_index_t  lane_ids[UCP_MAX_LANES];
    ucp_rma_bw_lane_t lanes[UCP_MAX_LANES];
    ucp_rma_bw_frag_t frags[UCP_RMA_BW_MAX_FRAGS];
};

void ucp_rma_bw_sample_start(ucp_request_t *req, ucp_lane_index_t num_lanes,
                             ucp_rma_bw_dir_t dir);
void ucp_rma_bw_sample_complete(uct_completion_t *comp);
void ucp_rma_bw_frag_complete(uct_completion_t *comp);
void ucp_rma_bw_sample_detach(ucp_request_t *req);
void ucp_rma_bw_ep_state_cleanup(ucp_ep_h ep);

ucp_rma_bw_reject_t
ucp_rma_bw_estimator_update(ucp_rma_bw_estimator_t *estimator,
                            const ucp_rma_bw_sample_t *sample, ucs_time_t now);

#endif
