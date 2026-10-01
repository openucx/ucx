/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_BW_H_
#define UCP_RMA_BW_H_

#include <ucp/core/ucp_request.h>

/* The first measurement stage records completed payload throughput only. */
#define UCP_RMA_BW_MAX_FRAGS       128
#define UCP_RMA_BW_MAX_ACTIVE      4
#define UCP_RMA_BW_SAMPLE_INTERVAL 1.0
#define UCP_RMA_BW_MIN_LANE_LENGTH (128 * UCS_KBYTE)

typedef struct ucp_rma_bw_sample ucp_rma_bw_sample_t;

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
    unsigned   num_frags;
    unsigned   num_async;
} ucp_rma_bw_lane_t;

struct ucp_rma_bw_sample {
    ucp_request_t     *req;
    unsigned          num_frags;
    unsigned          pending;
    unsigned          invalid;
    ucp_lane_index_t  num_lanes;
    ucp_rma_bw_lane_t lanes[UCP_MAX_LANES];
    ucp_rma_bw_frag_t frags[UCP_RMA_BW_MAX_FRAGS];
};

void ucp_rma_bw_sample_complete(uct_completion_t *comp);
void ucp_rma_bw_frag_complete(uct_completion_t *comp);
void ucp_rma_bw_sample_detach(ucp_request_t *req);

#endif
