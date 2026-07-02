/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCP_EP_FAILOVER_H_
#define UCP_EP_FAILOVER_H_

#include <ucp/core/ucp_ep.h>
#include <ucp/wireup/wireup.h>

typedef void (*ucp_ep_failover_lane_done_cb_t)(void *request,
                                               ucs_status_t status,
                                               void *user_data);

void ucp_ep_failover_init(ucp_ep_h ep);

void ucp_ep_failover_cleanup(ucp_ep_h ep);

uct_ep_h ucp_ep_failover_get_uct_ep(ucp_ep_h ep, ucp_lane_index_t lane);

ucs_status_t
ucp_ep_failover_add_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                          uct_ep_h *uct_eps, ucs_status_t discard_status,
                          ucp_ep_failover_lane_done_cb_t cb, void *arg,
                          ucp_lane_map_t *failover_lanes_p);

void ucp_ep_failover_abort_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                                 ucs_status_t status);

ucs_status_t ucp_ep_failover_query_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map);

/**
 * Extract outstanding zcopy requests from failed lanes and restart them
 * through their owning UCP requests.
 */
ucs_status_t
ucp_ep_failover_on_lane_state(ucp_ep_h ep,
                              const ucp_wireup_lane_state_t *lane_state);

ucp_lane_map_t ucp_ep_failover_test_query_lane_map(ucp_ep_h ep);

ucs_status_t
ucp_ep_failover_test_validate_lane_state(ucp_ep_h ep,
                                         const ucp_wireup_lane_state_t *state,
                                         size_t length);

#endif
