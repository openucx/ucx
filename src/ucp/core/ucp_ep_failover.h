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

typedef void (*ucp_ep_failover_lane_failed_cb_t)(ucs_status_t status,
                                                 void *user_data);

void ucp_ep_failover_init(ucp_ep_h ep);

void ucp_ep_failover_cleanup(ucp_ep_h ep);

uct_ep_h ucp_ep_failover_get_uct_ep(ucp_ep_h ep, ucp_lane_index_t lane);

int ucp_ep_failover_is_uct_ep(ucp_ep_h ep, ucp_lane_index_t lane,
                              uct_ep_h uct_ep);

ucs_status_t
ucp_ep_failover_add_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                          uct_ep_h *uct_eps, ucp_ep_failover_lane_done_cb_t cb,
                          ucp_ep_failover_lane_failed_cb_t failed_cb, void *arg,
                          ucp_lane_map_t *failover_lanes_p);

void ucp_ep_failover_cancel_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map);

/** Enable failover extraction on eligible lanes before posting operations. */
ucs_status_t ucp_ep_failover_enable_lanes(ucp_ep_h ep);

/** Lanes armed for failover that still need RX tokens. */
ucp_lane_map_t ucp_ep_failover_pending_rx_lanes(ucp_ep_h ep);

/** Correlate ADDR token trailers with the active recovery generation. */
void ucp_ep_failover_set_request_id(ucp_ep_h ep, uint64_t request_id);

uint64_t ucp_ep_failover_get_request_id(ucp_ep_h ep);

/**
 * Install peer RX tokens from an ADDR_REP/ACK trailer and schedule extract.
 * @a request_id must match the generation stored on the failover context
 * (0 skips correlation when the peer initiated asymmetric failover).
 */
ucs_status_t
ucp_ep_failover_apply_rx_tokens(ucp_ep_h ep, uint64_t request_id,
                                ucp_lane_map_t lane_map,
                                const uint8_t *token_lengths,
                                const void *tokens);

/** Abort in-progress failover (releases flush_ops / discard refs). */
void ucp_ep_failover_abort(ucp_ep_h ep, ucs_status_t status);

/** Notify that one extracted outstanding op finished (reposted or aborted). */
void ucp_ep_failover_replay_completed(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                                      ucs_status_t status);

ucp_lane_map_t ucp_ep_failover_test_pending_rx_lane_map(ucp_ep_h ep);

#endif
