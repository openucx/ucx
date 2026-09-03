/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCP_EP_FAILOVER_H_
#define UCP_EP_FAILOVER_H_

#include <ucp/core/ucp_ep.h>


void ucp_ep_failover_cleanup(ucp_ep_h ep);

/**
 * Whether this UCT ep's iface can keep outstanding ops for token-based
 * failover (@c UCT_IFACE_FLAG_V2_QUERY_TOKEN and a valid TX token length).
 */
int ucp_ep_failover_is_token_supported(uct_ep_h uct_ep);

/**
 * Take ownership of every UCT ep in @a lane_map for token-based failover.
 * Returns UCS_OK only if all lanes support QUERY_TOKEN and were armed.
 * @a cb is invoked with @a arg once per lane when failover is aborted,
 * matching ucp_worker_discard_uct_ep() completion.
 */
ucs_status_t
ucp_ep_failover_add_lanes(ucp_ep_h ep, ucp_lane_map_t lane_map,
                          uct_ep_h *uct_eps, ucp_send_nbx_callback_t cb,
                          void *arg);

/** Abort in-progress failover and release owned UCT endpoints. */
void ucp_ep_failover_abort(ucp_ep_h ep, ucs_status_t status);

/**
 * Abort from worker progress so UCT can finish its failure handler first.
 * Used until extract/replay takes over outstanding ops.
 */
void ucp_ep_failover_schedule_abort(ucp_ep_h ep, ucs_status_t status);

#endif
