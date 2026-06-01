/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WIREUP_LANE_INFO_H_
#define UCP_WIREUP_LANE_INFO_H_

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>


void ucp_wireup_log_ep_lanes(ucp_worker_h worker,
                             const ucp_ep_config_key_t *key,
                             ucp_worker_cfg_index_t cfg_index);


#endif
