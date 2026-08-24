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
#include <ucs/datastruct/string_buffer.h>


void ucp_wireup_log_ep_lanes(ucp_worker_h worker,
                             const ucp_ep_config_key_t *key,
                             ucp_worker_cfg_index_t cfg_index);


ucs_status_t ucp_wireup_render_ep_lanes(ucp_context_h context,
                                        const ucp_ep_config_key_t *key,
                                        ucp_worker_cfg_index_t cfg_index,
                                        ucs_string_buffer_t *strb);


#endif
