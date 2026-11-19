/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_H_
#define UCT_D2P_H_

#include <uct/api/device/uct_device_types.h>

#include "d2p_proto.h"

typedef struct UCS_V_ALIGNED(128) {
    unsigned long long pi;
    void               *queue_base;
    uint64_t           *done;
    uint64_t           *avail;
} uct_ib_d2p_gpu_channel_t;

typedef struct {
    uint32_t                 channel_mask;
    uint32_t                 depth_mask;
    uint32_t                 depth;
    uint32_t                 atomic_result_lkey;
    uint64_t                 atomic_result_va;
    uct_ib_d2p_gpu_channel_t channels[];
} uct_ib_d2p_gpu_iface_t;

typedef struct {
    uct_device_ep_t        super;
    uct_ib_d2p_gpu_iface_t *iface;
    uint64_t               ep_idx;
    uint8_t                pad[8];
} uct_ib_d2p_gpu_ep_t;

/**
 * @brief Completion object for device D2P operations.
 */
typedef struct {
    uint64_t comp_id;
    unsigned channel_id;
} uct_ib_d2p_completion_t;

#endif /* UCT_D2P_H_ */
