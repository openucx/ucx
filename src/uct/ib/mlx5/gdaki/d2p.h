/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_H_
#define UCT_D2P_H_

#include <uct/api/device/uct_device_types.h>

typedef struct {
    unsigned long long *pi;
    unsigned long long *ci;
    void               *queue_base;
} uct_ib_d2p_gpu_channel_t;

typedef struct {
    uint8_t                  channel_mask;
    uint8_t                  log_depth;
    uint8_t                  pad[2];
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

#endif /* UCT_D2P_H_ */
