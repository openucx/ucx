/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define UCT_CUDA_IPC_TL_NAME    "cuda_ipc"
#define UCT_CUDA_IPC_DEV_NAME   "cudaipc0"
#define UCT_CUDA_IPC_MAX_PEERS  16


typedef struct uct_cuda_ipc_iface {
    uct_base_iface_t super;
    ucs_mpool_t      event_desc;
    ucs_queue_head_t outstanding_d2d_event_q;
    int              device_count;
    int              streams_initialized;
    CUstream         stream_d2d[UCT_CUDA_IPC_MAX_PEERS];
    int              p2p_map[UCT_CUDA_IPC_MAX_PEERS][UCT_CUDA_IPC_MAX_PEERS];
    struct {
        unsigned     max_poll;
    } config;
} uct_cuda_ipc_iface_t;


typedef struct uct_cuda_ipc_iface_config {
    uct_iface_config_t super;
    unsigned           max_poll;
} uct_cuda_ipc_iface_config_t;


typedef struct uct_cuda_ipc_event_desc {
    CUevent           event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_ipc_event_desc_t;


ucs_status_t uct_cuda_ipc_iface_init_streams(uct_cuda_ipc_iface_t *iface);
#endif
