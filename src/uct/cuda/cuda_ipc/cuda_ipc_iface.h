/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda.h>

#include "cuda_ipc_md.h"
#include "cuda_ipc_ep.h"
#include "cuda_ipc_cache.h"


#define UCT_CUDA_IPC_MAX_PEERS 128


typedef struct {
    unsigned                max_poll;            /* query attempts w.o success */
    unsigned                max_streams;         /* # concurrent streams for || progress*/
    unsigned                max_cuda_ipc_events; /* max mpool entries */
    int                     enable_cache;        /* enable/disable ipc handle cache */
    ucs_on_off_auto_value_t enable_get_zcopy;    /* enable get_zcopy except for specific platforms */
    double                  bandwidth;           /* estimated bandwidth */
    double                  latency;             /* estimated latency */
    double                  overhead;            /* estimated CPU overhead */
} uct_cuda_ipc_iface_config_params_t;


typedef struct {
    uct_cuda_iface_t                   super;
    uct_cuda_ipc_iface_config_params_t config;
} uct_cuda_ipc_iface_t;


typedef struct {
    uct_iface_config_t                 super;
    uct_cuda_ipc_iface_config_params_t params;
} uct_cuda_ipc_iface_config_t;


typedef struct {
    uct_cuda_event_desc_t super;
    void                  *mapped_addr;
    uct_cuda_ipc_ep_t     *ep;
    uintptr_t             d_bptr;
    pid_t                 pid;
    CUdevice              cuda_device;
} uct_cuda_ipc_event_desc_t;


typedef struct {
    uct_cuda_ctx_rsc_t    super;
    uct_cuda_queue_desc_t queue_desc[UCT_CUDA_IPC_MAX_PEERS];
} uct_cuda_ipc_ctx_rsc_t;

#endif
