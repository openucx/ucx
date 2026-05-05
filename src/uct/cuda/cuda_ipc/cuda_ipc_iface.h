/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include "cuda_ipc_ep.h"
#include "ucs/config/types.h"
#include "ucs/sys/compiler_def.h"
#include "ucs/sys/sys.h"
#include "uct/api/uct_def.h"
#include "uct/base/uct_iface.h"
#include "uct/cuda/base/cuda_iface.h"

#include <cuda.h>

#include <stdint.h>
#include <sys/types.h>

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
    ucs_sys_ns_t          pid_ns;
    CUdevice              cuda_device;
} uct_cuda_ipc_event_desc_t;


typedef struct {
    uct_cuda_ctx_rsc_t    super;
    uct_cuda_queue_desc_t queue_desc[UCT_CUDA_IPC_MAX_PEERS];
} uct_cuda_ipc_ctx_rsc_t;


typedef struct {
    pid_t        pid;
    ucs_sys_ns_t pid_ns;
} UCS_S_PACKED uct_cuda_ipc_iface_address_t;


/**
 * Unpack the CUDA IPC interface address from the given interface address.
 */
uct_cuda_ipc_iface_address_t
uct_cuda_ipc_iface_address_unpack(const uct_iface_addr_t *iface_addr,
                                  size_t iface_addr_length);

#endif
