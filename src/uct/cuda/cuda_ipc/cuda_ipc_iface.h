/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define UCT_CUDA_IPC_TL_NAME    "cuda_ipc"
#define UCT_CUDA_DEV_NAME       "cudaipc0"


typedef struct uct_cuda_ipc_iface {
    uct_base_iface_t           super;
} uct_cuda_ipc_iface_t;


typedef struct uct_cuda_ipc_iface_config {
    uct_iface_config_t      super;
} uct_cuda_ipc_iface_config_t;


#endif
