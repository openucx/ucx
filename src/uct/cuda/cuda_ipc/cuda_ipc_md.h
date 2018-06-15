/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_MD_H
#define UCT_CUDA_IPC_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>
#include <uct/cuda/base/cuda_iface.h>


#define UCT_CUDA_IPC_MD_NAME      "cuda_ipc"
#define UCT_CUDA_IPC_MAX_ALLOC_SZ (1 << 30)


extern uct_md_component_t uct_cuda_ipc_md_component;


/**
 * @brief cuda ipc MD descriptor
 */
typedef struct uct_cuda_ipc_md {
    struct uct_md super;   /**< Domain info */
} uct_cuda_ipc_md_t;


/**
 * @brief cuda ipc domain configuration.
 */
typedef struct uct_cuda_ipc_md_config {
    uct_md_config_t super;
} uct_cuda_ipc_md_config_t;


/**
 * @brief cuda_ipc packed and remote key for put/get
 */
typedef struct uct_cuda_ipc_key {
    CUipcMemHandle ph;           /* Memory handle of GPU memory */
    CUdeviceptr    d_bptr;       /* Allocation base address */
    size_t         b_len;        /* Allocation size */
    int            dev_num;      /* GPU Device number */
} uct_cuda_ipc_key_t;

typedef uct_cuda_ipc_key_t uct_cuda_ipc_mem_t;

#define UCT_CUDA_IPC_GET_DEVICE(_cu_device)                             \
    do {                                                                \
        if (UCS_OK != UCT_CUDADRV_FUNC(cuCtxGetDevice(&_cu_device))) {  \
            return UCS_ERR_IO_ERROR;                                    \
        }                                                               \
    } while(0);

#endif
