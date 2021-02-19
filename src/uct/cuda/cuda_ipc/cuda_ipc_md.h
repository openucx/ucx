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
#include <ucs/type/spinlock.h>
#include <ucs/config/types.h>


/**
 * @brief cuda ipc MD descriptor
 */
typedef struct uct_cuda_ipc_md {
    struct uct_md            super;   /**< Domain info */
    CUuuid*                  uuid_map;
    ucs_ternary_auto_value_t *peer_accessible_cache;
    int                      uuid_map_size;
    int                      uuid_map_capacity;
} uct_cuda_ipc_md_t;

/**
 * @brief cuda ipc component extension
 */
typedef struct uct_cuda_ipc_component {
    uct_component_t    super;
    uct_cuda_ipc_md_t* md;
} uct_cuda_ipc_component_t;

extern uct_cuda_ipc_component_t uct_cuda_ipc_component;

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
    CUipcMemHandle ph;      /* Memory handle of GPU memory */
    pid_t          pid;     /* PID as key to resolve peer_map hash */
    CUdeviceptr    d_bptr;  /* Allocation base address */
    size_t         b_len;   /* Allocation size */
    int            dev_num; /* GPU Device number */
    CUuuid         uuid;    /* GPU Device UUID */
} uct_cuda_ipc_key_t;


#define UCT_CUDA_IPC_GET_DEVICE(_cu_device)                          \
    do {                                                             \
        if (UCS_OK !=                                                \
            UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetDevice(&_cu_device))) { \
            return UCS_ERR_IO_ERROR;                                 \
        }                                                            \
    } while(0);


#define UCT_CUDA_IPC_DEVICE_GET_COUNT(_num_device)                      \
    do {                                                                \
        if (UCS_OK !=                                                   \
            UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&_num_device))) { \
            return UCS_ERR_IO_ERROR;                                    \
        }                                                               \
    } while(0);

#endif
