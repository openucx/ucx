/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_MD_H
#define UCT_CUDA_IPC_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/spinlock.h>
#include <ucs/config/types.h>


#if HAVE_CUDA_FABRIC
typedef enum uct_cuda_ipc_key_handle {
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_ERROR = 0,
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY, /* cudaMalloc memory */
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM, /* cuMemCreate memory */
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_MEMPOOL /* cudaMallocAsync memory */
} uct_cuda_ipc_key_handle_t;


typedef struct uct_cuda_ipc_md_handle {
    uct_cuda_ipc_key_handle_t handle_type;
    union {
        CUipcMemHandle        legacy;        /* Legacy IPC handle */
        CUmemFabricHandle     fabric_handle; /* VMM/Mallocasync export handle */
    } handle;
    CUmemPoolPtrExportData    ptr;
    CUmemoryPool              pool;
} uct_cuda_ipc_md_handle_t;
#else
typedef CUipcMemHandle uct_cuda_ipc_md_handle_t;
#endif


/**
 * @brief cuda ipc MD descriptor
 */
typedef struct uct_cuda_ipc_md {
    uct_md_t                 super;   /**< Domain info */
    ucs_ternary_auto_value_t enable_mnnvl;
} uct_cuda_ipc_md_t;


typedef struct uct_cuda_ipc_uuid_hash_key {
    int     type;
    CUuuid  uuid;
} uct_cuda_ipc_uuid_hash_key_t;


typedef struct {
    /* GPU Device number */
    int     dev_num;
    /* Cache of accessible devices (ucs_ternary_auto_value_t) */
    uint8_t accessible[0];
} uct_cuda_ipc_dev_cache_t;


static UCS_F_ALWAYS_INLINE int
uct_cuda_ipc_uuid_equals(uct_cuda_ipc_uuid_hash_key_t key1,
                         uct_cuda_ipc_uuid_hash_key_t key2)
{
    int64_t *a64 = (int64_t *)key1.uuid.bytes;
    int64_t *b64 = (int64_t *)key2.uuid.bytes;

    return (key1.type == key2.type) && (a64[0] == b64[0]) && (a64[1] == b64[1]);
}


static UCS_F_ALWAYS_INLINE khint32_t
uct_cuda_ipc_uuid_hash_func(uct_cuda_ipc_uuid_hash_key_t key)
{
    int64_t *i64 = (int64_t *)key.uuid.bytes;
    return kh_int64_hash_func(i64[0] ^ i64[1] ^ key.type);
}


KHASH_INIT(cuda_ipc_uuid_hash, uct_cuda_ipc_uuid_hash_key_t,
           uct_cuda_ipc_dev_cache_t*, 1, uct_cuda_ipc_uuid_hash_func,
           uct_cuda_ipc_uuid_equals);


/**
 * @brief cuda ipc component extension
 */
typedef struct {
    uct_component_t             super;
    khash_t(cuda_ipc_uuid_hash) uuid_hash;
    pthread_mutex_t             lock;
} uct_cuda_ipc_component_t;

extern uct_cuda_ipc_component_t uct_cuda_ipc_component;

/**
 * @brief cuda ipc domain configuration.
 */
typedef struct uct_cuda_ipc_md_config {
    uct_md_config_t          super;
    ucs_ternary_auto_value_t enable_mnnvl;
} uct_cuda_ipc_md_config_t;


/**
 * @brief list of cuda ipc regions registered for memh
 */
typedef struct {
    pid_t           pid;     /* PID as key to resolve peer_map hash */
    int             dev_num; /* GPU Device number */
    ucs_list_link_t list;
} uct_cuda_ipc_memh_t;


/**
 * @brief cudar ipc region registered for exposure
 */
typedef struct {
    uct_cuda_ipc_md_handle_t  ph;     /* Memory handle of GPU memory */
    CUdeviceptr               d_bptr; /* Allocation base address */
    size_t                    b_len;  /* Allocation size */
    ucs_list_link_t           link;
} uct_cuda_ipc_lkey_t;


/**
 * @brief cuda ipc remote key for put/get
 */
typedef struct {
    uct_cuda_ipc_md_handle_t  ph;      /* Memory handle of GPU memory */
    pid_t                     pid;     /* PID as key to resolve peer_map hash */
    CUdeviceptr               d_bptr;  /* Allocation base address */
    size_t                    b_len;   /* Allocation size */
    int                       dev_num; /* GPU Device number */
    CUuuid                    uuid;    /* GPU Device UUID */
} uct_cuda_ipc_rkey_t;


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
