/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
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


typedef enum {
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_NO_IPC = 0,
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY, /* cudaMalloc memory */
#if HAVE_CUDA_FABRIC
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM, /* cuMemCreate memory */
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_MEMPOOL, /* cudaMallocAsync memory */
    UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI /* Multi-chunk VMM with metadata fetch */
#endif
} uct_cuda_ipc_key_handle_t;


typedef struct uct_cuda_ipc_md_handle {
    uct_cuda_ipc_key_handle_t handle_type;
    union {
        CUipcMemHandle        legacy;        /* Legacy IPC handle */
#if HAVE_CUDA_FABRIC
        CUmemFabricHandle     fabric_handle; /* VMM/Mallocasync export handle */
#endif
    } handle;
#if HAVE_CUDA_FABRIC
    CUmemPoolPtrExportData    ptr;
    CUmemoryPool              pool;
#endif
    unsigned long long        buffer_id;
} uct_cuda_ipc_md_handle_t;

/**
 * @brief cuda ipc MD descriptor
 */
typedef struct uct_cuda_ipc_md {
    uct_md_t                 super;             /**< Domain info */
    int                      enable_mnnvl;      /**< Multi-node NVLINK support status */
} uct_cuda_ipc_md_t;


typedef struct {
    uint8_t type;     /**< uct_cuda_ipc_key_handle_t */
    uint8_t is_local; /**< 1 if the key is local (PID+PID_NS), 0 otherwise */
    CUuuid  uuid;     /**< GPU Device UUID */
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

    return (key1.type == key2.type) && (key1.is_local == key2.is_local) &&
           (a64[0] == b64[0]) && (a64[1] == b64[1]);
}


static UCS_F_ALWAYS_INLINE khint32_t
uct_cuda_ipc_uuid_hash_func(uct_cuda_ipc_uuid_hash_key_t key)
{
    int64_t *i64 = (int64_t *)key.uuid.bytes;
    return kh_int64_hash_func(i64[0] ^ i64[1] ^ (key.type << 1) ^ key.is_local);
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
    unsigned long            cache_max_regions; /**< Max cached IPC regions per peer */
    size_t                   cache_max_size;    /**< Max total cached IPC mapping size */
} uct_cuda_ipc_md_config_t;


/**
 * @brief list of cuda ipc regions registered for memh
 */
typedef struct {
    pid_t           pid;     /* PID as key to resolve peer_map hash */
    ucs_sys_ns_t    pid_ns;  /* PID namespace */
    int             dev_num; /* GPU Device number */
    ucs_list_link_t list;
} uct_cuda_ipc_memh_t;


/**
 * @brief cuda ipc region registered for exposure
 */
typedef struct {
    uct_cuda_ipc_md_handle_t  ph;     /* Memory handle of GPU memory */
    CUdeviceptr               d_bptr; /* Allocation base address */
    size_t                    b_len;  /* Allocation size */
    ucs_list_link_t           link;
#if HAVE_CUDA_FABRIC
    CUdeviceptr               vmm_multi_header_dev_ptr;     /* GPU metadata header buffer VA */
    CUdeviceptr               vmm_multi_chunks_dev_ptr;     /* GPU metadata chunks buffer VA */
    CUmemFabricHandle         vmm_multi_header_fabric_handle;/* Fabric handle to header buf */
    size_t                    vmm_multi_header_alloc_size;  /* Header buffer alloc size */
    size_t                    vmm_multi_chunks_alloc_size;  /* Chunks buffer alloc size */
    CUdeviceptr               vmm_multi_d_bptr;             /* Expanded base, all chunks */
    size_t                    vmm_multi_b_len;              /* Expanded length, all chunks */
    uint16_t                  vmm_multi_meta_num_chunks;    /* Chunk count */
#endif
} uct_cuda_ipc_lkey_t;


/**
 * @brief cuda ipc remote key for put/get
 */
typedef struct {
    uct_cuda_ipc_md_handle_t  ph;      /* Memory handle of GPU memory */
    pid_t                     pid;     /* PID as key to resolve peer_map hash */
    CUdeviceptr               d_bptr;  /* Allocation base address */
    size_t                    b_len;   /* Allocation size */
    CUuuid                    uuid;    /* GPU Device UUID */
} uct_cuda_ipc_rkey_t;


/**
 * @brief cuda ipc extended remote key
 */
typedef struct {
    uct_cuda_ipc_rkey_t super;
    ucs_sys_ns_t        pid_ns; /* PID namespace */
} uct_cuda_ipc_extended_rkey_t;


#if HAVE_CUDA_FABRIC
typedef struct {
    uct_cuda_ipc_key_handle_t handle_type;
    union {
        CUmemFabricHandle     fabric;
    } handle;
} uct_cuda_ipc_vmm_handle_t;

typedef struct {
    uct_cuda_ipc_vmm_handle_t vmm_handle;
    CUdeviceptr               d_bptr;
    size_t                    b_len;
    unsigned long long        buffer_id;
} uct_cuda_ipc_vmm_chunk_desc_t;

typedef struct {
    uct_cuda_ipc_vmm_handle_t chunks_handle;
    uint16_t                  num_chunks;
} uct_cuda_ipc_vmm_meta_header_t;

static UCS_F_ALWAYS_INLINE void
uct_cuda_ipc_init_access_desc(CUmemAccessDesc *access_desc, CUdevice cu_dev)
{
    access_desc->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc->flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc->location.id   = cu_dev;
}

void uct_cuda_ipc_vmm_multi_meta_cleanup(uct_cuda_ipc_lkey_t *key);
#endif


typedef struct {
    uct_cuda_ipc_extended_rkey_t   super;
    int                            stream_id;
#if HAVE_CUDA_FABRIC
    uct_cuda_ipc_vmm_chunk_desc_t *chunks;
    uint16_t                       num_chunks;
#endif
} uct_cuda_ipc_unpacked_rkey_t;


#if HAVE_CUDA_FABRIC
ucs_status_t
uct_cuda_ipc_mkey_pack_vmm_multi_chunk(uct_cuda_ipc_memh_t *memh,
                                       uct_cuda_ipc_lkey_t *key, void *address,
                                       size_t length);

ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_chunks(uct_cuda_ipc_unpacked_rkey_t *rkey,
                                    CUdevice cu_dev,
                                    ucs_log_level_t log_level);
#endif

#endif
