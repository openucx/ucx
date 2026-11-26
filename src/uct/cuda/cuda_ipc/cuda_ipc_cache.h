/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_CACHE_H_
#define UCT_CUDA_IPC_CACHE_H_

#include "cuda_ipc_md.h"

#include <ucs/datastruct/interval_map.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/pgtable.h>
#include <ucs/type/init_once.h>
#include <ucs/type/spinlock.h>

#include <cuda.h>


typedef struct uct_cuda_ipc_cache        uct_cuda_ipc_cache_t;
typedef struct uct_cuda_ipc_cache_region uct_cuda_ipc_cache_region_t;
typedef struct uct_cuda_ipc_rem_memh     uct_cuda_ipc_rem_memh_t;


struct uct_cuda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    ucs_list_link_t         lru_list;     /**< LRU list element */
    uct_cuda_ipc_rkey_t     key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uint64_t                refcount;     /**< Track in-flight ops before unmapping*/
    CUdevice                cu_dev;       /**< CUDA device */
};


#if HAVE_CUDA_FABRIC
typedef struct {
    uct_cuda_ipc_cache_region_t super;
    ucs_interval_map_node_t     node;   /**< Entry in the multi-chunk views */
    uint8_t                     in_map; /**< Whether the view is in the map */
} uct_cuda_ipc_cache_view_t;
#endif


struct uct_cuda_ipc_cache {
    pthread_rwlock_t      lock;         /**< Protects the page table, lru, views */
    ucs_pgtable_t         pgtable;      /**< Page table to hold the regions */
#if HAVE_CUDA_FABRIC
    ucs_interval_map_t    views;        /**< Multi-chunk views */
#endif
    char                  *name;        /**< Name */
    ucs_list_link_t       lru_list;     /**< LRU list of all cached regions */
    unsigned long         num_regions;  /**< Current number of cached regions */
    size_t                total_size;   /**< Current total size of cached regions */
};


typedef struct {
    pid_t                       pid;           /**< Remote process ID */
    ucs_sys_ns_t                pid_ns;        /**< Remote PID namespace */
    uintptr_t                   d_bptr;        /**< Remote allocation base */
    const void                  *mapped_addr;  /**< Local mapped address */
    uct_cuda_ipc_cache_region_t *cache_region; /**< Multi-chunk view, or NULL */
} uct_cuda_ipc_mapping_t;


ucs_status_t uct_cuda_ipc_create_cache(uct_cuda_ipc_cache_t **cache,
                                       const char *name);


void uct_cuda_ipc_destroy_cache(uct_cuda_ipc_cache_t *cache, int close_handles);


#if HAVE_CUDA_FABRIC || HAVE_DECL_SYS_PIDFD_GETFD
/**
 * @brief Import a VMM allocation from a shareable handle and map it locally
 *
 * @param length            Length of the allocation to map
 * @param cu_dev            CUDA device which is granted access to the mapping
 * @param mapped_addr       Filled with the local mapped address on success
 * @param shareable_handle  Shareable handle exported by the peer
 * @param handle_type       Type of @a shareable_handle
 * @param log_level         Log level for failures
 *
 * @return UCS_OK on success, or an error status on failure
 */
ucs_status_t
uct_cuda_ipc_open_memhandle_vmm(size_t length, CUdevice cu_dev,
                                CUdeviceptr *mapped_addr,
                                void *shareable_handle,
                                CUmemAllocationHandleType handle_type,
                                ucs_log_level_t log_level);
#endif


#if HAVE_CUDA_FABRIC
/**
 * @brief Map a multi-chunk VMM key into one contiguous local reservation
 *
 * Fetches the chunk descriptors the peer published, reserves a single VA range
 * spanning the whole key and maps each chunk into its slot, so the caller gets
 * one base pointer for the entire range.
 *
 * @param key          Multi-chunk remote key
 * @param cu_dev       CUDA device which is granted access to the mapping
 * @param mapped_addr  Filled with the local address of the key base on success
 * @param log_level    Log level for failures
 *
 * @return UCS_OK on success, or an error status on failure
 */
ucs_status_t
uct_cuda_ipc_open_memhandle_vmm_multi(const uct_cuda_ipc_rkey_t *key,
                                      CUdevice cu_dev, CUdeviceptr *mapped_addr,
                                      ucs_log_level_t log_level);
#endif


/**
 * @brief Map an interprocess memory handle to a local address
 * 
 * This function maps an interprocess memory handle exported from another
 * process to a local virtual address that can be accessed by the current
 * process.
 * 
 * @param key          Pointer to the CUDA IPC remote memory key containing
 *                     the memory handle and other metadata needed for mapping
 * @param cu_dev       CUDA device handle where the memory should be mapped
 * @param mapped_addr  Pointer to store the local address of the key base
 * @param region_p     Filled with the referenced multi-chunk view, which must
 *                     be passed to @ref uct_cuda_ipc_unmap_memhandle, or NULL
 *                     for any other mapping. The view may be wider than the
 *                     key.
 * @param log_level    Log level for reporting failures during mapping operation
 * 
 * @return UCS_OK on success, or error status on failure
 */
ucs_status_t uct_cuda_ipc_map_memhandle(uct_cuda_ipc_extended_rkey_t *key,
                                        CUdevice cu_dev, void **mapped_addr,
                                        uct_cuda_ipc_cache_region_t **region_p,
                                        ucs_log_level_t log_level);


void uct_cuda_ipc_unmap_memhandle(const uct_cuda_ipc_mapping_t *mapping,
                                  CUdevice cu_dev, int cache_enabled);


#if HAVE_CUDA_FABRIC
/**
 * @brief Free the cached view of a multi-chunk key if nothing references it
 *
 * Used on release of a key which holds no mapping itself, after operations on
 * the key mapped and released its view.
 *
 * @param key     Multi-chunk remote key
 * @param cu_dev  CUDA device where the key was mapped
 */
void uct_cuda_ipc_unmap_unused_view(const uct_cuda_ipc_extended_rkey_t *key,
                                    CUdevice cu_dev);
#endif


/**
 * @brief Set global cache limits for newly created CUDA IPC remote caches
 *
 * @param max_regions  Max regions per cache (ULONG_MAX = unlimited)
 * @param max_size     Max total size per cache (SIZE_MAX = unlimited)
 */
void uct_cuda_ipc_cache_set_global_limits(unsigned long max_regions,
                                          size_t max_size);

#endif
