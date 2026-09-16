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

#if HAVE_CUDA_FABRIC
/**
 * @brief One chunk of a multi-chunk view, retained for staleness checking
 *
 * Remote addresses are recycled, so @ref start alone does not identify an
 * allocation; @ref buffer_id is what distinguishes "the same chunk" from "a
 * different allocation that happens to sit at the same address".
 */
typedef struct {
    uint64_t start;     /**< Chunk base address in the remote address space */
    uint64_t buffer_id; /**< Identity of the backing remote allocation */
} uct_cuda_ipc_chunk_t;
#endif


struct uct_cuda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< Page-table collection element */
    ucs_list_link_t         lru_list;     /**< LRU list element */
    uct_cuda_ipc_rkey_t     key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uint64_t                refcount;     /**< Active mapping references */
    CUdevice                cu_dev;       /**< CUDA device */
#if HAVE_CUDA_FABRIC
    ucs_interval_map_node_t node;         /**< Handle into cache->views */
    uint8_t                 indexed;      /**< Whether 'node' is in the map */
    uint16_t                num_chunks;   /**< Length of 'chunks' */
    uct_cuda_ipc_chunk_t    *chunks;      /**< Chunks, allocated with region */
#endif
};


struct uct_cuda_ipc_cache {
    pthread_rwlock_t      lock;         /**< Protects the cache */
    ucs_pgtable_t         pgtable;      /**< Single-view lookup index */
#if HAVE_CUDA_FABRIC
    ucs_interval_map_t    views;        /**< Multi-view lookup index */
#endif
    char                  *name;        /**< Name */
    ucs_list_link_t       lru_list;     /**< LRU of all cached regions */
    unsigned long         num_regions;  /**< Number of cached regions */
    size_t                total_size;   /**< Size of cached regions */
};


ucs_status_t uct_cuda_ipc_create_cache(uct_cuda_ipc_cache_t **cache,
                                       const char *name);


/**
 * @brief Drop unreferenced regions until the cache is within its limits
 */
void uct_cuda_ipc_cache_evict_lru(uct_cuda_ipc_cache_t *cache);


/**
 * @brief Unmap and release the local mapping of a cached region
 */
ucs_status_t uct_cuda_ipc_close_memhandle(uct_cuda_ipc_cache_region_t *region);


#if HAVE_CUDA_FABRIC
/**
 * @brief Map a multi-chunk VMM key into one contiguous local reservation
 *
 * Reserves a single VA range spanning the whole key and maps each chunk into
 * its slot, so the caller gets one base pointer for the entire range.
 */
ucs_status_t uct_cuda_ipc_open_memhandle_vmm_multi(
        const uct_cuda_ipc_unpacked_rkey_t *unpacked, CUdevice cu_dev,
        CUdeviceptr *mapped_addr, ucs_log_level_t log_level);


/*
 * Cache of multi-chunk views, implemented in cuda_ipc_multi_cache.c.
 *
 * Views overlap in the remote address space, so they are indexed by an
 * interval map rather than the page table. That is what makes a request for a
 * sub-range of an already-imported range a hit: the caller only ever passes a
 * whole rkey and cannot know it is nested inside one already cached.
 *
 * Get, put, and retire require cache->lock held for writing. Purge requires
 * either the write lock or exclusive ownership of the cache during teardown.
 */

/* Resolve a key to a local address, importing a new view if needed. The
 * returned view must be passed back to _put, since a view covering more than
 * the key is not findable by the key's address alone. */
ucs_status_t uct_cuda_ipc_multi_cache_get(
        uct_cuda_ipc_cache_t *cache,
        const uct_cuda_ipc_unpacked_rkey_t *unpacked, CUdevice cu_dev,
        void **mapped_addr_p, uct_cuda_ipc_cache_region_t **region_p,
        ucs_log_level_t log_level);

/* Drop one reference, tearing the view down once nothing references it and it
 * is either retired or caching is disabled. */
void uct_cuda_ipc_multi_cache_put(uct_cuda_ipc_cache_t *cache,
                                  uct_cuda_ipc_cache_region_t *region,
                                  int cache_enabled);

/* Make a view unfindable. Unmapping waits for outstanding references, which
 * point into its reservation. */
void uct_cuda_ipc_multi_cache_retire(uct_cuda_ipc_cache_t *cache,
                                     uct_cuda_ipc_cache_region_t *region);

/* Retire every view still in the index */
void uct_cuda_ipc_multi_cache_purge(uct_cuda_ipc_cache_t *cache,
                                    int close_handles);
#endif


void uct_cuda_ipc_destroy_cache(uct_cuda_ipc_cache_t *cache, int close_handles);


/**
 * @brief Map an interprocess memory handle to a local address
 *
 * This function maps an interprocess memory handle exported from another
 * process to a local virtual address that can be accessed by the current
 * process.
 *
 * @param key           Pointer to the CUDA IPC remote memory key containing
 *                      the memory handle and other metadata needed for mapping
 * @param cu_dev        CUDA device handle where the memory should be mapped
 * @param mapped_addr_p Pointer to store the resulting mapped local address
 * @param region_p      Pointer to store the referenced multi-view cache region,
 *                      or NULL for mappings managed by the original page table
 * @param log_level     Log level for reporting failures during mapping
 *                      operation
 *
 * @return UCS_OK on success, or error status on failure
 */
ucs_status_t uct_cuda_ipc_map_memhandle(uct_cuda_ipc_unpacked_rkey_t *key,
                                        CUdevice cu_dev, void **mapped_addr_p,
                                        uct_cuda_ipc_cache_region_t **region_p,
                                        ucs_log_level_t log_level);


/**
 * @brief Release a mapped interprocess memory handle reference
 *
 * @param pid            Remote process ID
 * @param pid_ns         Remote process namespace
 * @param d_bptr         Remote allocation base address
 * @param mapped_addr    Local mapped base address
 * @param cu_dev         CUDA device where the handle is mapped
 * @param region         Referenced multi-view region, or NULL for a
 *                       single-view mapping
 * @param cache_enabled  Whether to retain an unreferenced mapping in cache
 */
void uct_cuda_ipc_unmap_memhandle(pid_t pid, ucs_sys_ns_t pid_ns,
                                  uintptr_t d_bptr, const void *mapped_addr,
                                  CUdevice cu_dev,
                                  uct_cuda_ipc_cache_region_t *region,
                                  int cache_enabled);


/**
 * @brief Set global cache limits for newly created CUDA IPC remote caches
 *
 * @param max_regions  Max regions per cache (ULONG_MAX = unlimited)
 * @param max_size     Max total size per cache (SIZE_MAX = unlimited)
 */
void uct_cuda_ipc_cache_set_global_limits(unsigned long max_regions,
                                          size_t max_size);

#endif
