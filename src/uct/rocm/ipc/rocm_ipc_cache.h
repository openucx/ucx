/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_IPC_CACHE_H_
#define UCT_ROCM_IPC_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include "rocm_ipc_md.h"


typedef struct uct_rocm_ipc_cache {
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name */
} uct_rocm_ipc_cache_t;

typedef struct {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_rocm_ipc_key_t      key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uct_rocm_ipc_cache_t    *cache;       /**< Cache that owns the region */
    hsa_signal_t            signal;       /**< Attached IPC signal */
    hsa_signal_t            retired_signal; /**< Signal owned by claim winner */
    uint32_t                refcount;     /**< Pgtable, handler, watcher, pin */
    uint32_t                deactivate;   /**< 1 after unmap is claimed */
    uint32_t                handler_done; /**< Async handler entered once */
    int                     handler_armed; /**< Async handler owns signal */
    uint32_t                pin_count;    /**< Consumers holding mapped_addr,
                                               incremented under the lock */
    int                     in_pgtable;   /**< Region is linked in @a cache */
} uct_rocm_ipc_cache_region_t;

typedef struct {
    uct_rocm_ipc_key_t          super;
    uct_rocm_ipc_cache_region_t *region;      /**< Pinned until rkey release */
    void                        *mapped_addr; /**< Mapping of @a region */
} uct_rocm_ipc_unpacked_rkey_t;

ucs_status_t uct_rocm_ipc_create_cache(uct_rocm_ipc_cache_t **cache,
                                       const char *name);

void uct_rocm_ipc_destroy_cache(uct_rocm_ipc_cache_t *cache);

/**
 * Map @a key and pin it until @ref uct_rocm_ipc_cache_unpin. The pin keeps
 * the returned address valid even if the cache invalidates the region. A pin
 * does not keep the owner allocation alive.
 */
ucs_status_t
uct_rocm_ipc_cache_map_memhandle(uct_rocm_ipc_key_t *key, void **mapped_addr,
                                 uct_rocm_ipc_cache_region_t **region_p);

/**
 * Drop one pin. Detach the import when this is the last pin and
 * @a free_unused_mapping is set. A deactivated region is detached on the last
 * unpin either way.
 */
void uct_rocm_ipc_cache_unpin(uct_rocm_ipc_cache_region_t *region,
                              int free_unused_mapping);

void uct_rocm_ipc_export_init(void);

/**
 * Create or reuse an IPC signal for an exported buffer and remember it until
 * free. Returns an empty signal when invalidation is disabled.
 * The packed @a ipc_signal is the same for every peer that imports this key.
 */
ucs_status_t uct_rocm_ipc_export_add(uintptr_t address, size_t length,
                                     hsa_amd_ipc_signal_t *ipc_signal_p);

#endif
