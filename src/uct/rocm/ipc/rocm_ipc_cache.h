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


typedef struct {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_rocm_ipc_key_t      key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uct_rocm_ipc_cache_t    *cache;       /**< Cache that owns the region */
    hsa_signal_t            signal;       /**< Attached IPC signal */
    hsa_signal_t            retired_signal; /**< Signal owned by claim winner */
    uint32_t                refcount;     /**< Pgtable, handler, and watcher */
    uint32_t                deactivate;   /**< 1 after unmap is claimed */
    uint32_t                handler_done; /**< Async handler entered once */
    uint32_t                signal_owner; /**< 0 open, 1 claim, 2 handler */
    int                     in_pgtable;   /**< Region is linked in @a cache */
} uct_rocm_ipc_cache_region_t;

typedef struct uct_rocm_ipc_cache {
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name */
} uct_rocm_ipc_cache_t;

ucs_status_t uct_rocm_ipc_create_cache(uct_rocm_ipc_cache_t **cache,
                                       const char *name);

void uct_rocm_ipc_destroy_cache(uct_rocm_ipc_cache_t *cache);

ucs_status_t
uct_rocm_ipc_cache_map_memhandle(uct_rocm_ipc_key_t *key, void **mapped_addr);

/**
 * Create or reuse an IPC signal for an exported buffer and remember it until
 * free.
 * The packed @a ipc_signal is the same for every peer that imports this key.
 */
ucs_status_t uct_rocm_ipc_export_add(uintptr_t address, size_t length,
                                     hsa_amd_ipc_signal_t *ipc_signal_p);

/**
 * Destroy retired export signals. Importers attach the signal before mapping
 * memory, so any thread may safely reap them.
 */
void uct_rocm_ipc_export_progress(void);

#endif
