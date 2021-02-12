/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_CACHE_H_
#define UCT_CUDA_IPC_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/init_once.h>
#include <ucs/type/spinlock.h>
#include "cuda_ipc_md.h"
#include <cuda.h>
#include <cuda_runtime.h>


typedef struct uct_cuda_ipc_cache        uct_cuda_ipc_cache_t;
typedef struct uct_cuda_ipc_cache_region uct_cuda_ipc_cache_region_t;
typedef struct uct_cuda_ipc_rem_memh     uct_cuda_ipc_rem_memh_t;


struct uct_cuda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_cuda_ipc_key_t      key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
    uint64_t                refcount;     /**< Track inflight ops before unmapping*/
};


struct uct_cuda_ipc_cache {
    pthread_rwlock_t      lock;       /**< protests the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name */
};


ucs_status_t uct_cuda_ipc_create_cache(uct_cuda_ipc_cache_t **cache,
                                       const char *name);


void uct_cuda_ipc_destroy_cache(uct_cuda_ipc_cache_t *cache);


ucs_status_t
uct_cuda_ipc_map_memhandle(const uct_cuda_ipc_key_t *key, void **mapped_addr);
ucs_status_t uct_cuda_ipc_unmap_memhandle(pid_t pid, uintptr_t d_bptr,
                                          void *mapped_addr, int cache_enabled);
#endif
