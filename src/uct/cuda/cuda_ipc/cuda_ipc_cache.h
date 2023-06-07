/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
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


typedef struct uct_cuda_ipc_rem_memh     uct_cuda_ipc_rem_memh_t;


/**
 * Structure associated with rcache region. ipc_handle is returned along with
 * the start of the mapping to check if VA recycling has occurred.
 */
typedef struct uct_cuda_ipc_rcache_region {
    ucs_rcache_region_t     super;
    CUipcMemHandle          ipc_handle;
    void                    *mapping_start;
} uct_cuda_ipc_rcache_region_t;


ucs_status_t
uct_cuda_ipc_create_cache(uct_cuda_ipc_md_t *md, ucs_rcache_t **cache,
                          const char *name);


void uct_cuda_ipc_destroy_cache(ucs_rcache_t *cache);


ucs_status_t
uct_cuda_ipc_map_memhandle(uct_cuda_ipc_md_t *md,
                           uct_cuda_ipc_key_t *key,
                           void **mapped_addr,
                           uct_cuda_ipc_rcache_region_t **cuda_ipc_region);
ucs_status_t
uct_cuda_ipc_unmap_memhandle(uct_cuda_ipc_md_t *md, pid_t pid,
                             void *mapped_addr,
                             uct_cuda_ipc_rcache_region_t *cuda_ipc_region);

#endif
