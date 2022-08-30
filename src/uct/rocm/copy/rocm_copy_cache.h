/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2022. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_COPY_CACHE_H_
#define UCT_ROCM_COPY_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include "rocm_copy_md.h"

typedef struct uct_rocm_copy_cache_region {
    /* Base class - page table region */
    ucs_pgt_region_t super;

    /* List element */
    ucs_list_link_t  list;

    /* Local mapped address */
    void            *mapped_addr;

    /* Ptr used for the memory_lock operation */
    uint64_t         base_ptr;

    /* Length used for memory_lock operation */
    size_t           base_length;
} uct_rocm_copy_cache_region_t;

typedef struct uct_rocm_copy_cache {
    /* Page table to hold the regions */
    ucs_pgtable_t    pgtable;

    /* protects the page table */
    pthread_rwlock_t lock;

    /* Name */
    char            *name;
} uct_rocm_copy_cache_t;

ucs_status_t
uct_rocm_copy_create_cache(uct_rocm_copy_cache_t **cache, const char *name);

void uct_rocm_copy_destroy_cache(uct_rocm_copy_cache_t *cache);

ucs_status_t uct_rocm_copy_cache_map_memhandle(void *arg, const uint64_t addr,
					       size_t length,
					       void **mapped_addr);

#endif
