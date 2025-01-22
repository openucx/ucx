/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_IPC_CACHE_H_
#define UCT_ROCM_IPC_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include "rocm_ipc_md.h"


typedef struct uct_cuda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    uct_rocm_ipc_key_t      key;          /**< Remote memory key */
    void                    *mapped_addr; /**< Local mapped address */
} uct_rocm_ipc_cache_region_t;

typedef struct uct_rocm_ipc_cache {
    pthread_rwlock_t      lock;       /**< protests the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name */
} uct_rocm_ipc_cache_t;

ucs_status_t uct_rocm_ipc_create_cache(uct_rocm_ipc_cache_t **cache,
                                       const char *name);

void uct_rocm_ipc_destroy_cache(uct_rocm_ipc_cache_t *cache);

ucs_status_t uct_rocm_ipc_cache_map_memhandle(void *arg, uct_rocm_ipc_key_t *key,
                                              void **mapped_addr);
#endif
