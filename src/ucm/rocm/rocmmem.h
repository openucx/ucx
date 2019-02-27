/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCM_ROCMMEM_H_
#define UCM_ROCMMEM_H_

#include <ucm/api/ucm.h>
#include <hsa_ext_amd.h>

/* hsa_amd_memory_pool_allocate */
hsa_status_t ucm_override_hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size,
    uint32_t flags, void** ptr);
hsa_status_t ucm_orig_hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size,
    uint32_t flags, void** ptr);
hsa_status_t ucm_hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size,
    uint32_t flags, void** ptr);

/* hsa_amd_memory_pool_free */
hsa_status_t ucm_override_hsa_amd_memory_pool_free(void* ptr);
hsa_status_t ucm_orig_hsa_amd_memory_pool_free(void* ptr);
hsa_status_t ucm_hsa_amd_memory_pool_free(void* ptr);

#endif
