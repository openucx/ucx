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

#ifdef HAVE_ROCM_VMM_TYPE
/* hsa_amd_vmem_map */
hsa_status_t ucm_override_hsa_amd_vmem_map(void* va, size_t size,
                                           size_t in_offset,
                                           hsa_amd_vmem_alloc_handle_t handle,
                                           uint64_t flags);
hsa_status_t ucm_orig_hsa_amd_vmem_map(void* va, size_t size,
                                       size_t in_offset,
                                       hsa_amd_vmem_alloc_handle_t handle,
                                       uint64_t flags);
hsa_status_t ucm_hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                                   hsa_amd_vmem_alloc_handle_t handle,
                                   uint64_t flags);

/* hsa_amd_vmem_unmap */
hsa_status_t ucm_override_hsa_amd_vmem_unmap(void* va, size_t size);
hsa_status_t ucm_orig_hsa_amd_vmem_unmap(void* va, size_t size);
hsa_status_t ucm_hsa_amd_vmem_unmap(void* va, size_t size);
#endif

#endif
