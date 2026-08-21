/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_MEM_H
#define UCT_CUDA_MEM_H

#include <ucs/type/status.h>
#include <ucs/config/types.h>
#include <ucs/memory/memory_type.h>

#include <cuda.h>

typedef struct {
    CUdeviceptr ptr;
    size_t      length;
    uint8_t     is_vmm;
#if HAVE_CUDA_FABRIC
    CUmemGenericAllocationHandle generic_handle;
#endif
} uct_cuda_mem_t;


/**
 * Allocate CUDA device memory, optionally using fabric VMM allocation.
 *
 * @param [in]     log_level      Log level for CUDA driver API failures
 * @param [in]     mem_type       Memory type to allocate
 * @param [in]     enable_fabric  Controls fabric VMM allocation
 * @param [in]     length         The minimal size to allocate
 * @param [in,out] granularity_p  Allocation granularity (if fabric VMM is used)
 * @param [out]    mem_p          Filled with information about the allocated
 *                                memory
 */
ucs_status_t
uct_cuda_mem_alloc(ucs_log_level_t log_level, ucs_memory_type_t mem_type,
                   ucs_ternary_auto_value_t enable_fabric, size_t length,
                   size_t *granularity_p, uct_cuda_mem_t *mem_p);


/**
 * Release the memory allocated by @ref uct_cuda_mem_alloc.
 *
 * @param [in] mem  Description of allocated memory, as returned from
 *                  @ref uct_cuda_mem_alloc
 */
void uct_cuda_mem_free(uct_cuda_mem_t mem);


/**
 * Set the context flag to synchronize DMA operations.
 *
 * @param [in] log_level  Log level for CUDA driver API failures
 */
ucs_status_t uct_cuda_mem_set_ctx_sync_memops(ucs_log_level_t log_level);

#endif
