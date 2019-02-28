/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_COMMON_H
#define ROCM_COMMON_H

#include <hsa.h>
#include <hsa_ext_amd.h>

/**
 * @brief  Initialize ROCm support for UCX
 *
 *
 * @return HSA_STATUS_OK if ROCm support was initialized successfully
 *         HSA error code otherwise
*/
hsa_status_t uct_rocm_init();


/**
 * @brief Convert pointer to pointer which could be used for GPU access
 *
 * @param [in] ptr Pointer to memory
 * @param [out] gpu_ptr If not NULL return host address to be used for
 *                      GPU access.
 * @param [out] indicates if memory is locked for GPU access. If locked
 *              then hsa_amd_unlock() should be called
 * @return  HSA status
 *
*/
hsa_status_t uct_rocm_cma_ptr_to_gpu_ptr(void *ptr, void **gpu_address,
                                         size_t size, int any_memory,
                                         int *locked);


/**
 * @brief Import / lock system memory in ROCm address space for GPU access
 *
 * @param [in]  ptr Memory pointer
 * @param [in]  size Size to lock
 * @param [out] ptr Address to use for GPU access
 *
 * @return  HSA status
 *
*/
hsa_status_t uct_rocm_memory_lock(void *ptr, size_t size, void **gpu_ptr);


#endif


