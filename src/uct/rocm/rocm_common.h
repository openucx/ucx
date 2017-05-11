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
 * @brief Check if memory is GPU accessible
 *
 * @param [in] ptr Pointer to memory
 * @param [out] gpu_ptr If not NULL return host address to be used for
 *                      GPU access.
 *
 * @return  true if GPU accessible false otherwise
 *
*/
int uct_rocm_is_ptr_gpu_accessible(void *ptr, void **gpu_ptr);


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


