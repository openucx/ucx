/*
 * Copyright 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
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
bool uct_rocm_is_ptr_gpu_accessible(void *ptr, void **gpu_ptr);


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


