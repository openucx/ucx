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

#ifndef ROCM_CMA_MD_H
#define ROCM_CMA_MD_H

#include <uct/base/uct_md.h>


/** Define name of memory domain for GPU memory. Must not be larget than
    UCT_MD_COMPONENT_NAME_MAX.
*/
#define UCT_ROCM_CMA_MD_NAME   "rocm"

extern uct_md_component_t uct_rocm_cma_md_component;

/**
 * @brief ROCm MD descriptor
 */
typedef struct uct_rocm_cma_md {
    struct  uct_md super;  /**< Domain info */

    /* rocm specific data should be here if any. */
    int     any_memory;     /**< Support any memory */
    int     acc_dev;        /**< Flag if we want to register device as
                                 acceleration device */
} uct_rocm_cma_md_t;

/**
 * ROCm  CMA memory domain configuration.
 */
typedef struct uct_rocm_cma_md_config {
    uct_md_config_t super;
    int             any_memory; /**< Support any memory */
    int             acc_dev;    /**< Flag if we want to register device as
                                     acceleration device */
} uct_rocm_cma_md_config_t;


/**
 * @brief ROCm packed and remote key for CMA
 */
typedef struct uct_rocm_cma_key {
    size_t               length;      /**< Size of memory */
    uintptr_t            address;     /**< Local address of memory */
    int                  is_locked;   /**< If memory was "locked" in GPU space */
} uct_rocm_cma_key_t;

#endif
