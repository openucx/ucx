/*
 * Copyright (C) Advanced Micro Devices, Inc. 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_CMA_MD_H
#define ROCM_CMA_MD_H

#include <uct/base/uct_md.h>


/** Define name of memory domain for GPU memory. Must not be larget than
    UCT_MD_COMPONENT_NAME_MAX.
*/
#define UCT_ROCM_CMA_MD_NAME    "rocm"

extern uct_md_component_t uct_rocm_cma_md_component;

/**
 * @brief ROCm MD descriptor
 */
typedef struct uct_rocm_cma_md {
    struct      uct_md super;   /**< Domain info */

    /* rocm specific data should be here if any. */
    int         any_memory;     /**< Support any memory */
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
    size_t              length;     /**< Size of memory */
    uintptr_t           gpu_address;/**< GPU address of memory */
    uintptr_t           md_address; /**< MD address of memory */
    int                 is_locked;  /**< If memory was "locked" in GPU space */
} uct_rocm_cma_key_t;

#endif
