/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_MD_H
#define UCT_CUDA_COPY_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>
#include <cuda.h>


extern uct_component_t uct_cuda_copy_component;

typedef enum {
    UCT_CUDA_PREF_LOC_CPU,
    UCT_CUDA_PREF_LOC_GPU,
    UCT_CUDA_PREF_LOC_LAST
} uct_cuda_pref_loc_t;


/**
 * @brief cuda_copy MD descriptor
 */
typedef struct uct_cuda_copy_md {
    struct uct_md                super;           /* Domain info */
    int                          sync_memops_set;
    size_t                       granularity;     /* allocation granularity */
    struct {
        ucs_on_off_auto_value_t  alloc_whole_reg; /* force return of allocation
                                                     range even for small bar
                                                     GPUs*/
        double                   max_reg_ratio;
        int                      dmabuf_supported;
        ucs_ternary_auto_value_t enable_fabric;
        uct_cuda_pref_loc_t      pref_loc;
        int                      cuda_async_managed;
    } config;
} uct_cuda_copy_md_t;

/**
 * cuda_copy MD configuration.
 */
typedef struct uct_cuda_copy_md_config {
    uct_md_config_t             super;
    ucs_on_off_auto_value_t     alloc_whole_reg;
    double                      max_reg_ratio;
    ucs_ternary_auto_value_t    enable_dmabuf;
    ucs_ternary_auto_value_t    enable_fabric;
    uct_cuda_pref_loc_t         pref_loc;
    ucs_memory_type_t           cuda_async_mem_type;
} uct_cuda_copy_md_config_t;

/**
 * copy alloc handle.
 */
typedef struct uct_cuda_copy_alloc_handle {
    CUdeviceptr                 ptr;
    size_t                      length;
    uint8_t                     is_vmm;
#if HAVE_CUDA_FABRIC
    CUmemGenericAllocationHandle generic_handle;
#endif
} uct_cuda_copy_alloc_handle_t;


ucs_status_t uct_cuda_copy_md_detect_memory_type(uct_md_h md,
                                                 const void *address,
                                                 size_t length,
                                                 ucs_memory_type_t *mem_type_p);

#endif
