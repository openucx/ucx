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

typedef enum {
    UCT_CUDA_ALLOC_TYPE_FABRIC,
    UCT_CUDA_ALLOC_TYPE_PINNED,
    UCT_CUDA_ALLOC_TYPE_MANAGED,
    UCT_CUDA_ALLOC_TYPE_LAST = UCT_CUDA_ALLOC_TYPE_MANAGED
} uct_cuda_alloc_type_t;


/**
 * @brief cuda_copy MD descriptor
 */
typedef struct uct_cuda_copy_md {
    struct uct_md               super;           /* Domain info */
    struct {
        ucs_on_off_auto_value_t alloc_whole_reg; /* force return of allocation
                                                    range even for small bar
                                                    GPUs*/
        double                  max_reg_ratio;
        int                     dmabuf_supported;
        uct_cuda_pref_loc_t     pref_loc;
        uct_cuda_alloc_type_t   pref_alloc;
    } config;
} uct_cuda_copy_md_t;

/**
 * gdr copy domain configuration.
 */
typedef struct uct_cuda_copy_md_config {
    uct_md_config_t             super;
    ucs_on_off_auto_value_t     alloc_whole_reg;
    double                      max_reg_ratio;
    ucs_ternary_auto_value_t    enable_dmabuf;
    uct_cuda_pref_loc_t         pref_loc;
    uct_cuda_alloc_type_t       pref_alloc;
} uct_cuda_copy_md_config_t;

/**
 * copy alloc handle.
 */
typedef struct uct_cuda_copy_alloc_handle {
    uct_cuda_alloc_type_t        alloc_type;
    void                         *ptr;
    size_t                       length;
#if HAVE_CUDA_FABRIC
    CUmemGenericAllocationHandle generic_handle;
#endif
} uct_cuda_copy_alloc_handle_t;


ucs_status_t uct_cuda_copy_md_detect_memory_type(uct_md_h md,
                                                 const void *address,
                                                 size_t length,
                                                 ucs_memory_type_t *mem_type_p);

#endif
