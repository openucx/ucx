/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_MD_H
#define UCT_CUDA_COPY_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>


extern uct_component_t uct_cuda_copy_component;

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
    } config;
} uct_cuda_copy_md_t;

/**
 * gdr copy domain configuration.
 */
typedef struct uct_cuda_copy_md_config {
    uct_md_config_t             super;
    ucs_on_off_auto_value_t     alloc_whole_reg;
    double                      max_reg_ratio;
} uct_cuda_copy_md_config_t;

#endif
