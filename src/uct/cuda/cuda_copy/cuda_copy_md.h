/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_MD_H
#define UCT_CUDA_COPY_MD_H

#include <uct/base/uct_md.h>

#define UCT_CUDA_COPY_MD_NAME           "cuda_cpy"

extern uct_md_component_t uct_cuda_copy_md_component;

/**
 * @brief cuda_copy MD descriptor
 */
typedef struct uct_cuda_copy_md {
    struct uct_md super;   /**< Domain info */
} uct_cuda_copy_md_t;

/**
 * gdr copy domain configuration.
 */
typedef struct uct_cuda_copy_md_config {
    uct_md_config_t super;
} uct_cuda_copy_md_config_t;

#endif
