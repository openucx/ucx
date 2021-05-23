/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_MD_H
#define UCT_CUDA_COPY_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>
#include <ucs/memory/memory_type.h>


extern uct_component_t uct_cuda_copy_component;

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

/**
 * @brief cuda_copy packed and remote key for put/get
 */
typedef struct uct_cuda_copy_key {
    void              *address;        /* Address needed for unregistering */
    uint8_t           host_registered; /* Is cuHostRegister called on address */
    ucs_memory_type_t mem_type;        /* Memory type */
} uct_cuda_copy_key_t;
#endif
