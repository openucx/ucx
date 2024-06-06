/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDR_COPY_MD_H
#define UCT_GDR_COPY_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>
#include "gdrapi.h"


extern uct_component_t uct_gdr_copy_component;


/**
 * @brief gdr_copy MD descriptor
 */
typedef struct {
    uct_md_t            super;      /**< Domain info */
    gdr_t               gdrcpy_ctx; /**< gdr copy context */
    ucs_linear_func_t   reg_cost;   /**< Memory registration cost */
    ucs_rcache_t        *rcache;    /**< Registration cache */
} uct_gdr_copy_md_t;


/**
 * gdr copy domain configuration.
 */
typedef struct uct_gdr_copy_md_config {
    uct_md_config_t         super;
    int                     shared;       /**< Shared MD instance */
    int                     enable_rcache;/**< Enable registration cache */
    ucs_linear_func_t       uc_reg_cost;  /**< Memory registration cost estimation
                                             without using the cache */
} uct_gdr_copy_md_config_t;


/**
 * @brief gdr copy mem handle
 */
typedef struct uct_gdr_copy_mem {
    gdr_mh_t    mh;         /**< Memory handle of GPU memory */
    gdr_info_t  info;       /**< Info of GPU memory mapping */
    void        *bar_ptr;   /**< BAR address of GPU mapping */
    size_t      reg_size;   /**< Size of mapping */
} uct_gdr_copy_mem_t;


/**
 * @brief gdr copy  packed and remote key for put
 */
typedef struct uct_gdr_copy_key {
    uint64_t    vaddr;      /**< Mapped GPU address */
    void        *bar_ptr;   /**< BAR address of GPU mapping */
    gdr_mh_t    mh;         /**< Memory handle of GPU memory */
} uct_gdr_copy_key_t;

#endif
