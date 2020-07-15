/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDR_COPY_MD_H
#define UCT_GDR_COPY_MD_H

#include <uct/base/uct_md.h>
#include <uct/cuda/base/cuda_md.h>
#include <ucs/memory/rcache.h>
#include "gdrapi.h"


extern uct_component_t uct_gdr_copy_component;


/**
 * @brief gdr_copy MD descriptor
 */
typedef struct uct_gdr_copy_md {
    uct_md_t            super;      /**< Domain info */
    gdr_t               gdrcpy_ctx; /**< gdr copy context */
    ucs_rcache_t        *rcache;    /**< Registration cache (can be NULL) */
    ucs_linear_func_t   reg_cost;   /**< Memory registration cost */
} uct_gdr_copy_md_t;


/**
 * gdr copy domain configuration.
 */
typedef struct uct_gdr_copy_md_config {
    uct_md_config_t         super;
    int                     enable_rcache;/**< Enable registration cache */
    uct_md_rcache_config_t  rcache;       /**< Registration cache config */
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


/**
 * cuda memory region in the registration cache.
 */
typedef struct uct_gdr_copy_rcache_region {
    ucs_rcache_region_t  super;
    uct_gdr_copy_mem_t   memh;      /**<  mr exposed to the user as the memh */
} uct_gdr_copy_rcache_region_t;

#endif
