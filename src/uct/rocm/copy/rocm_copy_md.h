/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_COPY_MD_H
#define UCT_ROCM_COPY_MD_H

#include <uct/base/uct_md.h>
#include <ucs/config/types.h>
#include <ucs/memory/rcache.h>


extern uct_component_t uct_rocm_copy_component;

/*
 * @brief rocm_copy MD descriptor
 */
typedef struct uct_rocm_copy_md {
    uct_md_t            super;      /**< Domain info */
    ucs_rcache_t        *rcache;    /**< Registration cache (can be NULL) */
    ucs_linear_func_t   reg_cost;   /**< Memory registration cost */
} uct_rocm_copy_md_t;


/**
 * rocm copy domain configuration.
 */
typedef struct uct_rocm_copy_md_config {
    uct_md_config_t             super;
    ucs_ternary_auto_value_t    enable_rcache;/**< Enable registration cache */
    uct_md_rcache_config_t      rcache;       /**< Registration cache config */
    ucs_linear_func_t           uc_reg_cost;  /**< Memory registration cost estimation
                                                   without using the cache */
} uct_rocm_copy_md_config_t;


/**
 * @brief rocm copy mem handle
 */
typedef struct uct_rocm_copy_mem {
    void        *vaddr;
    void        *dev_ptr;
    size_t      reg_size;
} uct_rocm_copy_mem_t;


/**
 * @brief rocm copy packed and remote key for get/put
 */
typedef struct uct_rocm_copy_key {
    uint64_t    vaddr;      /**< CPU address being mapped */
    void        *dev_ptr;   /**< GPU accessible address */
} uct_rocm_copy_key_t;


/**
 * rocm memory region in the registration cache.
 */
typedef struct uct_rocm_copy_rcache_region {
    ucs_rcache_region_t  super;
    uct_rocm_copy_mem_t  memh;      /**<  mr exposed to the user as the memh */
} uct_rocm_copy_rcache_region_t;

#endif
