/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_KNEM_MD_H_
#define UCT_KNEM_MD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>
#include <ucs/sys/rcache.h>
#include <uct/base/uct_md.h>

extern uct_md_component_t uct_knem_md_component;
ucs_status_t uct_knem_md_query(uct_md_h md, uct_md_attr_t *md_attr);

/**
 * @brief KNEM MD descriptor
 */
typedef struct uct_knem_md {
    struct uct_md       super;    /**< Domain info */
    int                 knem_fd;  /**< File descriptor for /dev/knem */
    ucs_rcache_t       *rcache;   /**< Registration cache (can be NULL) */
    uct_linear_growth_t reg_cost; /**< Memory registration cost */
} uct_knem_md_t;

/**
 * @brief KNEM packed and remote key
 */
typedef struct uct_knem_key {
    uint64_t cookie;   /**< knem cookie */
    uintptr_t address; /**< base addr for the registration */
} uct_knem_key_t;

/**
 * KNEM memory domain configuration.
 */
typedef struct uct_knem_md_config {
    uct_md_config_t        super;
    ucs_ternary_value_t    rcache_enable;
    uct_md_rcache_config_t rcache;
} uct_knem_md_config_t;

/**
 * KNEM memory region in the registration cache.
 */
typedef struct uct_knem_rcache_region {
    ucs_rcache_region_t super;
    uct_knem_key_t      key;      /**<  exposed to the user as the memh */
} uct_knem_rcache_region_t;

#endif
