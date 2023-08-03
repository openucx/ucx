/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_KNEM_MD_H_
#define UCT_KNEM_MD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/status.h>
#include <uct/base/uct_md.h>
#include <uct/api/v2/uct_v2.h>

extern uct_component_t uct_knem_component;
ucs_status_t uct_knem_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr);

/**
 * @brief KNEM MD descriptor
 */
typedef struct uct_knem_md {
    struct uct_md       super;    /**< Domain info */
    int                 knem_fd;  /**< File descriptor for /dev/knem */
    ucs_linear_func_t   reg_cost; /**< Memory registration cost */
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
    uct_md_config_t          super;
} uct_knem_md_config_t;

#endif
