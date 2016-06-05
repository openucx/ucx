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
#include <uct/base/uct_md.h>

extern uct_md_component_t uct_knem_md_component;
ucs_status_t uct_knem_md_query(uct_md_h md, uct_md_attr_t *md_attr);

/**
 * @brief KNEM MD descriptor
 */
typedef struct uct_knem_md {
    struct uct_md super; /**< Domain info */
    int knem_fd;         /**< File descriptor for /dev/knem */
} uct_knem_md_t;

/**
 * @brief KNEM packed and remote key
 */
typedef struct uct_knem_key {
    uint64_t cookie;   /**< knem cookie */
    uintptr_t address; /**< base addr for the registration */
} uct_knem_key_t;

#endif
