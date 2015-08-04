/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_KNEM_PD_H_
#define UCT_KNEM_PD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>
#include <uct/tl/context.h>

extern uct_pd_component_t uct_knem_pd_component;
ucs_status_t uct_knem_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);

/**
 * @brief KNEM PD descriptor
 */
typedef struct uct_knem_pd {
    struct uct_pd super; /**< Domain info */
    int knem_fd;         /**< File descriptor for /dev/knem */
} uct_knem_pd_t;

/**
 * @brief KNEM packed and remote key
 */
typedef struct uct_knem_key {
    uint64_t cookie;   /**< knem cookie */
    uintptr_t address; /**< base addr for the registration */
} uct_knem_key_t;

#endif
