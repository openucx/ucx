/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SELF_IFACE_H
#define UCT_SELF_IFACE_H

#include <uct/base/uct_iface.h>

typedef uint64_t uct_self_iface_addr_t;

typedef struct uct_self_iface {
    uct_base_iface_t      super;
    uct_self_iface_addr_t id;    /* Unique identifier for the instance */
} uct_self_iface_t;

typedef struct uct_self_iface_config {
    uct_iface_config_t super;
} uct_self_iface_config_t;


#endif
