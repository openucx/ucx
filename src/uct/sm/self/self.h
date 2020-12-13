/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SELF_H
#define UCT_SELF_H

#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>


typedef uint64_t uct_self_iface_addr_t;


typedef struct uct_self_iface_config {
    uct_iface_config_t    super;
    size_t                seg_size;      /* Maximal send size */
} uct_self_iface_config_t;


/**
 * @brief self device MD descriptor
 */
typedef struct uct_self_md {
    uct_md_t super;
    size_t   num_devices; /* Number of devices to create */
} uct_self_md_t;


/**
 * @brief self device MD configuration
 */
typedef struct uct_self_md_config {
    uct_md_config_t super;
    size_t          num_devices; /* Number of devices to create */
} uct_self_md_config_t;


typedef struct uct_self_iface {
    uct_base_iface_t      super;
    uct_self_iface_addr_t id;           /* Unique identifier for the instance */
    size_t                send_size;    /* Maximum size for payload */
    ucs_mpool_t           msg_mp;       /* Messages memory pool */
} uct_self_iface_t;


typedef struct uct_self_ep {
    uct_base_ep_t         super;
} uct_self_ep_t;


#endif
