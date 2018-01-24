/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SELF_IFACE_H
#define UCT_SELF_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>

typedef uint64_t uct_self_iface_addr_t;

typedef struct uct_self_iface {
    uct_base_iface_t      super;
    uct_self_iface_addr_t id;           /* Unique identifier for the instance */
    size_t                rx_headroom;  /* User data size precedes payload */
    unsigned              data_length;  /* Maximum size for payload */
    uct_recv_desc_t       release_desc; /* Callback to desc release func */
    ucs_mpool_t           msg_desc_mp;  /* Messages memory pool */
} UCS_V_ALIGNED(UCS_SYS_CACHE_LINE_SIZE) uct_self_iface_t;

typedef struct uct_self_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mp;
} uct_self_iface_config_t;


#endif
