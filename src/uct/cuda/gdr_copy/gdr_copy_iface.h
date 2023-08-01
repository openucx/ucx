/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDR_COPY_IFACE_H
#define UCT_GDR_COPY_IFACE_H

#include <uct/base/uct_iface.h>


typedef uint64_t uct_gdr_copy_iface_addr_t;


typedef struct uct_gdr_copy_iface {
    uct_base_iface_t            super;
    uct_gdr_copy_iface_addr_t   id;
} uct_gdr_copy_iface_t;


typedef struct uct_gdr_copy_iface_config {
    uct_iface_config_t      super;
} uct_gdr_copy_iface_config_t;

int uct_gdr_copy_iface_is_reachable(const uct_iface_h tl_iface,
                                    const uct_device_addr_t *dev_addr,
                                    const uct_iface_addr_t *iface_addr);
#endif
