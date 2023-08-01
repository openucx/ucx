/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CMA_IFACE_H
#define UCT_CMA_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/sm/scopy/base/scopy_iface.h>


#define UCT_CMA_IFACE_ADDR_FLAG_PID_NS UCS_BIT(31) /* use PID NS in address */


typedef struct uct_cma_iface_config {
    uct_scopy_iface_config_t      super;
} uct_cma_iface_config_t;


typedef struct uct_cma_iface {
    uct_scopy_iface_t             super;
} uct_cma_iface_t;

int uct_cma_iface_is_reachable(const uct_iface_h tl_iface,
                               const uct_device_addr_t *dev_addr,
                               const uct_iface_addr_t *tl_iface_addr);

#endif
