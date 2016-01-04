/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CMA_IFACE_H
#define UCT_CMA_IFACE_H

#include <uct/base/uct_iface.h>

#define UCT_CMA_TL_NAME "cma"


typedef struct uct_cma_iface_config {
    uct_iface_config_t      super;
} uct_cma_iface_config_t;


typedef struct uct_cma_iface {
    uct_base_iface_t        super;
} uct_cma_iface_t;


extern uct_tl_component_t uct_cma_tl;

#endif
