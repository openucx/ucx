/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#ifndef ROCM_CMA_IFACE_H
#define ROCM_CMA_IFACE_H

#include "rocm_cma_md.h"
#include <uct/base/uct_iface.h>


/** Define name of transport used for memory operation. Must not be larger than
    UCT_TL_NAME_MAX
*/
#define UCT_ROCM_CMA_TL_NAME    "rocm"


typedef struct uct_rocm_cma_iface {
    uct_base_iface_t        super;
    uct_rocm_cma_md_t      *rocm_md;
} uct_rocm_cma_iface_t;


typedef struct uct_rocm_cma_iface_config {
    uct_iface_config_t      super;
} uct_rocm_cma_iface_config_t;

extern uct_tl_component_t uct_rocm_cma_tl;

#endif
