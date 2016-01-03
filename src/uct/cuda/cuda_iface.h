/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <uct/base/uct_iface.h>


#define UCT_CUDA_TL_NAME    "cuda"
#define UCT_CUDA_DEV_NAME   "gpu0"


typedef struct uct_cuda_iface {
    uct_base_iface_t        super;
} uct_cuda_iface_t;


typedef struct uct_cuda_iface_config {
    uct_iface_config_t      super;
} uct_cuda_iface_config_t;


#endif
