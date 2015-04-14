
/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <uct/tl/tl_base.h>
#include <ucs/sys/sys.h>
#include <stdbool.h>
#include "cuda_context.h"
#include "cuda_ep.h"


struct uct_cuda_iface;

typedef struct uct_cuda_iface_addr {
    uct_iface_addr_t    super;
    uint32_t            nic_addr;
} uct_cuda_iface_addr_t;

typedef struct uct_cuda_pd {
    uct_pd_t      super;
} uct_cuda_pd_t;

typedef struct uct_cuda_iface {
    uct_base_iface_t        super;
    uct_cuda_pd_t           pd;
    uct_cuda_iface_addr_t   addr;
    struct {
        unsigned            max_put;
    } config;
    /* list of ep */
} uct_cuda_iface_t;

typedef struct uct_cuda_iface_config {
    uct_iface_config_t       super;
} uct_cuda_iface_config_t;

typedef struct uct_cuda_key {
} uct_cuda_key_t;

extern ucs_config_field_t uct_cuda_iface_config_table[];
extern uct_tl_ops_t uct_cuda_tl_ops;

#endif
