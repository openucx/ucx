/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_COPY_IFACE_H
#define UCT_ROCM_COPY_IFACE_H

#include <uct/base/uct_iface.h>

#define UCT_ROCM_COPY_TL_NAME    "rocm_cpy"

typedef uint64_t uct_rocm_copy_iface_addr_t;

typedef struct uct_rocm_copy_iface {
    uct_base_iface_t            super;
    uct_rocm_copy_iface_addr_t  id;
    struct {
        size_t                  d2h_thresh;
        size_t                  h2d_thresh;
        int                     enable_async_zcopy;
    } config;
} uct_rocm_copy_iface_t;

typedef struct uct_rocm_copy_iface_config {
    uct_iface_config_t  super;
    size_t              d2h_thresh;
    size_t              h2d_thresh;
    size_t              short_d2h_thresh;
    size_t              short_h2d_thresh;
    int                 enable_async_zcopy;
} uct_rocm_copy_iface_config_t;

#endif
