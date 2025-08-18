/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_IFACE_H
#define UCT_GDAKI_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/datastruct/khash.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/rc/base/rc_iface.h>

#include <doca_gpunetio.h>

typedef struct uct_gdaki_iface {
    uct_ib_iface_t super;
    uct_rc_config_t rc_cfg;
    struct doca_gpu  *gpu_dev;
    CUdevice cuda_dev;
    uct_alloc_t alloc;
} uct_gdaki_iface_t;


ucs_status_t uct_gdaki_iface_open(uct_md_h md, uct_worker_h worker,
                                 const uct_iface_params_t *params,
                                 const uct_iface_config_t *tl_config,
                                 uct_iface_h *iface_p);

#endif /* UCT_GDAKI_IFACE_H */
