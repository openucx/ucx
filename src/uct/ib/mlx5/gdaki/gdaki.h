/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_H
#define UCT_GDAKI_H

#include <uct/ib/mlx5/rc/rc_mlx5_common.h>
#include <uct/base/uct_iface.h>

#include <cuda.h>
#include <doca_gpunetio.h>

#include "gdaki_dev.h"

#define UCT_DEVICE_CUDA_NAME     "cuda"
#define UCT_DEVICE_CUDA_NAME_LEN 4


typedef struct uct_rc_gdaki_iface {
    uct_rc_mlx5_iface_common_t super;
    struct doca_gpu            *gpu_dev;
    CUdevice                   cuda_dev;
    struct ibv_mr              *atomic_mr;
    uint64_t                   *atomic_buff;
    CUcontext                  cuda_ctx;
} uct_rc_gdaki_iface_t;


typedef struct uct_rc_gdaki_ep {
    uct_base_ep_t                super;
    uct_ib_mlx5_cq_t             cq;
    uct_ib_mlx5_txwq_t           qp;
    struct mlx5dv_devx_umem      *umem;
    struct doca_gpu_verbs_qp     *qp_cpu;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    uct_rc_gdaki_dev_ep_t        *ep_gpu;
} uct_rc_gdaki_ep_t;

#endif /* UCT_GDAKI_IFACE_H */
