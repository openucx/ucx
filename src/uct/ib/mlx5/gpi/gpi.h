/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GPI_H
#define UCT_GPI_H

#include <uct/ib/mlx5/rc/rc_mlx5_common.h>
#include <uct/base/uct_iface.h>

#include <cuda.h>

#define UCT_GPI_DEVICE_CUDA_NAME     "cuda"
#define UCT_GPI_DEVICE_CUDA_NAME_LEN 4


typedef struct uct_rc_gpi_iface {
    uct_rc_mlx5_iface_common_t   super;
    CUdevice                     cuda_dev;
    CUcontext                    cuda_ctx;
} uct_rc_gpi_iface_t;


typedef struct uct_rc_gpi_ep {
    uct_base_ep_t super;
} uct_rc_gpi_ep_t;

#endif /* UCT_GPI_H */
