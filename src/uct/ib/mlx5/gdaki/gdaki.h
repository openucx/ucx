/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_IFACE_H
#define UCT_GDAKI_IFACE_H

#include <uct/base/uct_iface.h>

#include <cuda.h>
#include <doca_gpunetio.h>

typedef struct uct_rc_gdaki_iface {
    struct doca_gpu *gpu_dev;
    CUdevice        cuda_dev;
} uct_rc_gdaki_iface_t;

#endif /* UCT_GDAKI_IFACE_H */
