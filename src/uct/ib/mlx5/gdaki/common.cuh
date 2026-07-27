/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEV_COMMON_CUH_
#define UCT_DEV_COMMON_CUH_

#include <ucs/sys/device_code.h>

#include <infiniband/mlx5dv.h> /* TODO add to gpunetio */
#include <cuda.h>              /* TODO add to gpunetio */
#include "gpunetio/device/doca_gpunetio_dev_verbs_common.cuh"
#include <cooperative_groups.h>


template<ucs_device_level_t level>
UCS_F_DEVICE void uct_dev_exec_init(unsigned &lane_id, unsigned &num_lanes)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        lane_id   = 0;
        num_lanes = 1;
        break;
    case UCS_DEVICE_LEVEL_WARP:
        lane_id   = doca_gpu_dev_verbs_get_lane_id();
        num_lanes = UCS_DEVICE_NUM_THREADS_IN_WARP;
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        lane_id   = threadIdx.x;
        num_lanes = blockDim.x;
        break;
    case UCS_DEVICE_LEVEL_GRID:
        lane_id   = threadIdx.x + blockIdx.x * blockDim.x;
        num_lanes = blockDim.x * gridDim.x;
        break;
    }
}

template<ucs_device_level_t level> UCS_F_DEVICE void uct_dev_sync(void)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_WARP:
        __syncwarp();
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        __syncthreads();
        break;
    case UCS_DEVICE_LEVEL_THREAD:
        break;
    case UCS_DEVICE_LEVEL_GRID:
        auto g = cooperative_groups::this_grid();
        g.sync();
    }
}

template<ucs_device_level_t level>
UCS_F_DEVICE int uct_dev_bcast(int value, unsigned lane_id)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_WARP:
        return __shfl_sync(0xffffffff, value, 0);
    case UCS_DEVICE_LEVEL_BLOCK: {
        __shared__ int shared_value;
        if (lane_id == 0) {
            shared_value = value;
        }
        __syncthreads();
        value = shared_value;
        __syncthreads();
        return value;
    }
    case UCS_DEVICE_LEVEL_THREAD:
    case UCS_DEVICE_LEVEL_GRID:
    default:
        return value;
    }
}

#endif /* UCT_DEV_COMMON_CUH_ */
