/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Host dispatch: switch on device level, call level-specific launch.
 */

#ifndef UCP_CUDA_KERNEL_LATENCY_DISPATCH_CUH_
#define UCP_CUDA_KERNEL_LATENCY_DISPATCH_CUH_

#include "ucp_cuda_kernel_latency.cuh"

static inline void ucp_perf_cuda_launch_pingpong_dispatch(const ucx_perf_context_t *perf,
                                                          ucx_perf_cuda_context_t *gpu_ctx,
                                                          const ucp_perf_cuda_params_t *params,
                                                          unsigned my_index)
{
    switch (perf->params.device_level) {
    case UCS_DEVICE_LEVEL_THREAD:
        ucp_perf_cuda_launch_pingpong_thread(perf, gpu_ctx, params, my_index);
        break;
    case UCS_DEVICE_LEVEL_WARP:
        ucp_perf_cuda_launch_pingpong_warp(perf, gpu_ctx, params, my_index);
        break;
    default:
        ucs_error("Unsupported device level: %d", perf->params.device_level);
        break;
    }
}

#endif /* UCP_CUDA_KERNEL_LATENCY_DISPATCH_CUH_ */
