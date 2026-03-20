/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Latency kernel WARP level: instantiations + launch entry.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_cuda_kernel_latency.cuh"
#include "ucp_cuda_kernel_latency_impl.cuh"
#include "cuda_kernel.cuh"

void ucp_perf_cuda_launch_pingpong_warp(const ucx_perf_context_t *perf,
                                        ucx_perf_cuda_context_t *gpu_ctx,
                                        const ucp_perf_cuda_params_t *params,
                                        unsigned my_index)
{
    ucp_perf_cuda_launch_pingpong_level<UCS_DEVICE_LEVEL_WARP>(perf, gpu_ctx,
                                                               params, my_index);
}
