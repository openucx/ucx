/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Latency host dispatch: entry point calls header dispatch.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_cuda_kernel_latency_dispatch.cuh"

extern "C" void ucp_perf_cuda_launch_pingpong(const ucx_perf_context_t *perf,
                                              ucx_perf_cuda_context_t *gpu_ctx,
                                              const ucp_perf_cuda_params_t *params,
                                              unsigned my_index)
{
    ucp_perf_cuda_launch_pingpong_dispatch(perf, gpu_ctx, params, my_index);
}
