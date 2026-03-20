/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Wait kernel + host entry point in same TU for correct linking.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_cuda_kernel_wait.cuh"
#include "cuda_kernel.cuh"
#include "ucp_cuda_kernel_common.cuh"

__global__ void
ucp_perf_cuda_wait_bw_kernel(ucx_perf_cuda_context_t &ctx,
                             ucp_perf_cuda_params_t params)
{
    volatile uint64_t *sn = params.counter_recv;
    while (*sn < ctx.max_iters) {
        __nanosleep(100000); // 100us
    }

    ctx.status = UCS_OK;
}

extern "C" void ucp_perf_cuda_launch_wait(ucx_perf_cuda_context_t *gpu_ctx,
                                          const ucp_perf_cuda_params_t *params)
{
    ucp_perf_cuda_wait_bw_kernel<<<1, 1>>>(*gpu_ctx, *params);
}
