/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * BW kernel THREAD level, fc=true.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_cuda_kernel_bw.cuh"
#include "ucp_cuda_kernel_bw_impl.cuh"

void ucp_perf_cuda_launch_bw_thread_fc(const ucx_perf_context_t *perf,
                                        ucx_perf_cuda_context_t *gpu_ctx,
                                        const ucp_perf_cuda_params_t *params)
{
    ucp_perf_cuda_launch_bw_level<UCS_DEVICE_LEVEL_THREAD, true>(perf, gpu_ctx,
                                                                 params);
}
