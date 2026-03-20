/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Host dispatch: switch on device level and fc, call level+fc-specific launch.
 */

#ifndef UCP_CUDA_KERNEL_BW_DISPATCH_CUH_
#define UCP_CUDA_KERNEL_BW_DISPATCH_CUH_

#include "ucp_cuda_kernel_bw.cuh"

static inline void ucp_perf_cuda_launch_bw_dispatch(const ucx_perf_context_t *perf,
                                                    ucx_perf_cuda_context_t *gpu_ctx,
                                                    const ucp_perf_cuda_params_t *params)
{
    bool fc = perf->params.device_fc_window > 1;

    switch (perf->params.device_level) {
    case UCS_DEVICE_LEVEL_THREAD:
        if (fc) {
            ucp_perf_cuda_launch_bw_thread_fc(perf, gpu_ctx, params);
        } else {
            ucp_perf_cuda_launch_bw_thread_nofc(perf, gpu_ctx, params);
        }
        break;
    case UCS_DEVICE_LEVEL_WARP:
        if (fc) {
            ucp_perf_cuda_launch_bw_warp_fc(perf, gpu_ctx, params);
        } else {
            ucp_perf_cuda_launch_bw_warp_nofc(perf, gpu_ctx, params);
        }
        break;
    default:
        ucs_error("Unsupported device level: %d", perf->params.device_level);
        break;
    }
}

#endif /* UCP_CUDA_KERNEL_BW_DISPATCH_CUH_ */
