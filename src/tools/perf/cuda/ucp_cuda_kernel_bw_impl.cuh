/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * BW kernel template: common implementation for all device levels.
 */

#ifndef UCP_CUDA_KERNEL_BW_IMPL_CUH_
#define UCP_CUDA_KERNEL_BW_IMPL_CUH_

#include "ucp_cuda_kernel_common.cuh"
#include <tools/perf/api/libperf.h>
#include <ucs/sys/math.h>

template<ucs_device_level_t level, ucx_perf_cmd_t cmd, bool fc>
__global__ void
ucp_perf_cuda_put_bw_kernel(ucx_perf_cuda_context_t &ctx,
                            ucp_perf_cuda_params_t params)
{
    extern __shared__ ucp_device_request_t shared_requests[];
    unsigned thread_index      = ucx_perf_cuda_thread_index<level>(threadIdx.x);
    unsigned reqs_count        = ucs_div_round_up(ctx.max_outstanding,
                                                  ctx.device_fc_window);
    unsigned global_thread_id  = ucx_perf_cuda_thread_index<level>(
        thread_index + blockIdx.x * blockDim.x);
    ucp_device_request_t *reqs = &shared_requests[reqs_count * thread_index];
    curandState rand_state;

    if (ctx.channel_mode == UCX_PERF_CHANNEL_MODE_RANDOM) {
        curand_init(ctx.channel_rand_seed, global_thread_id, 0, &rand_state);
    }

    ucp_perf_cuda_request_manager req_mgr(ctx, reqs, &rand_state);
    ctx.status = ucp_perf_cuda_put_bw_kernel_impl<level, cmd, fc>(
        ctx, params, req_mgr);
}

template<ucs_device_level_t level, bool fc>
static inline void ucp_perf_cuda_launch_bw_level(const ucx_perf_context_t *perf,
                                                 ucx_perf_cuda_context_t *gpu_ctx,
                                                 const ucp_perf_cuda_params_t *params)
{
    unsigned blocks     = perf->params.device_block_count;
    unsigned threads    = perf->params.device_thread_count;
    unsigned reqs_count = ucs_div_round_up(perf->params.max_outstanding,
                                          perf->params.device_fc_window);
    ucx_perf_cmd_t cmd  = perf->params.command;
    unsigned threads_per_level = (level == UCS_DEVICE_LEVEL_WARP) ?
        (threads / UCS_DEVICE_NUM_THREADS_IN_WARP) : threads;
    size_t shared = reqs_count * sizeof(ucp_device_request_t) * threads_per_level;

#define LAUNCH_BW(_cmd) \
    ucp_perf_cuda_put_bw_kernel<level, _cmd, fc> \
        <<<blocks, threads, shared>>>(*gpu_ctx, *params)

    switch (cmd) {
    case UCX_PERF_CMD_PUT_SINGLE: LAUNCH_BW(UCX_PERF_CMD_PUT_SINGLE); break;
    case UCX_PERF_CMD_PUT:         LAUNCH_BW(UCX_PERF_CMD_PUT); break;
    case UCX_PERF_CMD_PUT_MULTI:   LAUNCH_BW(UCX_PERF_CMD_PUT_MULTI); break;
    case UCX_PERF_CMD_PUT_PARTIAL: LAUNCH_BW(UCX_PERF_CMD_PUT_PARTIAL); break;
    default: ucs_error("Unsupported cmd: %d", cmd); break;
    }
#undef LAUNCH_BW
}

#endif /* UCP_CUDA_KERNEL_BW_IMPL_CUH_ */
