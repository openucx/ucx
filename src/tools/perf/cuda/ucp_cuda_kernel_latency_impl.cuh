/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Latency kernel template: common implementation for all device levels.
 */

#ifndef UCP_CUDA_KERNEL_LATENCY_IMPL_CUH_
#define UCP_CUDA_KERNEL_LATENCY_IMPL_CUH_

#include "ucp_cuda_kernel_common.cuh"
#include <ucs/sys/math.h>

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
__global__ void
ucp_perf_cuda_put_latency_kernel(ucx_perf_cuda_context_t &ctx,
                                 ucp_perf_cuda_params_t params,
                                 bool is_sender)
{
    extern __shared__ ucp_device_request_t shared_requests[];
    ucx_perf_counter_t max_iters = ctx.max_iters;
    ucs_status_t status          = UCS_OK;
    unsigned thread_index        = ucx_perf_cuda_thread_index<level>(threadIdx.x);
    unsigned global_thread_id    = ucx_perf_cuda_thread_index<level>(
        thread_index + blockIdx.x * blockDim.x);
    ucp_device_request_t *req    = &shared_requests[thread_index];
    curandState rand_state;

    if (ctx.channel_mode == UCX_PERF_CHANNEL_MODE_RANDOM) {
        curand_init(ctx.channel_rand_seed, global_thread_id, 0, &rand_state);
    }

    ucp_perf_cuda_request_manager req_mgr(ctx, req, &rand_state);
    ucx_perf_cuda_reporter reporter(ctx);

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        unsigned channel_id = req_mgr.get_channel_id<level>();
        if (is_sender) {
            status = ucp_perf_cuda_send_sync<level, cmd>(params, idx, req,
                                                        channel_id);
            if (status != UCS_OK) {
                ucs_device_error("sender send failed: %d", status);
                break;
            }
            ucx_perf_cuda_wait_sn(params.counter_recv, idx + 1);
        } else {
            ucx_perf_cuda_wait_sn(params.counter_recv, idx + 1);
            status = ucp_perf_cuda_send_sync<level, cmd>(params, idx, req,
                                                        channel_id);
            if (status != UCS_OK) {
                ucs_device_error("receiver send failed: %d", status);
                break;
            }
        }

        reporter.update_report(idx + 1);
    }

    ctx.status = status;
}

template<ucs_device_level_t level>
static inline void ucp_perf_cuda_launch_pingpong_level(const ucx_perf_context_t *perf,
                                                       ucx_perf_cuda_context_t *gpu_ctx,
                                                       const ucp_perf_cuda_params_t *params,
                                                       unsigned my_index)
{
    unsigned blocks          = perf->params.device_block_count;
    unsigned threads         = perf->params.device_thread_count;
    unsigned reqs_count      = ucs_div_round_up(perf->params.max_outstanding,
                                               perf->params.device_fc_window);
    ucx_perf_cmd_t cmd       = perf->params.command;
    unsigned threads_per_level = (level == UCS_DEVICE_LEVEL_WARP) ?
        (threads / UCS_DEVICE_NUM_THREADS_IN_WARP) : threads;
    size_t shared = reqs_count * sizeof(ucp_device_request_t) * threads_per_level;

#define LAUNCH_LATENCY(_cmd) \
    ucp_perf_cuda_put_latency_kernel<level, _cmd> \
        <<<blocks, threads, shared>>>(*gpu_ctx, *params, my_index)

    switch (cmd) {
    case UCX_PERF_CMD_PUT_SINGLE: LAUNCH_LATENCY(UCX_PERF_CMD_PUT_SINGLE); break;
    case UCX_PERF_CMD_PUT:         LAUNCH_LATENCY(UCX_PERF_CMD_PUT); break;
    case UCX_PERF_CMD_PUT_MULTI:   LAUNCH_LATENCY(UCX_PERF_CMD_PUT_MULTI); break;
    case UCX_PERF_CMD_PUT_PARTIAL: LAUNCH_LATENCY(UCX_PERF_CMD_PUT_PARTIAL); break;
    default: ucs_error("Unsupported cmd: %d", cmd); break;
    }
#undef LAUNCH_LATENCY
}

#endif /* UCP_CUDA_KERNEL_LATENCY_IMPL_CUH_ */
