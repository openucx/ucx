/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>
#include <uct/cuda/gdaki/gdaki_ep.h>
#include <uct/api/cuda/uct_dev.cuh>
#include <ucp/api/cuda/ucp_dev.cuh>
#include "libperf_gdaki.h"

static __device__ void inline ucx_perf_gdaki_dev_update_results_and_sync(
        ucx_perf_gdaki_context_t *ctx, ucx_perf_gdaki_time_t *next_report_time)
{
    ucx_perf_gdaki_dev_update_results(ctx, next_report_time);
    __syncthreads();
}

__global__ void uct_put_batch_bw_kernel(ucx_perf_gdaki_context_t *ctx,
                                        uct_dev_ep_h ep, uct_batch_h batch,
                                        uint64_t flags)
{
    ucx_perf_gdaki_time_t next_report_time;
    __shared__ uct_dev_completion_t comp;

    ucx_perf_gdaki_dev_init_test_time(ctx, &next_report_time);

    for (uint64_t idx = 0; idx < ctx->params.max_iter; idx++) {
        while (comp.count >= ctx->params.max_outstanding) {
            uct_dev_ep_progress(ep);
        }

        if (threadIdx.x == 0) {
            comp.count++;
        }

        for (;;) {
            ucs_status_t status = uct_dev_batch_execute(batch, flags, 1, &comp);
            if (status == UCS_OK) {
                break;
            } else if (status == UCS_ERR_NO_RESOURCE) {
                uct_dev_ep_progress(ep);
            } else {
                return;
            }
        }

        ucx_perf_gdaki_dev_update_results_and_sync(ctx, &next_report_time);
    }

    while (comp.count > 0) {
        uct_dev_ep_progress(ep);
    }

    ucx_perf_gdaki_dev_complete_test(ctx);
}


__global__ void
ucp_put_batch_bw_kernel(ucx_perf_gdaki_context_t *ctx, ucp_batch_h batch,
                        uint64_t flags, uint64_t signal_inc)
{
    ucx_perf_gdaki_time_t next_report_time;
    __shared__ ucp_dev_request_t request;
    ucs_status_t status;

    ucx_perf_gdaki_dev_init_test_time(ctx, &next_report_time);

    for (uint64_t idx = 0; idx < ctx->params.max_iter; idx++) {
        if (ctx->params.m_sends_outstanding > ctx->params.max_outstanding) {
            status = ucp_dev_request_progress(&request);
            if (status != UCS_OK) {
                printf("Failed to progress request %d\n", status);
                return;
            }
            if (threadIdx.x == 0) {
                ctx->params.m_sends_outstanding--;
            }
        }

        ucp_dev_batch_execute(batch, flags, signal_inc, &request);
        ucx_perf_gdaki_dev_update_results_and_sync(ctx, &next_report_time);
    }

    status = ucp_dev_request_progress(&request);
    if (status != UCS_OK) {
        printf("Failed to progress request %d\n", status);
        return;
    }

    ucx_perf_gdaki_dev_complete_test(ctx);
}


ucs_status_t ucx_perf_gdaki_execute_uct_put_batch_bw_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        uct_dev_ep_h ep, uct_batch_h batch, uint64_t flags)
{
    ucs_status_t ret = UCS_OK;
    cudaError_t err;

    uct_put_batch_bw_kernel<<<1, cuda_threads>>>(gpu_ctx, ep, batch, flags);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ucs_error("Failed to launch kernel");
        ret = UCS_ERR_NO_DEVICE;
    }

    return ret;
}

ucs_status_t ucx_perf_gdaki_execute_ucp_put_batch_bw_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        ucp_batch_h batch, uint64_t flags, uint64_t signal_inc)
{
    ucs_status_t ret = UCS_OK;
    cudaError_t err;

    ucp_put_batch_bw_kernel<<<1, cuda_threads>>>(gpu_ctx, batch, flags,
                                                 signal_inc);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ucs_error("Failed to launch kernel: %s", cudaGetErrorString(err));
        ret = UCS_ERR_NO_DEVICE;
    }

    return ret;
}
