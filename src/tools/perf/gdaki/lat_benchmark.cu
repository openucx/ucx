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

__device__ void inline uct_put_batch_lat_send(
        ucx_perf_gdaki_context_t *ctx, uct_dev_ep_h ep, uct_batch_h batch,
        uint64_t flags, ucx_perf_gdaki_time_t *next_report_time)
{
    __shared__ uct_dev_completion_t comp;

    comp.count = 1;
    uct_dev_batch_execute(batch, flags, 1, &comp);
    ucx_perf_gdaki_dev_update_results(ctx, next_report_time);

    while (comp.count != 0) {
        uct_dev_ep_progress(ep);
    }
}

__device__ void ucx_put_batch_lat_wait_for_signal(ucx_perf_gdaki_context_t *ctx,
                                                  volatile uint64_t *signal,
                                                  uint64_t expected_value)
{
    if (threadIdx.x == 0) {
        while (*signal == expected_value)
            ;
    }

    __syncthreads();
}

__global__ void uct_put_batch_lat_kernel_receiver(ucx_perf_gdaki_context_t *ctx,
                                                  uct_dev_ep_h ep,
                                                  uct_batch_h batch,
                                                  uint64_t flags,
                                                  volatile uint64_t *signal)
{
    ucx_perf_gdaki_time_t next_report_time;

    ucx_perf_gdaki_dev_init_test_time(ctx, &next_report_time);

    for (uint64_t idx = 0; idx < ctx->params.max_iter; idx++) {
        ucx_put_batch_lat_wait_for_signal(ctx, signal, idx);
        uct_put_batch_lat_send(ctx, ep, batch, flags, &next_report_time);
    }

    ucx_perf_gdaki_dev_complete_test(ctx);
}

__global__ void
uct_put_batch_lat_kernel_sender(ucx_perf_gdaki_context_t *ctx, uct_dev_ep_h ep,
                                uct_batch_h batch, uint64_t flags,
                                volatile uint64_t *signal)
{
    ucx_perf_gdaki_time_t next_report_time;

    ucx_perf_gdaki_dev_init_test_time(ctx, &next_report_time);

    for (uint64_t idx = 0; idx < ctx->params.max_iter; idx++) {
        uct_put_batch_lat_send(ctx, ep, batch, flags, &next_report_time);
        ucx_put_batch_lat_wait_for_signal(ctx, signal, idx);
    }

    ucx_perf_gdaki_dev_complete_test(ctx);
}

__device__ void inline ucp_put_batch_lat_send(
        ucx_perf_gdaki_context_t *ctx, ucp_batch_h batch,
        uint64_t flags, ucx_perf_gdaki_time_t *next_report_time)
{
    __shared__ ucp_dev_request_t request;
    ucs_status_t status;

    ucp_dev_batch_execute(batch, flags, 1, &request);
    ucx_perf_gdaki_dev_update_results(ctx, next_report_time);

    status = ucp_dev_request_progress(&request);
    if (status != UCS_OK) {
        printf("Failed to progress request %d\n", status);
        return;
    }
}

__global__ void ucp_put_batch_lat_kernel_receiver(ucx_perf_gdaki_context_t *ctx,
                                                  ucp_batch_h batch,
                                                  uint64_t flags,
                                                  volatile uint64_t *signal)
{
    ucx_perf_gdaki_time_t next_report_time;

    ucx_perf_gdaki_dev_init_test_time(ctx, &next_report_time);

    for (uint64_t idx = 0; idx < ctx->params.max_iter; idx++) {
        ucx_put_batch_lat_wait_for_signal(ctx, signal, idx);
        ucp_put_batch_lat_send(ctx, batch, flags, &next_report_time);
    }

    ucx_perf_gdaki_dev_complete_test(ctx);
}

__global__ void ucp_put_batch_lat_kernel_sender(ucx_perf_gdaki_context_t *ctx,
                                                ucp_batch_h batch,
                                                uint64_t flags,
                                                volatile uint64_t *signal)
{
    ucx_perf_gdaki_time_t next_report_time;

    ucx_perf_gdaki_dev_init_test_time(ctx, &next_report_time);

    for (uint64_t idx = 0; idx < ctx->params.max_iter; idx++) {
        ucp_put_batch_lat_send(ctx, batch, flags, &next_report_time);
        ucx_put_batch_lat_wait_for_signal(ctx, signal, idx);
    }

    ucx_perf_gdaki_dev_complete_test(ctx);
}

ucs_status_t ucx_perf_gdaki_execute_uct_put_batch_lat_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        uct_dev_ep_h ep, uct_batch_h batch, uint64_t flags,
        void *signal, int is_sender)
{
    ucs_status_t ret = UCS_OK;
    cudaError_t err;

    if (is_sender) {
        uct_put_batch_lat_kernel_sender<<<1, cuda_threads>>>(
                gpu_ctx, ep, batch, flags, (volatile uint64_t*)signal);
    } else {
        uct_put_batch_lat_kernel_receiver<<<1, cuda_threads>>>(
                gpu_ctx, ep, batch, flags, (volatile uint64_t*)signal);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ucs_error("Failed to launch kernel");
        ret = UCS_ERR_NO_DEVICE;
    }

    return ret;
}

ucs_status_t ucx_perf_gdaki_execute_ucp_put_batch_lat_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        ucp_batch_h batch, uint64_t flags, void *signal,
        int is_sender)
{
    ucs_status_t ret = UCS_OK;
    cudaError_t err;

    if (is_sender) {
        ucp_put_batch_lat_kernel_sender<<<1, cuda_threads>>>(
                gpu_ctx, batch, flags, (volatile uint64_t*)signal);
    } else {
        ucp_put_batch_lat_kernel_receiver<<<1, cuda_threads>>>(
                gpu_ctx, batch, flags, (volatile uint64_t*)signal);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ucs_error("Failed to launch kernel");
        ret = UCS_ERR_NO_DEVICE;
    }

    return ret;
}
