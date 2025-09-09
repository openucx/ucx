/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef GDAKI_BENCHMARK_H
#define GDAKI_BENCHMARK_H

#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include "libperf_gdaki.h"


// C wrapper function to launch the kernel
ucs_status_t ucx_perf_gdaki_execute_uct_put_batch_bw_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        uct_dev_ep_h ep, uct_batch_h batch, uint64_t flags);

ucs_status_t ucx_perf_gdaki_execute_uct_put_batch_lat_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        uct_dev_ep_h ep, uct_batch_h batch, uint64_t flags,
        void *signal, int is_sender);

ucs_status_t ucx_perf_gdaki_execute_ucp_put_batch_bw_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        ucp_batch_h batch, uint64_t flags, uint64_t signal_inc);

ucs_status_t ucx_perf_gdaki_execute_ucp_put_batch_lat_kernel(
        unsigned cuda_threads, ucx_perf_gdaki_context_t *gpu_ctx,
        ucp_batch_h batch, uint64_t flags, void *signal,
        int is_sender);

#endif // GDAKI_BENCHMARK_H
