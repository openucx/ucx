/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * C-compatible header for CUDA perf context.
 */

#ifndef CUDA_CONTEXT_H_
#define CUDA_CONTEXT_H_

#include <tools/perf/api/libperf.h>
#include <ucs/type/status.h>

typedef unsigned long long ucx_perf_cuda_time_t;

typedef struct ucx_perf_cuda_context {
    ucx_perf_channel_mode_t channel_mode;
    unsigned long long      channel_rand_seed;
    unsigned                max_outstanding;
    unsigned                device_fc_window;
    ucx_perf_counter_t      max_iters;
    ucx_perf_cuda_time_t    report_interval_ns;
    ucx_perf_counter_t      completed_iters;
    ucs_status_t            status;
} ucx_perf_cuda_context_t;

#endif /* CUDA_CONTEXT_H_ */
