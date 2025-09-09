/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef LIBPERF_GDAKI_H_
#define LIBPERF_GDAKI_H_

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <ucs/time/time.h>
#include <ucs/type/status.h>
// TODO: Maybe replace with Doca doca_gpu_mem_alloc.
#include "gdaki_mem_handle.h"
#include "../lib/libperf_int.h"

#define NS_TO_SEC(ns)      ((ns)*1.0 / (UCS_NSEC_PER_SEC))
#define BYTES_TO_MB(bytes) ((bytes) / 1048576.0)

typedef unsigned long long ucx_perf_gdaki_time_t;

//TODO: Replace with real packed batch API
typedef struct uct_gdaki_packed_batch {
    void     *qp;
    uint64_t batch_length;
    uint64_t batch_total_size;
    uint8_t  **src_buf;
    uint64_t *sizes;
    uint32_t *src_mkey;
    uint8_t  **dst_buf;
    uint32_t *dst_mkey;
} uct_gdaki_packed_batch_t;

//TODO: Improve code resue by resusing CPU perftest code.
typedef struct ucx_perf_gdaki_result {
    uint64_t iters;
    double   elapsed_time;
    uint64_t bytes;
    struct {
        double percentile;
        double moment_average; /* Average since last report */
        double total_average; /* Average of the whole test */
    } latency, bandwidth, msgrate;
} ucx_perf_gdaki_result_t;

/**
 * Describes a performance test.
 */
typedef struct ucx_perf_gdaki_params {
    unsigned max_outstanding; /* Maximal number of outstanding sends */
    unsigned m_sends_outstanding;
    uint64_t warmup_iter; /* Number of warm-up iterations */
    double   warmup_time; /* Approximately how long to warm-up */
    uint64_t max_iter; /* Iterations limit, 0 - unlimited */
    ucx_perf_gdaki_time_t max_time; /* Time limit (seconds), 0 - unlimited */
    ucx_perf_gdaki_time_t
             report_interval; /* Interval at which to call the report callback in nanoseconds */
    uint64_t length; // batch length

} ucx_perf_gdaki_params_t;

typedef struct ucx_perf_gdaki_context {
    ucx_perf_gdaki_params_t params;

    /* Measurements */
    ucx_perf_gdaki_time_t   start_time; /* inaccurate end time (upper bound) */
    ucx_perf_gdaki_time_t   end_time; /* inaccurate end time (upper bound) */
    ucx_perf_gdaki_time_t   prev_time; /* time of previous iteration */
    uint64_t                last_report; /* last report to CPU */
    volatile int            test_completed; // Signal test completion

    int                     active_buffer;
    /* Measurements of current/previous **report** */
    struct {
        uint64_t msgs; /* number of messages */
        uint64_t bytes; /* number of bytes */
        uint64_t iters; /* number of iterations */
        ucx_perf_gdaki_time_t
                time; /* inaccurate time (for median and report interval) */
    } current[2], prev[2];

    volatile int results_ready; // Signal CPU to calculate and print
} ucx_perf_gdaki_context_t;

void inline ucx_perf_gdaki_mirror_result(ucx_perf_context_t *perf,
                                         ucx_perf_gdaki_context_t *gdaki_perf)
{
    const int report_buffer = gdaki_perf->active_buffer ^ 1;

    perf->current.iters    = gdaki_perf->current[report_buffer].iters;
    perf->current.bytes    = gdaki_perf->current[report_buffer].bytes;
    perf->current.msgs     = gdaki_perf->current[report_buffer].msgs;
    perf->current.time_acc = NS_TO_SEC(gdaki_perf->current[report_buffer].time);

    perf->prev.iters    = gdaki_perf->prev[report_buffer].iters;
    perf->prev.bytes    = gdaki_perf->prev[report_buffer].bytes;
    perf->prev.msgs     = gdaki_perf->prev[report_buffer].msgs;
    perf->prev.time_acc = NS_TO_SEC(gdaki_perf->prev[report_buffer].time);

    perf->start_time_acc = NS_TO_SEC(gdaki_perf->start_time);
}

#ifdef __CUDACC__

__device__ inline unsigned long long ucx_perf_gdaki_dev_get_time_ns()
{
    unsigned long long globaltimer;
    // 64-bit GPU global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

__device__ inline void ucx_perf_gdaki_dev_sleep_ns(ucx_perf_gdaki_time_t ns)
{
    const ucx_perf_gdaki_time_t start_time = ucx_perf_gdaki_dev_get_time_ns();
    while (ucx_perf_gdaki_dev_get_time_ns() - start_time < ns) {
        // Do nothing
    }
}

__device__ static inline void
ucx_perf_gdaki_dev_update_metrics(ucx_perf_gdaki_context_t *perf,
                                  ucx_perf_gdaki_time_t current_time,
                                  uint64_t bytes, uint64_t iters)
{
    const int active_buffer = perf->active_buffer;

    perf->current[active_buffer].time   = current_time; // TODO: capture time
    perf->current[active_buffer].iters += iters;
    perf->current[active_buffer].bytes += bytes;
    perf->current[active_buffer].msgs  += 1;

    perf->prev_time = perf->current[active_buffer].time;
}

__device__ inline void
ucx_perf_gdaki_dev_report_results(ucx_perf_gdaki_context_t *ctx,
                                  ucx_perf_gdaki_time_t current_time)
{
    int report_buffer = ctx->active_buffer;

    ctx->active_buffer              ^= 1;
    ctx->current[report_buffer].time = current_time;
    ctx->current[ctx->active_buffer] = ctx->current[report_buffer];
    ctx->prev[ctx->active_buffer]    = ctx->current[report_buffer];
    ctx->results_ready               = 1;
}

__device__ void inline ucx_perf_gdaki_dev_init_test_time(
        ucx_perf_gdaki_context_t *ctx, ucx_perf_gdaki_time_t *next_report_time)
{
    if (threadIdx.x == 0) {
        ctx->start_time = ucx_perf_gdaki_dev_get_time_ns();
        ctx->current[ctx->active_buffer].time = ctx->start_time;
        ctx->prev[ctx->active_buffer].time =
                ctx->current[ctx->active_buffer].time;
        *next_report_time = ctx->start_time + ctx->params.report_interval;
    }

    __syncthreads();
}

__device__ void inline ucx_perf_gdaki_dev_update_results(
        ucx_perf_gdaki_context_t *ctx, ucx_perf_gdaki_time_t *next_report_time)
{
    if (threadIdx.x == 0) {
        ctx->params.m_sends_outstanding++;
        ucx_perf_gdaki_time_t current_time = ucx_perf_gdaki_dev_get_time_ns();
        ucx_perf_gdaki_dev_update_metrics(ctx, current_time, ctx->params.length,
                                          1);
        if (current_time >= *next_report_time) {
            ucx_perf_gdaki_dev_report_results(ctx, current_time);
            *next_report_time = current_time + ctx->params.report_interval;
        }
    }
}

__device__ void inline ucx_perf_gdaki_dev_complete_test(
        ucx_perf_gdaki_context_t *ctx)
{
    if (threadIdx.x == 0) {
        ucx_perf_gdaki_time_t current_time = ucx_perf_gdaki_dev_get_time_ns();
        ucx_perf_gdaki_dev_update_metrics(ctx, current_time, 0, 0);
        ucx_perf_gdaki_dev_report_results(ctx, current_time);
        ctx->end_time                   = current_time;
        ctx->test_completed             = 1;
        ctx->params.m_sends_outstanding = 0;
    }
}

#endif // __CUDACC__

#endif // LIBPERF_GDAKI_H_
