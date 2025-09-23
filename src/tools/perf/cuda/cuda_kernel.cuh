/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_KERNEL_CUH_
#define CUDA_KERNEL_CUH_

#include "cuda_common.h"

#include <tools/perf/lib/libperf_int.h>
#include <ucs/sys/device_code.h>
#include <cuda_runtime.h>


typedef unsigned long long ucx_perf_cuda_time_t;

struct ucx_perf_cuda_context {
    unsigned             max_outstanding;
    ucx_perf_counter_t   max_iters;
    ucx_perf_cuda_time_t report_interval_ns;
    ucx_perf_counter_t   completed_iters;
    ucs_status_t         status;
};

UCS_F_DEVICE ucx_perf_cuda_time_t ucx_perf_cuda_get_time_ns()
{
    ucx_perf_cuda_time_t globaltimer;
    /* 64-bit GPU global nanosecond timer */
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

UCS_F_DEVICE void
ucx_perf_cuda_update_report(ucx_perf_cuda_context &ctx,
                            ucx_perf_counter_t completed,
                            ucx_perf_counter_t max_iters,
                            ucx_perf_cuda_time_t &last_report_time)
{
    if (threadIdx.x == 0) {
        ucx_perf_cuda_time_t current_time = ucx_perf_cuda_get_time_ns();
        if (((current_time - last_report_time) >= ctx.report_interval_ns) ||
            (completed >= max_iters)) {
            ctx.completed_iters = completed;
            last_report_time    = current_time;
            __threadfence();
        }
    }
}

static UCS_F_ALWAYS_INLINE uint64_t *
ucx_perf_cuda_get_sn(const void *address, size_t length)
{
    return (uint64_t*)UCS_PTR_BYTE_OFFSET(address, length);
}

UCS_F_DEVICE void ucx_perf_cuda_wait_sn(const uint64_t *sn, uint64_t value)
{
    if (threadIdx.x == 0) {
        while (ucs_device_atomic64_read(sn) < value);
    }
    __syncthreads();
}

/* Simple bitset */
#define UCX_BIT_MASK(bit)       (1 << ((bit) & (CHAR_BIT - 1)))
#define UCX_BIT_SET(set, bit)   (set[(bit)/CHAR_BIT] |= UCX_BIT_MASK(bit))
#define UCX_BIT_RESET(set, bit) (set[(bit)/CHAR_BIT] &= ~UCX_BIT_MASK(bit))
#define UCX_BIT_GET(set, bit)   (set[(bit)/CHAR_BIT] &  UCX_BIT_MASK(bit))
#define UCX_BITSET_SIZE(bits)   ((bits + CHAR_BIT - 1) / CHAR_BIT)

UCS_F_DEVICE size_t ucx_bitset_popcount(const uint8_t *set, size_t bits) {
    size_t count = 0;
    for (size_t i = 0; i < bits; i++) {
        if (UCX_BIT_GET(set, i)) {
            count++;
        }
    }
    return count;
}

UCS_F_DEVICE size_t
ucx_bitset_ffns(const uint8_t *set, size_t bits, size_t from)
{
    for (size_t i = from; i < bits; i++) {
        if (!UCX_BIT_GET(set, i)) {
            return i;
        }
    }
    return bits;
}

#define UCX_KERNEL_CMD(level, cmd, blocks, threads, shared_size, func, ...) \
    do { \
        switch (cmd) { \
        case UCX_PERF_CMD_PUT_SINGLE: \
            func<level, UCX_PERF_CMD_PUT_SINGLE><<<blocks, threads, shared_size>>>(__VA_ARGS__); \
            break; \
        case UCX_PERF_CMD_PUT_MULTI: \
            func<level, UCX_PERF_CMD_PUT_MULTI><<<blocks, threads, shared_size>>>(__VA_ARGS__); \
            break; \
        case UCX_PERF_CMD_PUT_PARTIAL: \
            func<level, UCX_PERF_CMD_PUT_PARTIAL><<<blocks, threads, shared_size>>>(__VA_ARGS__); \
            break; \
        default: \
            ucs_error("Unsupported cmd: %d", cmd); \
            break; \
        } \
    } while (0)

#define UCX_KERNEL_DISPATCH(perf, func, ...) \
    do { \
        ucs_device_level_t _level = perf.params.device_level; \
        ucx_perf_cmd_t _cmd       = perf.params.command; \
        unsigned _blocks          = perf.params.device_block_count; \
        unsigned _threads         = perf.params.device_thread_count; \
        size_t _shared_size       = _threads * perf.params.max_outstanding * \
                                    sizeof(ucp_device_request_t); \
        switch (_level) { \
        case UCS_DEVICE_LEVEL_THREAD: \
            UCX_KERNEL_CMD(UCS_DEVICE_LEVEL_THREAD, _cmd, _blocks, _threads,\
                           _shared_size, func, __VA_ARGS__); \
            break; \
        case UCS_DEVICE_LEVEL_WARP: \
            UCX_KERNEL_CMD(UCS_DEVICE_LEVEL_WARP, _cmd, _blocks, _threads,\
                           _shared_size, func, __VA_ARGS__); \
            break; \
        case UCS_DEVICE_LEVEL_BLOCK: \
            UCX_KERNEL_CMD(UCS_DEVICE_LEVEL_BLOCK, _cmd, _blocks, _threads,\
                           _shared_size, func, __VA_ARGS__); \
            break; \
        case UCS_DEVICE_LEVEL_GRID: \
            UCX_KERNEL_CMD(UCS_DEVICE_LEVEL_GRID, _cmd, _blocks, _threads,\
                           _shared_size, func, __VA_ARGS__); \
            break; \
        default: \
            ucs_error("Unsupported level: %d", _level); \
            break; \
        } \
    } while (0)

class ucx_perf_cuda_test_runner {
public:
    ucx_perf_cuda_test_runner(ucx_perf_context_t &perf) : m_perf(perf)
    {
        init_ctx();

        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->max_iters          = perf.max_iter;
        m_cpu_ctx->completed_iters    = 0;
        m_cpu_ctx->report_interval_ns = (perf.report_interval == ULONG_MAX) ?
                                        ULONG_MAX :
                                        ucs_time_to_nsec(perf.report_interval) / 100;
        m_cpu_ctx->status             = UCS_ERR_NOT_IMPLEMENTED;
    }

    ~ucx_perf_cuda_test_runner()
    {
        CUDA_CALL_WARN(cudaFreeHost, m_cpu_ctx);
    }

    void wait_for_kernel()
    {
        size_t msg_length                 = ucx_perf_get_message_size(&m_perf.params);
        ucx_perf_counter_t last_completed = 0;
        ucx_perf_counter_t completed      = m_cpu_ctx->completed_iters;
        unsigned thread_count             = m_perf.params.device_thread_count;
        while (true) {
            ucx_perf_counter_t delta = completed - last_completed;
            if (delta > 0) {
                // TODO: calculate latency percentile on kernel
                ucx_perf_update(&m_perf, delta, delta * thread_count, msg_length);
            } else if (completed >= m_perf.max_iter) {
                break;
            }
            last_completed = completed;
            completed      = m_cpu_ctx->completed_iters;
            // TODO: use cuStreamWaitValue64 if available
            usleep(100);
        }
    }

protected:
    ucx_perf_context_t &m_perf;
    ucx_perf_cuda_context *m_cpu_ctx;
    ucx_perf_cuda_context *m_gpu_ctx;

private:
    void init_ctx()
    {
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaHostAlloc, &m_cpu_ctx,
                  sizeof(ucx_perf_cuda_context), cudaHostAllocMapped);
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaHostGetDevicePointer,
                  &m_gpu_ctx, m_cpu_ctx, 0);
    }
};


template<typename Runner> ucs_status_t
ucx_perf_cuda_dispatch(ucx_perf_context_t *perf)
{
    Runner runner(*perf);
    if ((perf->params.command == UCX_PERF_CMD_PUT_MULTI) ||
        (perf->params.command == UCX_PERF_CMD_PUT_SINGLE) ||
        (perf->params.command == UCX_PERF_CMD_PUT_PARTIAL)) {
        if (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) {
            return runner.run_pingpong();
        } else if (perf->params.test_type == UCX_PERF_TEST_TYPE_STREAM_UNI) {
            return runner.run_stream_uni();
        }
    }
    return UCS_ERR_INVALID_PARAM;
}

extern ucx_perf_device_dispatcher_t ucx_perf_cuda_dispatcher;

#endif /* CUDA_KERNEL_CUH_ */
