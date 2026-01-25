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
    ucx_perf_channel_mode_t channel_mode;
    unsigned long long      channel_rand_seed;
    unsigned                max_outstanding;
    unsigned                device_fc_window;
    ucx_perf_counter_t      max_iters;
    ucx_perf_cuda_time_t    report_interval_ns;
    ucx_perf_counter_t      completed_iters;
    ucs_status_t            status;
};

UCS_F_DEVICE ucx_perf_cuda_time_t ucx_perf_cuda_get_time_ns()
{
    ucx_perf_cuda_time_t globaltimer;
    /* 64-bit GPU global nanosecond timer */
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

class ucx_perf_cuda_reporter {
public:
    /* Number of updates per report interval */
    static const unsigned UPDATES_PER_INTERVAL = 5;

    __device__
    ucx_perf_cuda_reporter(ucx_perf_cuda_context &ctx) :
        m_ctx(ctx),
        m_max_iters(ctx.max_iters),
        m_next_report_iter(1),
        m_last_completed(0),
        m_last_report_time(ucx_perf_cuda_get_time_ns()),
        m_report_interval_ns(ctx.report_interval_ns / UPDATES_PER_INTERVAL)
    {
    }

    __device__ inline void
    update_report(ucx_perf_counter_t completed)
    {
        if ((blockIdx.x == 0) && (threadIdx.x == 0) && ucs_unlikely(completed >= m_next_report_iter)) {
            assert(completed - m_last_completed > 0);
            ucx_perf_cuda_time_t cur_time  = ucx_perf_cuda_get_time_ns();
            ucx_perf_cuda_time_t iter_time = (cur_time - m_last_report_time) /
                                             (completed - m_last_completed);
            assert(iter_time > 0);
            m_last_completed               = completed;
            m_last_report_time             = cur_time;
            m_ctx.completed_iters          = completed * gridDim.x;
            __threadfence_system();

            m_next_report_iter = ucs_min(completed + (m_report_interval_ns / iter_time),
                                         m_max_iters);
        }
    }

private:
    ucx_perf_cuda_context &m_ctx;
    ucx_perf_counter_t    m_max_iters;
    ucx_perf_counter_t    m_next_report_iter;
    ucx_perf_counter_t    m_last_completed;
    ucx_perf_cuda_time_t  m_last_report_time;
    ucx_perf_cuda_time_t  m_report_interval_ns;
};

static UCS_F_ALWAYS_INLINE uint64_t *
ucx_perf_cuda_get_sn(const void *address, size_t length)
{
    return (uint64_t*)UCS_PTR_BYTE_OFFSET(address, length);
}

UCS_F_DEVICE void ucx_perf_cuda_wait_sn(const uint64_t *sn, uint64_t value)
{
    if (threadIdx.x == 0) {
        /* TODO support host memory */
        while (ucs_device_atomic64_read(sn) < value);
    }
    __syncthreads();
}

template<ucs_device_level_t level>
__host__ UCS_F_DEVICE unsigned ucx_perf_cuda_thread_index(size_t tid)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD: return tid;
    case UCS_DEVICE_LEVEL_WARP:   return tid / UCS_DEVICE_NUM_THREADS_IN_WARP;
    default:                      return 0;
    }
}

#define UCX_PERF_THREAD_INDEX_SET(_level, _tid, _outval) \
    (_outval) = ucx_perf_cuda_thread_index<_level>(_tid)

#define UCX_PERF_SWITCH_CMD(_cmd, _func, ...) \
    switch (_cmd) { \
    case UCX_PERF_CMD_PUT_SINGLE: \
        _func(UCX_PERF_CMD_PUT_SINGLE, __VA_ARGS__); \
        break; \
    case UCX_PERF_CMD_PUT_MULTI: \
        _func(UCX_PERF_CMD_PUT_MULTI, __VA_ARGS__); \
        break; \
    case UCX_PERF_CMD_PUT_PARTIAL: \
        _func(UCX_PERF_CMD_PUT_PARTIAL, __VA_ARGS__); \
        break; \
    default: \
        ucs_error("Unsupported cmd: %d", _cmd); \
        break; \
    }

#define UCX_PERF_SWITCH_LEVEL(_level, _func, ...) \
    switch (_level) { \
        case UCS_DEVICE_LEVEL_THREAD: \
            _func(UCS_DEVICE_LEVEL_THREAD, __VA_ARGS__); \
            break; \
        case UCS_DEVICE_LEVEL_WARP: \
            _func(UCS_DEVICE_LEVEL_WARP, __VA_ARGS__); \
            break; \
        case UCS_DEVICE_LEVEL_BLOCK: \
        case UCS_DEVICE_LEVEL_GRID: \
        default: \
            ucs_error("Unsupported level: %d", _level); \
            break; \
    }

#define UCX_PERF_KERNEL_DISPATCH_CMD_LEVEL(_cmd, _level, _perf, _kernel, ...) \
    do { \
        unsigned _blocks     = _perf.params.device_block_count; \
        unsigned _threads    = _perf.params.device_thread_count; \
        unsigned _reqs_count = ucs_div_round_up(_perf.params.max_outstanding, \
                                                _perf.params.device_fc_window); \
        size_t _shared_size  = _reqs_count * sizeof(ucp_device_request_t) * \
                               ucx_perf_cuda_thread_index<_level>(_threads); \
        _kernel<_level, _cmd><<<_blocks, _threads, _shared_size>>>(__VA_ARGS__); \
    } while (0)

#define UCX_PERF_KERNEL_DISPATCH_CMD(_level, _perf, _kernel, ...) \
    UCX_PERF_SWITCH_CMD(_perf.params.command, UCX_PERF_KERNEL_DISPATCH_CMD_LEVEL, \
                        _level, _perf, _kernel, __VA_ARGS__);

#define UCX_PERF_KERNEL_DISPATCH(_perf, _kernel, ...) \
    UCX_PERF_SWITCH_LEVEL(_perf.params.device_level, UCX_PERF_KERNEL_DISPATCH_CMD, \
                          _perf, _kernel, __VA_ARGS__);


class ucx_perf_cuda_test_runner {
public:
    ucx_perf_cuda_test_runner(ucx_perf_context_t &perf) : m_perf(perf)
    {
        init_ctx();

        m_cpu_ctx->channel_mode       = perf.params.device_channel_mode;
        m_cpu_ctx->channel_rand_seed  = perf.params.channel_rand_seed;
        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->device_fc_window   = perf.params.device_fc_window;
        m_cpu_ctx->max_iters          = perf.max_iter;
        m_cpu_ctx->completed_iters    = 0;
        m_cpu_ctx->report_interval_ns = (perf.report_interval == ULONG_MAX) ?
                                        ULONG_MAX :
                                        ucs_time_to_nsec(perf.report_interval);
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
        unsigned block_count              = m_perf.params.device_block_count;
        ucs_device_level_t level          = m_perf.params.device_level;
        unsigned msgs_per_iter;
        UCX_PERF_SWITCH_LEVEL(level, UCX_PERF_THREAD_INDEX_SET, thread_count,
                              msgs_per_iter);

        while (true) {
            ucx_perf_counter_t delta = completed - last_completed;
            if (delta > 0) {
                // TODO: calculate latency percentile on kernel
                ucx_perf_update(&m_perf, delta, delta * msgs_per_iter, msg_length);
            } else if (completed >= (m_perf.max_iter * block_count)) {
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
