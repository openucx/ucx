/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef LIBPERF_CUDA_H_
#define LIBPERF_CUDA_H_

#include <tools/perf/lib/libperf_int.h>
#include "cuda_device_mem.h"

#include <cuda_runtime.h>

#include <functional>


typedef unsigned long long ucx_perf_cuda_time_t;

struct ucx_perf_cuda_context {
    unsigned             max_outstanding;
    ucx_perf_counter_t   max_iters;
    ucx_perf_cuda_time_t report_interval_ns;
    ucx_perf_counter_t   completed_iters;
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

template <typename Base>
class ucx_perf_cuda_test_runner: public Base {
public:
    using Base::m_perf;

    ucx_perf_cuda_test_runner(ucx_perf_context_t &perf) : Base(perf)
    {
        ucs_status_t status;
        status = ucx_perf_cuda_device_mem_create(&m_device_mem,
                                                 sizeof(m_device_mem));
        if (status != UCS_OK) {
            ucs_fatal("ucx_perf_cuda_device_mem_create() failed: %s",
                      ucs_status_string(status));
        }

        m_cpu_ctx = static_cast<ucx_perf_cuda_context*>(m_device_mem.cpu_ptr);
        m_gpu_ctx = static_cast<ucx_perf_cuda_context*>(m_device_mem.gpu_ptr);

        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->max_iters          = perf.params.max_iter;
        m_cpu_ctx->report_interval_ns = perf.params.report_interval *
                                        UCS_NSEC_PER_SEC;
        m_cpu_ctx->completed_iters    = 0;
        m_poll_interval               = perf.params.report_interval / 10000;
    }

    ~ucx_perf_cuda_test_runner() noexcept
    {
        ucx_perf_cuda_device_mem_destroy(&m_device_mem);
    }

    ucx_perf_cuda_context &gpu_ctx() const { return *m_gpu_ctx; }

    void wait_for(std::function<bool()> cond)
    {
        while (!cond()) {
            // TODO: use cuStreamWaitValue64 if available
            usleep(m_poll_interval);
        }
    }

    void wait_for_kernel(size_t length)
    {
        ucx_perf_counter_t last_completed = 0;
        wait_for([this, length, &last_completed]() {
            ucx_perf_counter_t completed = m_cpu_ctx->completed_iters;
            ucx_perf_counter_t delta     = completed - last_completed;
            if (delta > 0) {
                // TODO: calculate latency percentile on kernel
                ucx_perf_update_multi(&m_perf, delta, delta * length, 0);
            }
            last_completed = completed;
            return completed == m_perf.params.max_iter;
        });
    }

private:
    ucx_perf_cuda_device_mem_t m_device_mem;
    ucx_perf_cuda_context      *m_cpu_ctx;
    ucx_perf_cuda_context      *m_gpu_ctx;
    double                     m_poll_interval;
};


template<typename Runner> ucs_status_t
ucx_perf_cuda_dispatch(ucx_perf_context_t *perf)
{
    Runner runner(*perf);
    if (perf->params.command == UCX_PERF_CMD_PUT_MULTI) {
        if (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) {
            return runner.run_pingpong();
        } else if (perf->params.test_type == UCX_PERF_TEST_TYPE_STREAM_UNI) {
            return runner.run_stream_uni();
        }
    }
    return UCS_ERR_INVALID_PARAM;
}

extern ucx_perf_device_dispatcher_t ucx_perf_cuda_dispatcher;

#endif /* LIBPERF_CUDA_H_ */
