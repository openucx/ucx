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


typedef unsigned long long cuda_perf_time_t;

struct cuda_perf_context {
    unsigned           max_outstanding;
    ucx_perf_counter_t max_iters;
    cuda_perf_time_t   report_interval_ns;
    ucx_perf_counter_t completed_iters;
};

UCS_F_DEVICE cuda_perf_time_t cuda_perf_get_time_ns()
{
    cuda_perf_time_t globaltimer;
    /* 64-bit GPU global nanosecond timer */
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

UCS_F_DEVICE void cuda_update_perf_report(cuda_perf_context &ctx,
                                          ucx_perf_counter_t completed,
                                          ucx_perf_counter_t max_iters,
                                          cuda_perf_time_t &last_report_time)
{
    if (threadIdx.x == 0) {
        cuda_perf_time_t current_time = cuda_perf_get_time_ns();
        if (((current_time - last_report_time) >= ctx.report_interval_ns) ||
            (completed >= max_iters)) {
            ctx.completed_iters = completed;
            last_report_time    = current_time;
            __threadfence();
        }
    }
}

template <typename Base>
class cuda_ucx_test_runner: public Base {
public:
    using Base::m_perf;

    cuda_ucx_test_runner(ucx_perf_context_t &perf) : Base(perf)
    {
        ucs_status_t status;
        status = cuda_device_mem_create(&m_device_mem, sizeof(cuda_perf_context));
        if (status != UCS_OK) {
            ucs_fatal("cuda_device_mem_create() failed: %s",
                      ucs_status_string(status));
        }

        m_cpu_ctx = static_cast<cuda_perf_context *>(m_device_mem.cpu_ptr);
        m_gpu_ctx = static_cast<cuda_perf_context *>(m_device_mem.gpu_ptr);

        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->max_iters          = perf.params.max_iter;
        m_cpu_ctx->report_interval_ns = perf.params.report_interval *
                                        UCS_NSEC_PER_SEC;
        m_cpu_ctx->completed_iters    = 0;

        m_poll_interval = perf.params.report_interval / 10000;
    }

    ~cuda_ucx_test_runner() noexcept
    {
        cuda_device_mem_destroy(&m_device_mem);
    }

    cuda_perf_context &gpu_ctx() const { return *m_gpu_ctx; }

    // TODO: remove once real GDAKI is used
    void send_signal(size_t length)
    {

        ucs_memory_type_t mem_type = m_perf.params.send_mem_type;
        Base::write_sn(m_perf.send_buffer, mem_type, length,
                       m_perf.params.max_iter, m_perf.ucp.self_send_rkey);

        ucs_status_ptr_t request;
        ucp_request_param_t param = {0};
        request = ucp_put_nbx(m_perf.ucp.ep, m_perf.send_buffer, length,
                              m_perf.ucp.remote_addr, m_perf.ucp.rkey, &param);
        Base::request_wait(request, mem_type, "write_sn");
        request = ucp_ep_flush_nbx(m_perf.ucp.self_ep, &param);
        Base::request_wait(request, mem_type, "flush write_sn");
    }

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
    cuda_device_mem_t m_device_mem;
    cuda_perf_context *m_cpu_ctx;
    cuda_perf_context *m_gpu_ctx;
    double            m_poll_interval;
};


template<typename Runner> ucs_status_t
cuda_ucx_perf_dispatch(ucx_perf_context_t *perf)
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

extern ucx_perf_device_dispatcher_t cuda_dispatcher;

#endif /* LIBPERF_CUDA_H_ */
