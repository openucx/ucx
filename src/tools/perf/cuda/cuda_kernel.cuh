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

class ucx_perf_cuda_test_runner {
public:
    ucx_perf_cuda_test_runner(ucx_perf_context_t &perf) : m_perf(perf)
    {
        ucs_status_t status = init_ctx();
        if (status != UCS_OK) {
            ucs_fatal("failed to allocate device memory context: %s",
                      ucs_status_string(status));
        }

        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->max_iters          = perf.max_iter;
        m_cpu_ctx->completed_iters    = 0;
        if (perf.report_interval == ULONG_MAX) {
            m_cpu_ctx->report_interval_ns = ULONG_MAX;
        } else {
            m_cpu_ctx->report_interval_ns = ucs_time_to_nsec(
                                                    perf.report_interval) /
                                            100;
        }
    }

    ~ucx_perf_cuda_test_runner()
    {
        destroy_ctx();
    }

    ucx_perf_cuda_context &gpu_ctx() const { return *m_gpu_ctx; }

    void wait_for_kernel(size_t msg_length)
    {
        ucx_perf_counter_t last_completed = 0;
        ucx_perf_counter_t completed      = m_cpu_ctx->completed_iters;
        while (1) {
            ucx_perf_counter_t delta = completed - last_completed;
            if (delta > 0) {
                // TODO: calculate latency percentile on kernel
                ucx_perf_update(&m_perf, delta, msg_length);
            } else if (completed >= m_perf.max_iter) {
                break;
            }
            last_completed = completed;
            completed      = m_cpu_ctx->completed_iters;
            usleep(100);
        }
    }

protected:
    ucx_perf_context_t &m_perf;

private:
    ucs_status_t init_ctx()
    {
        CUDA_CALL(UCS_ERR_NO_MEMORY, cudaHostAlloc, &m_cpu_ctx,
                  sizeof(ucx_perf_cuda_context), cudaHostAllocMapped);

        cudaError_t err = cudaHostGetDevicePointer(&m_gpu_ctx, m_cpu_ctx, 0);
        if (err != cudaSuccess) {
            ucs_error("cudaHostGetDevicePointer() failed: %s",
                      cudaGetErrorString(err));
            cudaFreeHost(m_cpu_ctx);
            return UCS_ERR_IO_ERROR;
        }

        return UCS_OK;
    }

    void destroy_ctx()
    {
        CUDA_CALL_HANDLER(ucs_warn, , cudaFreeHost, m_cpu_ctx);
    }

    ucx_perf_cuda_context *m_cpu_ctx;
    ucx_perf_cuda_context *m_gpu_ctx;
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

#endif /* CUDA_KERNEL_CUH_ */
