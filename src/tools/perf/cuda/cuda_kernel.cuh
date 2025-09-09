/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_KERNEL_CUH_
#define CUDA_KERNEL_CUH_

#include "cuda_common.h"
#include <tools/perf/lib/libperf_int.h>

#include <cuda_runtime.h>
#include <cuda.h>


typedef unsigned long long ucx_perf_cuda_time_t;

struct ucx_perf_cuda_context {
    unsigned             max_outstanding;
    ucx_perf_counter_t   max_iters;
    ucx_perf_cuda_time_t report_interval_ns;
    ucx_perf_counter_t   completed_iters;
    ucs_status_t         status;
};

template <typename T>
class ucx_perf_cuda_mem {
public:
    ucx_perf_cuda_mem(size_t size = sizeof(T))
    {
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaMalloc, &m_ptr, size);
    }

    ~ucx_perf_cuda_mem()
    {
        CUDA_CALL_WARN(cudaFree, m_ptr);
    }

    T *ptr() const { return m_ptr; }

private:
    T *m_ptr;
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
    using psn_t = typename Base::psn_t;
    using Base::m_perf;

    ucx_perf_cuda_test_runner(ucx_perf_context_t &perf) : Base(perf)
    {
        init_ctx();

        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->max_iters          = perf.params.max_iter;
        m_cpu_ctx->report_interval_ns = perf.params.report_interval *
                                        UCS_NSEC_PER_SEC;
        m_cpu_ctx->completed_iters    = 0;
    }

    ~ucx_perf_cuda_test_runner()
    {
        CU_CALL_WARN(cuStreamDestroy, m_stream);
        CUDA_CALL_WARN(cudaFreeHost, m_cpu_ctx);
    }

    ucx_perf_cuda_context &gpu_ctx() const { return *m_gpu_ctx; }

    UCS_F_ALWAYS_INLINE psn_t get_sn(const psn_t *gpu_ptr, const psn_t *cpu_ptr)
    {
        if (cpu_ptr != nullptr) {
            return *cpu_ptr;
        }

        unsigned my_index          = rte_call(&m_perf, group_index);
        ucs_memory_type_t mem_type = my_index ? m_perf.params.send_mem_type :
                                                m_perf.params.recv_mem_type;
        auto allocator             = my_index ? m_perf.send_allocator :
                                                m_perf.recv_allocator;
        return Base::get_sn(gpu_ptr, mem_type, allocator);
    }

    psn_t wait_sn_geq(const psn_t *gpu_ptr, const psn_t *cpu_ptr, psn_t value)
    {
        psn_t sn = get_sn(gpu_ptr, cpu_ptr);
        if (sn >= value) {
            return sn;
        }

        CU_CALL_ERR(cuStreamWaitValue64, m_stream, (CUdeviceptr)gpu_ptr, value,
                    CU_STREAM_WAIT_VALUE_GEQ);
        CU_CALL_ERR(cuStreamSynchronize, m_stream);
        return get_sn(gpu_ptr, cpu_ptr);
    }

    void wait_for_kernel(size_t length)
    {
        psn_t last_completed = 0;
        while (last_completed < m_perf.params.max_iter) {
            psn_t completed = wait_sn_geq(&m_gpu_ctx->completed_iters,
                                          &m_cpu_ctx->completed_iters,
                                          last_completed);
            psn_t delta     = completed - last_completed;
            if (delta > 0) {
                // TODO: calculate latency percentile on kernel
                ucx_perf_update_multi(&m_perf, delta, delta * length);
            }
            last_completed = completed;
        }
    }

    void wait_for_sn(size_t length)
    {
        const psn_t *ptr = Base::sn_ptr(m_perf.recv_buffer, length);
        while (wait_sn_geq(ptr, nullptr, m_perf.params.max_iter)
               < m_perf.params.max_iter);
    }

private:
    void init_ctx()
    {
        CU_CALL(, UCS_LOG_LEVEL_FATAL, cuStreamCreate, &m_stream,
                CU_STREAM_NON_BLOCKING);
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaHostAlloc, &m_cpu_ctx,
                  sizeof(ucx_perf_cuda_context), cudaHostAllocMapped);
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaHostGetDevicePointer,
                  &m_gpu_ctx, m_cpu_ctx, 0);
    }

    ucx_perf_cuda_context *m_cpu_ctx;
    ucx_perf_cuda_context *m_gpu_ctx;
    CUstream              m_stream;
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
