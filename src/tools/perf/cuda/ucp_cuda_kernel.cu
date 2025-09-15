/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_kernel.cuh"
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>
#include <tools/perf/lib/ucp_tests.h>


template<ucp_device_level_t level>
__global__ void
ucp_perf_cuda_put_multi_bw_kernel(ucx_perf_cuda_context &ctx)
{
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        // TODO: replace with actual put multi call
        __nanosleep(100000); // 100us

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }
}

template<ucp_device_level_t level>
__global__ void
ucp_perf_cuda_put_multi_latency_kernel(ucx_perf_cuda_context &ctx, bool is_sender)
{
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        // TODO: replace with actual put multi call
        // TODO: wait for completion
        __nanosleep(100000); // 100us

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }
}

class ucp_perf_cuda_test_runner : public ucx_perf_cuda_test_runner {
public:
    ucp_perf_cuda_test_runner(ucx_perf_context_t &perf) :
        ucx_perf_cuda_test_runner(perf)
    {
        size_t length = ucx_perf_get_message_size(&m_perf.params);

        m_perf.send_allocator->memset(m_perf.send_buffer, 0, length);
        m_perf.recv_allocator->memset(m_perf.recv_buffer, 0, length);
    }

    ucs_status_t run_pingpong()
    {
        size_t length         = ucx_perf_get_message_size(&m_perf.params);
        unsigned thread_count = m_perf.params.device_thread_count;
        unsigned my_index     = rte_call(&m_perf, group_index);

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        ucp_perf_cuda_put_multi_latency_kernel
            <UCP_DEVICE_LEVEL_BLOCK><<<1, thread_count>>>(gpu_ctx(), my_index);
        CUDA_CALL(UCS_ERR_NO_DEVICE, cudaGetLastError);
        wait_for_kernel(length);

        CUDA_CALL(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);
        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        size_t length     = ucx_perf_get_message_size(&m_perf.params);
        unsigned my_index = rte_call(&m_perf, group_index);

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        if (my_index == 1) {
            unsigned thread_count = m_perf.params.device_thread_count;
            ucp_perf_cuda_put_multi_bw_kernel<UCP_DEVICE_LEVEL_BLOCK>
                                             <<<1, thread_count>>>(gpu_ctx());
            CUDA_CALL(UCS_ERR_NO_DEVICE, cudaGetLastError);
            wait_for_kernel(length);
        }
        // TODO run receiver kernel

        CUDA_CALL(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);
        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return UCS_OK;
    }
};

ucx_perf_device_dispatcher_t ucx_perf_cuda_dispatcher;

UCS_STATIC_INIT {
    ucx_perf_cuda_dispatcher.ucp_dispatch = ucx_perf_cuda_dispatch<ucp_perf_cuda_test_runner>;

    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = &ucx_perf_cuda_dispatcher;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = &ucx_perf_cuda_dispatcher;
}

UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;
}
