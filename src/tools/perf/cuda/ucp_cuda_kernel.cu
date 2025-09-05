/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf_cuda.h"
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>
#include <tools/perf/lib/ucp_tests.h>


template<ucp_device_level_t level>
__global__ void
cuda_ucp_put_multi_bw_kernel(cuda_perf_context &ctx)
{
    cuda_perf_time_t last_report_time = cuda_perf_get_time_ns();
    ucx_perf_counter_t max_iters      = ctx.max_iters;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        // TODO: replace with actual put multi call
        __nanosleep(1000000); // 1ms

        cuda_update_perf_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }
}

template<ucp_device_level_t level>
__global__ void
cuda_ucp_put_multi_latency_kernel(cuda_perf_context &ctx, bool is_sender)
{
    cuda_perf_time_t last_report_time = cuda_perf_get_time_ns();
    ucx_perf_counter_t max_iters      = ctx.max_iters;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        // TODO: replace with actual put multi call
        // TODO: wait for completion
        __nanosleep(1000000); // 1ms

        cuda_update_perf_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }
}

class cuda_ucp_test_runner:
    public cuda_ucx_test_runner<ucp_perf_test_runner_base<uint64_t>> {
public:
    using psn_t = uint64_t;

    cuda_ucp_test_runner(ucx_perf_context_t &perf) :
        cuda_ucx_test_runner<ucp_perf_test_runner_base<uint64_t>>(perf)
    {
    }

    ucs_status_t run_pingpong()
    {
        size_t length         = ucx_perf_get_message_size(&m_perf.params);
        unsigned my_index     = rte_call(&m_perf, group_index);
        unsigned thread_count = m_perf.params.device_thread_count;

        before();

        cuda_ucp_put_multi_latency_kernel<UCP_DEVICE_LEVEL_BLOCK>
                                         <<<1, thread_count>>>(gpu_ctx(), my_index);
        CUDA_CALL(UCS_ERR_NO_DEVICE, cudaGetLastError);

        wait_for_kernel(length);
        after();
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        size_t length     = ucx_perf_get_message_size(&m_perf.params);
        unsigned my_index = rte_call(&m_perf, group_index);

        before();

        if (my_index == 1) {
            unsigned thread_count = m_perf.params.device_thread_count;
            cuda_ucp_put_multi_bw_kernel<UCP_DEVICE_LEVEL_BLOCK>
                                        <<<1, thread_count>>>(gpu_ctx());
            CUDA_CALL(UCS_ERR_NO_DEVICE, cudaGetLastError);

            wait_for_kernel(length);

            // TODO: remove once real GDAKI is used
            send_signal(length);
        } else if (my_index == 0) {
            wait_for([this, length]() {
                psn_t sn = read_sn(m_perf.recv_buffer, length);
                return sn == m_perf.params.max_iter;
            });
        }

        after();
        return UCS_OK;
    }

private:
    void before()
    {
        size_t length = ucx_perf_get_message_size(&m_perf.params);
        ucs_assert(length >= sizeof(psn_t));

        m_perf.send_allocator->memset(m_perf.send_buffer, 0, length);
        m_perf.recv_allocator->memset(m_perf.recv_buffer, 0, length);

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);
    }

    void after()
    {
        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
    }
};

ucx_perf_device_dispatcher_t cuda_dispatcher;

UCS_STATIC_INIT {
    cuda_dispatcher.ucp_dispatch = cuda_ucx_perf_dispatch<cuda_ucp_test_runner>;

    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = &cuda_dispatcher;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = &cuda_dispatcher;
}

UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;
}
