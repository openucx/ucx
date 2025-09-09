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
        __nanosleep(1000000); // 1ms

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
        __nanosleep(1000000); // 1ms

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }
}

class ucp_perf_cuda_test_runner:
    public ucx_perf_cuda_test_runner<ucp_perf_test_runner_base<uint64_t>> {
public:
    using psn_t = uint64_t;

    ucp_perf_cuda_test_runner(ucx_perf_context_t &perf) :
        ucx_perf_cuda_test_runner<ucp_perf_test_runner_base<uint64_t>>(perf)
    {
        size_t length = ucx_perf_get_message_size(&m_perf.params);
        ucs_assert(length >= sizeof(psn_t));

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

            // TODO: remove once real GDAKI is used
            send_signal(length);
        } else if (my_index == 0) {
            wait_for_sn(length);
        }

        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return UCS_OK;
    }

private:
    // TODO: remove once real GDAKI is used
    void send_signal(size_t length)
    {
        ucs_memory_type_t mem_type = m_perf.params.send_mem_type;
        write_sn(m_perf.send_buffer, mem_type, length, m_perf.params.max_iter,
                 m_perf.ucp.self_send_rkey);

        ucs_status_ptr_t request;
        ucp_request_param_t param = {0};
        request = ucp_put_nbx(m_perf.ucp.ep, m_perf.send_buffer, length,
                              m_perf.ucp.remote_addr, m_perf.ucp.rkey, &param);
        request_wait(request, mem_type, "write_sn");
        request = ucp_ep_flush_nbx(m_perf.ucp.self_ep, &param);
        request_wait(request, mem_type, "flush write_sn");
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
