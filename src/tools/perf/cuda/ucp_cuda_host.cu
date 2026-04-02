/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Pure host dispatcher: C++ runner + registration only.
 * Kernel launches are in ucp_cuda_kernel_*.cu (each has kernel + host entry point).
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_kernel.cuh"
#include "ucp_cuda_impl.h"

class ucp_perf_cuda_test_runner : public ucx_perf_cuda_test_runner {
public:
    ucp_perf_cuda_test_runner(ucx_perf_context_t &perf) :
        ucx_perf_cuda_test_runner(perf)
    {
        size_t length = ucx_perf_get_message_size(&m_perf.params) + ONESIDED_SIGNAL_SIZE;

        m_perf.send_allocator->memset(m_perf.send_buffer, 0, length);
        m_perf.recv_allocator->memset(m_perf.recv_buffer, 0, length);
    }

    ucs_status_t run_pingpong()
    {
        return ucp_perf_cuda_run_pingpong(&m_perf, m_cpu_ctx, m_gpu_ctx);
    }

    ucs_status_t run_stream_uni()
    {
        return ucp_perf_cuda_run_stream_uni(&m_perf, m_cpu_ctx, m_gpu_ctx);
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
