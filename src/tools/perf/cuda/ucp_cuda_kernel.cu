/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_kernel.cuh"
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>
#include <tools/perf/lib/ucp_tests.h>

#include <memory>


class ucp_perf_cuda_request_manager {
public:
    __device__ ucp_perf_cuda_request_manager(size_t size) : m_size(size)
    {
        assert(m_size <= CAPACITY);
        for (size_t i = 0; i < m_size; ++i) {
            UCX_BIT_RESET(m_pending, i);
        }
    }

    template<ucs_device_level_t level>
    __device__ ucs_status_t progress(size_t max_completed)
    {
        ucs_status_t status = UCS_OK;
        size_t completed    = 0;

        for (size_t i = 0; i < m_size; i++) {
            if (UCX_BIT_GET(m_pending, i)) {
                status = ucp_device_progress_req<level>(&m_requests[i]);
                if (status == UCS_INPROGRESS) {
                    continue;
                }
                UCX_BIT_RESET(m_pending, i);
                if (status != UCS_OK) {
                    break;
                }
                if (++completed >= max_completed) {
                    break;
                }
            }
        }

        return status;
    }

    __device__ ucp_device_request_t &get_request()
    {
        assert(get_pending_count() < m_size);
        size_t index = ucx_bitset_ffs(m_pending, m_size, 0);
        UCX_BIT_SET(m_pending, index);
        return m_requests[index];
    }

    __device__ size_t get_pending_count() const
    {
        return ucx_bitset_popcount(m_pending, m_size);
    }

private:
    /* TODO: make it runtime configurable / alloc on host */
    static const size_t CAPACITY = 128;

    size_t m_size;
    ucp_device_request_t m_requests[CAPACITY];
    uint8_t m_pending[UCX_BITSET_SIZE(CAPACITY)];
};

template<ucs_device_level_t level>
__global__ void
ucp_perf_cuda_put_multi_bw_kernel(ucx_perf_cuda_context &ctx,
                                  ucp_device_mem_list_handle_h mem_list,
                                  unsigned mem_list_index, const void *address,
                                  uint64_t remote_address, size_t length)
{
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;
    uint64_t *sn = ucx_perf_cuda_get_sn(address, length);
    ucp_perf_cuda_request_manager request_mgr(ctx.max_outstanding);
    ucs_status_t status;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        while (request_mgr.get_pending_count() >= ctx.max_outstanding) {
            status = request_mgr.progress<level>(1);
            if (status != UCS_OK) {
                break;
            }
        }

        *sn                       = idx + 1;
        ucp_device_request_t &req = request_mgr.get_request();
        status = ucp_device_put_single<level>(mem_list, mem_list_index, address,
                                              remote_address, length, 0, &req);
        if (status != UCS_OK) {
            break;
        }

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }

    while (request_mgr.get_pending_count() > 0) {
        status = request_mgr.progress<level>(max_iters);
        if (status != UCS_OK) {
            break;
        }
    }

    ctx.status = status;
}

__global__ void ucp_perf_cuda_wait_multi_bw_kernel(ucx_perf_cuda_context &ctx,
                                                   const void *address,
                                                   size_t length)
{
    volatile uint64_t *sn = ucx_perf_cuda_get_sn(address, length);
    while (*sn < ctx.max_iters) {
        __nanosleep(100000); // 100us
    }

    ctx.status = UCS_OK;
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t ucp_perf_cuda_put_single(
        ucp_device_mem_list_handle_h mem_list, unsigned mem_list_index,
        const void *address, uint64_t remote_address, size_t length)
{
    ucp_device_request_t req;
    ucs_status_t status;

    status = ucp_device_put_single<level>(mem_list, mem_list_index, address,
                                          remote_address, length,
                                          UCP_DEVICE_FLAG_NODELAY, &req);
    if (status != UCS_OK) {
        return status;
    }

    do {
        status = ucp_device_progress_req<level>(&req);
    } while (status == UCS_INPROGRESS);

    return status;
}

template<ucs_device_level_t level>
__global__ void ucp_perf_cuda_put_multi_latency_kernel(
        ucx_perf_cuda_context &ctx, ucp_device_mem_list_handle_h mem_list,
        unsigned mem_list_index, const void *address, uint64_t remote_address,
        size_t length, const void *recv_address, bool is_sender)
{
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;
    uint64_t *sn        = ucx_perf_cuda_get_sn(address, length);
    uint64_t *recv_sn   = ucx_perf_cuda_get_sn(recv_address, length);
    ucs_status_t status = UCS_OK;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        if (is_sender) {
            *sn    = idx + 1;
            status = ucp_perf_cuda_put_single<level>(mem_list, mem_list_index,
                                                     address, remote_address,
                                                     length);
            if (status != UCS_OK) {
                break;
            }
            ucx_perf_cuda_wait_sn(recv_sn, idx + 1);
        } else {
            ucx_perf_cuda_wait_sn(recv_sn, idx + 1);
            *sn    = idx + 1;
            status = ucp_perf_cuda_put_single<level>(mem_list, mem_list_index,
                                                     address, remote_address,
                                                     length);
            if (status != UCS_OK) {
                break;
            }
        }

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }

    ctx.status = status;
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

        unique_mem_list_ptr handle = create_mem_list();
        if (!handle) {
            return UCS_ERR_NO_MEMORY;
        }

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        ucp_perf_cuda_put_multi_latency_kernel<UCS_DEVICE_LEVEL_THREAD>
                <<<1, thread_count>>>(gpu_ctx(), handle.get(), 0,
                                      m_perf.send_buffer,
                                      m_perf.ucp.remote_addr, length,
                                      m_perf.recv_buffer, my_index);
        CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);

        wait_for_kernel(length);

        CUDA_CALL_RET(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);

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
            unique_mem_list_ptr handle = create_mem_list();
            if (!handle) {
                return UCS_ERR_NO_MEMORY;
            }

            unsigned thread_count = m_perf.params.device_thread_count;
            ucp_perf_cuda_put_multi_bw_kernel<UCS_DEVICE_LEVEL_THREAD>
                    <<<1, thread_count>>>(gpu_ctx(), handle.get(), 0,
                                          m_perf.send_buffer,
                                          m_perf.ucp.remote_addr, length);
            CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);
            wait_for_kernel(length);
        } else if (my_index == 0) {
            ucp_perf_cuda_wait_multi_bw_kernel<<<1, 1>>>(gpu_ctx(),
                                                         m_perf.recv_buffer,
                                                         length);
        }

        CUDA_CALL_RET(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);
        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return UCS_OK;
    }

private:
    using unique_mem_list_ptr =
            std::unique_ptr<struct ucp_device_mem_list_handle,
                            decltype(&ucp_device_mem_list_release)>;

    unique_mem_list_ptr create_mem_list() const
    {
        ucp_device_mem_list_elem_t elem;
        elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
        elem.memh       = m_perf.ucp.send_memh;
        elem.rkey       = m_perf.ucp.rkey;

        ucp_device_mem_list_params_t params;
        params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
        params.element_size = sizeof(ucp_device_mem_list_elem_t);
        params.num_elements = 1;
        params.elements     = &elem;

        ucp_device_mem_list_handle_h mem_list;
        ucs_status_t status = ucp_device_mem_list_create(m_perf.ucp.ep, &params,
                                                         &mem_list);
        if (status != UCS_OK) {
            return unique_mem_list_ptr(nullptr, nullptr);
        }

        return unique_mem_list_ptr(mem_list, ucp_device_mem_list_release);
    }
};

ucx_perf_device_dispatcher_t ucx_perf_cuda_dispatcher;

UCS_STATIC_INIT
{
    ucx_perf_cuda_dispatcher.ucp_dispatch =
            ucx_perf_cuda_dispatch<ucp_perf_cuda_test_runner>;

    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA] =
            &ucx_perf_cuda_dispatcher;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] =
            &ucx_perf_cuda_dispatcher;
}

UCS_STATIC_CLEANUP
{
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;
}
