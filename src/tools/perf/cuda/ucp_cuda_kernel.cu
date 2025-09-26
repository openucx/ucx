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

#include <vector>
#include <stdexcept>


class ucp_perf_cuda_request_manager {
public:
    __device__
    ucp_perf_cuda_request_manager(size_t size, ucp_device_request_t *requests)
        : m_size(size), m_requests(&requests[size * threadIdx.x])
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
        size_t index = ucx_bitset_ffns(m_pending, m_size, 0);
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

    size_t               m_size;
    ucp_device_request_t *m_requests;
    uint8_t              m_pending[UCX_BITSET_SIZE(CAPACITY)];
};

struct ucp_perf_cuda_params {
    ucp_device_mem_list_handle_h mem_list;
    size_t                       length;
    unsigned                     *indices;
    void                         **addresses;
    uint64_t                     *remote_addresses;
    size_t                       *lengths;
    uint64_t                     counter_remote;
    uint64_t                     *counter_send;
    uint64_t                     *counter_recv;
    ucp_device_flags_t           flags;
};

class ucp_perf_cuda_params_handler {
public:
    ucp_perf_cuda_params_handler(const ucx_perf_context_t &perf)
    {
        init_mem_list(perf);
        init_elements(perf);
        init_counters(perf);
    }

    ~ucp_perf_cuda_params_handler()
    {
        ucp_device_mem_list_release(m_params.mem_list);
        CUDA_CALL_WARN(cudaFree, m_params.indices);
        CUDA_CALL_WARN(cudaFree, m_params.addresses);
        CUDA_CALL_WARN(cudaFree, m_params.remote_addresses);
        CUDA_CALL_WARN(cudaFree, m_params.lengths);
    }

    const ucp_perf_cuda_params &get_params() const { return m_params; }

private:
    void init_mem_list(const ucx_perf_context_t &perf)
    {
        /* +1 for the counter */
        size_t count = perf.params.msg_size_cnt + 1;
        ucp_device_mem_list_elem_t elems[count];
        for (size_t i = 0; i < count; ++i) {
            elems[i].field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                                  UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
            elems[i].memh       = perf.ucp.send_memh;
            elems[i].rkey       = perf.ucp.rkey;
        }

        ucp_device_mem_list_params_t params;
        params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
        params.element_size = sizeof(ucp_device_mem_list_elem_t);
        params.num_elements = count;
        params.elements     = elems;

        ucs_status_t status = ucp_device_mem_list_create(perf.ucp.ep, &params,
                                                         &m_params.mem_list);
        if (status != UCS_OK) {
            throw std::runtime_error("Failed to create memory list");
        }
    }

    void init_elements(const ucx_perf_context_t &perf)
    {
        /* +1 for the counter */
        size_t count = perf.params.msg_size_cnt + 1;

        std::vector<unsigned> indices(count);
        std::vector<void*> addresses(count);
        std::vector<uint64_t> remote_addresses(count);
        std::vector<size_t> lengths(count);
        for (unsigned i = 0, offset = 0; i < count; ++i) {
            indices[i]          = i;
            addresses[i]        = (char *)perf.send_buffer + offset;
            remote_addresses[i] = perf.ucp.remote_addr + offset;
            lengths[i]          = (i == count - 1) ? ONESIDED_SIGNAL_SIZE :
                                                     perf.params.msg_size_list[i];
            offset             += lengths[i];
        }

        device_clone(&m_params.indices, indices.data(), count);
        device_clone(&m_params.addresses, addresses.data(), count);
        device_clone(&m_params.remote_addresses, remote_addresses.data(), count);
        device_clone(&m_params.lengths, lengths.data(), count);
    }

    void init_counters(const ucx_perf_context_t &perf)
    {
        m_params.length         = ucx_perf_get_message_size(&perf.params);
        m_params.counter_remote = (uint64_t)ucx_perf_cuda_get_sn(
                                        (void*)perf.ucp.remote_addr,
                                        m_params.length);
        m_params.counter_send   = ucx_perf_cuda_get_sn(perf.send_buffer,
                                                       m_params.length);
        m_params.counter_recv   = ucx_perf_cuda_get_sn(perf.recv_buffer,
                                                       m_params.length);
        m_params.flags          = UCP_DEVICE_FLAG_NODELAY;
    }

    template<typename T>
    void device_clone(T **dst, const T *src, size_t count)
    {
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaMalloc, dst, count * sizeof(T));
        CUDA_CALL_ERR(cudaMemcpy, *dst, src, count * sizeof(T),
                      cudaMemcpyHostToDevice);
    }

    ucp_perf_cuda_params m_params;
};

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_send_nbx(ucp_perf_cuda_params &params, ucx_perf_counter_t idx,
                       ucp_device_request_t &req)
{
    switch (cmd) {
    case UCX_PERF_CMD_PUT_SINGLE:
        /* TODO: Change to ucp_device_counter_write */
        *params.counter_send = idx + 1;
        return ucp_device_put_single<level>(params.mem_list, params.indices[0],
                                            params.addresses[0],
                                            params.remote_addresses[0],
                                            params.length + ONESIDED_SIGNAL_SIZE,
                                            params.flags, &req);
    case UCX_PERF_CMD_PUT_MULTI:
        return ucp_device_put_multi<level>(params.mem_list, params.addresses,
                                           params.remote_addresses,
                                           params.lengths, 1,
                                           params.counter_remote, params.flags,
                                           &req);
    case UCX_PERF_CMD_PUT_PARTIAL:{
        unsigned counter_index = params.mem_list->mem_list_length - 1;
        return ucp_device_put_multi_partial<level>(params.mem_list,
                                                   params.indices,
                                                   counter_index,
                                                   params.addresses,
                                                   params.remote_addresses,
                                                   params.lengths,
                                                   counter_index, 1,
                                                   params.counter_remote,
                                                   params.flags, &req);
        }
    }

    return UCS_ERR_INVALID_PARAM;
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_send_sync(ucp_perf_cuda_params &params, ucx_perf_counter_t idx,
                        ucp_device_request_t &req)
{
    ucs_status_t status = ucp_perf_cuda_send_nbx<level, cmd>(params, idx, req);
    if (status != UCS_OK) {
        return status;
    }

    do {
        status = ucp_device_progress_req<level>(&req);
    } while (status == UCS_INPROGRESS);

    return status;
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
__global__ void
ucp_perf_cuda_put_multi_bw_kernel(ucx_perf_cuda_context &ctx,
                                  ucp_perf_cuda_params params)
{
    // TODO: use thread-local memory once we support it
    extern __shared__ ucp_device_request_t requests[];
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;
    ucs_status_t status                   = UCS_OK;
    ucp_perf_cuda_request_manager request_mgr(ctx.max_outstanding, requests);

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        while (request_mgr.get_pending_count() >= ctx.max_outstanding) {
            status = request_mgr.progress<level>(1);
            if (UCS_STATUS_IS_ERR(status)) {
                ucs_device_error("progress failed: %d", status);
                goto out;
            }
        }

        ucp_device_request_t &req = request_mgr.get_request();
        status = ucp_perf_cuda_send_nbx<level, cmd>(params, idx, req);
        if (status != UCS_OK) {
            ucs_device_error("send failed: %d", status);
            goto out;
        }

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }

    while (request_mgr.get_pending_count() > 0) {
        status = request_mgr.progress<level>(max_iters);
        if (UCS_STATUS_IS_ERR(status)) {
            ucs_device_error("final progress failed: %d", status);
            goto out;
        }
    }

out:
    ctx.status = status;
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
__global__ void
ucp_perf_cuda_put_multi_latency_kernel(ucx_perf_cuda_context &ctx,
                                       ucp_perf_cuda_params params,
                                       bool is_sender)
{
    // TODO: use thread-local memory once we support it
    extern __shared__ ucp_device_request_t requests[];
    ucp_device_request_t &req             = requests[threadIdx.x];
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;
    ucs_status_t status                   = UCS_OK;

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        if (is_sender) {
            status = ucp_perf_cuda_send_sync<level, cmd>(params, idx, req);
            if (status != UCS_OK) {
                ucs_device_error("sender send failed: %d", status);
                break;
            }
            ucx_perf_cuda_wait_sn(params.counter_recv, idx + 1);
        } else {
            ucx_perf_cuda_wait_sn(params.counter_recv, idx + 1);
            status = ucp_perf_cuda_send_sync<level, cmd>(params, idx, req);
            if (status != UCS_OK) {
                ucs_device_error("receiver send failed: %d", status);
                break;
            }
        }

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }

    ctx.status = status;
}

__global__ void
ucp_perf_cuda_wait_multi_bw_kernel(ucx_perf_cuda_context &ctx,
                                   ucp_perf_cuda_params params)
{
    // TODO: we can use ucp_device_counter_read, but it adds latency
    volatile uint64_t *sn = params.counter_recv;
    while (*sn < ctx.max_iters) {
        __nanosleep(100000); // 100us
    }

    ctx.status = UCS_OK;
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
        unsigned my_index = rte_call(&m_perf, group_index);
        ucp_perf_cuda_params_handler params_handler(m_perf);

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        UCX_KERNEL_DISPATCH(m_perf, ucp_perf_cuda_put_multi_latency_kernel,
                            *m_gpu_ctx, params_handler.get_params(), my_index);
        CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);

        wait_for_kernel();

        CUDA_CALL_RET(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);

        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return m_cpu_ctx->status;
    }

    ucs_status_t run_stream_uni()
    {
        unsigned my_index = rte_call(&m_perf, group_index);
        ucp_perf_cuda_params_handler params_handler(m_perf);

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        if (my_index == 1) {
            UCX_KERNEL_DISPATCH(m_perf, ucp_perf_cuda_put_multi_bw_kernel,
                                *m_gpu_ctx, params_handler.get_params());
            CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);
            wait_for_kernel();
        } else if (my_index == 0) {
            ucp_perf_cuda_wait_multi_bw_kernel<<<1, 1>>>(
                    *m_gpu_ctx, params_handler.get_params());
        }

        CUDA_CALL_RET(UCS_ERR_IO_ERROR, cudaDeviceSynchronize);
        ucx_perf_get_time(&m_perf);
        ucp_perf_barrier(&m_perf);
        return m_cpu_ctx->status;
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
