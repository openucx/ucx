/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_kernel.cuh"
#include "curand_kernel.h"
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>
#include <tools/perf/lib/ucp_tests.h>

#include <vector>
#include <stdexcept>


class ucp_perf_cuda_request_manager {
public:
    using size_type = uint8_t;

    __device__
    ucp_perf_cuda_request_manager(const ucx_perf_cuda_context &ctx,
                                  ucp_device_request_t *requests,
                                  curandState *rand_state)
        : m_size(ctx.max_outstanding),
          m_fc_window(ctx.device_fc_window),
          m_reqs_count(ucs_div_round_up(m_size, m_fc_window)),
          m_channel_mode(ctx.channel_mode),
          m_pending_count(0),
          m_requests(requests),
          m_pending_map(0),
          m_rand_state(rand_state)
    {
        assert(m_size <= CAPACITY);
        for (size_type i = 0; i < m_reqs_count; ++i) {
            m_pending[i] = 0;
        }
    }

    template<ucs_device_level_t level, bool fc, size_type reuse>
    __device__ inline ucs_status_t progress_one(size_type &index)
    {
        for (size_type i = 0; i < m_reqs_count; i++) {
            if (!reuse && !UCS_BIT_GET(m_pending_map, i)) {
                continue;
            }
            ucs_status_t status = ucp_device_progress_req<level>(&m_requests[i]);
            if (status == UCS_INPROGRESS) {
                continue;
            }
            index = i;
            if constexpr (fc || !reuse) {
                m_pending_count -= (m_pending[index] - reuse);
                m_pending[index] = reuse;
                m_pending_map   &= ~UCS_BIT(index);
            }
            return status;
        }
        return UCS_INPROGRESS;
    }

    template<ucs_device_level_t level, bool fc>
    __device__ inline ucs_status_t get_request(ucp_device_request_t *&req,
                                               ucp_device_flags_t &flags)
    {
        size_type index;
        if (m_pending_count == m_size) {
            ucs_status_t status;
            do {
                status = progress_one<level, fc, 1>(index);
            } while (status == UCS_INPROGRESS);

            if (ucs_unlikely(status != UCS_OK)) {
                ucs_device_error("progress failed: %d", status);
                return status;
            }
        } else {
            index = __ffs(~m_pending_map) - 1;
            ++m_pending[index];
            ++m_pending_count;
        }

        if (fc && (m_pending_count < m_size) && (m_pending[index] < m_fc_window)) {
            req   = nullptr;
            flags = static_cast<ucp_device_flags_t>(0);
        } else {
            req            = &m_requests[index];
            m_pending_map |= UCS_BIT(index);
        }
        return UCS_OK;
    }

    __device__ inline size_type get_pending_count() const
    {
        return m_pending_count;
    }

    template<ucs_device_level_t level>
    __device__ inline unsigned get_channel_id() const
    {
        switch (m_channel_mode) {
        case UCX_PERF_CHANNEL_MODE_SINGLE:
            return 0;
        case UCX_PERF_CHANNEL_MODE_RANDOM:
            return curand(m_rand_state) % (gridDim.x * blockDim.x);
        case UCX_PERF_CHANNEL_MODE_PER_THREAD:
        default:
            return ucx_perf_cuda_thread_index<level>(threadIdx.x +
                                                     blockIdx.x * blockDim.x);
        }
    }

private:
    static const size_type CAPACITY = 32;

    const size_type               m_size;
    const size_type               m_fc_window;
    const size_type               m_reqs_count;
    const ucx_perf_channel_mode_t m_channel_mode;
    size_type                     m_pending_count;
    ucp_device_request_t          *m_requests;
    uint32_t                      m_pending_map;
    uint8_t                       m_pending[CAPACITY];
    curandState                   *m_rand_state;
};

struct ucp_perf_cuda_params {
    ucp_device_mem_list_handle_h mem_list;
    size_t                       length;
    unsigned                     *indices;
    size_t                       *local_offsets;
    size_t                       *remote_offsets;
    size_t                       *lengths;
    uint64_t                     *counter_send;
    uint64_t                     *counter_recv;
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
        CUDA_CALL_WARN(cudaFree, m_params.local_offsets);
        CUDA_CALL_WARN(cudaFree, m_params.remote_offsets);
        CUDA_CALL_WARN(cudaFree, m_params.lengths);
    }

    const ucp_perf_cuda_params &get_params() const { return m_params; }

private:
    static bool has_counter(const ucx_perf_context_t &perf)
    {
        return (perf.params.command != UCX_PERF_CMD_PUT_SINGLE);
    }

    void init_mem_list(const ucx_perf_context_t &perf)
    {
        size_t data_count = perf.params.msg_size_cnt;
        size_t count      = data_count + (has_counter(perf) ? 1 : 0);
        size_t offset     = 0;
        ucp_device_mem_list_elem_t elems[count];

        for (size_t i = 0; i < data_count; ++i) {
            elems[i].field_mask  = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                                   UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                                   UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
                                   UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                                   UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
            elems[i].memh        = perf.ucp.send_memh;
            elems[i].rkey        = perf.ucp.rkey;
            elems[i].local_addr  = UCS_PTR_BYTE_OFFSET(perf.send_buffer, offset);
            elems[i].remote_addr = perf.ucp.remote_addr + offset;
            elems[i].length      = perf.params.msg_size_list[i];
            offset              += elems[i].length;
        }

        if (has_counter(perf)) {
            elems[data_count].field_mask  = UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                                            UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                                            UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;
            elems[data_count].rkey        = perf.ucp.rkey;
            elems[data_count].remote_addr = perf.ucp.remote_addr + offset;
            elems[data_count].length      = ONESIDED_SIGNAL_SIZE;
        }

        ucp_device_mem_list_params_t params;
        params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                              UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
        params.element_size = sizeof(ucp_device_mem_list_elem_t);
        params.num_elements = count;
        params.elements     = elems;

        ucs_status_t status;
        ucs_time_t deadline = ucs_get_time() + ucs_time_from_sec(60.0);
        do {
            if (ucs_get_time() > deadline) {
                ucs_warn("timeout creating device memory list");
                deadline = ULONG_MAX;
            }

            ucp_worker_progress(perf.ucp.worker);
            status = ucp_device_mem_list_create(perf.ucp.ep, &params,
                                                &m_params.mem_list);
        } while (status == UCS_ERR_NOT_CONNECTED);

        if (status != UCS_OK) {
            throw std::runtime_error("Failed to create memory list");
        }
    }

    void init_elements(const ucx_perf_context_t &perf)
    {
        size_t data_count = perf.params.msg_size_cnt;
        size_t count      = data_count + (has_counter(perf) ? 1 : 0);

        std::vector<unsigned> indices(count);
        std::vector<size_t> local_offsets(count, 0);
        std::vector<size_t> remote_offsets(count, 0);
        std::vector<size_t> lengths(count);

        for (unsigned i = 0; i < data_count; ++i) {
            indices[i] = i;
            lengths[i] = perf.params.msg_size_list[i];
        }

        if (has_counter(perf)) {
            indices[data_count] = data_count;
            lengths[data_count] = ONESIDED_SIGNAL_SIZE;
        }

        m_params.indices        = device_vector(indices);
        m_params.local_offsets  = device_vector(local_offsets);
        m_params.remote_offsets = device_vector(remote_offsets);
        m_params.lengths        = device_vector(lengths);
    }

    void init_counters(const ucx_perf_context_t &perf)
    {
        m_params.length       = ucx_perf_get_message_size(&perf.params);
        m_params.counter_send = ucx_perf_cuda_get_sn(perf.send_buffer,
                                                     m_params.length);
        m_params.counter_recv = ucx_perf_cuda_get_sn(perf.recv_buffer,
                                                     m_params.length);
    }

    template<typename T>
    T* device_vector(const std::vector<T> &src)
    {
        size_t size = src.size() * sizeof(T);
        T *dst;
        CUDA_CALL(, UCS_LOG_LEVEL_FATAL, cudaMalloc, &dst, size);
        CUDA_CALL_ERR(cudaMemcpy, dst, src.data(), size, cudaMemcpyHostToDevice);
        return dst;
    }

    ucp_perf_cuda_params m_params;
};

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_send_async(const ucp_perf_cuda_params &params,
                         ucx_perf_counter_t idx, ucp_device_request_t *req,
                         unsigned channel_id,
                         ucp_device_flags_t flags = UCP_DEVICE_FLAG_NODELAY)
{
    switch (cmd) {
    case UCX_PERF_CMD_PUT_SINGLE:
        *params.counter_send = idx + 1;
        return ucp_device_put_single<level>(params.mem_list, params.indices[0],
                                            0, 0,
                                            params.length + ONESIDED_SIGNAL_SIZE,
                                            channel_id, flags, req);
    case UCX_PERF_CMD_PUT_MULTI:
        return ucp_device_put_multi<level>(params.mem_list, 1, channel_id,
                                           flags, req);
    case UCX_PERF_CMD_PUT_PARTIAL: {
        unsigned counter_index = params.mem_list->mem_list_length - 1;
        return ucp_device_put_multi_partial<level>(params.mem_list,
                                                   params.indices,
                                                   counter_index,
                                                   params.local_offsets,
                                                   params.remote_offsets,
                                                   params.lengths,
                                                   counter_index, 1, 0,
                                                   channel_id, flags, req);
        }
    }

    return UCS_ERR_INVALID_PARAM;
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_send_sync(ucp_perf_cuda_params &params, ucx_perf_counter_t idx,
                        ucp_device_request_t *req, unsigned channel_id)
{
    ucs_status_t status = ucp_perf_cuda_send_async<level, cmd>(
                                params, idx, req, channel_id,
                                UCP_DEVICE_FLAG_NODELAY);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    if (nullptr == req) {
        return UCS_OK;
    }

    do {
        status = ucp_device_progress_req<level>(req);
    } while (status == UCS_INPROGRESS);

    return status;
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd, bool fc>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_put_bw_iter(const ucp_perf_cuda_params &params,
                          ucp_perf_cuda_request_manager &req_mgr,
                          ucx_perf_cuda_context &ctx, ucx_perf_counter_t idx)
{
    ucp_device_flags_t flags = UCP_DEVICE_FLAG_NODELAY;
    ucp_device_request_t *req;

    ucs_status_t status = req_mgr.get_request<level, fc>(req, flags);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    unsigned channel_id = req_mgr.get_channel_id<level>();
    return ucp_perf_cuda_send_async<level, cmd>(params, idx, req, channel_id, flags);
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd, bool fc>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_put_bw_kernel_impl(ucx_perf_cuda_context &ctx,
                                 const ucp_perf_cuda_params &params,
                                 ucp_perf_cuda_request_manager &req_mgr)
{
    ucx_perf_counter_t max_iters = ctx.max_iters;
    ucx_perf_cuda_reporter reporter(ctx);
    ucs_status_t status;

    for (ucx_perf_counter_t idx = 0; idx < (max_iters - 1); idx++) {
        status = ucp_perf_cuda_put_bw_iter<level, cmd, fc>(params, req_mgr, ctx,
                                                           idx);
        if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
            ucs_device_error("send failed: %d", status);
            return status;
        }

        reporter.update_report(idx + 1);
        __syncthreads();
    }

    /* Last iteration */
    status = ucp_perf_cuda_put_bw_iter<level, cmd, false>(params, req_mgr, ctx,
                                                          max_iters - 1);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_device_error("final send failed: %d", status);
        return status;
    }

    while (req_mgr.get_pending_count() > 0) {
        uint8_t index;
        status = req_mgr.progress_one<level, fc, 0>(index);
        if (UCS_STATUS_IS_ERR(status)) {
            ucs_device_error("final progress failed: %d", status);
            return status;
        }
    }

    reporter.update_report(max_iters);
    return UCS_OK;
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
__global__ void
ucp_perf_cuda_put_bw_kernel(ucx_perf_cuda_context &ctx,
                            ucp_perf_cuda_params params)
{
    extern __shared__ ucp_device_request_t shared_requests[];
    unsigned thread_index      = ucx_perf_cuda_thread_index<level>(threadIdx.x);
    unsigned reqs_count        = ucs_div_round_up(ctx.max_outstanding,
                                                  ctx.device_fc_window);
    unsigned global_thread_id  = ucx_perf_cuda_thread_index<level>(
        thread_index + blockIdx.x * blockDim.x);
    ucp_device_request_t *reqs = &shared_requests[reqs_count * thread_index];
    curandState rand_state;

    if (ctx.channel_mode == UCX_PERF_CHANNEL_MODE_RANDOM) {
        curand_init(ctx.channel_rand_seed, global_thread_id, 0, &rand_state);
    }

    ucp_perf_cuda_request_manager req_mgr(ctx, reqs, &rand_state);

    if (ctx.device_fc_window > 1) {
        ctx.status = ucp_perf_cuda_put_bw_kernel_impl<level, cmd, true>(
                                                        ctx, params, req_mgr);
    } else {
        ctx.status = ucp_perf_cuda_put_bw_kernel_impl<level, cmd, false>(
                                                        ctx, params, req_mgr);
    }
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
__global__ void
ucp_perf_cuda_put_latency_kernel(ucx_perf_cuda_context &ctx,
                                 ucp_perf_cuda_params params, bool is_sender)
{
    extern __shared__ ucp_device_request_t shared_requests[];
    ucx_perf_counter_t max_iters = ctx.max_iters;
    ucs_status_t status          = UCS_OK;
    unsigned thread_index        = ucx_perf_cuda_thread_index<level>(threadIdx.x);
    unsigned global_thread_id    = ucx_perf_cuda_thread_index<level>(
        thread_index + blockIdx.x * blockDim.x);
    ucp_device_request_t *req    = &shared_requests[thread_index];
    curandState rand_state;

    if (ctx.channel_mode == UCX_PERF_CHANNEL_MODE_RANDOM) {
        curand_init(ctx.channel_rand_seed, global_thread_id, 0, &rand_state);
    }

    ucp_perf_cuda_request_manager req_mgr(ctx, req, &rand_state);
    ucx_perf_cuda_reporter reporter(ctx);

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        unsigned channel_id = req_mgr.get_channel_id<level>();
        if (is_sender) {
            status = ucp_perf_cuda_send_sync<level, cmd>(params, idx, req,
                                                         channel_id);
            if (status != UCS_OK) {
                ucs_device_error("sender send failed: %d", status);
                break;
            }
            ucx_perf_cuda_wait_sn(params.counter_recv, idx + 1);
        } else {
            ucx_perf_cuda_wait_sn(params.counter_recv, idx + 1);
            status = ucp_perf_cuda_send_sync<level, cmd>(params, idx, req,
                                                         channel_id);
            if (status != UCS_OK) {
                ucs_device_error("receiver send failed: %d", status);
                break;
            }
        }

        reporter.update_report(idx + 1);
    }

    ctx.status = status;
}

__global__ void
ucp_perf_cuda_wait_bw_kernel(ucx_perf_cuda_context &ctx,
                             ucp_perf_cuda_params params)
{
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
        size_t length = ucx_perf_get_message_size(&m_perf.params) + ONESIDED_SIGNAL_SIZE;

        m_perf.send_allocator->memset(m_perf.send_buffer, 0, length);
        m_perf.recv_allocator->memset(m_perf.recv_buffer, 0, length);
    }

    ucs_status_t run_pingpong()
    {
        unsigned my_index = rte_call(&m_perf, group_index);
        ucp_perf_cuda_params_handler params_handler(m_perf);

        ucp_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        UCX_PERF_KERNEL_DISPATCH(m_perf, ucp_perf_cuda_put_latency_kernel,
                                 *m_gpu_ctx, params_handler.get_params(),
                                 my_index);
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
            UCX_PERF_KERNEL_DISPATCH(m_perf, ucp_perf_cuda_put_bw_kernel,
                                     *m_gpu_ctx, params_handler.get_params());
            CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);
            wait_for_kernel();
        } else if (my_index == 0) {
            ucp_perf_cuda_wait_bw_kernel<<<1, 1>>>(
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
