/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_CUDA_KERNEL_COMMON_CUH_
#define UCP_CUDA_KERNEL_COMMON_CUH_

#include "cuda_kernel.cuh"
#include "curand_kernel.h"
#include "ucp_cuda_impl.h"

#include <ucp/api/device/ucp_device_impl.h>

class ucp_perf_cuda_request_manager {
public:
    using size_type = uint8_t;

    __device__
    ucp_perf_cuda_request_manager(const ucx_perf_cuda_context_t &ctx,
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

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_send_async(const ucp_perf_cuda_params_t &params,
                         ucx_perf_counter_t idx, ucp_device_request_t *req,
                         unsigned channel_id,
                         ucp_device_flags_t flags = UCP_DEVICE_FLAG_NODELAY)
{
    switch (cmd) {
    case UCX_PERF_CMD_PUT:
        *params.counter_send = idx + 1;
        return ucp_device_put<level>(params.local_mem_list, 0, 0,
                                     params.remote_mem_list, 0, 0,
                                     params.length + ONESIDED_SIGNAL_SIZE,
                                     channel_id, flags, req);
    default:
        return UCS_ERR_UNSUPPORTED;
    }
}

template<ucs_device_level_t level, ucx_perf_cmd_t cmd>
UCS_F_DEVICE ucs_status_t
ucp_perf_cuda_send_sync(ucp_perf_cuda_params_t &params, ucx_perf_counter_t idx,
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
ucp_perf_cuda_put_bw_iter(const ucp_perf_cuda_params_t &params,
                          ucp_perf_cuda_request_manager &req_mgr,
                          ucx_perf_cuda_context_t &ctx, ucx_perf_counter_t idx)
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
ucp_perf_cuda_put_bw_kernel_impl(ucx_perf_cuda_context_t &ctx,
                                 const ucp_perf_cuda_params_t &params,
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

#endif /* UCP_CUDA_KERNEL_COMMON_CUH_ */
