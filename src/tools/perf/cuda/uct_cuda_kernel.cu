/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_kernel.cuh"

#include <uct/api/v2/uct_v2.h>
#include <uct/api/device/uct_device_types.h>
#include <uct/api/device/uct_device_impl.h>
#include <tools/perf/lib/uct_tests.h>

#include <memory>

template<uct_device_level_t level>
__global__ void
uct_perf_cuda_put_multi_bw_kernel(ucx_perf_cuda_context &ctx,
                                  uct_device_ep_h device_ep,
                                  const uct_device_mem_element_t *mem_elem,
                                  const void *address, uint64_t remote_address,
                                  size_t length, ucx_perf_counter_t *sn_ptr)
{
    ucx_perf_cuda_time_t last_report_time = ucx_perf_cuda_get_time_ns();
    ucx_perf_counter_t max_iters          = ctx.max_iters;
    __shared__ uct_device_completion_t comp;

    uct_device_completion_init(&comp);

    for (ucx_perf_counter_t idx = 0; idx < max_iters; idx++) {
        while (comp.count >= ctx.max_outstanding) {
            uct_device_ep_progress<level>(device_ep);
        }

        if (threadIdx.x == 0) {
            comp.count++;
            *sn_ptr = idx + 1;
        }

        ctx.status = uct_device_ep_put_single<level>(device_ep, mem_elem,
                                                     address, remote_address,
                                                     length, 0, &comp);
        if (ctx.status != UCS_OK) {
            break;
        }

        ucx_perf_cuda_update_report(ctx, idx + 1, max_iters, last_report_time);
        __syncthreads();
    }

    while (comp.count > 0) {
        uct_device_ep_progress<level>(device_ep);
    }
}

template<uct_device_level_t level>
__global__ void
uct_perf_cuda_put_multi_latency_kernel(ucx_perf_cuda_context &ctx, bool is_sender)
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

class uct_perf_cuda_test_runner:
    public ucx_perf_cuda_test_runner<uct_perf_test_runner_base<uint64_t>> {
public:
    using mem_elem_t = ucx_perf_cuda_mem<uct_device_mem_element_t>;

    uct_perf_cuda_test_runner(ucx_perf_context_t &perf) :
        ucx_perf_cuda_test_runner<uct_perf_test_runner_base<uint64_t>>(perf)
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

        uct_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        uct_perf_cuda_put_multi_latency_kernel
            <UCT_DEVICE_LEVEL_BLOCK><<<1, thread_count>>>(gpu_ctx(), my_index);
        CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);

        wait_for_kernel(length);
        ucx_perf_get_time(&m_perf);
        uct_perf_barrier(&m_perf);
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        size_t length     = ucx_perf_get_message_size(&m_perf.params);
        unsigned my_index = rte_call(&m_perf, group_index);

        uct_perf_barrier(&m_perf);
        ucx_perf_test_start_clock(&m_perf);

        if (my_index == 1) {
            unsigned group_size   = rte_call(&m_perf, group_size);
            unsigned peer_index   = rte_peer_index(group_size, my_index);
            unsigned thread_count = m_perf.params.device_thread_count;
            uint64_t remote_addr  = m_perf.uct.peers[peer_index].remote_addr;
            uct_rkey_t rkey       = m_perf.uct.peers[peer_index].rkey.rkey;
            uct_ep_h ep           = m_perf.uct.peers[peer_index].ep;
            uct_mem_h memh        = m_perf.uct.send_mem.memh;
            uct_iface_h iface     = m_perf.uct.iface;
            void *address         = m_perf.send_buffer;
            psn_t *ptr            = sn_ptr(m_perf.send_buffer, length);

            ucs_status_t status;
            uct_device_ep_h device_ep;
            status = uct_ep_get_device_ep(ep, &device_ep);
            if (status != UCS_OK) {
                return status;
            }

            auto mem_elem = create_mem_elem(iface, memh, rkey, status);
            if (!mem_elem) {
                return UCS_ERR_NO_MEMORY;
            }

            ucs_diag("before uct_perf_cuda_put_multi_bw_kernel");
            uct_perf_cuda_put_multi_bw_kernel
                <UCT_DEVICE_LEVEL_BLOCK><<<1, thread_count>>>(
                gpu_ctx(), device_ep, mem_elem.get()->ptr(), address,
                remote_addr, length, ptr);
            CUDA_CALL_RET(UCS_ERR_NO_DEVICE, cudaGetLastError);

            wait_for_kernel(length);
        } else if (my_index == 0) {
            wait_for_sn(length);
        }

        ucx_perf_get_time(&m_perf);
        uct_perf_barrier(&m_perf);
        return UCS_OK;
    }

private:
    static std::unique_ptr<mem_elem_t>
    create_mem_elem(uct_iface_h iface, uct_mem_h memh, uct_rkey_t rkey,
                    ucs_status_t &status)
    {
        uct_iface_attr_v2_t iface_attr;

        iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
        status                = uct_iface_query_v2(iface, &iface_attr);
        if (status != UCS_OK) {
            return nullptr;
        }

        auto mem_elem = 
            std::make_unique<mem_elem_t>(iface_attr.device_mem_element_size);
        status        = uct_iface_mem_element_pack(iface, memh, rkey,
                                                   mem_elem.get()->ptr());
        if (status != UCS_OK) {
            return nullptr;
        }

        return mem_elem;
    }
};

UCS_STATIC_INIT {
    ucx_perf_cuda_dispatcher.uct_dispatch = ucx_perf_cuda_dispatch<uct_perf_cuda_test_runner>;
}
