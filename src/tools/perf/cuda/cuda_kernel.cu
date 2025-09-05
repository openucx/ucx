/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>
#include <ucs/sys/compiler_def.h>
#include "cuda_device_mem.h"
#include <tools/perf/lib/ucp_tests.h>

#include <functional>

#include <cuda_runtime.h>

typedef unsigned long long cuda_perf_time_t;

struct cuda_perf_context {
    unsigned           max_outstanding;
    ucx_perf_counter_t max_iters;
    cuda_perf_time_t   report_interval_ns;
    ucx_perf_counter_t completed_iters;
};

UCS_F_DEVICE cuda_perf_time_t cuda_perf_get_time_ns()
{
    cuda_perf_time_t globaltimer;
    /* 64-bit GPU global nanosecond timer */
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

UCS_F_DEVICE void cuda_update_perf_report(cuda_perf_context &ctx,
                                          ucx_perf_counter_t completed,
                                          ucx_perf_counter_t max_iters,
                                          cuda_perf_time_t &last_report_time)
{
    if (threadIdx.x == 0) {
        cuda_perf_time_t current_time = cuda_perf_get_time_ns();
        if (((current_time - last_report_time) >= ctx.report_interval_ns) ||
            (completed >= max_iters)) {
            ctx.completed_iters = completed;
            last_report_time    = current_time;
            __threadfence();
        }
    }
}

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

template <typename Base>
class cuda_ucx_test_runner: public Base {
public:
    using Base::m_perf;

    cuda_ucx_test_runner(ucx_perf_context_t &perf) : Base(perf)
    {
        ucs_status_t status;
        status = cuda_device_mem_create(&m_device_mem, sizeof(cuda_perf_context));
        if (status != UCS_OK) {
            ucs_fatal("cuda_device_mem_create() failed: %s",
                      ucs_status_string(status));
        }

        m_cpu_ctx = static_cast<cuda_perf_context *>(m_device_mem.cpu_ptr);
        m_gpu_ctx = static_cast<cuda_perf_context *>(m_device_mem.gpu_ptr);

        m_cpu_ctx->max_outstanding    = perf.params.max_outstanding;
        m_cpu_ctx->max_iters          = perf.params.max_iter;
        m_cpu_ctx->report_interval_ns = perf.params.report_interval *
                                        UCS_NSEC_PER_SEC;
        m_cpu_ctx->completed_iters    = 0;

        m_poll_interval = perf.params.report_interval / 10000;
    }

    ~cuda_ucx_test_runner() noexcept
    {
        cuda_device_mem_destroy(&m_device_mem);
    }

    // TODO: remove once real GDAKI is used
    void send_signal(size_t length)
    {

        ucs_memory_type_t mem_type = m_perf.params.send_mem_type;
        Base::write_sn(m_perf.send_buffer, mem_type, length,
                       m_perf.params.max_iter, m_perf.ucp.self_send_rkey);

        ucs_status_ptr_t request;
        ucp_request_param_t param = {0};
        request = ucp_put_nbx(m_perf.ucp.ep, m_perf.send_buffer, length,
                              m_perf.ucp.remote_addr, m_perf.ucp.rkey, &param);
        Base::request_wait(request, mem_type, "write_sn");
        request = ucp_ep_flush_nbx(m_perf.ucp.self_ep, &param);
        Base::request_wait(request, mem_type, "flush write_sn");
    }

    void wait_for(std::function<bool()> cond)
    {
        while (!cond()) {
            // TODO: use cuStreamWaitValue64 if available
            usleep(m_poll_interval);
        }
    }

    void wait_for_kernel(size_t length)
    {
        ucx_perf_counter_t last_completed = 0;
        wait_for([this, length, &last_completed]() {
            ucx_perf_counter_t completed = m_cpu_ctx->completed_iters;
            ucx_perf_counter_t delta     = completed - last_completed;
            if (delta > 0) {
                // TODO: calculate latency percentile on kernel
                ucx_perf_update_multi(&m_perf, delta, delta * length, 0);
            }
            last_completed = completed;
            return completed == m_perf.params.max_iter;
        });
    }

    cuda_perf_context &gpu_ctx() const { return *m_gpu_ctx; }

private:
    cuda_device_mem_t m_device_mem;
    cuda_perf_context *m_cpu_ctx;
    cuda_perf_context *m_gpu_ctx;
    double            m_poll_interval;
};

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

static ucs_status_t
ucp_perf_cuda_dispatch(ucx_perf_context_t *perf)
{
    cuda_ucp_test_runner runner(*perf);
    if (perf->params.command == UCX_PERF_CMD_PUT_MULTI) {
        if (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) {
            return runner.run_pingpong();
        } else if (perf->params.test_type == UCX_PERF_TEST_TYPE_STREAM_UNI) {
            return runner.run_stream_uni();
        }
    }
    return UCS_ERR_INVALID_PARAM;
}

UCS_STATIC_INIT {
    static ucx_perf_device_dispatcher_t cuda_dispatcher = {
        .ucp_dispatch = ucp_perf_cuda_dispatch,
    };

    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = &cuda_dispatcher;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = &cuda_dispatcher;
}

UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;
}
