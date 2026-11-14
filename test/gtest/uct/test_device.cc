/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string>

#include <uct/api/uct.h>
#include "uct_test.h"

extern "C" {
#include <uct/cuda/base/cuda_util.h>
}

#include "cuda/test_kernels.h"

/* Ensure a CUDA context is active before UCX enumerates resources so that
 * cuda_ipc sys_device is set properly. */
class test_device_cuda_ctx_guard {
public:
    static test_device_cuda_ctx_guard &instance()
    {
        static test_device_cuda_ctx_guard guard;
        return guard;
    }

    test_device_cuda_ctx_guard(const test_device_cuda_ctx_guard&) = delete;

    test_device_cuda_ctx_guard &
    operator=(const test_device_cuda_ctx_guard&) = delete;

private:
    CUdevice  m_dev;
    CUcontext m_ctx;
    bool      m_is_active;

    test_device_cuda_ctx_guard() : m_dev(0), m_ctx(NULL), m_is_active(false)
    {
        (void)UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));

        if (UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&m_dev, 0)) != UCS_OK) {
            return;
        }

        init();

        // In case of fork, cleanup the current context and re-initialize it
        // only in the parent process.
        pthread_atfork([]() { instance().cleanup(); },
                       []() { instance().init(); }, NULL);
    }

    ~test_device_cuda_ctx_guard()
    {
        cleanup();
    }

    void init()
    {
        if (m_is_active) {
            return;
        }

        if (UCT_CUDADRV_FUNC_LOG_ERR(cuDevicePrimaryCtxRetain(&m_ctx, m_dev)) !=
            UCS_OK) {
            return;
        }

        if (UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(m_ctx)) != UCS_OK) {
            (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(m_dev));
            return;
        }

        m_is_active = true;
    }

    void cleanup()
    {
        if (!m_is_active) {
            return;
        }

        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(m_dev));
        m_is_active = false;
    }
};

static auto &g_test_device_cuda_ctx_guard =
        test_device_cuda_ctx_guard::instance();

class test_device : public uct_test {
protected:
    void skip_if_no_cuda() const
    {
        if (!(m_sender->md_attr().reg_mem_types &
              UCS_BIT(UCS_MEMORY_TYPE_CUDA))) {
            UCS_TEST_SKIP_R("CUDA registration not supported");
        }
    }

    void skip_if_not_rc_gda() const
    {
        if (!has_transport("rc_gda")) {
            UCS_TEST_SKIP_R("rc_gda transport not supported");
        }
    }

    void
    device_put(uint64_t send_seed, uint64_t recv_seed, unsigned ep_index = 0)
    {
        constexpr size_t length = 1024;
        mapped_buffer sendbuf(length, send_seed, *m_sender, 0,
                              UCS_MEMORY_TYPE_CUDA);
        mapped_buffer recvbuf(length, recv_seed, *m_receiver, 0,
                              UCS_MEMORY_TYPE_CUDA);

        device_mem_elem_handle elem(*m_sender, sendbuf.memh(), recvbuf.rkey());

        uct_device_ep_h dev_ep;
        ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(ep_index), &dev_ep));
        ASSERT_UCS_OK(ucx_cuda::launch_uct_put(dev_ep, elem.dev_ptr(),
                                               elem.dev_ptr(), sendbuf.ptr(),
                                               (uintptr_t)recvbuf.ptr(),
                                               length));

        recvbuf.pattern_check(send_seed);
    }

    void reconnect_test()
    {
        skip_if_no_cuda();
        skip_if_not_rc_gda();

        device_put(0x1111111111111111lu, 0x2222222222222222lu);

        m_sender->destroy_ep(0);
        m_receiver->destroy_ep(0);
        short_progress_loop();

        m_sender->create_ep(0);
        m_receiver->create_ep(0);

        m_sender->connect_p2p_ep(m_sender->ep(0), m_receiver->ep(0));
        m_receiver->connect_p2p_ep(m_receiver->ep(0), m_sender->ep(0));
        short_progress_loop();

        device_put(0x3333333333333333lu, 0x4444444444444444lu);
    }

    void connect_ep(unsigned index)
    {
        m_sender->create_ep(index);
        m_receiver->create_ep(index);
        m_sender->connect_p2p_ep(m_sender->ep(index), m_receiver->ep(index));
        m_receiver->connect_p2p_ep(m_receiver->ep(index), m_sender->ep(index));
    }

    void init()
    {
        CUcontext ctx;
        ucs_status_t status;

        uct_test::init();
        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);

        m_cuda_dev = uct_cuda_get_cuda_device(
                m_sender->iface_attr().ctl_device);
        if (m_cuda_dev == CU_DEVICE_INVALID) {
            return;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuDevicePrimaryCtxRetain(&ctx, m_cuda_dev));
        ASSERT_UCS_OK(status);

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(ctx));
        ASSERT_UCS_OK(status);
    }

    void cleanup()
    {
        uct_test::cleanup();

        if (m_cuda_dev == CU_DEVICE_INVALID) {
            return;
        }

        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(m_cuda_dev));
    }

    struct device_mem_elem_handle {
        const entity          &m_entity;
        uct_device_mem_elem_t m_host_buf;
        mapped_buffer         m_dev_buf;
        void                  *m_release_handle;

        device_mem_elem_handle(const entity &e, uct_mem_h memh,
                               uct_rkey_t rkey) :
            m_entity(e),
            m_dev_buf(sizeof(uct_device_mem_elem_t), 0, e, 0,
                      UCS_MEMORY_TYPE_CUDA),
            m_release_handle(nullptr)
        {
            EXPECT_UCS_OK(uct_md_mem_elem_pack(m_entity.md(), memh, rkey,
                                               &m_host_buf, &m_release_handle));
            EXPECT_EQ(CUDA_SUCCESS,
                      cuMemcpyHtoD((CUdeviceptr)m_dev_buf.ptr(), &m_host_buf,
                                   sizeof(uct_device_mem_elem_t)));
        }

        uct_device_mem_elem_t *dev_ptr()
        {
            return (uct_device_mem_elem_t*)m_dev_buf.ptr();
        }

        ~device_mem_elem_handle()
        {
            if (m_release_handle != nullptr) {
                uct_md_mem_elem_release(m_entity.md(), m_release_handle);
            }
        }
    };

    entity *m_sender;
    entity *m_receiver;

private:
    CUdevice m_cuda_dev = CU_DEVICE_INVALID;
};

UCS_TEST_P(test_device, put)
{
    skip_if_no_cuda();
    device_put(0x1111111111111111lu, 0x2222222222222222lu);
}

UCS_TEST_P(test_device, reconnect)
{
    reconnect_test();
}

UCS_TEST_P(test_device, reconnect_multi_channel, "RC_GDA_NUM_CHANNELS?=4")
{
    reconnect_test();
}

UCS_TEST_P(test_device, destroy_one_of_multiple_eps)
{
    skip_if_no_cuda();
    skip_if_not_rc_gda();

    connect_ep(1);
    device_put(0x1111111111111111lu, 0x2222222222222222lu, 0);
    device_put(0x3333333333333333lu, 0x4444444444444444lu, 1);

    m_sender->destroy_ep(0);
    m_receiver->destroy_ep(0);
    short_progress_loop();

    device_put(0x5555555555555555lu, 0x6666666666666666lu, 1);

    connect_ep(0);
    device_put(0x7777777777777777lu, 0x8888888888888888lu, 0);
}

UCS_TEST_P(test_device, atomic)
{
    if (!(m_sender->md_attr().reg_mem_types & UCS_BIT(UCS_MEMORY_TYPE_CUDA))) {
        UCS_TEST_SKIP_R("CUDA registration not supported");
    }

    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0,
                         UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val = 0;
    size_t i;

    device_mem_elem_handle elem(*m_sender, nullptr, signal.rkey());

    uct_device_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));

    for (i = 0; i < 100; i++) {
        ASSERT_UCS_OK(ucx_cuda::launch_uct_atomic(dev_ep, elem.dev_ptr(),
                                                  (uintptr_t)signal.ptr(), 4));
        signal_val += 4;
        while (!mem_buffer::compare(&signal_val, signal.ptr(),
                                    sizeof(signal_val), UCS_MEMORY_TYPE_CUDA))
            ;
    }
}

UCS_TEST_P(test_device, flush)
{
    if (!(m_sender->md_attr().reg_mem_types & UCS_BIT(UCS_MEMORY_TYPE_CUDA))) {
        UCS_TEST_SKIP_R("CUDA registration not supported");
    }

    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0,
                         UCS_MEMORY_TYPE_CUDA);
    constexpr unsigned num_ops = 8;
    constexpr uint64_t add     = 4;
    uint64_t signal_val        = num_ops * add;

    device_mem_elem_handle elem(*m_sender, nullptr, signal.rkey());

    uct_device_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));
    ASSERT_UCS_OK(ucx_cuda::launch_uct_flush(dev_ep, elem.dev_ptr(),
                                             (uintptr_t)signal.ptr(), add,
                                             num_ops));
    EXPECT_TRUE(mem_buffer::compare(&signal_val, signal.ptr(),
                                    sizeof(signal_val), UCS_MEMORY_TYPE_CUDA));
}

_UCT_INSTANTIATE_TEST_CASE(test_device, rc_gda)
_UCT_INSTANTIATE_TEST_CASE(test_device, cuda_ipc)

#if HAVE_GDA
class test_gda_device : public ucs::test {
};

UCS_TEST_SKIP_COND_F(test_gda_device, reserve_wqe_block,
                     !mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA))
{
    scoped_cuda_device device_guard;
    for (unsigned num_threads : {32u, 33u, 128u}) {
        unsigned errors = 0;
        ASSERT_UCS_OK(ucx_cuda::launch_gda_reserve_wqe(
                UCS_DEVICE_LEVEL_BLOCK, num_threads, errors));
        EXPECT_EQ(0u, errors) << "BLOCK threads=" << num_threads;
    }
}

UCS_TEST_SKIP_COND_F(test_gda_device, reserve_wqe_warp,
                     !mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA))
{
    scoped_cuda_device device_guard;
    unsigned errors = 0;
    ASSERT_UCS_OK(ucx_cuda::launch_gda_reserve_wqe(
            UCS_DEVICE_LEVEL_WARP, UCS_DEVICE_NUM_THREADS_IN_WARP, errors));
    EXPECT_EQ(0u, errors);
}

UCS_TEST_SKIP_COND_F(test_gda_device, reserve_wqe_thread,
                     !mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA))
{
    scoped_cuda_device device_guard;
    unsigned errors = 0;
    ASSERT_UCS_OK(ucx_cuda::launch_gda_reserve_wqe(
            UCS_DEVICE_LEVEL_THREAD, 1, errors));
    EXPECT_EQ(0u, errors);
}

UCS_TEST_SKIP_COND_F(test_gda_device, grid_unsupported,
                     !mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA))
{
    scoped_cuda_device device_guard;
    unsigned errors = 0;
    ASSERT_UCS_OK(ucx_cuda::launch_gda_grid_unsupported(errors));
    EXPECT_EQ(0u, errors);
}
#endif
