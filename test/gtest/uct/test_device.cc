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

    bool is_connected_to(const entity &remote, uct_ep_h ep) const
    {
        uct_iface_attr_t iface_attr;
        uct_ep_is_connected_params_t params;
        std::string dev_addr, ep_addr;

        ASSERT_UCS_OK(uct_iface_query(remote.iface(), &iface_attr));
        dev_addr.resize(iface_attr.device_addr_len);
        ep_addr.resize(iface_attr.ep_addr_len);

        ASSERT_UCS_OK(uct_iface_get_device_address(
                remote.iface(), (uct_device_addr_t*)dev_addr.data()));
        ASSERT_UCS_OK(uct_ep_get_address(remote.ep(0),
                                         (uct_ep_addr_t*)ep_addr.data()));

        params.field_mask  = UCT_EP_IS_CONNECTED_FIELD_DEVICE_ADDR |
                             UCT_EP_IS_CONNECTED_FIELD_EP_ADDR;
        params.device_addr = (uct_device_addr_t*)dev_addr.data();
        params.ep_addr     = (uct_ep_addr_t*)ep_addr.data();

        return uct_ep_is_connected(ep, &params);
    }

    void device_put(uint64_t send_seed, uint64_t recv_seed)
    {
        constexpr size_t length = 1024;
        mapped_buffer sendbuf(length, send_seed, *m_sender, 0,
                              UCS_MEMORY_TYPE_CUDA);
        mapped_buffer recvbuf(length, recv_seed, *m_receiver, 0,
                              UCS_MEMORY_TYPE_CUDA);

        uct_device_mem_elem_t src_elem_host;
        ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), sendbuf.memh(),
                                           recvbuf.rkey(), &src_elem_host));

        mapped_buffer src_elembuf(sizeof(src_elem_host), 0, *m_sender, 0,
                                  UCS_MEMORY_TYPE_CUDA);
        ASSERT_EQ(CUDA_SUCCESS,
                  cuMemcpyHtoD((CUdeviceptr)src_elembuf.ptr(), &src_elem_host,
                               sizeof(src_elem_host)));

        mapped_buffer rem_elembuf(sizeof(src_elem_host), 0, *m_sender, 0,
                                  UCS_MEMORY_TYPE_CUDA);
        ASSERT_EQ(CUDA_SUCCESS,
                  cuMemcpyHtoD((CUdeviceptr)rem_elembuf.ptr(), &src_elem_host,
                               sizeof(src_elem_host)));

        uct_device_ep_h dev_ep;
        ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));
        ASSERT_UCS_OK(ucx_cuda::launch_uct_put(
                dev_ep, (const uct_device_mem_elem_t*)src_elembuf.ptr(),
                (const uct_device_mem_elem_t*)rem_elembuf.ptr(), sendbuf.ptr(),
                (uintptr_t)recvbuf.ptr(), length));

        recvbuf.pattern_check(send_seed);
        recvbuf.pattern_fill(recv_seed);
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
    skip_if_no_cuda();
    skip_if_not_rc_gda();

    EXPECT_TRUE(is_connected_to(*m_receiver, m_sender->ep(0)));
    EXPECT_TRUE(is_connected_to(*m_sender, m_receiver->ep(0)));

    device_put(0x1111111111111111lu, 0x2222222222222222lu);

    m_sender->destroy_ep(0);
    m_receiver->destroy_ep(0);
    short_progress_loop();

    m_sender->create_ep(0);
    m_receiver->create_ep(0);
    EXPECT_FALSE(is_connected_to(*m_receiver, m_sender->ep(0)));
    EXPECT_FALSE(is_connected_to(*m_sender, m_receiver->ep(0)));

    m_sender->connect_p2p_ep(m_sender->ep(0), m_receiver->ep(0));
    EXPECT_TRUE(is_connected_to(*m_receiver, m_sender->ep(0)));

    m_receiver->connect_p2p_ep(m_receiver->ep(0), m_sender->ep(0));
    EXPECT_TRUE(is_connected_to(*m_sender, m_receiver->ep(0)));
    short_progress_loop();

    device_put(0x3333333333333333lu, 0x4444444444444444lu);
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

    mapped_buffer elembuf_host(sizeof(uct_device_mem_elem_t), 0, *m_sender, 0,
                               UCS_MEMORY_TYPE_HOST);
    mapped_buffer elembuf(sizeof(uct_device_mem_elem_t), 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    uct_device_mem_elem_t *mem_elem_host = (uct_device_mem_elem_t*)
                                                   elembuf_host.ptr();
    uct_device_mem_elem_t *mem_elem = (uct_device_mem_elem_t*)elembuf.ptr();
    ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), nullptr, signal.rkey(),
                                       mem_elem_host));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)mem_elem, mem_elem_host,
                                         sizeof(uct_device_mem_elem_t)));

    uct_device_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));

    for (i = 0; i < 100; i++) {
        ASSERT_UCS_OK(ucx_cuda::launch_uct_atomic(dev_ep, mem_elem,
                                                  (uintptr_t)signal.ptr(), 4));
        signal_val += 4;
        while (!mem_buffer::compare(&signal_val, signal.ptr(),
                                    sizeof(signal_val), UCS_MEMORY_TYPE_CUDA))
            ;
    }
}

_UCT_INSTANTIATE_TEST_CASE(test_device, rc_gda)
_UCT_INSTANTIATE_TEST_CASE(test_device, cuda_ipc)
