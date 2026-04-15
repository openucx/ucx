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
        ASSERT_NE(m_cuda_dev, CU_DEVICE_INVALID)
                << " sys_device "
                << static_cast<int>(m_sender->iface_attr().ctl_device);

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuDevicePrimaryCtxRetain(&ctx, m_cuda_dev));
        ASSERT_UCS_OK(status);

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(ctx));
        ASSERT_UCS_OK(status);
    }

    void cleanup()
    {
        uct_test::cleanup();
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(m_cuda_dev));
    }

    entity *m_sender;
    entity *m_receiver;

private:
    CUdevice m_cuda_dev;
};

UCS_TEST_P(test_device, single)
{
    constexpr uint64_t SEED1 = 0x1111111111111111lu;
    constexpr uint64_t SEED2 = 0x2222222222222222lu;
    constexpr size_t length  = 1024;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    mapped_buffer elembuf_host(sizeof(uct_device_mem_element_t), 0, *m_sender,
                               0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer elembuf(sizeof(uct_device_mem_element_t), 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    uct_device_mem_element_t *mem_elem_host = (uct_device_mem_element_t*)
                                                      elembuf_host.ptr();
    uct_device_mem_element_t *mem_elem      = (uct_device_mem_element_t*)
                                                      elembuf.ptr();
    ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), sendbuf.memh(),
                                       recvbuf.rkey(), mem_elem_host));
    /* Copy packed element from host to GPU */
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)mem_elem, mem_elem_host,
                                         sizeof(uct_device_mem_element_t)));

    uct_device_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));
    ASSERT_UCS_OK(
            ucx_cuda::launch_uct_put_single(dev_ep, mem_elem, sendbuf.ptr(),
                                            (uintptr_t)recvbuf.ptr(), length));

    recvbuf.pattern_check(SEED1);
    recvbuf.pattern_fill(SEED2);
}

UCS_TEST_P(test_device, atomic)
{
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0,
                         UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val = 0;
    size_t i;

    mapped_buffer elembuf_host(sizeof(uct_device_mem_element_t), 0, *m_sender,
                               0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer elembuf(sizeof(uct_device_mem_element_t), 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    uct_device_mem_element_t *mem_elem_host = (uct_device_mem_element_t*)
                                                      elembuf_host.ptr();
    uct_device_mem_element_t *mem_elem      = (uct_device_mem_element_t*)
                                                      elembuf.ptr();
    ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), nullptr, signal.rkey(),
                                       mem_elem_host));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)mem_elem, mem_elem_host,
                                         sizeof(uct_device_mem_element_t)));

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

UCS_TEST_P(test_device, multi)
{
    constexpr uint64_t SEED1 = 0x1111111111111111lu;
    constexpr uint64_t SEED2 = 0x2222222222222222lu;
    constexpr size_t iovcnt  = 31;
    constexpr size_t length  = iovcnt * 32;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0,
                         UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val = 0;
    size_t i;

    size_t total_elem_size = sizeof(uct_device_mem_element_t) * (iovcnt + 1);
    mapped_buffer elembuf_host(total_elem_size, 0, *m_sender, 0,
                               UCS_MEMORY_TYPE_HOST);
    mapped_buffer elembuf(total_elem_size, 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    for (i = 0; i < iovcnt; i++) {
        ASSERT_UCS_OK(uct_md_mem_elem_pack(
                m_sender->md(), sendbuf.memh(), recvbuf.rkey(),
                (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                        elembuf_host.ptr(),
                        sizeof(uct_device_mem_element_t) * i)));
    }

    ASSERT_UCS_OK(uct_md_mem_elem_pack(
            m_sender->md(), NULL, signal.rkey(),
            (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                    elembuf_host.ptr(),
                    sizeof(uct_device_mem_element_t) * iovcnt)));

    /* Copy all packed elements from host to GPU in one operation */
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)elembuf.ptr(),
                                         elembuf_host.ptr(), total_elem_size));

    uct_device_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));
    for (i = 0; i < 100; i++) {
        ASSERT_UCS_OK(ucx_cuda::launch_uct_put_multi<iovcnt>(
                dev_ep, (uct_device_mem_element_t*)elembuf.ptr(), sendbuf.ptr(),
                (uintptr_t)recvbuf.ptr(), (uintptr_t)signal.ptr(), length));

        signal_val += 4;
        while (!mem_buffer::compare(&signal_val, signal.ptr(),
                                    sizeof(signal_val), UCS_MEMORY_TYPE_CUDA))
            ;
        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }
}

UCS_TEST_P(test_device, partial)
{
    constexpr uint64_t SEED1 = 0x1111111111111111lu;
    constexpr uint64_t SEED2 = 0x2222222222222222lu;
    constexpr size_t iovcnt  = 31;
    constexpr size_t length  = iovcnt * 32;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0,
                         UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val = 0;
    size_t i;

    size_t total_elem_size = sizeof(uct_device_mem_element_t) * (iovcnt + 1);
    mapped_buffer elembuf_host(total_elem_size, 0, *m_sender, 0,
                               UCS_MEMORY_TYPE_HOST);
    mapped_buffer elembuf(total_elem_size, 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    for (i = 0; i < iovcnt; i++) {
        ASSERT_UCS_OK(uct_md_mem_elem_pack(
                m_sender->md(), sendbuf.memh(), recvbuf.rkey(),
                (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                        elembuf_host.ptr(),
                        sizeof(uct_device_mem_element_t) * i)));
    }

    ASSERT_UCS_OK(uct_md_mem_elem_pack(
            m_sender->md(), NULL, signal.rkey(),
            (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                    elembuf_host.ptr(),
                    sizeof(uct_device_mem_element_t) * iovcnt)));

    /* Copy all packed elements from host to GPU in one operation */
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)elembuf.ptr(),
                                         elembuf_host.ptr(), total_elem_size));

    uct_device_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &dev_ep));
    for (i = 0; i < 100; i++) {
        ASSERT_UCS_OK(ucx_cuda::launch_uct_put_partial<iovcnt>(
                dev_ep, (uct_device_mem_element_t*)elembuf.ptr(), sendbuf.ptr(),
                (uintptr_t)recvbuf.ptr(), (uintptr_t)signal.ptr(), length));

        signal_val += 4;
        while (!mem_buffer::compare(&signal_val, signal.ptr(),
                                    sizeof(signal_val), UCS_MEMORY_TYPE_CUDA))
            ;
        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }
}

_UCT_INSTANTIATE_TEST_CASE(test_device, rc_gda)
_UCT_INSTANTIATE_TEST_CASE(test_device, cuda_ipc)
