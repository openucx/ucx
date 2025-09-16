/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string>

#include <uct/api/uct.h>
#include "uct_test.h"

extern "C" {
#include <uct/cuda/base/cuda_iface.h>
}

#include "cuda/test_kernels.h"

/* Ensure a CUDA context is active before UCX enumerates resources so that
 * cuda_ipc sys_device is set properly. */
static struct test_device_cuda_ctx_guard {
    CUdevice  dev;
    CUcontext ctx;
    int       active;

    test_device_cuda_ctx_guard() : dev(0), ctx(NULL), active(0) {
        (void)UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));
        if (UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&dev, 0)) == UCS_OK) {
            if (UCT_CUDADRV_FUNC_LOG_ERR(cuDevicePrimaryCtxRetain(&ctx, dev)) == UCS_OK) {
                if (UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(ctx)) == UCS_OK) {
                    active = 1;
                } else {
                    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(dev));
                }
            }
        }
    }

    ~test_device_cuda_ctx_guard() {
        if (active) {
            (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
            (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(dev));
        }
    }
} g_test_device_cuda_ctx_guard;

class test_device : public uct_test {
protected:
    void init()
    {
        CUcontext ctx;
        ucs_status_t status;

        uct_test::init();
        status = uct_cuda_base_get_cuda_device(GetParam()->sys_device,
                                               &m_cuda_dev);
        ASSERT_UCS_OK(status, << " sys_device "
            << static_cast<int>(GetParam()->sys_device));

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuDevicePrimaryCtxRetain(&ctx, m_cuda_dev));
        ASSERT_UCS_OK(status);

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(ctx));
        ASSERT_UCS_OK(status);

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);
    }

    void cleanup()
    {
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(m_cuda_dev));
        uct_test::cleanup();
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

    uct_iface_attr_v2_t iface_attr;
    iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
    ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &iface_attr));
    mapped_buffer elembuf(iface_attr.device_mem_element_size, 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    uct_device_mem_element_t *mem_elem = (uct_device_mem_element_t*)
                                                 elembuf.ptr();
    ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), sendbuf.memh(),
                                             recvbuf.rkey(), mem_elem));

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

    uct_iface_attr_v2_t iface_attr;
    iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
    ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &iface_attr));

    mapped_buffer elembuf(iface_attr.device_mem_element_size, 0, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    uct_device_mem_element_t *mem_elem = (uct_device_mem_element_t*)
                                                 elembuf.ptr();
    ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), nullptr,
                                             signal.rkey(), mem_elem));

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

    uct_iface_attr_v2_t iface_attr;
    iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
    ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &iface_attr));

    mapped_buffer elembuf(iface_attr.device_mem_element_size * (iovcnt + 1), 0,
                          *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    for (i = 0; i < iovcnt; i++) {
        ASSERT_UCS_OK(uct_iface_mem_element_pack(
                m_sender->iface(), sendbuf.memh(), recvbuf.rkey(),
                (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                        elembuf.ptr(),
                        iface_attr.device_mem_element_size * i)));
    }

    ASSERT_UCS_OK(uct_iface_mem_element_pack(
            m_sender->iface(), NULL, signal.rkey(),
            (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                    elembuf.ptr(),
                    iface_attr.device_mem_element_size * iovcnt)));

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

    uct_iface_attr_v2_t iface_attr;
    iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
    ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &iface_attr));

    mapped_buffer elembuf(iface_attr.device_mem_element_size * (iovcnt + 1), 0,
                          *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    for (i = 0; i < iovcnt; i++) {
        ASSERT_UCS_OK(uct_iface_mem_element_pack(
                m_sender->iface(), sendbuf.memh(), recvbuf.rkey(),
                (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                        elembuf.ptr(),
                        iface_attr.device_mem_element_size * i)));
    }

    ASSERT_UCS_OK(uct_iface_mem_element_pack(
            m_sender->iface(), NULL, signal.rkey(),
            (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(
                    elembuf.ptr(),
                    iface_attr.device_mem_element_size * iovcnt)));

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
