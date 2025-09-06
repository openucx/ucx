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

class test_device : public uct_test {
protected:
    void init()
    {
        CUcontext ctx;
        ucs_status_t status;

        uct_test::init();
        status = uct_cuda_base_get_cuda_device(GetParam()->sys_device,
                                               &m_cuda_dev);
        if (status != UCS_OK) {
            return;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuDevicePrimaryCtxRetain(&ctx, m_cuda_dev));
        if (status != UCS_OK) {
            return;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(ctx));
        if (status != UCS_OK) {
            return;
        }

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

_UCT_INSTANTIATE_TEST_CASE(test_device, gdaki)
