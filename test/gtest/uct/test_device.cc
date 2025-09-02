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
        CUresult res_drv;
        CUcontext ctx;
        int cuda_id;
        ucs_status_t status;

        uct_test::init();

        cuda_id = std::stoi(
                GetParam()->dev_name.substr(UCT_DEVICE_CUDA_NAME_LEN));
        res_drv = cuDeviceGet(&m_cuda_dev, cuda_id);
        if (res_drv != CUDA_SUCCESS) {
            ucs_error("cuDeviceGet returned %d.", res_drv);
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
        (void)UCT_CUDADRV_FUNC(cuCtxPopCurrent(NULL), UCS_LOG_LEVEL_WARN);
        (void)UCT_CUDADRV_FUNC(cuDevicePrimaryCtxRelease(m_cuda_dev),
                               UCS_LOG_LEVEL_WARN);
        uct_test::cleanup();
    }

    entity *m_sender;
    entity *m_receiver;

private:
    CUdevice m_cuda_dev;
};

UCS_TEST_P(test_device, single)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    size_t length               = 1024;
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
            uct_cuda::launch_single_kernel(dev_ep, mem_elem, sendbuf.ptr(),
                                           (uintptr_t)recvbuf.ptr(), length));

    recvbuf.pattern_check(SEED1);
    recvbuf.pattern_fill(SEED2);
}

_UCT_INSTANTIATE_TEST_CASE(test_device, gdaki)
