/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "common/test_helpers.h"
#include <uct/uct_test.h>
extern "C" {
    #include <uct/base/uct_md.h>
    #include <uct/cuda/base/cuda_iface.h>
    #include <uct/api/v2/uct_v2.h>
    #include <uct/cuda/cuda_ipc/cuda_ipc_device.h>
    }
#include <cuda.h>
#include "test_kernels_uct.h"

class test_cuda_ipc_rma : public uct_test {
    protected:
    void init() {
        CUresult res_drv;
        int cuda_id;

        uct_test::init();

        cuda_id = 0;
        res_drv = cuDeviceGet(&m_cuda_dev, cuda_id);
        if (res_drv != CUDA_SUCCESS) {
            ucs_error("cuDeviceGet returned %d.", res_drv);
            return;
        }

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);
    }

    void cleanup() {
        uct_test::cleanup();
    }

    size_t get_mem_elem_size() {
        uct_iface_attr_v2_t iface_attr;
        iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
        ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &iface_attr));
        return iface_attr.device_mem_element_size;
    }

    entity * m_sender;
    entity * m_receiver;

    CUdevice m_cuda_dev;
};

UCS_TEST_P(test_cuda_ipc_rma, put_zcopy)
{
    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
    size_t length = 1024;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(m_sender->ep(0),
                                                 sendbuf.iov(), 1,
                                                 (uint64_t)recvbuf.ptr(),
                                                 recvbuf.rkey(), NULL));
    m_sender->flush();
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_cuda_ipc_rma, has_device_ep_capability)
{
    uct_iface_attr_t iface_attr;

    ASSERT_UCS_OK(uct_iface_query(m_sender->iface(), &iface_attr));
    EXPECT_EQ(iface_attr.cap.flags & UCT_IFACE_FLAG_DEVICE_EP,
              UCT_IFACE_FLAG_DEVICE_EP);
}

UCS_TEST_P(test_cuda_ipc_rma, mem_elem_size)
{
    EXPECT_EQ(get_mem_elem_size(), sizeof(uct_cuda_ipc_device_mem_element_t));
}

UCS_TEST_P(test_cuda_ipc_rma, get_device_ep)
{
    uct_device_ep_h device_ep;

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
}

UCS_TEST_P(test_cuda_ipc_rma, get_mem_elem_pack)
{
    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
    size_t length = 1024;
    size_t mem_elem_size = get_mem_elem_size();
    uct_device_mem_element_t *mem_elem;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size));
    EXPECT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), sendbuf.memh(),
                                             recvbuf.rkey(), mem_elem));
    cuMemFree((CUdeviceptr)&mem_elem);
}

UCS_TEST_P(test_cuda_ipc_rma, put_zcopy_device)
{
    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
    size_t length = 1024;
    size_t mem_elem_size = get_mem_elem_size();
    uct_device_ep_h device_ep;
    uct_device_mem_element_t *mem_elem;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size));
    ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), sendbuf.memh(),
                                             recvbuf.rkey(), mem_elem));

    cuda_uct::launch_uct_put_single(device_ep, mem_elem, sendbuf.ptr(),
                                    (uint64_t)recvbuf.ptr(), length);

    recvbuf.pattern_check(SEED1);
    cuMemFree((CUdeviceptr)&mem_elem);
}


_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma, cuda_ipc)
