/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/test_p2p_rma.h>

#include <cuda_runtime.h>

class test_p2p_switch_cuda_device : public uct_p2p_rma_test {
public:
    void test_xfer(send_func_t send, size_t length, unsigned flags,
                   ucs_memory_type_t mem_type) override;
};

void test_p2p_switch_cuda_device::test_xfer(send_func_t send, size_t length,
                                            unsigned flags,
                                            ucs_memory_type_t mem_type)
{
    int num_devices;
    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);

    if (num_devices < 2) {
        UCS_TEST_SKIP_R("less than two cuda devices available");
    }

    uct_p2p_rma_test::test_xfer(send, length, flags, mem_type);

    int current_device;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((current_device + 1) % num_devices), cudaSuccess);

    uct_p2p_rma_test::test_xfer(send, length, flags, mem_type);

    EXPECT_EQ(cudaSetDevice(current_device), cudaSuccess);
}

UCS_TEST_P(test_p2p_switch_cuda_device, put_short)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
              sender().iface_attr().cap.put.max_short / 2,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_switch_cuda_device, get_short)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_short),
              sender().iface_attr().cap.put.max_short / 2,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_switch_cuda_device, put_zcopy)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
              (sender().iface_attr().cap.put.min_zcopy +
               sender().iface_attr().cap.put.max_zcopy) / 2,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_switch_cuda_device, get_zcopy)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
              (sender().iface_attr().cap.get.min_zcopy +
               sender().iface_attr().cap.get.max_zcopy) / 2,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

_UCT_INSTANTIATE_TEST_CASE(test_p2p_switch_cuda_device, cuda_copy)
