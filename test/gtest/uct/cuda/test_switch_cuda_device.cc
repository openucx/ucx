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

    mapped_buffer sendbuf(length, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);

    int current_device;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((current_device + 1) % num_devices), cudaSuccess);

    mapped_buffer recvbuf(length, SEED2, receiver(), 0, mem_type);

    EXPECT_EQ(cudaSetDevice(current_device), cudaSuccess);

    blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
    if (flags & TEST_UCT_FLAG_SEND_ZCOPY) {
        sendbuf.memset(0);
        wait_for_remote();
        recvbuf.pattern_check(SEED1);
    } else if (flags & TEST_UCT_FLAG_RECV_ZCOPY) {
        recvbuf.memset(0);
        sendbuf.pattern_check(SEED2);
        wait_for_remote();
    }
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

_UCT_INSTANTIATE_TEST_CASE(test_p2p_switch_cuda_device, cuda_copy)
