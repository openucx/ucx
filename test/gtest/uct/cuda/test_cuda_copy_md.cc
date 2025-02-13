/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/test_md.h>

#include <cuda_runtime.h>

class test_cuda_copy_md : public test_md {
};

UCS_TEST_P(test_cuda_copy_md, switch_gpu) {
    if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("cuda is not supported");
    }

    int num_devices;
    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);

    if (num_devices < 2) {
        UCS_TEST_SKIP_R("less than two cuda devices available");
    }

    int device;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((device + 1) % num_devices), cudaSuccess);

    const size_t size = 16;
    mem_buffer buffer(size, UCS_MEMORY_TYPE_CUDA);

    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);

    ucs_memory_type_t mem_type;
    ASSERT_EQ(uct_md_detect_memory_type(m_md, buffer.ptr(), size, &mem_type),
              UCS_OK);
    EXPECT_EQ(mem_type, UCS_MEMORY_TYPE_CUDA);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_copy_md, cuda_cpy);
