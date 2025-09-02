/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/ucp_test.h>

#include "cuda/test_kernels.h"

class test_ucp_device : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_RMA | UCP_FEATURE_AMO64);
    }

    virtual void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
    }
};

UCS_TEST_P(test_ucp_device, mapped_buffer_kernel_memcmp)
{
    size_t size = 100 * UCS_MBYTE;

    mapped_buffer dst(size, receiver(), 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer src(size, sender(), 0, UCS_MEMORY_TYPE_CUDA);

    src.pattern_fill(0x1234, size);
    src.pattern_check(0x1234, size);

    ASSERT_EQ(cudaSuccess, cudaMemset(src.ptr(), 0x11, size));
    ASSERT_EQ(cudaSuccess, cudaMemset(dst.ptr(), 0xde, size));

    ASSERT_EQ(1, ucp_cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
    ASSERT_EQ(cudaSuccess, cudaMemset(dst.ptr(), 0x11, size));
    ASSERT_EQ(0, ucp_cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
    ASSERT_EQ(cudaSuccess,
              cudaMemset(UCS_PTR_BYTE_OFFSET(dst.ptr(), size / 10), 0xfa, 10));
    ASSERT_EQ(1, ucp_cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
}

UCS_TEST_P(test_ucp_device, put_single)
{
    const size_t size = 16 * UCS_KBYTE;

    mapped_buffer src(size, sender(), 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer dst(size, receiver(), 0, UCS_MEMORY_TYPE_CUDA);

    src.pattern_fill(0x1234, size);
    src.pattern_check(0x1234, size);

    // TODO create mem list
    ucp_device_mem_list_handle_h mem_list = nullptr;

    ucs_status_t status = ucp_cuda::launch_ucp_put_single(mem_list, src.ptr(),
                                                          (uint64_t)dst.ptr(),
                                                          size);
    EXPECT_EQ(UCS_ERR_NOT_IMPLEMENTED, status);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device, gdaki, "rc,gdaki")
