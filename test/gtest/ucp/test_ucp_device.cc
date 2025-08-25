/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/ucp_test.h>

#include <ucp/cuda/test_kernels.h>

class test_ucp_device: public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_RMA | UCP_FEATURE_AMO64);
    }

    static void cuda_host_alloc(int *&host_ptr, int *&dev_ptr)
    {
        ASSERT_EQ(cudaSuccess,
                  cudaHostAlloc(&host_ptr, sizeof(host_ptr),
                                cudaHostAllocMapped));
        ASSERT_EQ(cudaSuccess,
                  cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    }

protected:
    // Compare generic CUDA buffers without copying them
    int cuda_memcmp(const void *s1, const void *s2, size_t size)
    {
        int *h_result, *d_result;
        int result;

        cuda_host_alloc(h_result, d_result);

        *h_result = 0;
        launch_cuda_memcmp(s1, s2, d_result, size);
        cudaDeviceSynchronize();
        result = *h_result;

        cudaFreeHost(h_result);
        return result;
    }
};

UCS_TEST_P(test_ucp_device, cuda_kernel_memcmp)
{
    size_t size = 100 * UCS_MBYTE;
    uint8_t *data;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&data, 2 * size));
    uint8_t *src = data;
    uint8_t *dst = static_cast<uint8_t*>(UCS_PTR_BYTE_OFFSET(data, size));

    EXPECT_EQ(cudaSuccess, cudaMemset(src, 0x11, size));
    EXPECT_EQ(cudaSuccess, cudaMemset(dst, 0xde, size));

    EXPECT_EQ(1, cuda_memcmp(src, dst, size));
    EXPECT_EQ(cudaSuccess, cudaMemset(dst, 0x11, size));
    EXPECT_EQ(0, cuda_memcmp(src, dst, size));
    EXPECT_EQ(cudaSuccess, cudaMemset(dst + size/10, 0xfa, 10));
    EXPECT_EQ(1, cuda_memcmp(src, dst, size));

    EXPECT_EQ(cudaSuccess, cudaFree(data));
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device, rc_v, "rc_v")
