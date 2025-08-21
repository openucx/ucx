/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/ucp_test.h>

static __global__ void memcmp_kernel(const void* a, const void* b,
                                     int* result, size_t size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        if (reinterpret_cast<const uint8_t*>(a)[i]
            != reinterpret_cast<const uint8_t*>(b)[i]) {
            result[0] = 1;
            break;
        }
    }
}

class test_ucp_cuda_device: public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_RMA | UCP_FEATURE_AMO64);
    }

protected:
    // Compare generic CUDA buffers without copying them
    static int cuda_memcmp(const void *a, const void *b, size_t size)
    {
        int h_result = 0;
        int* d_result;

        if (cudaMalloc(&d_result, sizeof(*d_result)) != cudaSuccess) {
            return -1;
        }

        if (cudaMemcpy(d_result, &h_result, sizeof(h_result), cudaMemcpyHostToDevice)
            != cudaSuccess) {
            h_result = -1;
            goto out;
        }

        memcmp_kernel<<<16, 64>>>(a, b, d_result, size);
        cudaDeviceSynchronize();
        if (cudaMemcpy(&h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
            h_result = -1;
            goto out;
        }

out:
        cudaFree(d_result);
        return h_result;
    }
};

UCS_TEST_P(test_ucp_cuda_device, cuda_kernel_memcmp)
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

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_cuda_device, rc_v, "rc_v")
