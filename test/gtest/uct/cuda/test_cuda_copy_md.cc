/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/test_md.h>

#include <cuda_runtime.h>

class test_cuda_copy_md_multi_gpu : public test_md {
public:
    uct_allocated_memory_t mem_alloc(size_t size) const;
};

uct_allocated_memory_t test_cuda_copy_md_multi_gpu::mem_alloc(size_t size) const
{
    uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
    uct_md_h md               = m_md.get();

    uct_mem_alloc_params_t params;
    params.field_mask = UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                        UCT_MEM_ALLOC_PARAM_FIELD_MDS      |
                        UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.mem_type   = UCS_MEMORY_TYPE_CUDA;
    params.mds.mds    = &md;
    params.mds.count  = 1;
    params.name       = "test_cuda_copy_md_multi_gpu";

    uct_allocated_memory_t mem;
    EXPECT_EQ(uct_mem_alloc(size, &method, 1, &params, &mem), UCS_OK);
    return mem;
}

UCS_TEST_P(test_cuda_copy_md_multi_gpu, mem_query) {
    int num_devices;
    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);

    if (num_devices < 2) {
        UCS_TEST_SKIP_R("less than two cuda devices available");
    }

    int device;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((device + 1) % num_devices), cudaSuccess);

    const int size = 16;
    auto mem       = mem_alloc(size);

    EXPECT_EQ(cudaSetDevice(device), cudaSuccess);

    uct_md_mem_attr_t mem_attr = {};
    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;
    EXPECT_EQ(uct_md_mem_query(m_md.get(), mem.address, size, &mem_attr),
              UCS_OK);
    EXPECT_EQ(mem_attr.mem_type, UCS_MEMORY_TYPE_CUDA);
    EXPECT_EQ(uct_mem_free(&mem), UCS_OK);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_copy_md_multi_gpu, cuda_cpy);
