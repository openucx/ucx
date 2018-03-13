/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
extern "C" {
#include <ucs/sys/memtype_cache.h>
}


class test_memtype_cache : public ucs::test {
protected:

    virtual void init() {
        ucs_status_t status;

        ucs::test::init();
        status = ucs_memtype_cache_create(&m_memtype_cache);
        ASSERT_UCS_OK(status);
    }

    virtual void cleanup() {
        ucs_memtype_cache_destroy(m_memtype_cache);
        ucs::test::cleanup();
    }

    ucs_memtype_cache_t *m_memtype_cache;
};

#if HAVE_CUDA
UCS_TEST_F(test_memtype_cache, basic_cuda) {
    cudaError_t cerr;
    void *ptr;
    ucm_mem_type_t ucm_mem_type;
    ucs_status_t status;

    /* set cuda device */
    if (cudaSetDevice(0) != cudaSuccess) {
        UCS_TEST_SKIP_R("can't set cuda device");
    }

    cerr = cudaMalloc(&ptr, 64);
    EXPECT_EQ(cerr, cudaSuccess);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 64, &ucm_mem_type);
    EXPECT_UCS_OK(status);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 32, &ucm_mem_type);
    EXPECT_UCS_OK(status);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    status = ucs_memtype_cache_lookup(m_memtype_cache, (void *)((uintptr_t)ptr + 1), 7, &ucm_mem_type);
    EXPECT_UCS_OK(status);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 1, &ucm_mem_type);
    EXPECT_UCS_OK(status);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    status = ucs_memtype_cache_lookup(m_memtype_cache, (void *)((uintptr_t) ptr + 63), 1, &ucm_mem_type);
    EXPECT_UCS_OK(status);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 0, &ucm_mem_type);
    EXPECT_UCS_OK(status);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 65, &ucm_mem_type);
    EXPECT_TRUE(status == UCS_ERR_NO_ELEM);
    status = ucs_memtype_cache_lookup(m_memtype_cache, (void *)((uintptr_t) ptr + 64), 1, &ucm_mem_type);
    EXPECT_TRUE(status == UCS_ERR_NO_ELEM);

    cerr = cudaFree(ptr);
    EXPECT_EQ(cerr, cudaSuccess);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 64, &ucm_mem_type);
    EXPECT_TRUE(status == UCS_ERR_NO_ELEM);
    status = ucs_memtype_cache_lookup(m_memtype_cache, ptr, 1, &ucm_mem_type);
    EXPECT_TRUE(status == UCS_ERR_NO_ELEM);
}
#endif
