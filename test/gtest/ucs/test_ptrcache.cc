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
#include <ucs/sys/ptrcache.h>
}


class test_ptrcache : public ucs::test {
protected:

    virtual void init() {
        ucs::test::init();
        UCS_TEST_CREATE_HANDLE(ucs_ptrcache_t*, m_ptrcache, ucs_ptrcache_destroy,
                               ucs_ptrcache_create, "test");
    }
    virtual void cleanup() {
        m_ptrcache.reset();
        ucs::test::cleanup();
    }

    ucs::handle<ucs_ptrcache_t*> m_ptrcache;
};

#if HAVE_CUDA
UCS_TEST_F(test_ptrcache, basic_cuda) {
    cudaError_t cerr;
    void *ptr;
    int ret;
    ucm_mem_type_t ucm_mem_type;

    /* set cuda device */
    if (cudaSetDevice(0) != cudaSuccess) {
        UCS_TEST_SKIP_R("can't set cuda device");
    }

    cerr = cudaMalloc(&ptr, 64);
    EXPECT_EQ(cerr, cudaSuccess);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 64, &ucm_mem_type);
    EXPECT_EQ(ret, 1);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 32, &ucm_mem_type);
    EXPECT_EQ(ret, 1);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    ret = ucs_ptrcache_lookup(m_ptrcache, (void *)((uintptr_t)ptr + 1), 7, &ucm_mem_type);
    EXPECT_EQ(ret, 1);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 1, &ucm_mem_type);
    EXPECT_EQ(ret, 1);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    ret = ucs_ptrcache_lookup(m_ptrcache, (void *)((uintptr_t) ptr + 63), 1, &ucm_mem_type);
    EXPECT_EQ(ret, 1);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 0, &ucm_mem_type);
    EXPECT_EQ(ret, 1);
    EXPECT_EQ(ucm_mem_type, UCM_MEM_TYPE_CUDA);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 65, &ucm_mem_type);
    EXPECT_EQ(ret, 0);
    ret = ucs_ptrcache_lookup(m_ptrcache, (void *)((uintptr_t) ptr + 64), 1, &ucm_mem_type);
    EXPECT_EQ(ret, 0);

    cerr = cudaFree(ptr);
    EXPECT_EQ(cerr, cudaSuccess);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 64, &ucm_mem_type);
    EXPECT_EQ(ret, 0);
    ret = ucs_ptrcache_lookup(m_ptrcache, ptr, 1, &ucm_mem_type);
    EXPECT_EQ(ret, 0);
}
#endif
