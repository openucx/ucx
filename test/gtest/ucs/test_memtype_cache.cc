/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <ucs/sys/sys.h>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
extern "C" {
#include <ucs/memory/memtype_cache.h>
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

    void test_lookup_found(void *ptr, size_t size) {
        ucs_memory_type_t mem_type;
        ucs_status_t status = ucs_memtype_cache_lookup(m_memtype_cache, ptr,
                                                       size, &mem_type);
        EXPECT_UCS_OK(status);
        EXPECT_EQ(UCS_MEMORY_TYPE_CUDA, mem_type)
              << "ptr=" << ptr << " size=" << size;
    }

    void test_lookup_notfound(void *ptr, size_t size) {
        ucs_memory_type_t mem_type;
        ucs_status_t status = ucs_memtype_cache_lookup(m_memtype_cache, ptr,
                                                       size, &mem_type);
        /* memory type should be not-found or unknown */
        EXPECT_TRUE((status == UCS_ERR_NO_ELEM) ||
                    ((status == UCS_OK) && (mem_type == UCS_MEMORY_TYPE_LAST)))
              << "ptr=" << ptr << " size=" << size << ": "
              << ucs_status_string(status);
    }

private:
    ucs_memtype_cache_t *m_memtype_cache;
};

#if HAVE_CUDA
UCS_TEST_SKIP_COND_F(test_memtype_cache, basic_cuda,
                     /* skip if unable to set CUDA device */
                     (cudaSetDevice(0) != cudaSuccess)) {
    const size_t size = 64;
    cudaError_t cerr;
    void *ptr;

    cerr = cudaMalloc(&ptr, size);
    EXPECT_EQ(cerr, cudaSuccess);

    test_lookup_found(ptr, size);
    test_lookup_found(ptr, size / 2);
    test_lookup_found((char*)ptr + 1, 7);
    test_lookup_found(ptr, 1);
    test_lookup_found((char*)ptr + size - 1, 1);
    test_lookup_found(ptr, 0);

    /* memtype cache is page-aligned, so need to step by page size
     * to make something not found
     */
    test_lookup_notfound(ptr, size + ucs_get_page_size());
    test_lookup_notfound((char*)ptr + size + ucs_get_page_size(), 1);

    cerr = cudaFree(ptr);
    EXPECT_EQ(cerr, cudaSuccess);

    /* buffer is released */
    test_lookup_notfound(ptr, size);
    test_lookup_notfound(ptr, 1);
}
#endif
