/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <common/mem_buffer.h>

#include <ucs/sys/sys.h>
#include <ucs/memory/memtype_cache.h>


class test_memtype_cache : public ucs::test_with_param<ucs_memory_type_t> {
protected:
    test_memtype_cache() : m_memtype_cache(NULL) {
    }

    virtual void init() {
        ucs::test_with_param<ucs_memory_type_t>::init();
        ucs_status_t status = ucs_memtype_cache_create(&m_memtype_cache);
        ASSERT_UCS_OK(status);
    }

    virtual void cleanup() {
        ucs_memtype_cache_destroy(m_memtype_cache);
        ucs::test_with_param<ucs_memory_type_t>::cleanup();
    }

    void test_lookup_found(void *ptr, size_t size) {
        ucs_memory_type_t mem_type;
        ucs_status_t status = ucs_memtype_cache_lookup(m_memtype_cache, ptr,
                                                       size, &mem_type);
        EXPECT_UCS_OK(status);
        EXPECT_EQ(GetParam(), mem_type) << "ptr=" << ptr << " size=" << size;
    }

    void test_lookup_notfound(void *ptr, size_t size) {
        ucs_memory_type_t mem_type;
        ucs_status_t status = ucs_memtype_cache_lookup(m_memtype_cache, ptr,
                                                       size, &mem_type);
        /* memory type should be not-found or unknown */
        EXPECT_TRUE((status == UCS_ERR_NO_ELEM) ||
                    ((status == UCS_OK) && (mem_type == UCS_MEMORY_TYPE_LAST)))
              << "ptr=" << ptr << " size=" << size << ": "
              << ucs_status_string(status)
              << " memtype=" << mem_buffer::mem_type_name(mem_type);
    }

private:
    ucs_memtype_cache_t *m_memtype_cache;
};

UCS_TEST_P(test_memtype_cache, basic) {
    const size_t size = 64;
    void *ptr;

    {
        mem_buffer b(size, GetParam());

        if (GetParam() != UCS_MEMORY_TYPE_HOST) {
            test_lookup_found(b.ptr(), size);
            test_lookup_found(b.ptr(), size / 2);
            test_lookup_found((char*)b.ptr() + 1, 7);
            test_lookup_found(b.ptr(), 1);
            test_lookup_found((char*)b.ptr() + size - 1, 1);
            test_lookup_found(b.ptr(), 0);
        }

        /* memtype cache is page-aligned, so need to step by page size
         * to make something not found
         */
        test_lookup_notfound(b.ptr(), size + ucs_get_page_size());
        test_lookup_notfound((char*)b.ptr() + size + ucs_get_page_size(), 1);

        ptr = b.ptr();
    }

    /* buffer is released */
    test_lookup_notfound(ptr, size);
    test_lookup_notfound(ptr, 1);
}

INSTANTIATE_TEST_CASE_P(mem_type, test_memtype_cache,
                        ::testing::ValuesIn(mem_buffer::supported_mem_types()));
