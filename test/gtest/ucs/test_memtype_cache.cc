/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <common/mem_buffer.h>

#include <ucs/sys/sys.h>
#include <ucs/memory/memtype_cache.h>
#include <ucm/api/ucm.h>

extern "C" {
#include <ucm/event/event.h>
}


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

    void test_lookup_found(void *ptr, size_t size,
                           ucs_memory_type_t expected_type) const {
        if (!size) {
            return;
        }

        ucs_memory_type_t mem_type;
        ucs_status_t status = ucs_memtype_cache_lookup(m_memtype_cache, ptr,
                                                       size, &mem_type);
        EXPECT_UCS_OK(status);
        EXPECT_EQ(expected_type, mem_type) << "ptr=" << ptr << " size=" << size;
    }

    void test_lookup_notfound(void *ptr, size_t size) const {
        if (!size) {
            return;
        }

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

    void test_ptr_found(void *ptr, size_t size,
                        ucs_memory_type_t expected_type) const {
        if (expected_type == UCS_MEMORY_TYPE_HOST) {
            return;
        }

        test_lookup_found(ptr, size, expected_type);
        test_lookup_found(ptr, size / 2, expected_type);
        test_lookup_found(ptr, 1, expected_type);
        test_lookup_found(UCS_PTR_BYTE_OFFSET(ptr, size - 1),
                          1, expected_type);
        test_lookup_found(ptr, 0, expected_type);
    }

    void test_region_found(mem_buffer &b, size_t size) const {
        test_ptr_found(b.ptr(), size, b.mem_type());
    }

    void test_region_not_found(mem_buffer &b, size_t size) const {
        test_ptr_not_found(b.ptr(), size);
    }

    void test_ptr_not_found(void *ptr, size_t size) const {
        /* memtype cache is aligned by Page Table defined constant,
         * so need to step by this value to make something not found */
        test_lookup_notfound(ptr, size + UCS_PGT_ADDR_ALIGN);
        test_lookup_notfound(UCS_PTR_BYTE_OFFSET(ptr, size), 1 + UCS_PGT_ADDR_ALIGN);
    }

    void test_ptr_released(void *ptr, size_t size) const {
        test_lookup_notfound(ptr, size);
        test_lookup_notfound(ptr, 1);
    }

    mem_buffer* allocate_mem_buffer(size_t size, ucs_memory_type_t mem_type,
                                    std::vector<mem_buffer*> *allocated_buffers = NULL,
                                    bool test_not_found = true) const {
        mem_buffer *buf = new mem_buffer(size, mem_type);

        if (allocated_buffers != NULL) {
            allocated_buffers->push_back(buf);
        }

        test_region_found(*buf, buf->size());

        if (test_not_found) {
            test_region_not_found(*buf, buf->size());
        }

        return buf;
    }

    void release_mem_buffer(mem_buffer *buf,
                            std::vector<std::pair<void*, size_t> > *released_ptrs,
                            std::vector<mem_buffer*> *allocated_buffers = NULL) const {
        if (allocated_buffers != NULL) {
            allocated_buffers->pop_back();
        }

        released_ptrs->push_back(std::make_pair(buf->ptr(), buf->size()));

        delete buf;
    }

    void test_ptrs_released(std::vector<std::pair<void*, size_t> > *released_ptrs) const {
        while (!released_ptrs->empty()) {
            void *ptr   = released_ptrs->back().first;
            size_t size = released_ptrs->back().second;

            test_ptr_released(ptr, size);
            test_ptr_not_found(ptr, size);

            released_ptrs->pop_back();
        }
    }

    void release_buffers(std::vector<mem_buffer*> *allocated_buffers) const {
        std::vector<std::pair<void*, size_t> > released_ptrs;

        while (!allocated_buffers->empty()) {
            release_mem_buffer(allocated_buffers->back(),
                               &released_ptrs, allocated_buffers);
        }

        test_ptrs_released(&released_ptrs);
    }

    size_t get_test_step(size_t portions = 64) const {
        return (RUNNING_ON_VALGRIND ?
                (ucs_get_page_size() / 2 - 1) :
                (ucs_get_page_size() / portions));
    }

    void test_memtype_cache_alloc_diff_mem_types(bool keep_buffers,
                                                 bool same_size_buffers) {
        const size_t step       = get_test_step();
        const size_t inner_step = (same_size_buffers ?
                                   ucs_get_page_size() : step);
        std::vector<std::pair<void*, size_t> > released_ptrs;
        std::vector<mem_buffer*> allocated_buffers;

        const std::vector<ucs_memory_type_t> supported_mem_types =
            mem_buffer::supported_mem_types();    

        /* The tests try to allocate two buffers with different memory types */
        for (std::vector<ucs_memory_type_t>::const_iterator iter =
                 supported_mem_types.begin();
             iter != supported_mem_types.end(); ++iter) {
            for (size_t i = 1; i <= ucs_get_page_size(); i += step) {
                mem_buffer *buf1 = allocate_mem_buffer(i, GetParam(),
                                                       &allocated_buffers, 0);

                for (size_t j = 1; j <= ucs_get_page_size(); j += inner_step) {
                    mem_buffer *buf2 = allocate_mem_buffer(j, *iter,
                                                           &allocated_buffers,
                                                           0);
                    if (!keep_buffers) {
                        release_mem_buffer(buf2, &released_ptrs);
                    }
                }

                if (!keep_buffers) {
                    release_mem_buffer(buf1, &released_ptrs);
                }
            }

            if (keep_buffers) {
                /* release allocated buffers */
                release_buffers(&allocated_buffers);
            } else {
                /* test released buffers */
                test_ptrs_released(&released_ptrs);
            }
        }
    }

private:
    ucs_memtype_cache_t *m_memtype_cache;
};

UCS_TEST_P(test_memtype_cache, basic) {
    const size_t size = 64;
    void *ptr;

    {
        mem_buffer b(size, GetParam());

        test_region_found(b, size);
        test_region_not_found(b, size);

        ptr = b.ptr();
    }

    /* buffer is released */
    test_ptr_released(ptr, size);
    test_ptr_not_found(ptr, size);
}

UCS_TEST_P(test_memtype_cache, shared_page_regions) {
    const std::vector<ucs_memory_type_t> supported_mem_types =
        mem_buffer::supported_mem_types();
    const size_t size = 1000000;

    for (std::vector<ucs_memory_type_t>::const_iterator iter =
             supported_mem_types.begin();
         iter != supported_mem_types.end(); ++iter) {

        std::vector<std::pair<void*, size_t> > released_ptrs;

        /* Create two buffers that possibly will share one page
         *
         *                         < shared page >
         *                            ||    ||
         *                            \/    ||
         *        +----------------------+  ||
         * buf1:  |    |    |    |    |  |  \/
         *        +----------------------+----------------------+
         *                        buf2:  |  |    |    |    |    |
         *                               +----------------------+
         */
        mem_buffer *buf1 = allocate_mem_buffer(size, GetParam());
        mem_buffer *buf2 = allocate_mem_buffer(size, *iter);

        test_region_found(*buf1, size);
        test_region_found(*buf2, size);

        release_mem_buffer(buf2, &released_ptrs);

        /* check that `buf1` was not released accidentally
         * after releasing `buf2` */
        test_region_found(*buf1, size);

        release_mem_buffer(buf1, &released_ptrs);

        /* buffer `buf1` and `buf2` are released */
        test_ptrs_released(&released_ptrs);
    }
}

UCS_TEST_P(test_memtype_cache, diff_mem_types_same_bufs) {
    test_memtype_cache_alloc_diff_mem_types(false, true);
}

UCS_TEST_P(test_memtype_cache, diff_mem_types_same_bufs_keep_mem) {
    test_memtype_cache_alloc_diff_mem_types(true, true);
}

UCS_TEST_P(test_memtype_cache, diff_mem_types_diff_bufs) {
    test_memtype_cache_alloc_diff_mem_types(false, false);
}

UCS_TEST_P(test_memtype_cache, diff_mem_types_diff_bufs_keep_mem) {
    test_memtype_cache_alloc_diff_mem_types(true, false);
}

INSTANTIATE_TEST_CASE_P(mem_type, test_memtype_cache,
                        ::testing::ValuesIn(mem_buffer::supported_mem_types()));
