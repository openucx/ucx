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
    test_memtype_cache() {
    }

    virtual void init() {
        ucs::test_with_param<ucs_memory_type_t>::init();
        // Trigger on-demand create of the global memtype cache instance
        test_lookup_notfound(NULL, ucs_get_page_size());
    }

    virtual void cleanup() {

        ucs::test_with_param<ucs_memory_type_t>::cleanup();
    }

    void
    check_lookup(const void *ptr, size_t size, bool expect_found,
                 ucs_memory_type_t expected_type = UCS_MEMORY_TYPE_UNKNOWN) const
    {
        if (!size) {
            return;
        }

        ucs_memory_info_t mem_info;
        ucs_status_t status = ucs_memtype_cache_lookup(ptr, size, &mem_info);

        if (!expect_found || (expected_type == UCS_MEMORY_TYPE_HOST)) {
            /* memory type should be not found or unknown */
            if (status != UCS_ERR_NO_ELEM) {
                ASSERT_UCS_OK(status, << " ptr=" << ptr << " size=" << size);
                EXPECT_EQ(UCS_MEMORY_TYPE_UNKNOWN, mem_info.type)
                        << "ptr=" << ptr << " size=" << size
                        << mem_buffer::mem_type_name(mem_info.type);
            }
        } else {
            ASSERT_UCS_OK(status, << " ptr=" << ptr << " size=" << size);
            EXPECT_TRUE((UCS_MEMORY_TYPE_UNKNOWN == mem_info.type) ||
                        (expected_type == mem_info.type))
                    << "ptr=" << ptr << " size=" << size
                    << " type=" << mem_buffer::mem_type_name(mem_info.type);
        }
    }

    void test_lookup_found(const void *ptr, size_t size,
                           ucs_memory_type_t expected_type) const {
        check_lookup(ptr, size, true, expected_type);
    }

    void test_lookup_notfound(const void *ptr, size_t size) const {
        check_lookup(ptr, size, false);
    }

    void test_ptr_found(const void *ptr, size_t size,
                        ucs_memory_type_t expected_type) const {
        test_lookup_found(ptr, size, expected_type);
        test_lookup_found(ptr, size / 2, expected_type);
        test_lookup_found(ptr, 1, expected_type);
        test_lookup_found(UCS_PTR_BYTE_OFFSET(ptr, size - 1),
                          1, expected_type);
        test_lookup_found(ptr, 0, expected_type);
    }

    void test_region_found(const mem_buffer &b) const {
        test_ptr_found(b.ptr(), b.size(), b.mem_type());
    }

    void test_region_not_found(const mem_buffer &b) const {
        test_ptr_not_found(b.ptr(), b.size());
    }

    void test_ptr_not_found(const void *ptr, size_t size) const {
        /* memtype cache is aligned by Page Table defined constant,
         * so need to step by this value to make something not found */
        test_lookup_notfound(ptr, size + UCS_PGT_ADDR_ALIGN);
        test_lookup_notfound(UCS_PTR_BYTE_OFFSET(ptr, size), 1 + UCS_PGT_ADDR_ALIGN);
    }

    void test_ptr_released(const void *ptr, size_t size) const {
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

        test_region_found(*buf);

        if (test_not_found) {
            test_region_not_found(*buf);
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

        /* The tests try to allocate two buffers with different memory types */
        for (auto mem_type : mem_buffer::supported_mem_types()) {
            for (size_t i = 1; i <= ucs_get_page_size(); i += step) {
                mem_buffer *buf1 = allocate_mem_buffer(i, GetParam(),
                                                       &allocated_buffers, 0);

                for (size_t j = 1; j <= ucs_get_page_size(); j += inner_step) {
                    mem_buffer *buf2 = allocate_mem_buffer(j, mem_type,
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

    struct region_info {
        void              *start;
        void              *end;
        ucs_memory_type_t mem_type;

        region_info(size_t start, size_t end,
                    ucs_memory_type_t mem_type) :
            start(reinterpret_cast<void*>(start)),
            end(reinterpret_cast<void*>(end)),
            mem_type(mem_type) {}
    };

    void generate_test_remove_subintervals(
            const std::vector<region_info> &insert_regions,
            size_t interval_start_offset, size_t interval_end_offset,
            std::vector<region_info> &remove_regions) {
        // add regions that will be removed as intervals
        for (std::vector<region_info>::const_iterator iter =
                 insert_regions.begin(); iter != insert_regions.end(); ++iter) {
            remove_regions.push_back(region_info(reinterpret_cast<size_t>(iter->start) +
                                                 interval_start_offset,
                                                 reinterpret_cast<size_t>(iter->end) -
                                                 interval_end_offset,
                                                 UCS_MEMORY_TYPE_LAST));
        }

        // add regions that will be removed as remaining intervals
        for (std::vector<region_info>::const_iterator iter =
                 insert_regions.begin(); iter != insert_regions.end(); ++iter) {
            if (interval_start_offset) {
                remove_regions.push_back(region_info(reinterpret_cast<size_t>(iter->start),
                                                     reinterpret_cast<size_t>(iter->start) +
                                                     interval_start_offset,
                                                     UCS_MEMORY_TYPE_LAST));
            }

            if (interval_end_offset) {
                remove_regions.push_back(region_info(reinterpret_cast<size_t>(iter->end) -
                                                     interval_end_offset,
                                                     reinterpret_cast<size_t>(iter->end),
                                                     UCS_MEMORY_TYPE_LAST));
            }
        }
    }

    void test_region_insert_and_remove_subintervals(const std::vector<region_info> &regions,
                                                    size_t interval_start_offset,
                                                    size_t interval_end_offset,
                                                    std::vector<region_info> &remove_regions) {
        generate_test_remove_subintervals(regions, interval_start_offset,
                                          interval_end_offset, remove_regions);

        // insert new regions
        for (std::vector<region_info>::const_iterator iter =
                 regions.begin(); iter != regions.end(); ++iter) {
            size_t size = UCS_PTR_BYTE_DIFF(iter->start, iter->end);
            memtype_cache_update(iter->start, size, iter->mem_type);
            test_ptr_found(iter->start, size, iter->mem_type);
        }

        // remove subintervals
        for (std::vector<region_info>::const_iterator iter =
                 remove_regions.begin(); iter != remove_regions.end(); ++iter) {
            size_t size = UCS_PTR_BYTE_DIFF(iter->start, iter->end);
            memtype_cache_remove(iter->start, size);
            test_ptr_released(iter->start, size);
        }

        // now all buffers released, check that can't find them
        for (std::vector<region_info>::const_iterator iter =
                 regions.begin(); iter != regions.end(); ++iter) {
            size_t size = UCS_PTR_BYTE_DIFF(iter->start, iter->end);
            test_ptr_released(iter->start, size);
            test_ptr_not_found(iter->start, size);
        }
    }

    void memtype_cache_update(const void *ptr, size_t size,
                              ucs_memory_type_t mem_type) {
        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            return;
        }

        ucs_memtype_cache_update(ptr, size, mem_type,
                                 UCS_SYS_DEVICE_ID_UNKNOWN);
    }

    void memtype_cache_update(const mem_buffer &b) {
        memtype_cache_update(b.ptr(), b.size(), b.mem_type());
    }

    void memtype_cache_remove(const void *ptr, size_t size) {
        ucs_memtype_cache_remove(ptr, size);
    }
};

UCS_TEST_P(test_memtype_cache, basic) {
    const size_t size = 64;
    void *ptr;

    {
        mem_buffer b(size, GetParam());

        test_region_found(b);
        test_region_not_found(b);

        ptr = b.ptr();
    }

    /* buffer is released */
    test_ptr_released(ptr, size);
    test_ptr_not_found(ptr, size);
}

UCS_TEST_P(test_memtype_cache, update_non_contig_regions_and_remove_subintervals) {
    std::vector<test_memtype_cache::region_info> insert_regions;
    std::vector<test_memtype_cache::region_info> remove_regions;
    size_t start, end;

    const size_t region_size           = UCS_BIT(28);
    const size_t interval_start_offset = UCS_BIT(27);

    // insert [0x7f6ef0000000 .. 0x7f6f00000000]
    start = 0x7f6ef0000000;
    end   = start + region_size;
    test_memtype_cache::region_info region_info1(start, end, GetParam());
    insert_regions.push_back(region_info1);

    // insert [0x7f6f2c021000 .. 0x7f6f3c021000]
    start = 0x7f6f2c021000;
    end   = start + region_size;
    test_memtype_cache::region_info region_info2(start, end,
                                                 UCS_MEMORY_TYPE_LAST);
    insert_regions.push_back(region_info2);

    // insert [0x7f6f42000000 .. 0x7f6f52000000]
    start = 0x7f6f42000000;
    end   = start + region_size;
    test_memtype_cache::region_info region_info3(start, end,
                                                 UCS_MEMORY_TYPE_LAST);
    insert_regions.push_back(region_info3);

    test_region_insert_and_remove_subintervals(insert_regions,
                                               interval_start_offset,
                                               0, remove_regions);
}

UCS_TEST_P(test_memtype_cache, update_adjacent_regions_and_remove_subintervals) {
    std::vector<test_memtype_cache::region_info> insert_regions;
    std::vector<test_memtype_cache::region_info> remove_regions;
    size_t start, end;

    const size_t region_size           = UCS_BIT(28);
    const size_t interval_start_offset = UCS_BIT(27);

    // insert [0x7f6ef0000000 .. 0x7f6f00000000]
    start = 0x7f6ef0000000;
    end   = start + region_size;
    test_memtype_cache::region_info region_info1(0x7f6ef0000000, 0x7f6f00000000,
                                                 GetParam());
    insert_regions.push_back(region_info1);

    // insert [0x7f6f00000000 .. 0x7f6f10000000]
    start = end;
    end   = start + region_size;
    test_memtype_cache::region_info region_info2(reinterpret_cast<size_t>
                                                 (region_info1.end),
                                                 0x7f6f40000000, GetParam());
    insert_regions.push_back(region_info2);

    // insert [0x7f6f10000000 .. 0x7f6f20000000]
    start = end;
    end   = start + region_size;
    test_memtype_cache::region_info region_info3(reinterpret_cast<size_t>
                                                 (region_info2.end),
                                                 0x7f6f48000000, GetParam());
    insert_regions.push_back(region_info3);

    test_region_insert_and_remove_subintervals(insert_regions,
                                               interval_start_offset,
                                               0, remove_regions);
}

UCS_TEST_P(test_memtype_cache, shared_page_regions) {
    const size_t size = 1000000;

    for (auto mem_type : mem_buffer::supported_mem_types()) {
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
        mem_buffer *buf2 = allocate_mem_buffer(size, mem_type);

        test_region_found(*buf1);
        test_region_found(*buf2);

        release_mem_buffer(buf2, &released_ptrs);

        /* check that `buf1` was not released accidentally
         * after releasing `buf2` */
        test_region_found(*buf1);

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

INSTANTIATE_TEST_SUITE_P(mem_type, test_memtype_cache,
                        ::testing::ValuesIn(mem_buffer::supported_mem_types()));

class test_memtype_cache_deferred_create : public test_memtype_cache {
protected:
    virtual void init() {
        /* do nothing */
    }

    void test_unknown_region_found(const mem_buffer &b) const {
        test_ptr_found(b.ptr(), b.size(),
                       ((b.mem_type() == UCS_MEMORY_TYPE_HOST) ?
                        UCS_MEMORY_TYPE_HOST :
                        UCS_MEMORY_TYPE_LAST));
    }

    void test_alloc_before_init(size_t buf_size, bool test_adjacent,
                                size_t overlap_size) {
        void *ptr;

        {
            mem_buffer b(buf_size, GetParam());

            test_memtype_cache::init();

            test_unknown_region_found(b);
            test_region_not_found(b);

            if (test_adjacent) {
                /* add two adjacent regions: */
                memtype_cache_update(b.ptr(), b.size() / 2, b.mem_type());
                test_ptr_found(b.ptr(), b.size() / 2, b.mem_type());
                memtype_cache_update(UCS_PTR_BYTE_OFFSET(b.ptr(),
                                                         b.size() / 2 - overlap_size),
                                     b.size() / 2 + 1, b.mem_type());
                test_ptr_found(b.ptr(), b.size() / 2, b.mem_type());
            } else {
                memtype_cache_update(b);
            }

            /* check that able to find the entire region */
            test_region_found(b);

            ptr = b.ptr();
        }

        /* buffer is released */
        test_ptr_released(ptr, buf_size);
        test_ptr_not_found(ptr, buf_size);
    }
};

UCS_TEST_P(test_memtype_cache_deferred_create, allocate_and_update) {
    test_alloc_before_init(1000000, false, 0);
}

UCS_TEST_P(test_memtype_cache_deferred_create, lookup_adjacent_regions) {
    test_alloc_before_init(1000000, false, 0);
}

UCS_TEST_P(test_memtype_cache_deferred_create, lookup_overlapped_regions) {
    test_alloc_before_init(1000000, false, 1);
}

INSTANTIATE_TEST_SUITE_P(mem_type, test_memtype_cache_deferred_create,
                        ::testing::ValuesIn(mem_buffer::supported_mem_types()));
