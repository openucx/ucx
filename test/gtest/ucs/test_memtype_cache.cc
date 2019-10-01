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

#include <deque>

class test_memtype_cache : public ucs::test_with_param<ucs_memory_type_t> {
protected:
    test_memtype_cache() : m_memtype_cache(NULL) {
    }

    virtual void init() {
        ucs::test_with_param<ucs_memory_type_t>::init();
        ucs_status_t status = ucs_memtype_cache_create(&m_memtype_cache);
        ASSERT_UCS_OK(status);

        m_type = GetParam();
    }

    virtual void cleanup() {
        ucs_memtype_cache_destroy(m_memtype_cache);
        ucs::test_with_param<ucs_memory_type_t>::cleanup();
    }

    void test_lookup_found(void *ptr, size_t size,
                           ucs_memory_type_t expected_type = m_type) const {
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
        if (expected_type != UCS_MEMORY_TYPE_HOST) {
            test_lookup_found(ptr, size, expected_type);
            test_lookup_found(ptr, size / 2, expected_type);
            test_lookup_found(ptr, 1, expected_type);
            test_lookup_found(UCS_PTR_BYTE_OFFSET(ptr, size - 1),
                              1, expected_type);
            test_lookup_found(ptr, 0, expected_type);
        }
    }

    void test_region_found(mem_buffer &b, size_t size) const {
        test_ptr_found(b.ptr(), size, b.mem_type());
    }

    void test_region_not_found(mem_buffer &b, size_t size) const {
        test_ptr_not_found(b.ptr(), size);
    }

    void test_ptr_not_found(void *ptr, size_t size) const {
        /* memtype cache is page-aligned, so need to step
         * by page size to make something not found */        
        test_lookup_notfound(ptr, size + ucs_get_page_size());
        test_lookup_notfound(UCS_PTR_BYTE_OFFSET(ptr, size + ucs_get_page_size()), 1);
    }

    void test_ptr_released(void *ptr, size_t size,
                           /* the flag indicates that the first/last page
                            * maybe shared between two regions */
                           bool page_maybe_shared = false) const {
        if (!page_maybe_shared) {
            test_lookup_notfound(ptr, size);
            test_lookup_notfound(ptr, 1);
        } else {
            test_lookup_notfound(ptr, size - ucs_min(ucs_get_page_size(), size));
            test_lookup_notfound(ptr, 1 + ucs_get_page_size());
        }
    }

    void simulate_ucm_mem_type_event(void *ptr, size_t size,
                                     ucs_memory_type_t mem_type,
                                     ucm_event_type_t event_type) const {
        ASSERT_TRUE((event_type == UCM_EVENT_MEM_TYPE_ALLOC) ||
                    (event_type == UCM_EVENT_MEM_TYPE_FREE));

        ucm_event_t event;

        memset(&event, 0, sizeof(event));

        event.mem_type.address  = ptr;
        event.mem_type.size     = size;
        event.mem_type.mem_type = mem_type;

        ucm_event_enter();
        ucm_event_dispatch(event_type, &event);
        ucm_event_leave();
    }

    mem_buffer* allocate_mem_buffer(void *ptr, size_t size, ucs_memory_type_t mem_type,
                                    std::deque<mem_buffer*> *allocated_buffers = NULL,
                                    bool insert_inversed = false,
                                    bool test_not_found = true) const {
        mem_buffer *buf;

        if (ptr != NULL) {
            buf = new mem_buffer(ptr, size, mem_type);
            simulate_ucm_mem_type_event(ptr, size, mem_type,
                                        UCM_EVENT_MEM_TYPE_ALLOC);
        } else {
            buf = new mem_buffer(size, mem_type);
        }

        if (allocated_buffers != NULL) {
            if (!insert_inversed) {
                allocated_buffers->push_back(buf);
            } else {
                allocated_buffers->push_front(buf);
            }
        }

        test_region_found(*buf, buf->size());

        if (test_not_found) {
            test_region_not_found(*buf, buf->size());
        }

        return buf;
    }

    mem_buffer* allocate_mem_buffer(size_t size, ucs_memory_type_t mem_type,
                                    std::deque<mem_buffer*> *allocated_buffers = NULL,
                                    bool insert_inversed = false,
                                    bool test_not_found = true) const {
        return allocate_mem_buffer(NULL, size, mem_type, allocated_buffers,
                                   insert_inversed, test_not_found);
    }

    void release_mem_buffer(mem_buffer *buf,
                            std::vector<std::pair<void*, size_t> > *released_ptrs,
                            std::deque<mem_buffer*> *allocated_buffers = NULL) const {
        if (allocated_buffers != NULL) {
            if (buf == allocated_buffers->back()) {
                allocated_buffers->pop_back();
            } else if (buf == allocated_buffers->front()) {
                allocated_buffers->pop_front();
            } else {
                UCS_TEST_ABORT("provided mem_bufer must be equal to either front() or back()");
            }
        }

        released_ptrs->push_back(std::make_pair(buf->ptr(), buf->size()));

        if (buf->external_mem()) {
            simulate_ucm_mem_type_event(buf->ptr(), buf->size(),
                                        buf->mem_type(), UCM_EVENT_MEM_TYPE_FREE);
        }

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

    void release_buffers(std::deque<mem_buffer*> *allocated_buffers, bool inversed) const {
        std::vector<std::pair<void*, size_t> > released_ptrs;

        while (!allocated_buffers->empty()) {
            release_mem_buffer(inversed ?
                               allocated_buffers->front() :
                               allocated_buffers->back(),
                               &released_ptrs, allocated_buffers);
        }

        test_ptrs_released(&released_ptrs);
    }

    void *get_next_ptr(void *cur_p, size_t offset) const {
        return ((cur_p == NULL) ? NULL :
                UCS_PTR_BYTE_OFFSET(cur_p, offset));
    }

    size_t get_test_step(size_t portions = 64) const {
        return (RUNNING_ON_VALGRIND ?
                (ucs_get_page_size() / 2 - 1) :
                (ucs_get_page_size() / portions));
    }

private:
    ucs_memtype_cache_t      *m_memtype_cache;
    static ucs_memory_type_t m_type;
};

ucs_memory_type_t test_memtype_cache::m_type = UCS_MEMORY_TYPE_LAST;

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
    void *buf2_ptr, *cur_ptr;
    std::vector<void*> initial_ptrs;

    initial_ptrs.push_back(NULL);
    initial_ptrs.push_back((void*)ucs_get_page_size());

    for (std::vector<void*>::const_iterator initial_ptr = initial_ptrs.begin();
         initial_ptr != initial_ptrs.end(); ++initial_ptr) {
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
            cur_ptr          = get_next_ptr(*initial_ptr, 0);
            mem_buffer *buf1 = allocate_mem_buffer(cur_ptr, size, GetParam());

            cur_ptr          = get_next_ptr(cur_ptr, size);            
            mem_buffer *buf2 = allocate_mem_buffer(size, *iter);

            test_region_found(*buf1, size);
            test_region_found(*buf2, size);

            buf2_ptr = buf2->ptr();
            release_mem_buffer(buf2, &released_ptrs);

            /* buffer `buf2` is released, but need to consider
             * a shared page with buffer `buf1` */
            test_ptr_released(buf2_ptr, size, 1);

            test_region_found(*buf1, size);

            release_mem_buffer(buf1, &released_ptrs);

            /* buffer `buf1` and `buf2` are released */
            test_ptrs_released(&released_ptrs);
        }
    }
}

UCS_TEST_P(test_memtype_cache, different_mem_types) {
    std::vector<void*> initial_ptrs;
    std::vector<bool> insert_type;

    initial_ptrs.push_back(NULL);
    initial_ptrs.push_back((void*)ucs_get_page_size());

    insert_type.push_back(false);
    insert_type.push_back(true);

    const std::vector<ucs_memory_type_t> supported_mem_types =
        mem_buffer::supported_mem_types();

    /* The tests try to allocate/simulate two buffers with different memory types */
    for (std::vector<void*>::const_iterator initial_ptr = initial_ptrs.begin();
         initial_ptr != initial_ptrs.end(); ++initial_ptr) {
        for (std::vector<ucs_memory_type_t>::const_iterator iter =
                 supported_mem_types.begin();
             iter != supported_mem_types.end(); ++iter) {
            /* 1. Allocate the same amount of memory for buffers */
            {
                const size_t step = get_test_step(1024);

                for (std::vector<bool>::const_iterator insert_inversed = insert_type.begin();
                     insert_inversed != insert_type.end(); ++insert_inversed) {
                    for (size_t i = 1; i <= ucs_get_page_size(); i += step) {
                        void *cur_ptr = get_next_ptr(*initial_ptr, 0);
                        std::deque<mem_buffer*> allocated_buffers;

                        allocate_mem_buffer(cur_ptr, i, GetParam(), &allocated_buffers,
                                            *insert_inversed, 1);

                        cur_ptr = get_next_ptr(cur_ptr, i);
                        allocate_mem_buffer(cur_ptr, i, *iter, &allocated_buffers,
                                            *insert_inversed, 1);

                        /* release allocated buffers */
                        release_buffers(&allocated_buffers, *insert_inversed);
                    }
                }
            }

            /* 2. Allocate the same amount of memory for buffers
             *    and keep them allocated until the end of the testing */
            {
                const size_t step = get_test_step();

                for (std::vector<bool>::const_iterator insert_inversed = insert_type.begin();
                     insert_inversed != insert_type.end(); ++insert_inversed) {
                    void *cur_ptr = get_next_ptr(*initial_ptr, 0);
                    std::deque<mem_buffer*> allocated_buffers;

                    for (size_t i = 1; i <= ucs_get_page_size(); i += step) {
                        allocate_mem_buffer(cur_ptr, i, GetParam(), &allocated_buffers,
                                            *insert_inversed, 0);
                        cur_ptr = get_next_ptr(cur_ptr, i);

                        allocate_mem_buffer(cur_ptr, i, *iter, &allocated_buffers,
                                            *insert_inversed, 0);
                        cur_ptr = get_next_ptr(cur_ptr, i);
                    }

                    /* release allocated buffers */
                    release_buffers(&allocated_buffers, *insert_inversed);
                }
            }

            /* 3. Allocate different amount of memory for buffers */
            {
                const size_t step = get_test_step(1024);
                /* this test doesn't support inversed insert */
                std::vector<std::pair<void*, size_t> > released_ptrs;

                for (size_t i = 1; i <= ucs_get_page_size(); i += step) {
                    void *cur_ptr = get_next_ptr(*initial_ptr, 0);

                    mem_buffer *buf1 = allocate_mem_buffer(cur_ptr, i, GetParam());
                    cur_ptr = get_next_ptr(cur_ptr, i);

                    for (size_t j = 1; j <= ucs_get_page_size(); j += step) {
                        mem_buffer *buf2 = allocate_mem_buffer(cur_ptr, j, *iter);
                        release_mem_buffer(buf2, &released_ptrs);
                    }

                    release_mem_buffer(buf1, &released_ptrs);
                }
            }

            /* 4. Allocate different amount of memory for buffers
             *    and keep them allocated until the end of testing */
            {
                const size_t step = get_test_step();

                for (std::vector<bool>::const_iterator insert_inversed = insert_type.begin();
                     insert_inversed != insert_type.end(); ++insert_inversed) {
                    void *cur_ptr = get_next_ptr(*initial_ptr, 0);
                    std::deque<mem_buffer*> allocated_buffers;

                    for (size_t i = 1; i <= ucs_get_page_size(); i += step) {
                        allocate_mem_buffer(cur_ptr, i, GetParam(), &allocated_buffers,
                                            *insert_inversed, 0);
                        cur_ptr = get_next_ptr(cur_ptr, i);

                        for (size_t j = 1; j <= ucs_get_page_size(); j += step) {
                            allocate_mem_buffer(j, *iter, &allocated_buffers,
                                                *insert_inversed, 0);
                            cur_ptr = get_next_ptr(cur_ptr, j);
                        }
                    }

                    /* release allocated buffers */
                    release_buffers(&allocated_buffers, *insert_inversed);
                }
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(mem_type, test_memtype_cache,
                        ::testing::ValuesIn(mem_buffer::supported_mem_types()));
