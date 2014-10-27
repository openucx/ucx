/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
extern "C" {
#include <ucs/datastruct/mpool.h>
}

#include <vector>
#include <queue>

class test_mpool : public ucs::test {
protected:
    static void *test_alloc(size_t *size, void *context UCS_MEMTRACK_ARG) {
        return (*(void**)context = malloc(*size));
    }

    static void test_free(void *ptr, void *context) {
        free(ptr);
        *(void**)context = NULL;
    }

    static const size_t header_size = 30;
    static const size_t data_size = 152;
    static const size_t align = 114;
};


UCS_TEST_F(test_mpool, no_allocs) {
    ucs_mpool_h mp;
    ucs_status_t status;

    status = ucs_mpool_create("test", header_size + data_size, header_size, align,
                             6, 18, NULL,
                             ucs_mpool_chunk_malloc,
                             ucs_mpool_chunk_free,
                             NULL, NULL, &mp);
    ASSERT_UCS_OK(status);
    ucs_mpool_destroy(mp);
}

UCS_TEST_F(test_mpool, basic) {
    ucs_status_t status;
    ucs_mpool_h mp;

    status = ucs_mpool_create("test", header_size + data_size, header_size, align,
                             6, 18, NULL,
                             ucs_mpool_chunk_malloc,
                             ucs_mpool_chunk_free,
                             NULL, NULL, &mp);
    ASSERT_UCS_OK(status);

    for (unsigned loop = 0; loop < 10; ++loop) {
        std::vector<void*> objs;
        for (unsigned i = 0; i < 18; ++i) {
            void *ptr = ucs_mpool_get(mp);
            ASSERT_TRUE(ptr != NULL);
            ASSERT_EQ(0ul, ((uintptr_t)ptr + header_size) % align) << ptr;
            memset(ptr, 0xAA, header_size + data_size);
            objs.push_back(ptr);
        }

        ASSERT_TRUE(NULL == ucs_mpool_get(mp));

        for (std::vector<void*>::iterator iter = objs.begin(); iter != objs.end(); ++iter) {
            ucs_mpool_put(*iter);
        }
    }

    ucs_mpool_destroy(mp);
}

UCS_TEST_F(test_mpool, custom_alloc) {
    ucs_status_t status;
    ucs_mpool_h mp;
    void *ptr = NULL;

    status = ucs_mpool_create("test", header_size + data_size, header_size, align, 5, 18,
                             &ptr, test_alloc, test_free, NULL, NULL, &mp);
    ASSERT_UCS_OK(status);

    void *obj = ucs_mpool_get(mp);
    EXPECT_TRUE(obj != NULL);
    EXPECT_TRUE(ptr != NULL);

    ucs_mpool_put(obj);

    ucs_mpool_destroy(mp);
    EXPECT_EQ(NULL, ptr);
}

UCS_TEST_F(test_mpool, infinite) {
    const unsigned NUM_ELEMS = 1000000 / ucs::test_time_multiplier();
    ucs_status_t status;
    ucs_mpool_h mp;

    status = ucs_mpool_create("test", header_size + data_size, header_size, align,
                             10000, UCS_MPOOL_INFINITE,
                             NULL,
                             ucs_mpool_chunk_malloc,
                             ucs_mpool_chunk_free,
                             NULL, NULL, &mp);
    ASSERT_UCS_OK(status);

    std::queue<void*> q;
    for (unsigned i = 0; i < NUM_ELEMS; ++i) {
        void *obj = ucs_mpool_get(mp);
        ASSERT_TRUE(obj != NULL);
        q.push(obj);
    }

    while (!q.empty()) {
        ucs_mpool_put(q.front());
        q.pop();
    }

    ucs_mpool_destroy(mp);
}
