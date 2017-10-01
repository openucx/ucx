/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/datastruct/mpool.h>
}

#include <limits.h>
#include <vector>
#include <queue>

class test_mpool : public ucs::test {
protected:
    static ucs_status_t test_alloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p) {
        *chunk_p = malloc(*size_p);
        return (*chunk_p == NULL) ? UCS_ERR_NO_MEMORY : UCS_OK;
    }

    static void test_free(ucs_mpool_t *mp, void *chunk) {
        free(chunk);
    }

    static const size_t header_size = 30;
    static const size_t data_size = 152;
    static const size_t align = 128;
};


UCS_TEST_F(test_mpool, no_allocs) {
    ucs_mpool_t mp;
    ucs_status_t status;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
       NULL
    };

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                             6, 18, &ops, "test");
    ASSERT_UCS_OK(status);
    ucs_mpool_cleanup(&mp, 1);
}

UCS_TEST_F(test_mpool, basic) {
    ucs_status_t status;
    ucs_mpool_t mp;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
       NULL
    };

    push_config();

    for (int mpool_fifo = 0; mpool_fifo <= 1; ++mpool_fifo) {
#if ENABLE_DEBUG_DATA
        modify_config("MPOOL_FIFO", ucs::to_string(mpool_fifo).c_str());
#else
        if (mpool_fifo == 1) {
            continue;
        }
#endif
        status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                                 6, 18, &ops, "test");
        ASSERT_UCS_OK(status);

        for (unsigned loop = 0; loop < 10; ++loop) {
            std::vector<void*> objs;
            for (unsigned i = 0; i < 18; ++i) {
                void *ptr = ucs_mpool_get(&mp);
                ASSERT_TRUE(ptr != NULL);
                ASSERT_EQ(0ul, ((uintptr_t)ptr + header_size) % align) << ptr;
                memset(ptr, 0xAA, header_size + data_size);
                objs.push_back(ptr);
            }

            ASSERT_TRUE(NULL == ucs_mpool_get(&mp));

            for (std::vector<void*>::iterator iter = objs.begin(); iter != objs.end(); ++iter) {
                ucs_mpool_put(*iter);
            }
        }

        ucs_mpool_cleanup(&mp, 1);
    }

    pop_config();
}

UCS_TEST_F(test_mpool, custom_alloc) {
    ucs_status_t status;
    ucs_mpool_t mp;

    ucs_mpool_ops_t ops = {
       test_alloc,
       test_free,
       NULL,
       NULL
    };

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                            5, 18, &ops, "test");
    ASSERT_UCS_OK(status);

    void *obj = ucs_mpool_get(&mp);
    EXPECT_TRUE(obj != NULL);

    ucs_mpool_put(obj);

    ucs_mpool_cleanup(&mp, 1);
}

UCS_TEST_F(test_mpool, grow) {
    ucs_status_t status;
    ucs_mpool_t mp;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
       NULL
    };

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                            1000, 2000, &ops, "test");
    ASSERT_UCS_OK(status);

    ucs_mpool_grow(&mp, 1);

    void *obj = ucs_mpool_get(&mp);
    EXPECT_TRUE(obj != NULL);

    ucs_mpool_put(obj);

    ucs_mpool_cleanup(&mp, 1);
}

UCS_TEST_F(test_mpool, infinite) {
    const unsigned NUM_ELEMS = 1000000 / ucs::test_time_multiplier();
    ucs_status_t status;
    ucs_mpool_t mp;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
       NULL
    };

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                            10000, UINT_MAX, &ops, "test");
    ASSERT_UCS_OK(status);

    std::queue<void*> q;
    for (unsigned i = 0; i < NUM_ELEMS; ++i) {
        void *obj = ucs_mpool_get(&mp);
        ASSERT_TRUE(obj != NULL);
        q.push(obj);
    }

    while (!q.empty()) {
        ucs_mpool_put(q.front());
        q.pop();
    }

    ucs_mpool_cleanup(&mp, 1);
}
