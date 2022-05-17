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

    static void obj_str(ucs_mpool_t *mp, void *obj, ucs_string_buffer_t *strb)
    {
        ucs_string_buffer_appendf(strb, "test-obj-%p", obj);
    }

    static ucs_log_func_rc_t
    mpool_log_handler(const char *file, unsigned line, const char *function,
                      ucs_log_level_t level,
                      const ucs_log_component_config_t *comp_conf,
                      const char *message, va_list ap)
    {
        // Ignore errors that invalid input parameters as it is expected
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            std::string exp_str = "Invalid memory pool parameter(s)";

            if (err_str == exp_str) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static bool is_leak_str(const std::string &str)
    {
        return (str.find("not returned to mpool test") != std::string::npos) &&
               (str.find("{test-obj-") != std::string::npos);
    }

    static ucs_log_func_rc_t
    mpool_log_leak_handler(const char *file, unsigned line,
                           const char *function, ucs_log_level_t level,
                           const ucs_log_component_config_t *comp_conf,
                           const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_WARN) {
            std::string msg = format_message(message, ap);
            if (is_leak_str(msg)) {
                UCS_TEST_MESSAGE << "< " << msg << " >";
                ++leak_count;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static const size_t header_size = 30;
    static const size_t data_size = 152;
    static const size_t align = 128;
    static size_t leak_count;
};

size_t test_mpool::leak_count;

UCS_TEST_F(test_mpool, no_allocs) {
    ucs_mpool_t mp;
    ucs_status_t status;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
       NULL,
       NULL
    };

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                            6, 18, &ops, "test");
    ASSERT_UCS_OK(status);
    ucs_mpool_cleanup(&mp, 1);
}

UCS_TEST_F(test_mpool, wrong_ops) {
    ucs_mpool_t mp;
    ucs_status_t status;
    ucs_mpool_ops_t ops = { 0 };
    scoped_log_handler log_handler(mpool_log_handler);

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                            6, 18, &ops, "test");
    EXPECT_TRUE(status == UCS_ERR_INVALID_PARAM);
}

UCS_TEST_F(test_mpool, basic) {
    ucs_status_t status;
    ucs_mpool_t mp;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
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

UCS_TEST_F(test_mpool, leak_check) {
    ucs_mpool_t mp;
    ucs_status_t status;

    ucs_mpool_ops_t ops = {
        ucs_mpool_chunk_malloc,
        ucs_mpool_chunk_free,
        NULL,
        NULL,
        obj_str
    };

    status = ucs_mpool_init(&mp, 0, header_size + data_size, header_size, align,
                            6, 18, &ops, "test");
    ASSERT_UCS_OK(status);

    for (int i = 0; i < 5; ++i) {
        void *obj = ucs_mpool_get(&mp);
        EXPECT_TRUE(obj != NULL);
    }
    // Do not release allocated objects

    leak_count = 0;
    scoped_log_handler log_handler(mpool_log_leak_handler);
    ucs_mpool_cleanup(&mp, 1);

    EXPECT_EQ(5u, leak_count);
}

UCS_TEST_SKIP_COND_F(test_mpool, alloc_4g, RUNNING_ON_VALGRIND) {
    const size_t elem_size         = 32 * UCS_MBYTE;
    const unsigned elems_per_chunk = ucs::limit_buffer_size(4 * UCS_GBYTE) /
                                     elem_size;
    ucs_mpool_ops_t mpool_ops = {ucs_mpool_chunk_malloc, ucs_mpool_chunk_free,
                                 NULL, NULL, NULL};
    ucs_mpool_t mp;

    ucs_status_t status = ucs_mpool_init(&mp, 0, header_size + elem_size,
                                         header_size, align, elems_per_chunk,
                                         elems_per_chunk, &mpool_ops, "test");
    ASSERT_UCS_OK(status);

    // Allocate objects per one chunk size
    std::vector<void*> objs;
    for (unsigned i = 0; i < elems_per_chunk; ++i) {
        void *obj = ucs_mpool_get(&mp);
        if (obj == NULL) {
            ADD_FAILURE() << "Could not allocate object [" << i
                          << "] from mpool";
            break;
        }
        objs.push_back(obj);
    }

    // Release allocated objects
    while (!objs.empty()) {
        ucs_mpool_put(objs.back());
        objs.pop_back();
    }

    ucs_mpool_cleanup(&mp, 1);
}
