/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
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

    ucs_status_t setup_mpool(ucs_mpool_t *mp, size_t elem_size,
                             unsigned elems_per_chunk, unsigned max_elems = 0)
    {
        static ucs_mpool_ops_t mpool_ops = {ucs_mpool_chunk_malloc,
                                            ucs_mpool_chunk_free, NULL, NULL,
                                            NULL};
        if (max_elems == 0) {
            max_elems = elems_per_chunk;
        }

        ucs_mpool_params_t mp_params;

        ucs_mpool_params_reset(&mp_params);
        mp_params.elem_size       = header_size + elem_size;
        mp_params.align_offset    = header_size;
        mp_params.alignment       = align;
        mp_params.max_chunk_size  = 4 * UCS_GBYTE;
        mp_params.elems_per_chunk = elems_per_chunk;
        mp_params.max_elems       = max_elems;
        mp_params.ops             = &mpool_ops;
        mp_params.name            = "tests";
        return ucs_mpool_init(&mp_params, mp);
    }
};

size_t test_mpool::leak_count;

UCS_TEST_F(test_mpool, no_allocs) {
    ucs_mpool_t mp;
    ucs_status_t status;
    ucs_mpool_params_t mp_params;

    ucs_mpool_ops_t ops = {
       ucs_mpool_chunk_malloc,
       ucs_mpool_chunk_free,
       NULL,
       NULL,
       NULL
    };

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 6;
    mp_params.max_elems       = 18;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    status = ucs_mpool_init(&mp_params, &mp);
    ASSERT_UCS_OK(status);
    ucs_mpool_cleanup(&mp, 1);
}

UCS_TEST_F(test_mpool, wrong_ops) {
    ucs_mpool_t mp;
    ucs_status_t status;
    ucs_mpool_ops_t ops = { 0 };
    scoped_log_handler log_handler(mpool_log_handler);
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 6;
    mp_params.max_elems       = 18;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    status = ucs_mpool_init(&mp_params, &mp);
    EXPECT_TRUE(status == UCS_ERR_INVALID_PARAM);
}

UCS_TEST_F(test_mpool, wrong_mpool_chuk_size) {
    ucs_mpool_t mp;
    ucs_mpool_ops_t ops = { 0 };
    scoped_log_handler log_handler(mpool_log_handler);
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 1;
    mp_params.max_chunk_size  = mp_params.elems_per_chunk * mp_params.elem_size;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    ucs_status_t status       = ucs_mpool_init(&mp_params, &mp);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
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
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 6;
    mp_params.max_elems       = 18;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    push_config();

    for (int mpool_fifo = 0; mpool_fifo <= 1; ++mpool_fifo) {
#if ENABLE_DEBUG_DATA
        modify_config("MPOOL_FIFO", ucs::to_string(mpool_fifo).c_str());
#else
        if (mpool_fifo == 1) {
            continue;
        }
#endif
        status = ucs_mpool_init(&mp_params, &mp);
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
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 5;
    mp_params.max_elems       = 18;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    status = ucs_mpool_init(&mp_params, &mp);
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
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 1000;
    mp_params.max_elems       = 2000;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    status = ucs_mpool_init(&mp_params, &mp);
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
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 10000;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    status = ucs_mpool_init(&mp_params, &mp);
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
    ucs_mpool_params_t mp_params;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = header_size + data_size;
    mp_params.align_offset    = header_size;
    mp_params.alignment       = align;
    mp_params.elems_per_chunk = 6;
    mp_params.max_elems       = 18;
    mp_params.ops             = &ops;
    mp_params.name            = "tests";
    status = ucs_mpool_init(&mp_params, &mp);
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

class test_mpool_grow : public test_mpool {
public:
    void run_grow_test(double grow_factor,
                       std::vector<unsigned> &num_elems_per_chunk)
    {
        ucs_mpool_ops_t ops = {ucs_mpool_chunk_malloc,
                               ucs_mpool_chunk_free,
                               NULL, NULL, NULL};
        unsigned chunk_num  = num_elems_per_chunk.size();
        ucs_mpool_chunk_t *chunk;
        ucs_mpool_params_t mp_params;
        ucs_mpool_t mp;
        unsigned i;

        ucs_mpool_params_reset(&mp_params);
        mp_params.elem_size       = header_size + elem_size;
        mp_params.align_offset    = header_size;
        mp_params.alignment       = align;
        mp_params.elems_per_chunk = num_elems_first_chunk;
        mp_params.max_elems       = total_num_elems;
        mp_params.max_chunk_size  = max_chunk_size;
        mp_params.grow_factor     = grow_factor;
        mp_params.ops             = &ops;
        mp_params.name            = "tests";
        ASSERT_UCS_OK(ucs_mpool_init(&mp_params, &mp));
        for (i = 0; i < total_num_elems; ++i) {
            EXPECT_NE(ucs_mpool_get(&mp), nullptr);
        }

        chunk = mp.data->chunks;
        EXPECT_NE(chunk, nullptr);

        for (; (chunk != NULL) && (chunk_num > 0);
             chunk_num--, chunk = chunk->next) {
            EXPECT_EQ(chunk->num_elems,
                      num_elems_per_chunk[chunk_num - 1]);
        }

        EXPECT_EQ(chunk_num, 0);
        EXPECT_EQ(chunk, nullptr);
        ucs_mpool_cleanup(&mp, 0);
    }

private:
    static const size_t elem_size = 1 * UCS_MBYTE;
    static const unsigned num_elems_first_chunk = 8;
    static const unsigned total_num_elems = 25;
    static const unsigned max_chunk_size = 32 * UCS_MBYTE;
};

UCS_TEST_F(test_mpool_grow, grow_factor1) {
    std::vector<unsigned> num_elems_per_chunk = {8, 8, 8, 1};
    run_grow_test(1.0, num_elems_per_chunk);
}

UCS_TEST_F(test_mpool_grow, grow_factor2) {
    std::vector<unsigned> num_elems_per_chunk = {8, 16, 1};
    run_grow_test(2.0, num_elems_per_chunk);
}

UCS_TEST_SKIP_COND_F(test_mpool, alloc_4g, RUNNING_ON_VALGRIND) {
    const size_t elem_size         = 32 * UCS_MBYTE;
    const unsigned elems_per_chunk = ucs::limit_buffer_size(4 * UCS_GBYTE) /
                                     elem_size;

    ucs_mpool_t mp;
    ucs_status_t status = setup_mpool(&mp, elem_size, elems_per_chunk);
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

class test_mpool_fifo : public test_mpool {
public:
    void init()
    {
        modify_config("MPOOL_FIFO", ucs::to_string(1).c_str());
        test_mpool::init();
    }
};

UCS_TEST_SKIP_COND_F(test_mpool_fifo, alloc_release, !ENABLE_DEBUG_DATA)
{
    const size_t elem_size         = 16;
    const unsigned elems_per_chunk = 32;
    const unsigned max_elems       = 128;
    ucs_mpool_t mp;
    std::vector<void*> objs;

    ucs_status_t status = setup_mpool(&mp, elem_size, elems_per_chunk,
                                      max_elems);
    ASSERT_UCS_OK(status);

    for (unsigned i = 0; i < 2 * elems_per_chunk; ++i) {
        objs.push_back(ucs_mpool_get(&mp));
    }

    void *obj = ucs_mpool_get(&mp);
    EXPECT_NE(nullptr, obj);
    ucs_mpool_put(obj);

    for (auto obj : objs) {
        ASSERT_NE(nullptr, obj);
        ucs_mpool_elem_t *elem = (ucs_mpool_elem_t*)obj - 1;
        VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
        ASSERT_EQ(&mp, elem->mpool);
    }

    ucs_mpool_cleanup(&mp, 0); // skip individual put as obj could be corrupted
}
