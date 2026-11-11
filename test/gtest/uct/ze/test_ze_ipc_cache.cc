/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/ze/ze_ipc/ze_ipc_cache.h>
}

#include <unistd.h>


/*
 * The cache tests use synthetic mapping operations to cover cache identity and
 * lifecycle behavior without requiring a Level Zero device.
 */
class test_ze_ipc_cache : public ucs::test {
protected:
    struct cache_ops_state {
        unsigned open_count;
        unsigned close_count;
        uintptr_t next_mapped_addr;
    };

    void SetUp() override
    {
        ucs::test::SetUp();
        m_state.open_count       = 0;
        m_state.close_count      = 0;
        m_state.next_mapped_addr = 0x10000000ul;
    }

    void TearDown() override
    {
        uct_ze_ipc_purge_cache_by_context(context());
        ucs::test::TearDown();
    }

    static ucs_status_t open_memhandle(uct_ze_ipc_key_t *,
                                       ze_context_handle_t ze_context,
                                       ze_device_handle_t,
                                       void **mapped_addr, int *dup_fd)
    {
        cache_ops_state *state = reinterpret_cast<cache_ops_state*>(ze_context);

        *mapped_addr = reinterpret_cast<void*>(state->next_mapped_addr);
        state->next_mapped_addr += 0x10000;
        state->open_count++;
        *dup_fd = -1;
        return UCS_OK;
    }

    static ucs_status_t close_memhandle(ze_context_handle_t ze_context,
                                        void *, int)
    {
        cache_ops_state *state = reinterpret_cast<cache_ops_state*>(ze_context);

        state->close_count++;
        return UCS_OK;
    }

    ze_context_handle_t context()
    {
        return reinterpret_cast<ze_context_handle_t>(&m_state);
    }

    uct_ze_ipc_key_t make_key(uint64_t alloc_id = 1,
                              uint64_t proc_create_time = 1)
    {
        uct_ze_ipc_key_t key = {};

        key.pid              = getpid();
        key.address          = 0x20000000ul;
        key.length           = 4096;
        key.alloc_id         = alloc_id;
        key.proc_create_time = proc_create_time;
        return key;
    }

    ucs_status_t map(uct_ze_ipc_key_t *key, void **mapped_addr, int *dup_fd)
    {
        static const uct_ze_ipc_cache_ops_t ops = {
            open_memhandle,
            close_memhandle
        };

        return uct_ze_ipc_map_memhandle(key, context(), NULL, &ops, mapped_addr,
                                        dup_fd);
    }

    cache_ops_state m_state;
};


UCS_TEST_F(test_ze_ipc_cache, create_destroy) {
    uct_ze_ipc_cache_t *cache = NULL;
    ASSERT_UCS_OK(uct_ze_ipc_create_cache(&cache, "test"));
    ASSERT_TRUE(cache != NULL);
    uct_ze_ipc_destroy_cache(cache);
}


UCS_TEST_F(test_ze_ipc_cache, create_destroy_repeated) {
    for (int i = 0; i < 16; ++i) {
        uct_ze_ipc_cache_t *cache = NULL;
        ASSERT_UCS_OK(uct_ze_ipc_create_cache(&cache, "test_repeat"));
        ASSERT_TRUE(cache != NULL);
        uct_ze_ipc_destroy_cache(cache);
    }
}


UCS_TEST_F(test_ze_ipc_cache, multiple_instances) {
    const int N                   = 8;
    uct_ze_ipc_cache_t *caches[N] = {};

    for (int i = 0; i < N; ++i) {
        ASSERT_UCS_OK(uct_ze_ipc_create_cache(&caches[i], "test_multi"));
        ASSERT_TRUE(caches[i] != NULL);
    }
    for (int i = 0; i < N; ++i) {
        uct_ze_ipc_destroy_cache(caches[i]);
    }
}


UCS_TEST_F(test_ze_ipc_cache, stale_allocation_is_remapped) {
    uct_ze_ipc_key_t key_a = make_key(1);
    uct_ze_ipc_key_t key_b = make_key(2);
    void *mapped_a = NULL;
    void *mapped_b = NULL;
    int dup_fd_a   = -1;
    int dup_fd_b   = -1;

    ASSERT_UCS_OK(map(&key_a, &mapped_a, &dup_fd_a));
    ASSERT_UCS_OK(map(&key_b, &mapped_b, &dup_fd_b));

    EXPECT_EQ(2u, m_state.open_count);

    EXPECT_UCS_OK(uct_ze_ipc_unmap_memhandle(key_a.pid, key_a.address, mapped_a,
                                             context(), dup_fd_a, 0));
    EXPECT_UCS_OK(uct_ze_ipc_unmap_memhandle(key_b.pid, key_b.address, mapped_b,
                                             context(), dup_fd_b, 0));
    EXPECT_EQ(2u, m_state.close_count);
}


UCS_TEST_F(test_ze_ipc_cache, pid_reuse_is_remapped) {
    uct_ze_ipc_key_t key_a = make_key(1, 1);
    uct_ze_ipc_key_t key_b = make_key(1, 2);
    void *mapped_a = NULL;
    void *mapped_b = NULL;
    int dup_fd_a   = -1;
    int dup_fd_b   = -1;

    ASSERT_UCS_OK(map(&key_a, &mapped_a, &dup_fd_a));
    ASSERT_UCS_OK(map(&key_b, &mapped_b, &dup_fd_b));

    EXPECT_EQ(2u, m_state.open_count);

    EXPECT_UCS_OK(uct_ze_ipc_unmap_memhandle(key_a.pid, key_a.address, mapped_a,
                                             context(), dup_fd_a, 0));
    EXPECT_UCS_OK(uct_ze_ipc_unmap_memhandle(key_b.pid, key_b.address, mapped_b,
                                             context(), dup_fd_b, 0));
    EXPECT_EQ(2u, m_state.close_count);
}


UCS_TEST_F(test_ze_ipc_cache, purge_by_context_then_remap) {
    uct_ze_ipc_key_t key = make_key();
    void *mapped_a       = NULL;
    void *mapped_b       = NULL;
    int dup_fd_a         = -1;
    int dup_fd_b         = -1;

    ASSERT_UCS_OK(map(&key, &mapped_a, &dup_fd_a));
    ASSERT_UCS_OK(uct_ze_ipc_unmap_memhandle(key.pid, key.address, mapped_a,
                                             context(), dup_fd_a, 1));
    EXPECT_EQ(1u, m_state.open_count);
    EXPECT_EQ(0u, m_state.close_count);

    uct_ze_ipc_purge_cache_by_context(context());
    EXPECT_EQ(1u, m_state.close_count);

    ASSERT_UCS_OK(map(&key, &mapped_b, &dup_fd_b));
    EXPECT_EQ(2u, m_state.open_count);

    ASSERT_UCS_OK(uct_ze_ipc_unmap_memhandle(key.pid, key.address, mapped_b,
                                             context(), dup_fd_b, 0));
    EXPECT_EQ(2u, m_state.close_count);
}
