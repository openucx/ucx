/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/ze/base/ze_base.h>
#include <uct/ze/ze_ipc/ze_ipc_cache.h>
}


/*
 * Lifecycle tests for the ze_ipc cache. The cache is a simple pgtable-
 * based handle map with no LRU; we only validate that it can be created
 * and destroyed cleanly, including back-to-back and multiple instances.
 */
class test_ze_ipc_cache : public ucs::test {
protected:
    void SetUp() override {
        ucs::test::SetUp();
        if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
            UCS_TEST_SKIP_R("Level Zero runtime not available");
        }
    }
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
    const int N = 8;
    uct_ze_ipc_cache_t *caches[N] = {};

    for (int i = 0; i < N; ++i) {
        ASSERT_UCS_OK(uct_ze_ipc_create_cache(&caches[i], "test_multi"));
        ASSERT_TRUE(caches[i] != NULL);
    }
    for (int i = 0; i < N; ++i) {
        uct_ze_ipc_destroy_cache(caches[i]);
    }
}



