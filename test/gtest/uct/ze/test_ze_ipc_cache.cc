/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ze/base/ze_base.h>
#include <uct/ze/ze_ipc/ze_ipc_cache.h>
#include <uct/ze/ze_ipc/ze_ipc_md.h>
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

    static uct_md_h open_first_md() {
        uct_component_attr_t attr;
        attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
        if (uct_component_query(&uct_ze_ipc_component, &attr) != UCS_OK) {
            return NULL;
        }

        unsigned count = attr.md_resource_count;
        if (count == 0) {
            return NULL;
        }

        uct_md_resource_desc_t *res = (uct_md_resource_desc_t *)
                ucs_calloc(count, sizeof(uct_md_resource_desc_t),
                           "ze_ipc_md_resources");
        if (res == NULL) {
            return NULL;
        }

        attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
        attr.md_resources = res;
        ucs_status_t status = uct_component_query(&uct_ze_ipc_component, &attr);
        if (status != UCS_OK) {
            ucs_free(res);
            return NULL;
        }

        uct_md_config_t *md_config = NULL;
        status = uct_md_config_read(&uct_ze_ipc_component, NULL, NULL,
                                    &md_config);
        if (status != UCS_OK) {
            ucs_free(res);
            return NULL;
        }

        uct_md_h md = NULL;
        status = uct_md_open(&uct_ze_ipc_component, res[0].md_name, md_config,
                             &md);
        uct_config_release(md_config);
        ucs_free(res);

        return (status == UCS_OK) ? md : NULL;
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


/*
 * Simulates a peer freeing an allocation and a later, different allocation
 * reusing the same VA: two distinct real IPC keys are forced to report the
 * same base address. The cache must detect the IPC-handle mismatch on the
 * second map and remap, rather than returning the first mapping.
 */
UCS_TEST_F(test_ze_ipc_cache, stale_address_is_remapped) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ZE_IPC MD on this system");
    }
    auto *ze_md = ucs_derived_of(md, uct_ze_ipc_md_t);

    const size_t size = 4096;
    void *ptr_a = NULL, *ptr_b = NULL;
    ze_device_mem_alloc_desc_t dev_desc = {};
    dev_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_md->ze_context, &dev_desc, size, 64,
                               ze_md->ze_device, &ptr_a));
    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_md->ze_context, &dev_desc, size, 64,
                               ze_md->ze_device, &ptr_b));

    uct_mem_h memh_a = NULL, memh_b = NULL;
    ASSERT_UCS_OK(md->ops->mem_reg(md, ptr_a, size, NULL, &memh_a));
    ASSERT_UCS_OK(md->ops->mem_reg(md, ptr_b, size, NULL, &memh_b));

    uct_ze_ipc_key_t key_a = {}, key_b = {};
    ASSERT_UCS_OK(md->ops->mkey_pack(md, memh_a, ptr_a, size, NULL, &key_a));
    ASSERT_UCS_OK(md->ops->mkey_pack(md, memh_b, ptr_b, size, NULL, &key_b));

    /* simulate allocation B being reported at the VA allocation A used */
    key_b.address = key_a.address;

    void *mapped_a = NULL, *mapped_b = NULL;
    int dup_fd_a = -1, dup_fd_b = -1;

    ASSERT_UCS_OK(uct_ze_ipc_map_memhandle(&key_a, ze_md->ze_context,
                                           ze_md->ze_device, &mapped_a,
                                           &dup_fd_a));
    ASSERT_UCS_OK(uct_ze_ipc_map_memhandle(&key_b, ze_md->ze_context,
                                           ze_md->ze_device, &mapped_b,
                                           &dup_fd_b));

    EXPECT_NE(mapped_a, mapped_b)
            << "cache returned a's stale mapping for a different allocation";

    /* the second map_memhandle() already invalidated and freed a's region
     * (both keys alias the same address), so only b's mapping is still
     * live in the cache to unmap */
    uct_ze_ipc_unmap_memhandle(key_b.pid, key_b.address, mapped_b,
                               ze_md->ze_context, dup_fd_b, 0);

    uct_md_mem_dereg_params_t dereg_params = {};
    dereg_params.field_mask                = UCT_MD_MEM_DEREG_FIELD_MEMH;
    dereg_params.memh                      = memh_a;
    EXPECT_UCS_OK(md->ops->mem_dereg(md, &dereg_params));
    dereg_params.memh = memh_b;
    EXPECT_UCS_OK(md->ops->mem_dereg(md, &dereg_params));

    EXPECT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_md->ze_context, ptr_a));
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_md->ze_context, ptr_b));
    uct_md_close(md);
}


/*
 * uct_ze_ipc_purge_cache_by_context() must drop cached regions for a
 * closed context without leaving a dangling pgtable entry: mapping the
 * same key again afterwards must go through a fresh open, not reuse a
 * freed region.
 */
UCS_TEST_F(test_ze_ipc_cache, purge_by_context_then_remap) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ZE_IPC MD on this system");
    }
    auto *ze_md = ucs_derived_of(md, uct_ze_ipc_md_t);

    const size_t size = 4096;
    void *ptr = NULL;
    ze_device_mem_alloc_desc_t dev_desc = {};
    dev_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;

    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_md->ze_context, &dev_desc, size, 64,
                               ze_md->ze_device, &ptr));

    uct_mem_h memh = NULL;
    ASSERT_UCS_OK(md->ops->mem_reg(md, ptr, size, NULL, &memh));

    uct_ze_ipc_key_t key = {};
    ASSERT_UCS_OK(md->ops->mkey_pack(md, memh, ptr, size, NULL, &key));

    void *mapped = NULL;
    int dup_fd = -1;
    ASSERT_UCS_OK(uct_ze_ipc_map_memhandle(&key, ze_md->ze_context,
                                           ze_md->ze_device, &mapped,
                                           &dup_fd));

    uct_ze_ipc_purge_cache_by_context(ze_md->ze_context);

    /* the purged region must not be referenced again; a fresh map/unmap
     * must succeed cleanly */
    ASSERT_UCS_OK(uct_ze_ipc_map_memhandle(&key, ze_md->ze_context,
                                           ze_md->ze_device, &mapped,
                                           &dup_fd));
    uct_ze_ipc_unmap_memhandle(key.pid, key.address, mapped, ze_md->ze_context,
                               dup_fd, 0);

    uct_md_mem_dereg_params_t dereg_params = {};
    dereg_params.field_mask                = UCT_MD_MEM_DEREG_FIELD_MEMH;
    dereg_params.memh                      = memh;
    EXPECT_UCS_OK(md->ops->mem_dereg(md, &dereg_params));

    EXPECT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_md->ze_context, ptr));
    uct_md_close(md);
}
