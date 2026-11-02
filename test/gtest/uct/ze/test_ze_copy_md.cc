/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/mem_buffer.h>
#include <uct/test_md.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ze/copy/ze_copy_md.h>
#include <ucs/memory/memtype_cache.h>
}


/*
 * Smoke tests for the ze_copy MD: lifecycle, capability query, allocation
 * and memory-type detection. The ze_copy MD is the path used by UCP for
 * VRAM<->host staging on Intel XPU.
 */
class test_ze_copy_md : public test_md {
protected:
    /* Allocate ZE_DEVICE memory through the MD, which advertises it */
    static void alloc_device(uct_md_h md, size_t *length_p, void **addr_p,
                             uct_mem_h *memh_p)
    {
        ASSERT_UCS_OK(md->ops->mem_alloc(md, length_p, addr_p,
                                         UCS_MEMORY_TYPE_ZE_DEVICE,
                                         UCS_SYS_DEVICE_ID_UNKNOWN, 0,
                                         "test_ze_copy", memh_p));
    }
};


UCS_TEST_P(test_ze_copy_md, open_close_md) {
    auto *ze_md = ucs_derived_of(md(), uct_ze_copy_md_t);
    EXPECT_NE(nullptr, ze_md->ze_device);
    EXPECT_NE(nullptr, ze_md->ze_context);
}


UCS_TEST_P(test_ze_copy_md, md_attr_advertises_alloc_caps) {
    /* ze_copy MD must be able to allocate ZE memory. */
    EXPECT_NE(0u, md_attr().alloc_mem_types);
    EXPECT_GT(md_attr().max_alloc, 0u);
}


/*
 * Allocate ZE_DEVICE memory through the MD, then free it.
 */
UCS_TEST_P(test_ze_copy_md, mem_alloc_free_device) {
    size_t length  = 4096;
    void *addr     = NULL;
    uct_mem_h memh = NULL;

    alloc_device(md(), &length, &addr, &memh);
    EXPECT_TRUE(addr != NULL);
    EXPECT_GE(length, 4096u);

    EXPECT_UCS_OK(md()->ops->mem_free(md(), memh));
}


/*
 * detect_memory_type must classify ZE device pointers as ZE_DEVICE.
 */
UCS_TEST_P(test_ze_copy_md, detect_memory_type_device) {
    size_t length  = 4096;
    void *addr     = NULL;
    uct_mem_h memh = NULL;
    alloc_device(md(), &length, &addr, &memh);

    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    EXPECT_UCS_OK(md()->ops->detect_memory_type(md(), addr, length, &mem_type));
    EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_type);

    EXPECT_UCS_OK(md()->ops->mem_free(md(), memh));
}


/*
 * mem_query reports base address and allocation length back.
 */
UCS_TEST_P(test_ze_copy_md, mem_query_base_and_length) {
    size_t length  = 8192;
    void *addr     = NULL;
    uct_mem_h memh = NULL;
    alloc_device(md(), &length, &addr, &memh);

    uct_md_mem_attr_v2_t mem_attr = {};
    mem_attr.field_mask           = UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE |
                                    UCT_MD_MEM_ATTR_V2_FIELD_BASE_ADDRESS |
                                    UCT_MD_MEM_ATTR_V2_FIELD_ALLOC_LENGTH;
    EXPECT_UCS_OK(md()->ops->mem_query(md(), addr, length, &mem_attr));
    EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_attr.mem_type);
    EXPECT_TRUE(mem_attr.base_address != NULL);
    EXPECT_GE(mem_attr.alloc_length, length);

    EXPECT_UCS_OK(md()->ops->mem_free(md(), memh));
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ze_copy_md, ze_cpy)


class test_ze_copy_memtype_cache : public test_ze_copy_md {
protected:
    void init() override
    {
        /* Drop the cache, so that the MD open has to create it again */
        ucs_memtype_cache_cleanup();
        ucs_memtype_cache_global_init();
        ASSERT_EQ(nullptr, ucs_memtype_cache_global_instance);

        test_ze_copy_md::init();
    }
};


/*
 * An application allocation made after the MD is opened must be tracked in
 * the memtype cache.
 */
UCS_TEST_P(test_ze_copy_memtype_cache, memtype_cache_created_at_md_open,
           "MEMTYPE_CACHE=y") {
    ASSERT_NE(nullptr, ucs_memtype_cache_global_instance);

    const size_t length = 4096;
    ucs_memory_info_t mem_info;
    void *addr;

    {
        mem_buffer buffer(length, UCS_MEMORY_TYPE_ZE_DEVICE);

        addr = buffer.ptr();
        ASSERT_UCS_OK(ucs_memtype_cache_lookup(addr, length, &mem_info));
        EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_info.type);
    }

    EXPECT_EQ(UCS_ERR_NO_ELEM,
              ucs_memtype_cache_lookup(addr, length, &mem_info));
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_ze_copy_memtype_cache, ze_cpy)
