/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ze/base/ze_base.h>
#include <uct/ze/copy/ze_copy_md.h>
}


/*
 * Smoke tests for the ze_copy MD: lifecycle, capability query, allocation
 * and memory-type detection. The ze_copy MD is the path used by UCP for
 * VRAM<->host staging on Intel XPU.
 */
class test_ze_copy_md : public ucs::test {
protected:
    void SetUp() override {
        ucs::test::SetUp();
        if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
            UCS_TEST_SKIP_R("Level Zero runtime not available");
        }
        if (uct_ze_base_get_num_devices() == 0) {
            UCS_TEST_SKIP_R("No Level Zero devices available");
        }
    }

    static uct_md_h open_first_md() {
        uct_md_resource_desc_t *resources = NULL;
        unsigned                num       = 0;
        ucs_status_t            status;

        status = uct_ze_copy_component.query_md_resources(&uct_ze_copy_component,
                                                          &resources, &num);
        EXPECT_UCS_OK(status);
        if (num == 0) {
            ucs_free(resources);
            return NULL;
        }

        uct_md_config_t *md_config = NULL;
        status = uct_md_config_read(&uct_ze_copy_component, NULL, NULL,
                                    &md_config);
        EXPECT_UCS_OK(status);

        uct_md_h md = NULL;
        status = uct_md_open(&uct_ze_copy_component, resources[0].md_name,
                             md_config, &md);
        uct_config_release(md_config);
        ucs_free(resources);

        return (status == UCS_OK) ? md : NULL;
    }
};


UCS_TEST_F(test_ze_copy_md, component_is_registered) {
    EXPECT_STREQ("ze_cpy", uct_ze_copy_component.name);
}


UCS_TEST_F(test_ze_copy_md, query_md_resources_returns_at_least_one) {
    uct_md_resource_desc_t *resources = NULL;
    unsigned                num       = 0;

    EXPECT_UCS_OK(uct_ze_copy_component.query_md_resources(
            &uct_ze_copy_component, &resources, &num));
    EXPECT_GE(num, 1u);
    ucs_free(resources);
}


UCS_TEST_F(test_ze_copy_md, open_close_md) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ze_cpy MD on this system");
    }

    auto *ze_md = ucs_derived_of(md, uct_ze_copy_md_t);
    EXPECT_TRUE(ze_md->ze_device  != NULL);
    EXPECT_TRUE(ze_md->ze_context != NULL);

    uct_md_close(md);
}


UCS_TEST_F(test_ze_copy_md, md_attr_advertises_alloc_caps) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ze_cpy MD on this system");
    }

    uct_md_attr_v2_t attr;
    attr.field_mask = UINT64_MAX;
    EXPECT_UCS_OK(uct_md_query_v2(md, &attr));

    /* ze_copy MD must be able to allocate ZE memory. */
    EXPECT_NE(0u, attr.alloc_mem_types);
    EXPECT_GT(attr.max_alloc, 0u);

    uct_md_close(md);
}


/*
 * Allocate ZE_DEVICE memory through the MD, then free it.
 */
UCS_TEST_F(test_ze_copy_md, mem_alloc_free_device) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ze_cpy MD on this system");
    }

    size_t    length = 4096;
    void     *addr   = NULL;
    uct_mem_h memh   = NULL;

    ucs_status_t status =
            md->ops->mem_alloc(md, &length, &addr, UCS_MEMORY_TYPE_ZE_DEVICE,
                               UCS_SYS_DEVICE_ID_UNKNOWN, 0,
                               "test_ze_copy", &memh);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("ZE_DEVICE alloc unsupported on this system");
    }
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(addr != NULL);
    ASSERT_GE(length, 4096u);

    EXPECT_UCS_OK(md->ops->mem_free(md, memh));
    uct_md_close(md);
}


/*
 * detect_memory_type must classify ZE device pointers as ZE_DEVICE.
 */
UCS_TEST_F(test_ze_copy_md, detect_memory_type_device) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ze_cpy MD on this system");
    }

    size_t    length = 4096;
    void     *addr   = NULL;
    uct_mem_h memh   = NULL;
    ucs_status_t status =
            md->ops->mem_alloc(md, &length, &addr, UCS_MEMORY_TYPE_ZE_DEVICE,
                               UCS_SYS_DEVICE_ID_UNKNOWN, 0,
                               "test_ze_copy", &memh);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("ZE_DEVICE alloc unsupported on this system");
    }
    ASSERT_UCS_OK(status);

    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    EXPECT_UCS_OK(md->ops->detect_memory_type(md, addr, length, &mem_type));
    EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_type);

    EXPECT_UCS_OK(md->ops->mem_free(md, memh));
    uct_md_close(md);
}


/*
 * mem_query reports base address and allocation length back.
 */
UCS_TEST_F(test_ze_copy_md, mem_query_base_and_length) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ze_cpy MD on this system");
    }

    size_t    length = 8192;
    void     *addr   = NULL;
    uct_mem_h memh   = NULL;
    ucs_status_t status =
            md->ops->mem_alloc(md, &length, &addr, UCS_MEMORY_TYPE_ZE_DEVICE,
                               UCS_SYS_DEVICE_ID_UNKNOWN, 0,
                               "test_ze_copy", &memh);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("ZE_DEVICE alloc unsupported on this system");
    }
    ASSERT_UCS_OK(status);

    uct_md_mem_attr_t mem_attr = {};
    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE       |
                          UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS   |
                          UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH;
    EXPECT_UCS_OK(md->ops->mem_query(md, addr, length, &mem_attr));
    EXPECT_EQ(UCS_MEMORY_TYPE_ZE_DEVICE, mem_attr.mem_type);
    EXPECT_TRUE(mem_attr.base_address != NULL);
    EXPECT_GE(mem_attr.alloc_length, length);

    EXPECT_UCS_OK(md->ops->mem_free(md, memh));
    uct_md_close(md);
}
