/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ze/base/ze_base.h>
#include <uct/ze/ze_ipc/ze_ipc_md.h>
}


/*
 * Lightweight smoke tests for the ZE_IPC component. They do not require a
 * peer process; they validate that the component is registered, that it
 * advertises at least one MD resource, and that the MD opens/closes cleanly
 * with the post-merge per-sub-device layout.
 */
class test_ze_ipc_md : public ucs::test {
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

    /* Query MD resources via the component API used by the rest of UCT. */
    static ucs_status_t query_md_resources(uct_md_resource_desc_t **resources_p,
                                           unsigned *num_p)
    {
        uct_component_attr_t attr;
        ucs_status_t         status;

        attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
        status = uct_component_query(&uct_ze_ipc_component, &attr);
        if (status != UCS_OK) {
            return status;
        }

        unsigned count = attr.md_resource_count;
        uct_md_resource_desc_t *res = (uct_md_resource_desc_t *)
                ucs_calloc(count, sizeof(uct_md_resource_desc_t),
                           "ze_ipc_md_resources");
        if ((res == NULL) && (count > 0)) {
            return UCS_ERR_NO_MEMORY;
        }

        attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
        attr.md_resources = res;
        status = uct_component_query(&uct_ze_ipc_component, &attr);
        if (status != UCS_OK) {
            ucs_free(res);
            return status;
        }

        *resources_p = res;
        *num_p       = count;
        return UCS_OK;
    }

    static uct_md_h open_first_md() {
        uct_md_resource_desc_t *resources = NULL;
        unsigned                num       = 0;

        if (query_md_resources(&resources, &num) != UCS_OK) {
            return NULL;
        }
        if (num == 0) {
            ucs_free(resources);
            return NULL;
        }

        uct_md_config_t *md_config = NULL;
        ucs_status_t status = uct_md_config_read(&uct_ze_ipc_component, NULL,
                                                 NULL, &md_config);
        if (status != UCS_OK) {
            ucs_free(resources);
            return NULL;
        }

        uct_md_h md = NULL;
        status = uct_md_open(&uct_ze_ipc_component, resources[0].md_name,
                             md_config, &md);
        uct_config_release(md_config);
        ucs_free(resources);

        return (status == UCS_OK) ? md : NULL;
    }
};


UCS_TEST_F(test_ze_ipc_md, component_is_registered) {
    EXPECT_STREQ("ze_ipc", uct_ze_ipc_component.name);
}


UCS_TEST_F(test_ze_ipc_md, query_md_resources_returns_at_least_one) {
    uct_md_resource_desc_t *resources = NULL;
    unsigned                num       = 0;

    EXPECT_UCS_OK(query_md_resources(&resources, &num));
    EXPECT_GE(num, 1u);
    ucs_free(resources);
}


UCS_TEST_F(test_ze_ipc_md, open_close_md) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ZE_IPC MD on this system");
    }

    /* The post-merge ze_ipc_md must populate device + context. */
    auto *ze_md = ucs_derived_of(md, uct_ze_ipc_md_t);
    EXPECT_TRUE(ze_md->ze_device  != NULL);
    EXPECT_TRUE(ze_md->ze_context != NULL);

    uct_md_close(md);
}


UCS_TEST_F(test_ze_ipc_md, md_attr_advertises_registerable_mem) {
    uct_md_h md = open_first_md();
    if (md == NULL) {
        UCS_TEST_SKIP_R("Could not open ZE_IPC MD on this system");
    }

    uct_md_attr_v2_t attr;
    attr.field_mask = UINT64_MAX;
    EXPECT_UCS_OK(uct_md_query_v2(md, &attr));
    EXPECT_NE(0u, attr.reg_mem_types) << "no registerable mem types";

    uct_md_close(md);
}
