/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <ucs/debug/memtrack_int.h>

#include <string>

extern "C" {
#include <musa_runtime_api.h>
#include <uct/api/uct.h>
#include <uct/base/uct_component.h>
}

extern "C" uct_component_t uct_musa_component;

class test_musa_copy_md : public ucs::test {
protected:
    void init() override
    {
        ucs_status_t status;
        unsigned count;
        int device_count = 0;
        musaError_t error;

        ucs::test::init();
        error = musaGetDeviceCount(&device_count);
        if ((error != musaSuccess) || (device_count == 0)) {
            UCS_TEST_SKIP_R("MUSA runtime has no device");
        }

        status = uct_musa_component.query_md_resources(
                &uct_musa_component, &md_resources, &count);
        if (status != UCS_OK) {
            UCS_TEST_SKIP_R(std::string("MUSA MD resources are unavailable: ") +
                            ucs_status_string(status));
        }
        if (count == 0) {
            ucs_free(md_resources);
            md_resources = NULL;
            UCS_TEST_SKIP_R("MUSA MD has no resources");
        }

        status = uct_md_config_read(&uct_musa_component, NULL, NULL,
                                    &md_config);
        ASSERT_UCS_OK(status);

        status = uct_md_open(&uct_musa_component,
                             md_resources[0].md_name, md_config, &md);
        ASSERT_UCS_OK(status);
    }

    void cleanup() override
    {
        if (md != NULL) {
            uct_md_close(md);
            md = NULL;
        }
        if (md_config != NULL) {
            uct_config_release(md_config);
            md_config = NULL;
        }
        if (md_resources != NULL) {
            ucs_free(md_resources);
            md_resources = NULL;
        }
        ucs::test::cleanup();
    }

    uct_md_h                  md             = NULL;
    uct_md_config_t           *md_config     = NULL;
    uct_md_resource_desc_t    *md_resources  = NULL;
};

UCS_TEST_F(test_musa_copy_md, reject_managed_memory_registration)
{
    void *address = NULL;
    uct_mem_h memh;
    ucs_status_t status;
    musaError_t error;

    error = musaMallocManaged(&address, 4096, musaMemAttachGlobal);
    if (error != musaSuccess) {
        UCS_TEST_SKIP_R(std::string("MUSA managed allocation is unavailable: ") +
                        musaGetErrorString(error));
    }

    status = uct_md_mem_reg(md, address, 4096, 0, &memh);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, status);
    if (status == UCS_OK) {
        EXPECT_EQ(UCS_OK, uct_md_mem_dereg(md, memh));
    }

    EXPECT_EQ(musaSuccess, musaFree(address));
}
