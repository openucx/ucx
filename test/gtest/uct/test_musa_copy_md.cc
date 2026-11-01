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

UCS_TEST_F(test_musa_copy_md, allocates_device_memory)
{
    uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
    uct_mem_alloc_params_t params;
    uct_allocated_memory_t mem;
    ucs_status_t status;
    uct_md_h md_ref = md;

    params.field_mask = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS     |
                        UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE  |
                        UCT_MEM_ALLOC_PARAM_FIELD_MDS       |
                        UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.flags      = UCT_MD_MEM_ACCESS_ALL;
    params.mem_type   = UCS_MEMORY_TYPE_MUSA;
    params.mds.mds    = &md_ref;
    params.mds.count  = 1;
    params.name       = "test_musa_copy_md";

    status = uct_mem_alloc(4096, &method, 1, &params, &mem);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(UCS_MEMORY_TYPE_MUSA, mem.mem_type);
    EXPECT_EQ(md, mem.md);
    EXPECT_NE(nullptr, mem.address);
    EXPECT_NE(nullptr, mem.memh);

    EXPECT_UCS_OK(uct_mem_free(&mem));
}

UCS_TEST_F(test_musa_copy_md, host_detection_preserves_runtime_error)
{
    char buffer[64];
    ucs_memory_type_t mem_type;

    EXPECT_NE(musaSuccess, musaSetDevice(-1));
    musaError_t error = musaPeekAtLastError();
    UCS_TEST_SCOPE_EXIT() {
        (void)musaGetLastError();
    } UCS_TEST_SCOPE_EXIT_END

    ASSERT_NE(musaSuccess, error);
    ASSERT_UCS_OK(uct_md_detect_memory_type(md, buffer, sizeof(buffer),
                                           &mem_type));
    EXPECT_EQ(UCS_MEMORY_TYPE_HOST, mem_type);
    EXPECT_EQ(error, musaPeekAtLastError());
}
