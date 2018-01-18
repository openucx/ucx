/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include "uct/api/uct.h"
#include "ucp/core/ucp_context.h"
#include "ucp/core/ucp_mm.h"


class test_ucp_mem_type : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG;
        return params;
    }
};

UCS_TEST_P(test_ucp_mem_type, detect_host) {
    ucs_status_t status;
    uct_memory_type_t mem_type;
    void *ptr;
    size_t size = 256;

    sender().connect(&sender(), get_ep_params());

    ptr = malloc(size);
    EXPECT_TRUE(ptr != NULL);

    status = ucp_memory_type_detect_mds(sender().ucph(), ptr, size, &mem_type);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(UCT_MD_MEM_TYPE_HOST, mem_type);

    free(ptr);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_mem_type)
