/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/ze/base/ze_base.h>
}


/*
 * Unit tests for the helpers added to make ZE_IPC compatible with the
 * post-merge upstream uct_ze_base data layout (root device per index).
 */
class test_ze_base : public ucs::test {
};


UCS_TEST_F(test_ze_base, init_succeeds_or_skip) {
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        UCS_TEST_SKIP_R("Level Zero runtime not available");
    }
}


UCS_TEST_F(test_ze_base, get_num_devices_consistent) {
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        UCS_TEST_SKIP_R("Level Zero runtime not available");
    }

    int n = uct_ze_base_get_num_devices();
    EXPECT_GE(n, 0);

    /* second call must be stable */
    EXPECT_EQ(n, uct_ze_base_get_num_devices());
}


UCS_TEST_F(test_ze_base, get_device_by_ordinal) {
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        UCS_TEST_SKIP_R("Level Zero runtime not available");
    }

    int n = uct_ze_base_get_num_devices();
    if (n == 0) {
        UCS_TEST_SKIP_R("No Level Zero devices available");
    }

    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(uct_ze_base_get_device(i) != NULL) << "ordinal " << i;
    }
}


UCS_TEST_F(test_ze_base, get_device_out_of_range) {
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        UCS_TEST_SKIP_R("Level Zero runtime not available");
    }

    int n = uct_ze_base_get_num_devices();

    EXPECT_TRUE(uct_ze_base_get_device(-1) == NULL);
    EXPECT_TRUE(uct_ze_base_get_device(n) == NULL);
    EXPECT_TRUE(uct_ze_base_get_device(n + 100) == NULL);
}
