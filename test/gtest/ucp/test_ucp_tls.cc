/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <ucp/core/ucp_context.h>

class test_ucp_tl : public test_ucp_context {
};

UCS_TEST_P(test_ucp_tl, check_ucp_tl, "SELF_NUM_DEVICES?=50")
{
    create_entity();
    EXPECT_GE((sender().ucph())->num_tls, 50);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tl, self, "self");
