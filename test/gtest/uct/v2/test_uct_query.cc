/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
}

#include <gtest/uct/uct_p2p_test.h>

class test_uct_query : public uct_p2p_test {
public:
    test_uct_query() : uct_p2p_test(0)
    {
    }
};

UCS_TEST_P(test_uct_query, query_perf)
{
    uct_iface_attr_t iface_attr;
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    perf_attr.field_mask         = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH;
    perf_attr.operation          = UCT_OP_AM_SHORT;
    perf_attr.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    perf_attr.remote_memory_type = UCS_MEMORY_TYPE_HOST;
    status                       = uct_iface_estimate_perf(sender().iface(),
                                                           &perf_attr);
    EXPECT_EQ(status, UCS_OK);

    status = uct_iface_query(sender().iface(), &iface_attr);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_EQ(iface_attr.bandwidth.dedicated, perf_attr.bandwidth.dedicated);
    EXPECT_EQ(iface_attr.bandwidth.shared, perf_attr.bandwidth.shared);
    EXPECT_EQ(iface_attr.overhead, perf_attr.overhead);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_query)
