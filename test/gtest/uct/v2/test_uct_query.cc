/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <gtest/uct/uct_p2p_test.h>

extern "C" {
#include <ucs/sys/topo/base/topo.h>
#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
}


class test_uct_query : public uct_p2p_test {
public:
    test_uct_query() : uct_p2p_test(0)
    {
    }
};

UCS_TEST_P(test_uct_query, query_perf)
{
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    perf_attr.field_mask         = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_LOCAL_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_REMOTE_MEMORY_TYPE |
                                   UCT_PERF_ATTR_FIELD_LOCAL_SYS_DEVICE |
                                   UCT_PERF_ATTR_FIELD_REMOTE_SYS_DEVICE |
                                   UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_RECV_OVERHEAD |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH;
    perf_attr.operation          = UCT_EP_OP_AM_SHORT;
    perf_attr.local_memory_type  = UCS_MEMORY_TYPE_HOST;
    perf_attr.remote_memory_type = UCS_MEMORY_TYPE_HOST;
    perf_attr.local_sys_device   = UCS_SYS_DEVICE_ID_UNKNOWN;
    perf_attr.remote_sys_device  = UCS_SYS_DEVICE_ID_UNKNOWN;
    status                       = uct_iface_estimate_perf(sender().iface(),
                                                           &perf_attr);
    EXPECT_EQ(status, UCS_OK);

    perf_attr.remote_memory_type = UCS_MEMORY_TYPE_CUDA;
    perf_attr.operation          = UCT_EP_OP_PUT_SHORT;
    status                       = uct_iface_estimate_perf(sender().iface(),
                                                           &perf_attr);

    /* At least one type of bandwidth must be non-zero */
    EXPECT_NE(0, perf_attr.bandwidth.shared + perf_attr.bandwidth.dedicated);

    if (has_transport("cuda_copy") ||
        has_transport("gdr_copy")  ||
        has_transport("rocm_copy")) {
        uct_perf_attr_t perf_attr_get;
        perf_attr_get.field_mask = UCT_PERF_ATTR_FIELD_OPERATION |
                                   UCT_PERF_ATTR_FIELD_BANDWIDTH;
        perf_attr_get.operation  = UCT_EP_OP_GET_SHORT;
        status = uct_iface_estimate_perf(sender().iface(), &perf_attr_get);
        EXPECT_EQ(status, UCS_OK);

        /* Put and get operations have different bandwidth in cuda_copy
           and gdr_copy transports */
        EXPECT_NE(perf_attr.bandwidth.shared, perf_attr_get.bandwidth.shared);
    }
}

UCT_INSTANTIATE_TEST_CASE(test_uct_query)
