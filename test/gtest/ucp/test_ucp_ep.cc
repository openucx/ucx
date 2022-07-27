/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <ucp/core/ucp_context.h>

class test_ucp_ep : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_TAG);
    }

    /// @override
    virtual void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
    }
};

UCS_TEST_P(test_ucp_ep, ucp_query_ep)
{
    ucp_ep_h ep;
    ucs_status_t status;
    ucp_ep_evaluate_perf_param_t param;
    ucp_ep_evaluate_perf_attr_t attr;
    double estimated_time_0, estimated_time_1000;

    param.field_mask   = UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE;
    attr.field_mask    = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME;
    param.message_size = 0;
    create_entity();

    ep     = sender().ep();
    status = ucp_ep_evaluate_perf(ep, &param, &attr);

    EXPECT_EQ(status, UCS_OK);
    EXPECT_GE(attr.estimated_time, 0);
    estimated_time_0 = attr.estimated_time;

    param.message_size = 1000;
    status             = ucp_ep_evaluate_perf(ep, &param, &attr);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_GT(attr.estimated_time, 0);
    EXPECT_LT(attr.estimated_time, 10);
    estimated_time_1000 = attr.estimated_time;

    param.message_size = 2000;
    status             = ucp_ep_evaluate_perf(ep, &param, &attr);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_GT(attr.estimated_time, 0);
    EXPECT_LT(attr.estimated_time, 10);

    /* Test time estimation sanity, by verifying constant increase per message
       size (which represents current calculation model) */
    EXPECT_FLOAT_EQ(attr.estimated_time - estimated_time_1000,
                    estimated_time_1000 - estimated_time_0);
}

UCS_TEST_P(test_ucp_ep, ucp_query_transport)
{
    ucs_status_t status;
    int i;
    ucp_ep_attr_t ep_attrs;
    std::vector<ucp_transport_entry_t> transport_entries(100);

    ep_attrs.field_mask             = UCP_EP_ATTR_FIELD_TRANSPORTS;
    ep_attrs.transports.entries     = &transport_entries[0];
    ep_attrs.transports.num_entries = transport_entries.size();
    ep_attrs.transports.entry_size  = sizeof(ucp_transport_entry_t);
    status                          = ucp_ep_query(sender().ep(), &ep_attrs);

    // Verify ucp_ep_query completed successfully, that the number of
    // transports is updated, since no system should reasonably have
    // 100 transport/device name pairs, and verify returned strings are
    // not empty.
    ASSERT_UCS_OK(status);
    EXPECT_LT(ep_attrs.transports.num_entries, 100);
    UCS_TEST_MESSAGE << "Number of transport/device name pairs: "
                     << ep_attrs.transports.num_entries;
    for (i = 0; i < ep_attrs.transports.num_entries; i++) {
        UCS_TEST_MESSAGE << "Transport[" << i << "] transport="
                         << ep_attrs.transports.entries[i].transport_name
                         << " device="
                         << ep_attrs.transports.entries[i].device_name;
        EXPECT_STRNE(ep_attrs.transports.entries[i].transport_name, "");
        EXPECT_STRNE(ep_attrs.transports.entries[i].device_name, "");
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_ep);
