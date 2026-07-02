/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp/ucp_test.h"

extern "C" {
#include <ucp/wireup/wireup.h>
#include <ucp/core/ucp_ep_failover.h>
}

class test_ucp_ep_failover : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_RMA);
    }

    void init() override
    {
        modify_config("PROTO_ENABLE", "y");
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }
};

class test_ucp_ep_failover_err_mode : public test_ucp_ep_failover {
public:
    ucp_ep_params_t get_ep_params() override
    {
        ucp_ep_params_t params = ucp_test::get_ep_params();
        params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        params.err_mode        = UCP_ERR_HANDLING_MODE_FAILOVER;
        return params;
    }
};

UCS_TEST_P(test_ucp_ep_failover_err_mode, query_lane_state_no_token_lanes)
{
    ucs_status_t status;

    status = ucp_wireup_send_query_lane_state(sender().ep(), UCS_BIT(0));
    EXPECT_TRUE((status == UCS_ERR_UNREACHABLE) || (status == UCS_ERR_NO_RESOURCE) ||
                (status == UCS_OK));
}

UCS_TEST_P(test_ucp_ep_failover_err_mode, query_lanes_is_retryable_state)
{
    ucs_status_t status;

    status = ucp_ep_failover_query_lanes(sender().ep(), UCS_BIT(0));
    ASSERT_UCS_OK(status);
    EXPECT_EQ(UCS_BIT(0), ucp_ep_failover_test_query_lane_map(sender().ep()));
}

UCS_TEST_P(test_ucp_ep_failover_err_mode, lane_state_validation)
{
    struct {
        ucp_wireup_lane_state_t state;
        uint8_t                 token_lengths[2];
    } msg;

    memset(&msg, 0, sizeof(msg));
    msg.state.lane_map       = UCS_BIT(0);
    msg.token_lengths[0]     = 0;
    EXPECT_UCS_OK(ucp_ep_failover_test_validate_lane_state(
            sender().ep(), &msg.state, sizeof(msg.state) + 1));

    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_ep_failover_test_validate_lane_state(
                      sender().ep(), &msg.state, sizeof(msg.state) + 2));

    msg.state.lane_map = UCS_BIT(UCP_MAX_LANES - 1);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_ep_failover_test_validate_lane_state(
                      sender().ep(), &msg.state, sizeof(msg.state) + 1));
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_ep_failover, self, "self")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_ep_failover_err_mode, self, "self")
