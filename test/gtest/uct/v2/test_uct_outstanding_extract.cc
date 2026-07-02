/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <gtest/uct/uct_p2p_test.h>

extern "C" {
#include <uct/api/v2/uct_v2.h>
#if HAVE_MLX5_DV
#include <uct/ib/mlx5/ib_mlx5_ext.h>
#endif
}

class test_uct_outstanding_extract : public uct_p2p_test {
public:
    test_uct_outstanding_extract() : uct_p2p_test(0) {
    }

    static void extract_cb(const uct_ep_op_info_t *op_info, void *arg)
    {
        (void)op_info;
        (void)arg;
    }
};

UCS_TEST_P(test_uct_outstanding_extract, unsupported_on_self)
{
    uct_ep_outstanding_extract_params_t params = {};
    uint64_t rx_token = 0;
    ucs_status_t status;

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                        UCT_EP_OUTSTANDING_FIELD_CB;
    params.rx_token   = &rx_token;
    params.cb         = extract_cb;
    params.arg        = NULL;

    status = uct_ep_outstanding_extract(sender().ep(0), &params);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, status);
}

#if HAVE_MLX5_DV
class test_uct_ib_mlx5_ext_outstanding_extract : public ucs::test {
public:
    static void extract_cb(const uct_ep_op_info_t *op_info, void *arg)
    {
        (void)op_info;
        (void)arg;
    }
};

UCS_TEST_F(test_uct_ib_mlx5_ext_outstanding_extract, invalid_params) {
    uct_ep_outstanding_extract_params_t params = {};
    uint64_t rx_token = 0;

    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_ext_ep_outstanding_extract(NULL, NULL));

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN;
    params.rx_token   = &rx_token;
    params.cb         = extract_cb;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_ext_ep_outstanding_extract(NULL, &params));

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_CB;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_ext_ep_outstanding_extract(NULL, &params));

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                        UCT_EP_OUTSTANDING_FIELD_CB;
    params.rx_token   = NULL;
    params.cb         = extract_cb;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_ext_ep_outstanding_extract(NULL, &params));

    params.rx_token = &rx_token;
    params.cb       = NULL;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_ext_ep_outstanding_extract(NULL, &params));

    params.field_mask |= UCT_EP_OUTSTANDING_FIELD_FLAGS;
    params.cb          = extract_cb;
    params.flags       = UCT_EP_OUTSTANDING_FLAG_COMPLETE_DELIVERED << 1;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ib_mlx5_ext_ep_outstanding_extract(NULL, &params));
}
#endif

_UCT_INSTANTIATE_TEST_CASE(test_uct_outstanding_extract, self)
