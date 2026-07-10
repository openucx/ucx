/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_rc.h"

extern "C" {
#include <uct/ib/mlx5/ib_mlx5_ext.h>
}


class test_uct_ib_mlx5_ext_rc : public test_rc {
public:
    void init() override
    {
        uct_ib_mlx5_ext_cleanup();
        reset_state();
        test_rc::init();
    }

    void cleanup() override
    {
        uct_ib_mlx5_ext_cleanup();
        uct_test::cleanup();
    }

protected:
    typedef struct {
        uct_iface_h                             iface = nullptr;
        uct_ep_h                                ep = nullptr;
        uint64_t                                tx_token_count = 0;
        uint64_t                                rx_token_count = 0;
        uint64_t                                purge_count = 0;
        const uct_ep_outstanding_purge_params_t *purge_params = nullptr;
    } state_t;

    static void reset_state()
    {
        m_state = state_t();
    }

    static constexpr uint64_t cap_flags()
    {
        return UCT_IFACE_FLAG_V2_QUERY_TOKEN;
    }

    static ucs_status_t
    iface_query(uct_iface_h iface, uct_ib_mlx5_ext_iface_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.iface, iface);

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS) {
            attr->cap.flags = cap_flags();
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN_LEN) {
            attr->tx_token_len = sizeof(uint64_t);
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN_LEN) {
            attr->rx_token_len = sizeof(uint64_t);
        }

        return UCS_OK;
    }

    static ucs_status_t
    ep_query(uct_ep_h ep, uct_ib_mlx5_ext_ep_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.ep, ep);

        if ((attr->field_mask & UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_TX_TOKEN) &&
            (attr->tx_token_len < sizeof(uint64_t))) {
            return UCS_ERR_INVALID_PARAM;
        }

        if ((attr->field_mask & UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_RX_TOKEN) &&
            (attr->rx_token_len < sizeof(uint64_t))) {
            return UCS_ERR_INVALID_PARAM;
        }

        if (attr->field_mask & UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_TX_TOKEN) {
            ++m_state.tx_token_count;
            *static_cast<uint64_t*>(attr->tx_token) = m_state.tx_token_count;
        }

        if (attr->field_mask & UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_RX_TOKEN) {
            ++m_state.rx_token_count;
            *static_cast<uint64_t*>(attr->rx_token) = m_state.rx_token_count;
        }

        return UCS_OK;
    }

    static void purge_cb(const uct_ep_op_info_t *op_info, void *arg)
    {
        uint64_t *purge_cb_count = static_cast<uint64_t*>(arg);
        ++*purge_cb_count;

        EXPECT_TRUE(op_info->field_mask & UCT_EP_OP_INFO_FIELD_OPERATION);
        EXPECT_EQ(UCT_EP_OP_FLUSH, op_info->operation);
    }

    static ucs_status_t
    ep_outstanding_purge(uct_ep_h ep,
                         const uct_ep_outstanding_purge_params_t *params)
    {
        uct_ep_op_info_t op_info = {};
        uint64_t rx_token = *static_cast<const uint64_t*>(params->rx_token);

        EXPECT_EQ(m_state.ep, ep);
        EXPECT_EQ(m_state.purge_params, params);
        EXPECT_EQ(m_state.rx_token_count, rx_token);

        ++m_state.purge_count;

        op_info.field_mask = UCT_EP_OP_INFO_FIELD_OPERATION;
        op_info.operation  = UCT_EP_OP_FLUSH;
        params->cb(&op_info, params->arg);

        return UCS_OK;
    }
    void register_plugin()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, "stub_plugin", sizeof(ops.name));
        ops.iface_query          = iface_query;
        ops.ep_query             = ep_query;
        ops.ep_outstanding_purge = ep_outstanding_purge;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static state_t m_state;
};

test_uct_ib_mlx5_ext_rc::state_t test_uct_ib_mlx5_ext_rc::m_state;

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, iface_query)
{
    uct_iface_attr_v2_t attr = {};

    m_state.iface = m_e1->iface();

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS;
    EXPECT_EQ(UCS_OK, uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(0, attr.cap.flags);

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH |
                      UCT_IFACE_ATTR_FIELD_RX_TOKEN_LENGTH;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, uct_iface_query_v2(m_e1->iface(), &attr));

    register_plugin();

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS;
    EXPECT_EQ(UCS_OK, uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(cap_flags(), attr.cap.flags & cap_flags());

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH |
                      UCT_IFACE_ATTR_FIELD_RX_TOKEN_LENGTH;
    EXPECT_EQ(UCS_OK, uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(cap_flags(), attr.cap.flags & cap_flags());
    EXPECT_EQ(sizeof(uint64_t), attr.tx_token_length);
    EXPECT_EQ(sizeof(uint64_t), attr.rx_token_length);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_query)
{
    uint64_t tx_token  = 0;
    uint64_t rx_token  = 0;
    uct_ep_attr_t attr = {};

    m_state.ep = m_e1->ep(0);

    register_plugin();

    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), NULL));

    attr.field_mask = UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, uct_ep_query(m_e1->ep(0), &attr));

    attr            = uct_ep_attr_t();
    attr.field_mask = UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, uct_ep_query(m_e1->ep(0), &attr));

    attr            = uct_ep_attr_t();
    attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));

    attr            = uct_ep_attr_t();
    attr.field_mask = UCT_EP_ATTR_FIELD_RX_TOKEN;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));

    attr                 = uct_ep_attr_t();
    attr.field_mask      = UCT_EP_ATTR_FIELD_TX_TOKEN;
    attr.tx_token        = &tx_token;
    attr.tx_token_length = sizeof(tx_token);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));

    attr                 = uct_ep_attr_t();
    attr.field_mask      = UCT_EP_ATTR_FIELD_RX_TOKEN;
    attr.rx_token        = &rx_token;
    attr.rx_token_length = sizeof(rx_token);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));

    attr                 = uct_ep_attr_t();
    attr.field_mask      = UCT_EP_ATTR_FIELD_TX_TOKEN |
                           UCT_EP_ATTR_FIELD_TX_TOKEN_LENGTH;
    attr.tx_token        = &tx_token;
    attr.tx_token_length = 0;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));

    attr                 = uct_ep_attr_t();
    attr.field_mask      = UCT_EP_ATTR_FIELD_RX_TOKEN |
                           UCT_EP_ATTR_FIELD_RX_TOKEN_LENGTH;
    attr.rx_token        = &rx_token;
    attr.rx_token_length = 0;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));

    attr.field_mask      = UCT_EP_ATTR_FIELD_TX_TOKEN |
                           UCT_EP_ATTR_FIELD_TX_TOKEN_LENGTH;
    attr.tx_token   = &tx_token;
    attr.tx_token_length = sizeof(tx_token);
    EXPECT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));
    EXPECT_EQ(tx_token, m_state.tx_token_count);

    attr.field_mask      = UCT_EP_ATTR_FIELD_RX_TOKEN |
                           UCT_EP_ATTR_FIELD_RX_TOKEN_LENGTH;
    attr.rx_token        = &rx_token;
    attr.rx_token_length = sizeof(rx_token);
    EXPECT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));
    EXPECT_EQ(rx_token, m_state.rx_token_count);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_outstanding_purge)
{
    uint64_t rx_token                        = 0;
    uint64_t purge_count                     = 0;
    uct_ep_outstanding_purge_params_t params = {};
    uct_ep_attr_t attr                       = {};

    m_state.ep = m_e1->ep(0);

    register_plugin();

    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ep_outstanding_purge(m_e1->ep(0), NULL));

    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ep_outstanding_purge(m_e1->ep(0), &params));

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                        UCT_EP_OUTSTANDING_FIELD_CB;
    params.cb         = purge_cb;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ep_outstanding_purge(m_e1->ep(0), &params));

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                        UCT_EP_OUTSTANDING_FIELD_CB;
    params.rx_token = &rx_token;
    params.cb       = NULL;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_ep_outstanding_purge(m_e1->ep(0), &params));

    // query rx token and then purge
    attr.field_mask      = UCT_EP_ATTR_FIELD_RX_TOKEN |
                           UCT_EP_ATTR_FIELD_RX_TOKEN_LENGTH;
    attr.rx_token        = &rx_token;
    attr.rx_token_length = sizeof(rx_token);
    EXPECT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));
    EXPECT_EQ(rx_token, m_state.rx_token_count);

    params.field_mask      = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                             UCT_EP_OUTSTANDING_FIELD_CB |
                             UCT_EP_OUTSTANDING_FIELD_ARG;
    params.cb              = purge_cb;
    params.arg             = &purge_count;
    m_state.purge_params   = &params;
    EXPECT_UCS_OK(uct_ep_outstanding_purge(m_e1->ep(0), &params));
    EXPECT_EQ(purge_count, m_state.purge_count);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_ib_mlx5_ext_rc, rc_mlx5)
