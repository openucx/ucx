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
        uint64_t                                iface_query_count = 0;
        uint64_t                                second_iface_query_count = 0;
        uint64_t                                failed_iface_query_count = 0;
        const uct_ep_outstanding_purge_params_t *purge_params = nullptr;
    } state_t;

    static const char *failing_plugin_name()
    {
        return "stub_fail";
    }

    static const char *other_plugin_name()
    {
        return "stub_other";
    }

    static const char *token_plugin_name()
    {
        return "stub_token";
    }

    static const char *second_token_plugin_name()
    {
        return "stub_token2";
    }

    static uint64_t get_rx_token(uint64_t tx_token)
    {
        return tx_token + 1;
    }

    static void reset_state()
    {
        m_state = state_t();
    }

    static constexpr uint64_t token_cap_flags()
    {
        return UCT_IFACE_FLAG_V2_QUERY_TOKEN;
    }

    static constexpr uint64_t other_cap_flags()
    {
        return UCT_IFACE_FLAG_V2_PUT_SGL_ZCOPY;
    }

    static ucs_status_t
    iface_query_fail(uct_iface_h iface,
                     uct_ib_mlx5_ext_iface_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.iface, iface);
        ++m_state.failed_iface_query_count;
        return UCS_ERR_IO_ERROR;
    }

    static ucs_status_t
    iface_query_other(uct_iface_h iface,
                      uct_ib_mlx5_ext_iface_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.iface, iface);

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS) {
            attr->cap.flags = other_cap_flags();
        }

        return UCS_OK;
    }

    static ucs_status_t
    iface_query_token(uct_iface_h iface,
                      uct_ib_mlx5_ext_iface_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.iface, iface);
        ++m_state.iface_query_count;

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS) {
            attr->cap.flags = token_cap_flags();
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN_LEN) {
            attr->tx_token_len = sizeof(uint64_t);
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN_LEN) {
            attr->rx_token_len = sizeof(uint64_t);
        }

        /* Derive the RX token from the TX token received from the sender. */
        if ((attr->field_mask &
             UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN) &&
            (attr->field_mask &
             UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN)) {
            if ((attr->tx_token == NULL) || (attr->rx_token == NULL)) {
                ADD_FAILURE() << "token buffers must be non-NULL";
                return UCS_ERR_INVALID_PARAM;
            }
            ++m_state.rx_token_count;
            *static_cast<uint64_t*>(attr->rx_token) = get_rx_token(
                    *static_cast<const uint64_t*>(attr->tx_token));
        }

        return UCS_OK;
    }

    static ucs_status_t
    ep_query(uct_ep_h ep, uct_ib_mlx5_ext_ep_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.ep, ep);

        if (attr->field_mask & UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_TX_TOKEN) {
            if (attr->tx_token == NULL) {
                ADD_FAILURE() << "TX token buffer must be non-NULL";
                return UCS_ERR_INVALID_PARAM;
            }
            ++m_state.tx_token_count;
            *static_cast<uint64_t*>(attr->tx_token) = m_state.tx_token_count;
        }

        return UCS_OK;
    }

    static void purge_cb(const uct_ep_op_info_t *op_info, void *arg)
    {
        uint64_t *purge_cb_count = static_cast<uint64_t*>(arg);
        ++*purge_cb_count;

        EXPECT_TRUE(op_info->field_mask & UCT_EP_OP_INFO_FIELD_OPERATION);
        EXPECT_TRUE(op_info->field_mask & UCT_EP_OP_INFO_FIELD_FLUSH);
        EXPECT_TRUE(op_info->flush.field_mask &
                    UCT_EP_OP_INFO_FLUSH_FIELD_FLAGS);
        EXPECT_EQ(UCT_EP_OP_FLUSH, op_info->operation);
        EXPECT_EQ(0u, op_info->flush.flags);
    }

    static ucs_status_t
    ep_outstanding_purge(uct_ep_h ep,
                         const uct_ep_outstanding_purge_params_t *params)
    {
        uct_ep_op_info_t op_info = {};
        uint64_t rx_token;

        if (params->rx_token == NULL) {
            ADD_FAILURE() << "RX token buffer must be non-NULL";
            return UCS_ERR_INVALID_PARAM;
        }
        rx_token = *static_cast<const uint64_t*>(params->rx_token);

        EXPECT_EQ(m_state.ep, ep);
        EXPECT_EQ(m_state.purge_params, params);
        EXPECT_EQ(get_rx_token(m_state.tx_token_count), rx_token);

        ++m_state.purge_count;

        op_info.field_mask       = UCT_EP_OP_INFO_FIELD_OPERATION |
                                   UCT_EP_OP_INFO_FIELD_FLUSH;
        op_info.operation        = UCT_EP_OP_FLUSH;
        op_info.flush.field_mask = UCT_EP_OP_INFO_FLUSH_FIELD_FLAGS;
        op_info.flush.flags      = 0;
        params->cb(&op_info, params->arg);

        return UCS_OK;
    }

    static ucs_status_t
    second_iface_query_token(uct_iface_h iface,
                             uct_ib_mlx5_ext_iface_query_attr_t *attr)
    {
        EXPECT_EQ(m_state.iface, iface);
        ++m_state.second_iface_query_count;

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS) {
            attr->cap.flags = token_cap_flags();
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN_LEN) {
            attr->tx_token_len = sizeof(uint64_t) * 2;
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN_LEN) {
            attr->rx_token_len = sizeof(uint64_t) * 2;
        }

        if ((attr->field_mask &
             UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN) &&
            (attr->field_mask &
             UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN)) {
            ADD_FAILURE() << "second token plugin should not derive tokens";
            return UCS_ERR_IO_ERROR;
        }

        return UCS_OK;
    }

    static ucs_status_t
    second_ep_query(uct_ep_h ep, uct_ib_mlx5_ext_ep_query_attr_t *attr)
    {
        ADD_FAILURE()
                << "second token plugin should not be selected for ep query";
        return UCS_ERR_IO_ERROR;
    }

    static ucs_status_t
    second_ep_outstanding_purge(uct_ep_h ep,
                                const uct_ep_outstanding_purge_params_t *params)
    {
        ADD_FAILURE() << "second token plugin should not be selected for purge";
        return UCS_ERR_IO_ERROR;
    }

    static void register_other_plugin()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, other_plugin_name(), sizeof(ops.name));
        ops.iface_query = iface_query_other;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static void register_failing_plugin()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, failing_plugin_name(), sizeof(ops.name));
        ops.iface_query = iface_query_fail;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static void register_token_plugin()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, token_plugin_name(), sizeof(ops.name));
        ops.iface_query          = iface_query_token;
        ops.ep_query             = ep_query;
        ops.ep_outstanding_purge = ep_outstanding_purge;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static void register_token_plugin_without_ep_query()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, token_plugin_name(), sizeof(ops.name));
        ops.iface_query          = iface_query_token;
        ops.ep_outstanding_purge = ep_outstanding_purge;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static void register_token_plugin_without_purge()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, token_plugin_name(), sizeof(ops.name));
        ops.iface_query = iface_query_token;
        ops.ep_query    = ep_query;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static void register_second_token_plugin()
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, second_token_plugin_name(),
                         sizeof(ops.name));
        ops.iface_query          = second_iface_query_token;
        ops.ep_query             = second_ep_query;
        ops.ep_outstanding_purge = second_ep_outstanding_purge;

        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

    static state_t m_state;
};

test_uct_ib_mlx5_ext_rc::state_t test_uct_ib_mlx5_ext_rc::m_state;

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, iface_query)
{
    uint64_t tx_token        = 0x1234;
    uint64_t rx_token        = 0;
    uct_iface_attr_v2_t attr = {};

    m_state.iface = m_e1->iface();

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                          UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH |
                          UCT_IFACE_ATTR_FIELD_RX_TOKEN_LENGTH;
        EXPECT_EQ(UCS_ERR_UNSUPPORTED,
                  uct_iface_query_v2(m_e1->iface(), &attr));
    }

    register_failing_plugin();
    register_other_plugin();

    uint64_t failed_iface_query_count = m_state.failed_iface_query_count;
    attr            = uct_iface_attr_v2_t();
    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS;
    EXPECT_UCS_OK(uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(other_cap_flags(), attr.cap.flags);
    EXPECT_EQ(failed_iface_query_count + 1, m_state.failed_iface_query_count);

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr.field_mask = UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH |
                          UCT_IFACE_ATTR_FIELD_RX_TOKEN_LENGTH;
        EXPECT_EQ(UCS_ERR_UNSUPPORTED,
                  uct_iface_query_v2(m_e1->iface(), &attr));
    }

    register_token_plugin();
    register_second_token_plugin();

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS;
    EXPECT_EQ(UCS_OK, uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(token_cap_flags() | other_cap_flags(), attr.cap.flags);

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH |
                      UCT_IFACE_ATTR_FIELD_RX_TOKEN_LENGTH;
    EXPECT_EQ(UCS_OK, uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(token_cap_flags() | other_cap_flags(), attr.cap.flags);
    EXPECT_EQ(sizeof(uint64_t), attr.tx_token_length);
    EXPECT_EQ(sizeof(uint64_t), attr.rx_token_length);

    /* TX and RX tokens must be requested together. */
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr            = uct_iface_attr_v2_t();
        attr.field_mask = UCT_IFACE_ATTR_FIELD_TX_TOKEN;
        attr.tx_token   = &tx_token;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  uct_iface_query_v2(m_e1->iface(), &attr));
    }

    /* Derive an RX token from the TX token. */
    uint64_t second_iface_query_count = m_state.second_iface_query_count;
    uint64_t iface_query_count        = m_state.iface_query_count;
    attr            = uct_iface_attr_v2_t();
    attr.field_mask = UCT_IFACE_ATTR_FIELD_TX_TOKEN |
                      UCT_IFACE_ATTR_FIELD_RX_TOKEN;
    attr.tx_token   = &tx_token;
    attr.rx_token   = &rx_token;
    EXPECT_UCS_OK(uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_EQ(get_rx_token(tx_token), rx_token);
    EXPECT_EQ(1ul, m_state.rx_token_count);
    EXPECT_EQ(iface_query_count + 2, m_state.iface_query_count);
    EXPECT_EQ(second_iface_query_count, m_state.second_iface_query_count);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_query)
{
    uint64_t tx_token  = 0;
    uct_ep_attr_t attr = {};

    m_state.iface = m_e1->iface();
    m_state.ep = m_e1->ep(0);

    register_token_plugin_without_ep_query();

    attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
    attr.tx_token   = &tx_token;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, uct_ep_query(m_e1->ep(0), &attr));

    uct_ib_mlx5_ext_unregister(token_plugin_name());

    register_other_plugin();
    register_token_plugin();
    register_second_token_plugin();

    attr            = uct_ep_attr_t();
    attr.field_mask = UCT_EP_ATTR_FIELD_LOCAL_SOCKADDR;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, uct_ep_query(m_e1->ep(0), &attr));

    attr            = uct_ep_attr_t();
    attr.field_mask = UCT_EP_ATTR_FIELD_REMOTE_SOCKADDR;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, uct_ep_query(m_e1->ep(0), &attr));

    /* TX token requested without a buffer is rejected. */
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr            = uct_ep_attr_t();
        attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));
    }

    /* No token requested is a no-op success. */
    attr = uct_ep_attr_t();
    EXPECT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));

    /* Query the TX token. */
    uint64_t second_iface_query_count = m_state.second_iface_query_count;
    attr            = uct_ep_attr_t();
    attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
    attr.tx_token   = &tx_token;
    EXPECT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));
    EXPECT_EQ(tx_token, m_state.tx_token_count);
    EXPECT_EQ(second_iface_query_count, m_state.second_iface_query_count);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_outstanding_purge)
{
    uint64_t tx_token                        = 0;
    uint64_t rx_token                        = 0;
    uint64_t purge_count                     = 0;
    uct_ep_outstanding_purge_params_t params = {};
    uct_ep_attr_t ep_attr                    = {};
    uct_iface_attr_v2_t iface_attr           = {};

    m_state.iface = m_e1->iface();
    m_state.ep    = m_e1->ep(0);

    params.field_mask     = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                            UCT_EP_OUTSTANDING_FIELD_CB |
                            UCT_EP_OUTSTANDING_FIELD_ARG;
    params.rx_token       = &rx_token;
    params.cb             = purge_cb;
    params.arg            = &purge_count;

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        EXPECT_EQ(UCS_ERR_UNSUPPORTED,
                  uct_ep_outstanding_purge(m_e1->ep(0), &params));
    }
    EXPECT_EQ(0ul, purge_count);

    register_token_plugin_without_purge();
    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_ep_outstanding_purge(m_e1->ep(0), &params));
    uct_ib_mlx5_ext_unregister(token_plugin_name());

    params = uct_ep_outstanding_purge_params_t();

    register_other_plugin();
    register_token_plugin();
    register_second_token_plugin();

    {
        scoped_log_handler wrap_err(wrap_errors_logger);

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
        params.rx_token   = &rx_token;
        params.cb         = NULL;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  uct_ep_outstanding_purge(m_e1->ep(0), &params));
    }

    /* Sender queries its TX token from the endpoint. */
    ep_attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
    ep_attr.tx_token   = &tx_token;
    EXPECT_UCS_OK(uct_ep_query(m_e1->ep(0), &ep_attr));

    /* Receiver derives the RX token from the peer's TX token. */
    iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_TX_TOKEN |
                            UCT_IFACE_ATTR_FIELD_RX_TOKEN;
    iface_attr.tx_token   = &tx_token;
    iface_attr.rx_token   = &rx_token;
    EXPECT_UCS_OK(uct_iface_query_v2(m_e1->iface(), &iface_attr));

    /* Purge the outstanding operations using the derived RX token. */
    params.field_mask    = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                           UCT_EP_OUTSTANDING_FIELD_CB |
                           UCT_EP_OUTSTANDING_FIELD_ARG;
    params.rx_token      = &rx_token;
    params.cb            = purge_cb;
    params.arg           = &purge_count;
    m_state.purge_params = &params;
    uint64_t second_iface_query_count = m_state.second_iface_query_count;
    EXPECT_UCS_OK(uct_ep_outstanding_purge(m_e1->ep(0), &params));
    EXPECT_EQ(purge_count, m_state.purge_count);
    EXPECT_EQ(second_iface_query_count, m_state.second_iface_query_count);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_ib_mlx5_ext_rc, rc_mlx5)
