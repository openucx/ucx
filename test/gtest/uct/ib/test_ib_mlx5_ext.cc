/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_rc.h"

extern "C" {
#include <ucs/sys/stubs.h>
#include <uct/ib/mlx5/ib_mlx5_ext.h>

extern ucs_list_link_t uct_ib_mlx5_ext_plugins;
}


class test_uct_ib_mlx5_ext_rc : public test_rc {
public:
    void init() override
    {
        save_plugins();
        test_rc::init();
    }

    void cleanup() override
    {
        uct_ib_mlx5_ext_cleanup();
        uct_test::cleanup();
        restore_plugins();
    }

protected:
    static uint64_t tx_token()
    {
        return 0x1234;
    }

    static uint64_t rx_token()
    {
        return tx_token() + 1;
    }

    void save_plugins()
    {
        if (ucs_list_is_empty(&uct_ib_mlx5_ext_plugins)) {
            ucs_list_head_init(&m_saved_plugins);
        } else {
            ucs_list_replace(&uct_ib_mlx5_ext_plugins, &m_saved_plugins);
            ucs_list_head_init(&uct_ib_mlx5_ext_plugins);
        }
    }

    void restore_plugins()
    {
        if (!ucs_list_is_empty(&m_saved_plugins)) {
            ucs_list_replace(&m_saved_plugins, &uct_ib_mlx5_ext_plugins);
            ucs_list_head_init(&m_saved_plugins);
        }
    }

    static ucs_status_t
    iface_query(uct_iface_h, uct_ib_mlx5_ext_iface_query_attr_t *attr)
    {
        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS) {
            attr->cap.flags = UCT_IFACE_FLAG_V2_QUERY_TOKEN;
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN_LEN) {
            attr->tx_token_len = sizeof(uint64_t);
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN_LEN) {
            attr->rx_token_len = sizeof(uint64_t);
        }

        if (attr->field_mask &
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN) {
            if ((attr->tx_token == NULL) || (attr->rx_token == NULL)) {
                ADD_FAILURE() << "token buffers must be non-NULL";
                return UCS_ERR_INVALID_PARAM;
            }
            *static_cast<uint64_t*>(attr->rx_token) =
                    *static_cast<const uint64_t*>(attr->tx_token) + 1;
        }

        return UCS_OK;
    }

    static ucs_status_t
    ep_query(uct_ep_h, uct_ib_mlx5_ext_ep_query_attr_t *attr)
    {
        *static_cast<uint64_t*>(attr->tx_token) = tx_token();
        return UCS_OK;
    }

    static ucs_status_t
    purge_fail(uct_ep_h, const uct_ep_outstanding_purge_params_t*)
    {
        return UCS_ERR_UNSUPPORTED;
    }

    static ucs_status_t
    purge(uct_ep_h, const uct_ep_outstanding_purge_params_t *params)
    {
        uct_ep_op_info_t op_info = {};

        EXPECT_EQ(rx_token(), *static_cast<const uint64_t*>(params->rx_token));
        params->cb(&op_info, params->arg);
        return UCS_OK;
    }

    static void purge_cb(const uct_ep_op_info_t*, void *arg)
    {
        *static_cast<bool*>(arg) = true;
    }

    static void
    register_plugin(const char *name,
                    uct_ib_mlx5_ext_iface_query_func_t iface_query_cb = NULL,
                    uct_ib_mlx5_ext_ep_query_func_t ep_query_cb = NULL,
                    uct_ep_outstanding_purge_func_t purge_cb = NULL)
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, name, sizeof(ops.name));
        ops.iface_query          = iface_query_cb;
        ops.ep_query             = ep_query_cb;
        ops.ep_outstanding_purge = purge_cb;
        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

private:
    ucs_list_link_t m_saved_plugins = UCS_LIST_INITIALIZER(&m_saved_plugins,
                                                           &m_saved_plugins);
};


UCS_TEST_P(test_uct_ib_mlx5_ext_rc, iface_query)
{
    uint64_t tx_token_value  = tx_token();
    uint64_t rx_token_value  = 0;
    uct_iface_attr_v2_t attr = {};

    register_plugin("token", iface_query, ep_query, purge);

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr.field_mask = UCT_IFACE_ATTR_FIELD_TX_TOKEN;
        attr.tx_token   = &tx_token_value;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  uct_iface_query_v2(m_e1->iface(), &attr));
    }

    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN_LENGTH |
                      UCT_IFACE_ATTR_FIELD_RX_TOKEN_LENGTH |
                      UCT_IFACE_ATTR_FIELD_TX_TOKEN |
                      UCT_IFACE_ATTR_FIELD_RX_TOKEN;
    attr.tx_token   = &tx_token_value;
    attr.rx_token   = &rx_token_value;

    ASSERT_UCS_OK(uct_iface_query_v2(m_e1->iface(), &attr));
    EXPECT_TRUE(attr.cap.flags & UCT_IFACE_FLAG_V2_QUERY_TOKEN);
    EXPECT_EQ(sizeof(uint64_t), attr.tx_token_length);
    EXPECT_EQ(sizeof(uint64_t), attr.rx_token_length);
    EXPECT_EQ(rx_token(), rx_token_value);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_query)
{
    uint64_t tx_token_value = 0;
    uct_ep_attr_t attr      = {};

    register_plugin("token", iface_query, ep_query, purge);

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));
    }

    attr.tx_token = &tx_token_value;
    ASSERT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));
    EXPECT_EQ(tx_token(), tx_token_value);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_outstanding_purge)
{
    uint64_t rx_token_value                  = rx_token();
    bool callback_invoked                    = false;
    uct_ep_outstanding_purge_params_t params = {};

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  uct_ep_outstanding_purge(m_e1->ep(0), &params));
    }

    register_plugin("stub",
                    (uct_ib_mlx5_ext_iface_query_func_t)
                            ucs_empty_function_return_unsupported,
                    NULL, NULL);
    register_plugin("fail", NULL, NULL, purge_fail);
    register_plugin("token", iface_query, ep_query, purge);

    params.field_mask = UCT_EP_OUTSTANDING_FIELD_RX_TOKEN |
                        UCT_EP_OUTSTANDING_FIELD_CB |
                        UCT_EP_OUTSTANDING_FIELD_ARG;
    params.rx_token   = &rx_token_value;
    params.cb         = purge_cb;
    params.arg        = &callback_invoked;

    ASSERT_UCS_OK(uct_ep_outstanding_purge(m_e1->ep(0), &params));
    EXPECT_TRUE(callback_invoked);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_ib_mlx5_ext_rc, rc_mlx5)
