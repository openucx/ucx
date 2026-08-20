/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_rc.h"

extern "C" {
#include <ucs/sys/stubs.h>
#include <uct/ib/mlx5/ib_mlx5_ext.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>

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
    static ucs_status_t am_counter(void *arg, void*, size_t, unsigned)
    {
        ++*static_cast<uint32_t*>(arg);
        return UCS_OK;
    }

    static void query_tx_token(uct_ep_h ep, uct_rc_mlx5_ft_tx_token_t *tx_token)
    {
        uct_ep_attr_t attr = {};

        attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
        attr.tx_token   = tx_token;
        ASSERT_UCS_OK(uct_ep_query(ep, &attr));
    }

    static void query_rx_token(uct_iface_h iface,
                               const uct_rc_mlx5_ft_tx_token_t *tx_token,
                               uct_rc_mlx5_ft_rx_token_t *rx_token)
    {
        uct_iface_attr_v2_t attr = {};

        attr.field_mask = UCT_IFACE_ATTR_FIELD_TX_TOKEN |
                          UCT_IFACE_ATTR_FIELD_RX_TOKEN;
        attr.tx_token   = tx_token;
        attr.rx_token   = rx_token;
        ASSERT_UCS_OK(uct_iface_query_v2(iface, &attr));
    }

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
    uct_rc_mlx5_ft_tx_token_t tx_token_value = {};
    uct_rc_mlx5_ft_rx_token_t rx_token_value = {};
    uct_iface_attr_v2_t attr = {};

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

    query_tx_token(m_e1->ep(0), &tx_token_value);
    ASSERT_UCS_OK(uct_iface_query_v2(m_e2->iface(), &attr));
    EXPECT_TRUE(attr.cap.flags & UCT_IFACE_FLAG_V2_QUERY_TOKEN);
    EXPECT_EQ(sizeof(tx_token_value), attr.tx_token_length);
    EXPECT_EQ(sizeof(rx_token_value), attr.rx_token_length);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, ep_query)
{
    uct_rc_mlx5_ft_tx_token_t tx_token_value = {};
    uct_rc_mlx5_base_ep_t *remote_ep         = ucs_derived_of(m_e2->ep(0),
                                                              uct_rc_mlx5_base_ep_t);
    uct_ep_attr_t attr                       = {};

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        attr.field_mask = UCT_EP_ATTR_FIELD_TX_TOKEN;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_ep_query(m_e1->ep(0), &attr));
    }

    attr.tx_token = &tx_token_value;
    ASSERT_UCS_OK(uct_ep_query(m_e1->ep(0), &attr));
    EXPECT_EQ(remote_ep->tx.wq.super.qp_num, tx_token_value.remote_qpn);
}

UCS_TEST_P(test_uct_ib_mlx5_ext_rc, token_query_psn)
{
    uct_rc_mlx5_ft_tx_token_t tx_token_value = {};
    uct_rc_mlx5_ft_rx_token_t rx_token_pre   = {};
    uct_rc_mlx5_ft_rx_token_t rx_token_post  = {};
    uct_rc_mlx5_base_ep_t *sender_ep         = ucs_derived_of(m_e1->ep(0),
                                                              uct_rc_mlx5_base_ep_t);
    const uint32_t num_messages = UCS_PTR_BYTE_DIFF(sender_ep->tx.wq.qstart,
                                                    sender_ep->tx.wq.qend) /
                                          MLX5_SEND_WQE_BB +
                                  1;
    uint32_t rx_count           = 0;

    ASSERT_UCS_OK(uct_iface_set_am_handler(m_e2->iface(), 0, am_counter,
                                           &rx_count, 0));
    query_tx_token(m_e1->ep(0), &tx_token_value);
    query_rx_token(m_e2->iface(), &tx_token_value, &rx_token_pre);

    EXPECT_EQ(sender_ep->tx.wq.next_first_psn, rx_token_pre.receiver_next_psn)
            << "initial HW receive PSN must match SW send PSN";

    for (uint32_t i = 0; i < num_messages; ++i) {
        ucs_status_t status;

        do {
            status = uct_ep_am_short(m_e1->ep(0), 0, i, NULL, 0);
            if (status == UCS_ERR_NO_RESOURCE) {
                progress();
            }
        } while (status == UCS_ERR_NO_RESOURCE);
        ASSERT_UCS_OK(status);
    }

    while (rx_count < num_messages) {
        progress();
    }

    query_rx_token(m_e2->iface(), &tx_token_value, &rx_token_post);
    EXPECT_EQ((rx_token_pre.receiver_next_psn + num_messages) &
                      UCS_MASK(UCT_IB_MLX5_PSN_BITS),
              rx_token_post.receiver_next_psn);
    EXPECT_EQ(sender_ep->tx.wq.next_first_psn, rx_token_post.receiver_next_psn)
            << "updated HW receive PSN must match SW send PSN";
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
