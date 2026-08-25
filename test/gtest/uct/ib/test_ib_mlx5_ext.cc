/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_rc.h"

extern "C" {
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
                    uct_ep_outstanding_purge_func_t purge_cb = NULL)
    {
        uct_ib_mlx5_ext_ops_t ops = {};

        ucs_strncpy_zero(ops.name, name, sizeof(ops.name));
        ops.ep_outstanding_purge = purge_cb;
        ASSERT_UCS_OK(uct_ib_mlx5_ext_register(&ops));
    }

private:
    ucs_list_link_t m_saved_plugins = UCS_LIST_INITIALIZER(&m_saved_plugins,
                                                           &m_saved_plugins);
};


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

    register_plugin("stub");
    register_plugin("fail", purge_fail);
    register_plugin("token", purge);

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
