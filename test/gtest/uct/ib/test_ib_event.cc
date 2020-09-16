/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <infiniband/verbs.h>
#include <common/test.h>

extern "C" {
#include <uct/ib/base/ib_device.h>
#if HAVE_MLX5_HW
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/rc/accel/rc_mlx5.h>
#endif
#include <uct/ib/rc/verbs/rc_verbs.h>
}

#include <uct/uct_p2p_test.h>

class uct_p2p_test_event : public uct_p2p_test {
private:
    uint32_t rc_mlx5_qp_num(entity &e) {
#if HAVE_MLX5_HW
        uct_rc_mlx5_ep_t *ep = (uct_rc_mlx5_ep_t *)e.ep(0);
        return ep->tx.wq.super.qp_num;
#else
        ucs_fatal("no mlx5 compile time support");
        return 0;
#endif
    }

    uint32_t rc_verbs_qp_num(entity &e) {
        uct_rc_verbs_ep_t *ep = (uct_rc_verbs_ep_t *)e.ep(0);
        return ep->qp->qp_num;
    }

    void rc_mlx5_ep_to_err(entity &e) {
#if HAVE_MLX5_HW
        uct_ib_mlx5_md_t *md = (uct_ib_mlx5_md_t *)e.md();
        uct_rc_mlx5_ep_t *ep = (uct_rc_mlx5_ep_t *)e.ep(0);
        uct_ib_mlx5_qp_t *qp = &ep->tx.wq.super;

        uct_ib_mlx5_modify_qp_state(md, qp, IBV_QPS_ERR);
#endif
    }

    void rc_verbs_ep_to_err(entity &e) {
        uct_rc_verbs_ep_t *ep = (uct_rc_verbs_ep_t *)e.ep(0);
        uct_ib_modify_qp(ep->qp, IBV_QPS_ERR);
    }

public:
    uct_p2p_test_event(): uct_p2p_test(0) {}

    static ucs_log_level_t orig_log_level;
    static volatile unsigned flushed_qp_num;

    static ucs_log_func_rc_t
    last_wqe_check_log(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level,
                       const ucs_log_component_config_t *comp_conf,
                       const char *message, va_list ap)
    {
        std::string msg = format_message(message, ap);

        UCS_TEST_MESSAGE << msg.c_str();
        sscanf(msg.c_str(),
               "IB Async event on %*s SRQ-attached QP 0x%x was flushed",
               &flushed_qp_num);

        return (level <= orig_log_level) ? UCS_LOG_FUNC_RC_CONTINUE
            : UCS_LOG_FUNC_RC_STOP;
    }

    void trigger_last_wqe_event(entity &e) {
        const resource *r = dynamic_cast<const resource*>(GetParam());

        if (r->tl_name == "rc_mlx5") {
            rc_mlx5_ep_to_err(e);
        } else {
            rc_verbs_ep_to_err(e);
        }
    }

    uint32_t get_qp_num(entity &e) {
        const resource *r = dynamic_cast<const resource*>(GetParam());

        if (r->tl_name == "rc_mlx5") {
            return rc_mlx5_qp_num(e);
        } else {
            return rc_verbs_qp_num(e);
        }
    }

    int wait_for_last_wqe_event_by_log(entity &e) {
        uint32_t qp_num = get_qp_num(e);
        flushed_qp_num = -1;
        int ret = 0;

        trigger_last_wqe_event(e);
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(ucs::test_time_multiplier());
        while (ucs_get_time() < deadline) {
            if (flushed_qp_num == qp_num) {
                ret = 1;
                break;
            }
            usleep(1000);
        }

        return ret;
    }

    static unsigned last_wqe_check_cb(void *arg) {
        volatile bool *got_event = (volatile bool *)arg;
        *got_event = true;
        return 1;
    }

    int wait_for_last_wqe_event_cb(entity &e, bool before) {
        volatile bool got_event = false;
        uint32_t qp_num = get_qp_num(e);
        ucs_status_t status;

        if (before) {
            trigger_last_wqe_event(e);
            usleep(1000);
        }

        status = uct_ib_device_async_event_wait(
                &ucs_derived_of(e.md(), uct_ib_md_t)->dev,
                IBV_EVENT_QP_LAST_WQE_REACHED, qp_num,
                last_wqe_check_cb, (void *)&got_event);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
        if (status == UCS_OK) {
            return 1;
        }

        if (!before) {
            trigger_last_wqe_event(e);
        }

        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(ucs::test_time_multiplier() * 10);
        while (!got_event && ucs_get_time() < deadline) {
            progress();
        }

        return got_event;
    }
};

UCS_TEST_P(uct_p2p_test_event, last_wqe)
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    ucs_log_push_handler(last_wqe_check_log);
    orig_log_level = ucs_global_opts.log_component.log_level;
    ucs_global_opts.log_component.log_level = UCS_LOG_LEVEL_DEBUG;
    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        UCS_TEST_SKIP_R("Debug logging is disabled");
    }

    UCS_TEST_SCOPE_EXIT() {
        ucs_global_opts.log_component.log_level = orig_log_level;
        ucs_log_pop_handler();
    } UCS_TEST_SCOPE_EXIT_END

    ASSERT_TRUE(wait_for_last_wqe_event_by_log(sender()));
    if (!r->loopback) {
        ASSERT_TRUE(wait_for_last_wqe_event_by_log(receiver()));
    }
}

UCS_TEST_P(uct_p2p_test_event, last_wqe_cb_after_subscribe)
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    ASSERT_TRUE(wait_for_last_wqe_event_cb(sender(), false));
    if (!r->loopback) {
        ASSERT_TRUE(wait_for_last_wqe_event_cb(receiver(), false));
    }
}

UCS_TEST_P(uct_p2p_test_event, last_wqe_cb_before_subscribe)
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    ASSERT_TRUE(wait_for_last_wqe_event_cb(sender(), true));
    if (!r->loopback) {
        ASSERT_TRUE(wait_for_last_wqe_event_cb(receiver(), true));
    }
}

ucs_log_level_t uct_p2p_test_event::orig_log_level;
volatile unsigned uct_p2p_test_event::flushed_qp_num;

UCT_INSTANTIATE_RC_TEST_CASE(uct_p2p_test_event);
