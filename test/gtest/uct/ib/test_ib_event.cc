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
#if HAVE_MLX5_HW
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/rc/accel/rc_mlx5.h>
#endif
#include <uct/ib/rc/verbs/rc_verbs.h>
}

#include <uct/uct_p2p_test.h>

class uct_p2p_test_event : public uct_p2p_test {
private:
    void rc_mlx5_ep_to_err(entity &e, uint32_t *qp_num_p) {
#if HAVE_MLX5_HW
        uct_ib_mlx5_md_t *md = (uct_ib_mlx5_md_t *)e.md();
        uct_rc_mlx5_ep_t *ep = (uct_rc_mlx5_ep_t *)e.ep(0);
        uct_ib_mlx5_qp_t *qp = &ep->tx.wq.super;

        uct_ib_mlx5_modify_qp_state(md, qp, IBV_QPS_ERR);

        *qp_num_p = qp->qp_num;
#endif
    }

    void rc_verbs_ep_to_err(entity &e, uint32_t *qp_num_p) {
        uct_rc_verbs_ep_t *ep = (uct_rc_verbs_ep_t *)e.ep(0);

        uct_ib_modify_qp(ep->qp, IBV_QPS_ERR);

        *qp_num_p = ep->qp->qp_num;
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

    int wait_for_last_wqe_event(entity &e) {
        const resource *r = dynamic_cast<const resource*>(GetParam());
        flushed_qp_num = -1;
        uint32_t qp_num = 0;

        if (r->tl_name == "rc_mlx5") {
            rc_mlx5_ep_to_err(e, &qp_num);
        } else {
            rc_verbs_ep_to_err(e, &qp_num);
        }

        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(ucs::test_time_multiplier());
        while (ucs_get_time() < deadline) {
            if (flushed_qp_num == qp_num) {
                return 1;
            }
            usleep(1000);
        }

        return 0;
    }
};

UCS_TEST_P(uct_p2p_test_event, last_wqe, "ASYNC_EVENTS=y")
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    mapped_buffer sendbuf(0, 0, sender());
    mapped_buffer recvbuf(0, 0, receiver());

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

    ASSERT_TRUE(wait_for_last_wqe_event(sender()));
    if (!r->loopback) {
        ASSERT_TRUE(wait_for_last_wqe_event(receiver()));
    }
}

ucs_log_level_t uct_p2p_test_event::orig_log_level;
volatile unsigned uct_p2p_test_event::flushed_qp_num;

UCT_INSTANTIATE_RC_TEST_CASE(uct_p2p_test_event);
