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
#include <uct/ib/mlx5/exp/ib_exp.h>
#endif
#include <uct/ib/rc/verbs/rc_verbs.h>
}

#include <uct/uct_p2p_test.h>

class uct_test_event_base : public uct_p2p_test {
public:
    uct_test_event_base(): uct_p2p_test(0), m_event() {}

    class qp {
    public:
        qp(entity &e) : m_e(e) {}
        virtual ~qp() {}

        virtual uint32_t qp_num() const = 0;
        virtual void to_err()           = 0;

        struct ibv_ah_attr ah_attr() const {
            uct_ib_iface_t *iface = ucs_derived_of(m_e.iface(), uct_ib_iface_t);
            struct ibv_ah_attr result = {};

            uct_ib_iface_fill_ah_attr_from_gid_lid(iface,
                    uct_ib_iface_port_attr(iface)->lid, &iface->gid_info.gid,
                    iface->gid_info.gid_index, 0, &result);
            return result;
        }

        enum ibv_mtu path_mtu() const {
            uct_ib_iface_t *iface = ucs_derived_of(m_e.iface(), uct_ib_iface_t);

            return iface->config.path_mtu;
        }

    protected:
        entity &m_e;
    };

    struct event_ctx {
        uct_ib_async_event_wait_t super;
        volatile bool             got;
        uct_ib_device_t           *dev;
    };

    static unsigned last_wqe_check_cb(void *arg) {
        event_ctx *event = (event_ctx *)arg;
        int cb_id;

        ucs_spin_lock(&event->dev->async_event_lock);
        cb_id              = event->super.cb_id;
        event->super.cb_id = UCS_CALLBACKQ_ID_NULL;
        ucs_spin_unlock(&event->dev->async_event_lock);

        EXPECT_FALSE(event->got);
        event->got = true;
        ucs_callbackq_remove_safe(event->super.cbq, cb_id);
        return 1;
    }

    virtual void init_qp(entity &e) = 0;

    uct_ib_device_t *dev(entity &e) {
        return &ucs_derived_of(e.md(), uct_ib_md_t)->dev;
    }

    bool wait_for_last_wqe_event(entity &e, bool before) {
        uint32_t qp_num = m_qp->qp_num();
        ucs_time_t deadline;
        ucs_status_t status;

        m_event.got       = false;
        m_event.super.cb  = last_wqe_check_cb;
        m_event.super.cbq = &e.worker()->progress_q;
        m_event.dev       = dev(e);

        if (before) {
            /* move QP to error state before scheduling event callback */
            m_qp->to_err();
            usleep(1000);
        }

        /* schedule event callback */
        status = uct_ib_device_async_event_wait(dev(e),
                IBV_EVENT_QP_LAST_WQE_REACHED, qp_num, &m_event.super);
        ASSERT_UCS_OK(status);

        /* event should not be called directly, but only from progress */
        usleep(1000);
        EXPECT_FALSE(m_event.got);

        if (!before) {
            /* move QP to error state after scheduling event callback */
            m_qp->to_err();
        }

        /* wait for callback */
        deadline = ucs_get_time() +
                   ucs_time_from_sec(ucs::test_time_multiplier() * 10);
        while (!m_event.got && (ucs_get_time() < deadline)) {
            e.progress();
        }

        return m_event.got;
    }

protected:
    ucs::auto_ptr<qp> m_qp;
    event_ctx         m_event;
};

class uct_ep_test_event : public uct_test_event_base {
protected:
    void init_qp(entity &e) {
        if (GetParam()->tl_name == "rc_mlx5") {
#if HAVE_MLX5_HW
            m_qp.reset(new mlx5_qp(e));
#else
            ucs_fatal("no mlx5 compile time support");
#endif
        } else {
            m_qp.reset(new verbs_qp(e));
        }
    }

private:
#if HAVE_MLX5_HW
    class mlx5_qp : public qp {
    public:
        mlx5_qp(entity &e) : qp(e) {}

        uint32_t qp_num() const {
            uct_rc_mlx5_ep_t *ep = (uct_rc_mlx5_ep_t *)m_e.ep(0);
            return ep->tx.wq.super.qp_num;
        }

        void to_err() {
            uct_ib_iface_t   *iface = (uct_ib_iface_t *)m_e.iface();
            uct_rc_mlx5_ep_t *ep    = (uct_rc_mlx5_ep_t *)m_e.ep(0);

            uct_ib_mlx5_modify_qp_state(iface, &ep->tx.wq.super, IBV_QPS_ERR);
        }
    };
#endif

    class verbs_qp : public qp {
    public:
        verbs_qp(entity &e) : qp(e) {}

        uint32_t qp_num() const {
            uct_rc_verbs_ep_t *ep = (uct_rc_verbs_ep_t *)m_e.ep(0);
            return ep->qp->qp_num;
        }

        void to_err() {
            uct_rc_verbs_ep_t *ep = (uct_rc_verbs_ep_t *)m_e.ep(0);
            uct_ib_modify_qp(ep->qp, IBV_QPS_ERR);
        }
    };
};

class uct_p2p_test_event_log : public uct_ep_test_event {
public:
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

    int wait_for_last_wqe_event_by_log(entity &e) {
        init_qp(e);

        uint32_t qp_num = m_qp->qp_num();
        int ret         = 0;

        flushed_qp_num = UINT_MAX;
        m_qp->to_err();
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
};

UCS_TEST_P(uct_p2p_test_event_log, last_wqe)
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

ucs_log_level_t uct_p2p_test_event_log::orig_log_level;
volatile unsigned uct_p2p_test_event_log::flushed_qp_num;

UCT_INSTANTIATE_RC_TEST_CASE(uct_p2p_test_event_log);


class uct_p2p_test_event : public uct_ep_test_event {
public:
    bool wait_for_last_wqe_event(entity& e, bool before) {
        init_qp(e); /* retrieve rc qp from connected entity */
        return uct_test_event_base::wait_for_last_wqe_event(e, before);
    }

};

UCS_TEST_P(uct_p2p_test_event, last_wqe_cb_after_subscribe)
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    ASSERT_TRUE(wait_for_last_wqe_event(sender(), false));
    if (!r->loopback) {
        ASSERT_TRUE(wait_for_last_wqe_event(receiver(), false));
    }
}

UCS_TEST_P(uct_p2p_test_event, last_wqe_cb_before_subscribe)
{
    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    ASSERT_TRUE(wait_for_last_wqe_event(sender(), true));
    if (!r->loopback) {
        ASSERT_TRUE(wait_for_last_wqe_event(receiver(), true));
    }
}

UCT_INSTANTIATE_RC_TEST_CASE(uct_p2p_test_event);


class uct_qp_test_event : public uct_test_event_base {
public:
    uct_qp_test_event() : uct_test_event_base() {}

    void init_qp(entity &e) {
        if (GetParam()->tl_name == "rc_mlx5") {
#if HAVE_MLX5_HW
            m_qp.reset(new mlx5_qp(e));
#else
            ucs_fatal("no mlx5 compile time support");
#endif
        } else {
            m_qp.reset(new verbs_qp(e));
        }
    }

    static std::vector<const resource*>
    enum_resources(const std::string& tl_name) {
        return uct_test::enum_resources(tl_name);
    }

    virtual void init() {
        uct_test::init();
        m_e.reset(create_entity(0));
    }

    bool wait_for_last_wqe_event(bool before) {
        ucs_status_t status;
        bool ret;

        init_qp(*m_e);

        status = uct_ib_device_async_event_register(dev(*m_e),
                IBV_EVENT_QP_LAST_WQE_REACHED, m_qp->qp_num());
        ASSERT_UCS_OK(status);

        ret = uct_test_event_base::wait_for_last_wqe_event(*m_e, before);

        uct_ib_device_async_event_unregister(dev(*m_e),
                IBV_EVENT_QP_LAST_WQE_REACHED, m_qp->qp_num());

        m_qp.reset();
        return ret;
    }

protected:
    ucs::auto_ptr<entity> m_e;

private:
#if HAVE_MLX5_HW
    class mlx5_qp : public qp {
    public:
        mlx5_qp(entity &e) : qp(e), m_txwq(), m_iface(), m_md() {
            m_iface = ucs_derived_of(m_e.iface(), uct_rc_mlx5_iface_common_t);
            m_md    = ucs_derived_of(m_e.md(), uct_ib_mlx5_md_t);
            uct_ib_mlx5_qp_attr_t attr = {};
            ucs_status_t status;

            uct_rc_mlx5_iface_fill_attr(m_iface, &attr,
                                        m_iface->super.config.tx_qp_len,
                                        &m_iface->rx.srq);
            uct_ib_exp_qp_fill_attr(&m_iface->super.super, &attr.super);
            status = uct_rc_mlx5_iface_create_qp(m_iface, &m_txwq.super,
                                                 &m_txwq, &attr);
            ASSERT_UCS_OK(status);

            if (m_txwq.super.type == UCT_IB_MLX5_OBJ_TYPE_VERBS) {
                status = uct_rc_iface_qp_init(&m_iface->super,
                                              m_txwq.super.verbs.qp);
                ASSERT_UCS_OK(status);
            }

            struct ibv_ah_attr ah = ah_attr();
            status = uct_rc_mlx5_ep_connect_qp(m_iface, &m_txwq.super,
                                               qp_num(), &ah, path_mtu(), 0);
            ASSERT_UCS_OK(status);
        }

        ~mlx5_qp() {
            uct_ib_mlx5_qp_mmio_cleanup(&m_txwq.super, m_txwq.reg);
            uct_ib_mlx5_destroy_qp(m_md, &m_txwq.super);
        }

        uint32_t qp_num() const {
            return m_txwq.super.qp_num;
        }

        void to_err() {
            uct_ib_mlx5_modify_qp_state(&m_iface->super.super, &m_txwq.super,
                                        IBV_QPS_ERR);
        }

    private:
        uct_ib_mlx5_txwq_t         m_txwq;
        uct_rc_mlx5_iface_common_t *m_iface;
        uct_ib_mlx5_md_t           *m_md;
    };
#endif

    class verbs_qp : public qp {
    public:
        verbs_qp(entity &e) : qp(e), m_ibqp(), m_iface() {
            m_iface = ucs_derived_of(m_e.iface(), uct_rc_verbs_iface_t);

            uct_ib_qp_attr_t attr = {};
            ucs_status_t status;

            status = uct_rc_iface_qp_create(&m_iface->super, &m_ibqp, &attr,
                                            m_iface->super.config.tx_qp_len,
                                            m_iface->srq);
            ASSERT_UCS_OK(status);

            status = uct_rc_iface_qp_init(&m_iface->super, m_ibqp);
            ASSERT_UCS_OK(status);

            struct ibv_ah_attr ah = ah_attr();
            status = uct_rc_iface_qp_connect(&m_iface->super, m_ibqp,
                                             qp_num(), &ah, path_mtu());
            ASSERT_UCS_OK(status);
        }

        ~verbs_qp() {
            uct_ib_destroy_qp(m_ibqp);
        }

        uint32_t qp_num() const {
            return m_ibqp->qp_num;
        }

        void to_err() {
            uct_ib_modify_qp(m_ibqp, IBV_QPS_ERR);
        }

    private:
        struct ibv_qp        *m_ibqp;
        uct_rc_verbs_iface_t *m_iface;
    };
};

UCS_TEST_P(uct_qp_test_event, last_wqe_cb_after_subscribe)
{
    ASSERT_TRUE(wait_for_last_wqe_event(false));
}

UCS_TEST_P(uct_qp_test_event, last_wqe_cb_before_subscribe)
{
    ASSERT_TRUE(wait_for_last_wqe_event(true));
}

UCT_INSTANTIATE_RC_TEST_CASE(uct_qp_test_event);
