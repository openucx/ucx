/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>
#include <ucs/time/time.h>
#include <poll.h>
#include <infiniband/verbs.h>

/* wait for 1 sec to get statistics */
static const unsigned long test_period_usec = (1ul * UCS_USEC_PER_SEC);
static const unsigned moderation_period_usec = 1000; /* usecs */
/* use multiplier 2 because we have same iface to send/recv which may produce 2x events */
static const unsigned event_limit = (2 * test_period_usec / moderation_period_usec);
static const unsigned max_repeats = 60; /* max 3 minutes per test */

class test_uct_cq_moderation : public uct_test {
protected:

    void init() {
        if (RUNNING_ON_VALGRIND) {
            UCS_TEST_SKIP_R("skipping on valgrind");
        }

        if (!has_rc() && !has_ud()) {
            UCS_TEST_SKIP_R("unsupported");
        }

        uct_test::init();

        if (has_rc()) {
            set_config("RC_FC_ENABLE=n");
        }

        set_config(std::string("IB_TX_EVENT_MOD_PERIOD=") +
                   ucs::to_string(moderation_period_usec) + "us");
        set_config(std::string("IB_RX_EVENT_MOD_PERIOD=") +
                   ucs::to_string(moderation_period_usec) + "us");

        m_sender = uct_test::create_entity(0, NULL, NULL, NULL, NULL, NULL,
                                           send_async_event_handler, this);
        m_entities.push_back(m_sender);

        m_receiver = uct_test::create_entity(0, NULL, NULL, NULL, NULL, NULL,
                                             recv_async_event_handler, this);
        m_entities.push_back(m_receiver);

        check_skip_test();

        m_send_async_event_ctx.wait_for_event(*m_sender, 0);
        m_recv_async_event_ctx.wait_for_event(*m_receiver, 0);
    }

    void connect() {
        m_sender->connect(0, *m_receiver, 0);
        short_progress_loop(10); /* Some transports need time to become ready */
    }

    void disconnect() {
        flush();
        if (m_receiver->iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            m_receiver->destroy_ep(0);
        }
        m_sender->destroy_ep(0);
    }

    static void send_async_event_handler(void *arg, unsigned flags) {
        test_uct_cq_moderation *self = static_cast<test_uct_cq_moderation*>(arg);
        self->m_send_async_event_ctx.signal();
    }

    static void recv_async_event_handler(void *arg, unsigned flags) {
        test_uct_cq_moderation *self = static_cast<test_uct_cq_moderation*>(arg);
        self->m_recv_async_event_ctx.signal();
    }

    void iface_arm(entity &test_e, async_event_ctx &ctx) {
        /* wait for all messages are arrived */
        while (m_recv < m_send) {
            progress();
        }

        do {
            /* arm all event types */
            while (1) {
                if (uct_iface_event_arm(test_e.iface(),
                                        UCT_EVENT_SEND_COMP |
                                        UCT_EVENT_RECV      |
                                        UCT_EVENT_RECV_SIG) != UCS_ERR_BUSY) {
                    break;
                }
                progress();
            }
            /* repeat till there are events */
        } while (ctx.wait_for_event(test_e, 0));
    }

    static ucs_status_t am_cb(void *arg, void *data, size_t len, unsigned flags) {
        ucs_assert_always(arg != NULL);
        test_uct_cq_moderation *self = static_cast<test_uct_cq_moderation*>(arg);

        self->m_recv++;

        return UCS_OK;
    }

    void run_test(entity &test_e, async_event_ctx &ctx);

    entity * m_sender;
    entity * m_receiver;

    unsigned m_send;
    unsigned m_recv;

    uct_test::async_event_ctx m_send_async_event_ctx;
    uct_test::async_event_ctx m_recv_async_event_ctx;
};

void test_uct_cq_moderation::run_test(entity &test_e, async_event_ctx &ctx) {
    unsigned events, i;
    ucs_status_t status;

    uct_iface_set_am_handler(m_receiver->iface(), 0, am_cb, this, 0);

    connect();

    m_send = 0;
    m_recv = 0;

    /* repeat test till at least one iteration is successful
     * to exclude random fluctuations */
    for (i = 0; i < max_repeats; i++) {
        events = 0;
        iface_arm(test_e, ctx);

        ucs_time_t tm = ucs_get_time();

        while ((ucs_time_to_usec(ucs_get_time()) -
                ucs_time_to_usec(tm)) < test_period_usec) {
            if (ctx.wait_for_event(test_e, 0)) {
                events++;
                iface_arm(test_e, ctx);
            }

            do {
                status = uct_ep_am_short(m_sender->ep(0), 0, 0, NULL, 0);
                progress();
            } while (status == UCS_ERR_NO_RESOURCE);
            m_send++;
            ASSERT_UCS_OK(status);
        }
        m_sender->flush();
        UCS_TEST_MESSAGE << "iteration: " << i + 1 << ", events: " << events
                         << ", limit: " << event_limit;
        if (events <= event_limit) {
            break;
        }
    }

    disconnect();

    EXPECT_LE(events, event_limit);
}

UCS_TEST_SKIP_COND_P(test_uct_cq_moderation, send_period,
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_SEND_COMP)) {
    run_test(*m_sender, m_send_async_event_ctx);
}

UCS_TEST_SKIP_COND_P(test_uct_cq_moderation, recv_period,
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV)) {
    run_test(*m_receiver, m_recv_async_event_ctx);
}

#if HAVE_DECL_IBV_EXP_CQ_MODERATION
UCT_INSTANTIATE_IB_TEST_CASE(test_uct_cq_moderation)
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */
