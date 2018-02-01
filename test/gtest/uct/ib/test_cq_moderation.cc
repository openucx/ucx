/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
}
#include <uct/uct_test.h>
#include <ucs/time/time.h>
#include <poll.h>
#include <infiniband/verbs.h>

static const unsigned nsec_per_usec = (UCS_NSEC_PER_SEC / UCS_USEC_PER_SEC);

/* wait for 3 usecs to get statistics */
static const unsigned long test_period = (3ul * UCS_USEC_PER_SEC);
static const unsigned moderation_period = 1000; /* usecs */
static const unsigned event_limit = (40 * test_period / moderation_period / nsec_per_usec);
static const unsigned max_repeats = 1000;

class test_uct_cq_moderation : public uct_test {
protected:

    void init() {
        uct_test::init();

        if (RUNNING_ON_VALGRIND) {
            UCS_TEST_SKIP_R("skipping on valgrind");
        }

        set_config("IB_TX_CQ_MODERATION=1");
        if ((GetParam()->tl_name == "rc") || (GetParam()->tl_name == "rc_mlx5") ||
            (GetParam()->tl_name == "dc") || (GetParam()->tl_name == "dc_mlx5")) {
            set_config("RC_FC_ENABLE=n");
        }

        set_config(std::string("IB_TX_EVENT_MOD_PERIOD=") + ucs::to_string(moderation_period) + "us");
        set_config(std::string("IB_RX_EVENT_MOD_PERIOD=") + ucs::to_string(moderation_period) + "us");

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);
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

    void iface_arm(uct_iface_h iface) {
        struct pollfd pfd;
        int fd;

        /* wait for all messages are arrived */
        while (m_recv < m_send) {
            progress();
        }

        uct_iface_event_fd_get(iface, &fd);

        pfd.fd = fd;
        pfd.events = POLLIN;

        do {
            /* arm all event types */
            while (1) {
                if (uct_iface_event_arm(iface,
                                        UCT_EVENT_SEND_COMP |
                                        UCT_EVENT_RECV      |
                                        UCT_EVENT_RECV_SIG) != UCS_ERR_BUSY) {
                    break;
                }
                progress();
            }
            /* repeat till FD is in active state */
        } while (poll(&pfd, 1, 0) > 0);
    }

    static ucs_status_t am_cb(void *arg, void *data, size_t len, unsigned flags) {
        ucs_assert_always(arg != NULL);
        test_uct_cq_moderation *self = static_cast<test_uct_cq_moderation*>(arg);

        self->m_recv++;

        return UCS_OK;
    }

    void run_test(uct_iface_h iface);

    entity * m_sender;
    entity * m_receiver;

    unsigned m_send;
    unsigned m_recv;
};

void test_uct_cq_moderation::run_test(uct_iface_h iface) {
    unsigned events;
    int fd;
    unsigned i;
    int polled;
    struct pollfd pfd;
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_EVENT_SEND_COMP);
    check_caps(UCT_IFACE_FLAG_EVENT_RECV);

    uct_iface_set_am_handler(m_receiver->iface(), 0, am_cb, this, UCT_CB_FLAG_SYNC);

    connect();

    m_send = 0;
    m_recv = 0;

    uct_iface_event_fd_get(iface, &fd);
    pfd.fd = fd;
    pfd.events = POLLIN;

    /* repeat test till at least one iteration is successful
     * to exclude random fluctuations */
    for (i = 0; i < max_repeats; i++) {
        events = 0;
        iface_arm(iface);

        ucs_time_t tm = ucs_get_time();

        while ((ucs_get_time() - tm) < test_period) {
            polled = poll(&pfd, 1, 0);
            if (polled > 0) {
                events++;
                iface_arm(iface);
            }

            do {
                status = uct_ep_am_short(m_sender->ep(0), 0, 0, NULL, 0);
                progress();
            } while (status == UCS_ERR_NO_RESOURCE);
            m_send++;
            ASSERT_UCS_OK(status);
        }
        m_sender->flush();
        if (events <= event_limit) {
            break;
        }
    }

    disconnect();

    EXPECT_LE(events, event_limit);
}

UCS_TEST_P(test_uct_cq_moderation, send_period) {
    run_test(m_sender->iface());
}

UCS_TEST_P(test_uct_cq_moderation, recv_period) {
    run_test(m_receiver->iface());
}

#if HAVE_DECL_IBV_EXP_CQ_MODERATION
UCT_INSTANTIATE_IB_TEST_CASE(test_uct_cq_moderation)
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */
