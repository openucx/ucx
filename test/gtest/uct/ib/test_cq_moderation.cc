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

#define TEST_PERIOD (3ul * UCS_NSEC_PER_SEC) /* wait for 3 secs to get statistics */
#define NSEC_PER_USEC (UCS_NSEC_PER_SEC / UCS_USEC_PER_SEC)

#define MODERATION_PERIOD 1000 /* usecs */
#define EVENT_LIMIT (TEST_PERIOD / MODERATION_PERIOD / NSEC_PER_USEC)

#define _STR(_arg) #_arg
#define TO_STR(_val) _STR(_val)

class test_uct_cq_moderation : public uct_test {
protected:

    void init() {
        uct_test::init();

        if (!(GetParam()->tl_name == "rc" || GetParam()->tl_name == "rc_mlx5" ||
              GetParam()->tl_name == "ud" || GetParam()->tl_name == "ud_mlx5" ||
              GetParam()->tl_name == "dc" || GetParam()->tl_name == "dc_mlx5")) {
            /* dc_mlx5 is masked due to unknown issue - CQ moderation doesn't work on DC */
            UCS_TEST_SKIP_R("unsupported");
        }

        set_config("IB_TX_CQ_MODERATION=1");
        if (!(GetParam()->tl_name == "ud" || GetParam()->tl_name == "ud_mlx5")) {
            set_config("RC_FC_ENABLE=n");
        }
#if HAVE_DECL_IBV_EXP_CQ_MODERATION
        set_config("IB_TX_EVENT_MOD_PERIOD=" TO_STR(MODERATION_PERIOD) "us");
        set_config("IB_RX_EVENT_MOD_PERIOD=" TO_STR(MODERATION_PERIOD) "us");
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */

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

    void iface_arm(uct_iface_h iface, unsigned events) {
        while (1) {
            if (uct_iface_event_arm(iface, events) != UCS_ERR_BUSY) {
                break;
            }
            progress();
        }
    }

    void test_period(uct_iface_h iface, unsigned event);

    entity * m_sender;
    entity * m_receiver;
};

static ucs_status_t am_cb(void *arg, void *data, size_t len, unsigned flags) {
    unsigned *count = (unsigned *)arg;
    (*count)++;
    return UCS_OK;
}

void test_uct_cq_moderation::test_period(uct_iface_h iface, unsigned event) {
    unsigned sent = 0;
    unsigned recv = 0;
    unsigned events = 0;
    int fd;

    check_caps(UCT_IFACE_FLAG_EVENT_SEND_COMP);
    check_caps(UCT_IFACE_FLAG_EVENT_RECV_AM);
#if !HAVE_DECL_IBV_EXP_CQ_MODERATION
    UCS_TEST_SKIP_R("unsupported");
#endif /* HAVE_DECL_IBV_EXP_CQ_MODERATION */

    uct_iface_set_am_handler(m_receiver->iface(), 0, am_cb, &recv, UCT_CB_FLAG_SYNC);

    connect();

    uct_iface_event_fd_get(iface, &fd);
    iface_arm(iface, event);

    ucs_time_t tm = ucs_get_time();

    while ((ucs_get_time() - tm) < TEST_PERIOD) {
        struct pollfd pfd = {.fd = fd, .events = POLLIN};

        int polled = poll(&pfd, 1, 0);
        if (polled > 0) {
            events++;
            iface_arm(iface, event);
        }

        uct_ep_am_short(m_sender->ep(0), 0, 0, NULL, 0);
        progress();
        sent++;
    }
    m_sender->flush();

    disconnect();

    EXPECT_LE(events, EVENT_LIMIT);
}

UCS_TEST_P(test_uct_cq_moderation, send_period) {
    test_period(m_sender->iface(), UCT_EVENT_SEND_COMP);
}

UCS_TEST_P(test_uct_cq_moderation, recv_period) {
    test_period(m_receiver->iface(), UCT_EVENT_RECV_AM);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_cq_moderation)
