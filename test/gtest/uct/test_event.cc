/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <ucs/time/time.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_uct_event : public uct_test {
public:
    void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0, NULL, NULL, NULL, NULL, NULL,
                                       async_event_handler, this);
        m_entities.push_back(m_e2);

        check_skip_test();

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        /* give a chance to finish connection for some transports (ib/ud, tcp) */
        flush();

        m_am_send_count = 0;
        m_am_recv_count = 0;
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static void async_event_handler(void *arg, unsigned flags) {
        test_uct_event *self = static_cast<test_uct_event*>(arg);
        self->m_async_event_ctx.signal();
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {
        recv_desc_t *my_desc  = (recv_desc_t *) arg;
        uint64_t *test_ib_hdr = (uint64_t *) data;
        uint64_t *actual_data = (uint64_t *) test_ib_hdr + 1;
        unsigned data_length  = length - sizeof(test_ib_hdr);

        my_desc->length = data_length;
        if (*test_ib_hdr == 0xbeef) {
            memcpy(my_desc + 1, actual_data , data_length);
        }

        ++m_am_recv_count;
        return UCS_OK;
    }

    void send_am_data(unsigned send_flags, bool progress_rx) {
        ssize_t res;

        m_send_data = 0xdeadbeef;
        do {
            res = uct_ep_am_bcopy(m_e1->ep(0), 0, pack_u64,
                                  &m_send_data, send_flags);
            m_e1->progress();
            if (progress_rx) {
                m_e2->progress();
            }
        } while (res == UCS_ERR_NO_RESOURCE);
        ASSERT_EQ((ssize_t)sizeof(m_send_data), res);
        ++m_am_send_count;
    }

    void test_recv_am(unsigned arm_flags, unsigned send_flags);

    static size_t pack_u64(void *dest, void *arg)
    {
        *reinterpret_cast<uint64_t*>(dest) = *reinterpret_cast<uint64_t*>(arg);
        return sizeof(uint64_t);
    }

    void arm(entity *e, unsigned arm_flags)
    {
        ucs_status_t status;
        unsigned progress_count;
        do {
            progress_count = e->progress();
            status         = uct_iface_event_arm(e->iface(), arm_flags);
        } while ((status == UCS_ERR_BUSY) || (progress_count != 0));
        ASSERT_EQ(UCS_OK, status);
    }

    double measure_am_loop(unsigned count, unsigned send_flags)
    {
        ucs_time_t start_time = ucs_get_time();
        for (unsigned i = 0; i < count; ++i) {
            send_am_data(send_flags, true);
            while (m_am_recv_count < m_am_send_count) {
                progress();
            }
        }
        return ucs_time_to_sec(ucs_get_time() - start_time);
    }

protected:
    static unsigned m_am_recv_count;

    entity *m_e1, *m_e2;
    unsigned m_am_send_count;
    uct_test::async_event_ctx m_async_event_ctx;
    uint64_t m_send_data;
};

unsigned test_uct_event::m_am_recv_count = 0;

void test_uct_event::test_recv_am(unsigned arm_flags, unsigned send_flags)
{
    static const unsigned count = 100000 / ucs::test_time_multiplier();
    recv_desc_t *recv_buffer;
    unsigned spurious_count = 0;

    recv_buffer = (recv_desc_t *)malloc(sizeof(*recv_buffer) +
                                        sizeof(m_send_data));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler, recv_buffer, 0);

    EXPECT_FALSE(m_async_event_ctx.wait_for_event(*m_e2, 0));

    double time1 = measure_am_loop(count, send_flags);

    for (int retry = 0; ; ++retry) {
        for (unsigned i = 0; i < count; ++i) {
            arm(m_e2, arm_flags);
            if (m_async_event_ctx.wait_for_event(*m_e2, 0)) {
                ++spurious_count;
            }

            /* send the data */
            send_am_data(send_flags, false);

            /* wait for the event */
            EXPECT_TRUE(m_async_event_ctx.wait_for_event(*m_e2, 60));

            /* recv the data */
            unsigned progress_count = 0;
            while (m_am_recv_count < m_am_send_count) {
                progress_count += progress();
            }
            EXPECT_GE(progress_count, 0);
        }

        /* Expect no events arrive unless when we asked for it */
        EXPECT_LT(spurious_count, ucs_max(100, count / 100));

        double time2 = measure_am_loop(count, send_flags);
        double ratio = time2 / time1;
        UCS_TEST_MESSAGE << "Send time: " << time1 << " after arm: " << time2
                         << " ratio: " << ratio
                         << " (" << retry << "/" << ucs::perf_retry_count << ")";

        if ((ucs::test_time_multiplier() > 1) ||
            (ucs::perf_retry_count == 0)) {
            UCS_TEST_MESSAGE << "(Not validating performance)";
            break;
        } else if (ratio < 1.4) {
            break; /* Success */
        } else if (retry >= ucs::perf_retry_count) {
            ADD_FAILURE() << "Sending after event is armed is too slow";
            break;
        }

        ucs::safe_sleep(ucs::perf_retry_interval);
    }

    m_e1->flush();

    free(recv_buffer);
}

UCS_TEST_SKIP_COND_P(test_uct_event, am_no_fc,
                     !check_caps(UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_AM_BCOPY) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV),
                     "RC_FC_ENABLE?=n")
{
    test_recv_am(UCT_EVENT_RECV, 0);
}

UCS_TEST_SKIP_COND_P(test_uct_event, am_with_fc,
                     !check_caps(UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_AM_BCOPY) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV),
                     "RC_FC_ENABLE~=y")
{
    test_recv_am(UCT_EVENT_RECV, 0);
}

UCS_TEST_SKIP_COND_P(test_uct_event, sig_am,
                     !check_caps(UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_AM_BCOPY) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV_SIG))
{
    test_recv_am(UCT_EVENT_RECV_SIG, UCT_SEND_FLAG_SIGNALED);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_event);
