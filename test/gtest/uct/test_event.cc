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

        m_am_count = 0;
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

        ++m_am_count;
        return UCS_OK;
    }

    void send_am_data(unsigned send_flags, int &am_send_count) {
        ssize_t res;

        m_send_data = 0xdeadbeef;
        do {
            res = uct_ep_am_bcopy(m_e1->ep(0), 0, pack_u64,
                                  &m_send_data, send_flags);
            m_e1->progress();
        } while (res == UCS_ERR_NO_RESOURCE);
        ASSERT_EQ((ssize_t)sizeof(m_send_data), res);

        ++am_send_count;
    }

    void test_recv_am(unsigned arm_flags, unsigned send_flags);

    static size_t pack_u64(void *dest, void *arg)
    {
        *reinterpret_cast<uint64_t*>(dest) = *reinterpret_cast<uint64_t*>(arg);
        return sizeof(uint64_t);
    }

    void arm(entity *e, unsigned arm_flags) {
        ucs_status_t status;
        for (int i = 0; i < 10; ++i) {
            /* have several retries for arming, in case a transport has spurious
             * events */
            status = uct_iface_event_arm(e->iface(), arm_flags);
            if (status == UCS_OK) {
                break;
            }
        }
        ASSERT_EQ(UCS_OK, status);
    }

protected:
    entity *m_e1, *m_e2;
    static int m_am_count;
    uct_test::async_event_ctx m_async_event_ctx;
    uint64_t m_send_data;
};

int test_uct_event::m_am_count = 0;

void test_uct_event::test_recv_am(unsigned arm_flags, unsigned send_flags)
{
    int am_send_count = 0;
    recv_desc_t *recv_buffer;
    ucs_status_t status;

    recv_buffer = (recv_desc_t *)malloc(sizeof(*recv_buffer) +
                                        sizeof(m_send_data));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler, recv_buffer, 0);
    EXPECT_FALSE(m_async_event_ctx.wait_for_event(*m_e2, 0));

    arm(m_e2, arm_flags);
    EXPECT_FALSE(m_async_event_ctx.wait_for_event(*m_e2, 0));

    /* send the data */
    send_am_data(send_flags, am_send_count);
    EXPECT_TRUE(m_async_event_ctx.wait_for_event(*m_e2,
                                                 1000 *
                                                 ucs::test_time_multiplier()));

    for (;;) {
        if ((progress() == 0) && (m_am_count == am_send_count)) {
            status = uct_iface_event_arm(m_e2->iface(), arm_flags);
            if (status != UCS_ERR_BUSY) {
                break;
            }
        }
    }
    ASSERT_EQ(UCS_OK, status);

    arm(m_e2, arm_flags);

    /* send the data again */
    send_am_data(send_flags, am_send_count);
    EXPECT_TRUE(m_async_event_ctx.wait_for_event(*m_e2,
                                                 1000 *
                                                 ucs::test_time_multiplier()));

    while (m_am_count < am_send_count) {
        progress();
    }

    m_e1->flush();

    free(recv_buffer);
}

UCS_TEST_SKIP_COND_P(test_uct_event, am,
                     !check_caps(UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_AM_BCOPY) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_RECV))
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
