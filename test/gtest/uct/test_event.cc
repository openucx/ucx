/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <poll.h>
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_uct_event_fd : public uct_test {
public:
    void initialize() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        m_am_count = 0;
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {
        recv_desc_t *my_desc = (recv_desc_t *) arg;

        my_desc->length = length;
        memcpy(my_desc + 1, data, length);

        ++m_am_count;
        return UCS_OK;
    }

    static size_t copy64(void *dest, void *arg)
    {
        *(uint64_t*)dest = *(uint64_t*)arg;
        return sizeof(uint64_t);
    }

    void cleanup() {
        uct_test::cleanup();
    }

protected:
    entity *m_e1, *m_e2;
    static int m_am_count;
};

int test_uct_event_fd::m_am_count = 0;

UCS_TEST_P(test_uct_event_fd, am)
{
    uint64_t send_data   = 0xdeadbeef;
    recv_desc_t *recv_buffer;
    struct pollfd wakeup_fd;
    ucs_status_t status;
    int am_send_count = 0;

    initialize();
    check_caps(UCT_IFACE_FLAG_EVENT_RECV_AM | UCT_IFACE_FLAG_AM_CB_SYNC |
               UCT_IFACE_FLAG_AM_BCOPY);

    recv_buffer = (recv_desc_t *) malloc(sizeof(*recv_buffer) + sizeof(send_data));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler, recv_buffer,
                             UCT_AM_CB_FLAG_SYNC);

    /* create receiver wakeup */
    status = uct_iface_event_fd_get(m_e2->iface(), &wakeup_fd.fd);
    ASSERT_EQ(UCS_OK, status);

    wakeup_fd.events = POLLIN;
    EXPECT_EQ(0, poll(&wakeup_fd, 1, 0));

    status = uct_iface_event_arm(m_e2->iface(), UCT_EVENT_RECV_AM);
    ASSERT_EQ(UCS_OK, status);

    EXPECT_EQ(0, poll(&wakeup_fd, 1, 0));

    /* send the data */
    uct_ep_am_bcopy(m_e1->ep(0), 0, copy64, &send_data, 0);
    ++am_send_count;

    /* make sure the file descriptor IS signaled ONCE */
    ASSERT_EQ(1, poll(&wakeup_fd, 1, 1000*ucs::test_time_multiplier()));

    for (;;) {
        status = uct_iface_event_arm(m_e2->iface(), UCT_EVENT_RECV_AM);
        if (status != UCS_ERR_BUSY) {
            break;
        }
        progress();
    }
    ASSERT_EQ(UCS_OK, status);


    status = uct_iface_event_arm(m_e2->iface(), UCT_EVENT_RECV_AM);
    ASSERT_EQ(UCS_OK, status);

    /* send the data again */
    uct_ep_am_bcopy(m_e1->ep(0), 0, copy64, &send_data, 0);
    ++am_send_count;

    /* make sure the file descriptor IS signaled */
    ASSERT_EQ(1, poll(&wakeup_fd, 1, 1000*ucs::test_time_multiplier()));

    while (m_am_count < am_send_count) {
        progress();
    }

    m_e1->flush();

    free(recv_buffer);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_event_fd);
