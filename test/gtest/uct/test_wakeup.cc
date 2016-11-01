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

class test_uct_wakeup : public uct_test {
public:
    void initialize() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static ucs_status_t ib_am_handler(void *arg, void *data, size_t length, void *desc) {
        recv_desc_t *my_desc  = (recv_desc_t *) arg;
        uint64_t *test_ib_hdr = (uint64_t *) data;
        uint64_t *actual_data = (uint64_t *) test_ib_hdr + 1;
        unsigned data_length  = length - sizeof(test_ib_hdr);

        my_desc->length = data_length;
        if (*test_ib_hdr == 0xbeef) {
            memcpy(my_desc + 1, actual_data , data_length);
        }

        return UCS_OK;
    }

    void cleanup() {
        uct_test::cleanup();
    }

protected:
    entity *m_e1, *m_e2;
};

UCS_TEST_P(test_uct_wakeup, am)
{
    uint64_t send_data   = 0xdeadbeef;
    uint64_t test_ib_hdr = 0xbeef;
    recv_desc_t *recv_buffer;
    uct_wakeup_h wakeup_handle;
    struct pollfd wakeup_fd;

    initialize();
    check_caps(UCT_IFACE_FLAG_WAKEUP);

    recv_buffer = (recv_desc_t *) malloc(sizeof(*recv_buffer) + sizeof(send_data));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, ib_am_handler, recv_buffer,
                             UCT_AM_CB_FLAG_SYNC);

    /* create receiver wakeup */
    ASSERT_EQ(uct_wakeup_open(m_e2->iface(), UCT_WAKEUP_RX_SIGNALED_AM,
              &wakeup_handle), UCS_OK);
    ASSERT_EQ(uct_wakeup_efd_get(wakeup_handle, &wakeup_fd.fd), UCS_OK);
    wakeup_fd.events = POLLIN;
    EXPECT_EQ(poll(&wakeup_fd, 1, 0), 0);
    ASSERT_EQ(uct_wakeup_efd_arm(wakeup_handle), UCS_OK);
    EXPECT_EQ(poll(&wakeup_fd, 1, 0), 0);

    /* send the data */
    uct_ep_am_short(m_e1->ep(0), 0, test_ib_hdr, &send_data, sizeof(send_data));

    /* make sure the file descriptor IS signaled ONCE */
    ASSERT_EQ(poll(&wakeup_fd, 1, 1), 1);
    ASSERT_EQ(uct_wakeup_efd_arm(wakeup_handle), UCS_OK);
    wakeup_fd.revents = 0;
    EXPECT_EQ(poll(&wakeup_fd, 1, 0), 0);

    /* send the data again */
    uct_ep_am_short(m_e1->ep(0), 0, test_ib_hdr, &send_data, sizeof(send_data));

    /* make sure the file descriptor IS signaled */
    ASSERT_EQ(poll(&wakeup_fd, 1, 1), 1);

    uct_wakeup_close(wakeup_handle);
    free(recv_buffer);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_wakeup);
