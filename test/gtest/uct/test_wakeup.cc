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
#include <uct/ib/base/ib_iface.h>
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

        test_uct_wakeup::am_handler_count = 0;
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
        ++test_uct_wakeup::am_handler_count;

        return UCS_OK;
    }

    void cleanup() {
        uct_test::cleanup();
    }

protected:
    entity *m_e1, *m_e2;
    static size_t am_handler_count;
};

size_t test_uct_wakeup::am_handler_count = 0;

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
    EXPECT_EQ(uct_wakeup_wait(wakeup_handle), UCS_OK);
    uct_wakeup_close(wakeup_handle);
    free(recv_buffer);
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_wakeup);


class test_uct_wakeup_ib : public test_uct_wakeup {
public:
    test_uct_wakeup_ib() {
        length            = 8;
        wakeup_handle     = NULL;
        wakeup_fd.revents = 0;
        wakeup_fd.events  = POLLIN;
        wakeup_fd.fd      = 0;
        test_ib_hdr       = 0xbeef;
        m_buf1            = NULL;
        m_buf2            = NULL;
    }
    void initialize() {
        ucs_status_t status;

        test_uct_wakeup::initialize();

        check_caps(UCT_IFACE_FLAG_PUT_SHORT | UCT_IFACE_FLAG_WAKEUP);

        /* create receiver wakeup */
        status = uct_wakeup_open(m_e1->iface(),
                                 UCT_WAKEUP_RX_SIGNALED_AM | UCT_WAKEUP_TX_COMPLETION,
                                 &wakeup_handle);
        ASSERT_EQ(status, UCS_OK);

        status = uct_wakeup_efd_get(wakeup_handle, &wakeup_fd.fd);
        ASSERT_EQ(status, UCS_OK);

        EXPECT_EQ(poll(&wakeup_fd, 1, 0), 0);

        m_buf1 = new mapped_buffer(length, 0x1, *m_e1);
        m_buf2 = new mapped_buffer(length, 0x2, *m_e2);

        /* set a callback for the uct to invoke for receiving the data */
        uct_iface_set_am_handler(m_e1->iface(), 0, ib_am_handler, m_buf1->ptr(),
                                 UCT_AM_CB_FLAG_SYNC);

        test_uct_wakeup_ib::bcopy_pack_count = 0;
    }

    static size_t pack_cb(void *dest, void *arg) {
        const mapped_buffer *buf = (const mapped_buffer *)arg;
        memcpy(dest, buf->ptr(), buf->length());
        ++test_uct_wakeup_ib::bcopy_pack_count;
        return buf->length();
    }

    /* Use put_bcopy here to provide send_cq entry */
    void send_msg_e1_e2(size_t count = 1) {
        for (size_t i = 0; i < count; ++i) {
            ssize_t status = uct_ep_put_bcopy(m_e1->ep(0), pack_cb, (void *)m_buf1,
                                              m_buf2->addr(), m_buf2->rkey());
            if (status < 0) {
                ASSERT_UCS_OK((ucs_status_t)status);
            }
        }
    }

    void send_msg_e2_e1(size_t count = 1) {
        for (size_t i = 0; i < count; ++i) {
            ucs_status_t status = uct_ep_am_short(m_e2->ep(0), 0, test_ib_hdr,
                                                  m_buf2->ptr(), m_buf2->length());
            ASSERT_UCS_OK(status);
        }
    }

    void check_send_cq(uct_iface_t *iface, size_t val) {
        uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
        struct ibv_cq  *send_cq = ib_iface->send_cq;

        if (val != send_cq->comp_events_completed) {
            uint32_t completed_evt = send_cq->comp_events_completed;
            /* need this call to acknowledge the completion to prevent iface dtor hung*/
            ibv_ack_cq_events(ib_iface->send_cq, 1);
            UCS_TEST_ABORT("send_cq->comp_events_completed have to be 1 but the value "
                           << completed_evt);
        }
    }

    void check_recv_cq(uct_iface_t *iface, size_t val) {
        uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
        struct ibv_cq  *recv_cq = ib_iface->recv_cq;

        if (val != recv_cq->comp_events_completed) {
            uint32_t completed_evt = recv_cq->comp_events_completed;
            /* need this call to acknowledge the completion to prevent iface dtor hung*/
            ibv_ack_cq_events(ib_iface->recv_cq, 1);
            UCS_TEST_ABORT("recv_cq->comp_events_completed have to be 1 but the value "
                           << completed_evt);
        }
    }

    void cleanup() {
        delete(m_buf1);
        delete(m_buf2);
        if (wakeup_handle) {
            uct_wakeup_close(wakeup_handle);
        }
        test_uct_wakeup::cleanup();
    }

protected:
    uct_wakeup_h wakeup_handle;
    struct pollfd wakeup_fd;
    size_t length;
    uint64_t test_ib_hdr;
    mapped_buffer *m_buf1, *m_buf2;
    static size_t bcopy_pack_count;
};

size_t test_uct_wakeup_ib::bcopy_pack_count = 0;


UCS_TEST_P(test_uct_wakeup_ib, tx_cq)
{
    ucs_status_t status;

    initialize();

    status = uct_wakeup_efd_arm(wakeup_handle);
    ASSERT_EQ(status, UCS_OK);

    /* check initial state of the fd and [send|recv]_cq */
    EXPECT_EQ(poll(&wakeup_fd, 1, 0), 0);
    check_send_cq(m_e1->iface(), 0);
    check_recv_cq(m_e1->iface(), 0);

    /* send the data */
    send_msg_e1_e2();

    /* make sure the file descriptor is signaled once */
    ASSERT_EQ(poll(&wakeup_fd, 1, 1), 1);

    status = uct_wakeup_efd_arm(wakeup_handle);
    ASSERT_EQ(status, UCS_OK);

    /* make sure [send|recv]_cq handled properly */
    check_send_cq(m_e1->iface(), 1);
    check_recv_cq(m_e1->iface(), 0);
}


UCS_TEST_P(test_uct_wakeup_ib, txrx_cq)
{
    const size_t msg_count = 1;
    ucs_status_t status;

    initialize();

    status = uct_wakeup_efd_arm(wakeup_handle);
    ASSERT_EQ(status, UCS_OK);

    /* check initial state of the fd and [send|recv]_cq */
    EXPECT_EQ(poll(&wakeup_fd, 1, 0), 0);
    check_send_cq(m_e1->iface(), 0);
    check_recv_cq(m_e1->iface(), 0);

    /* send the data */
    send_msg_e1_e2(msg_count);
    send_msg_e2_e1(msg_count);

    twait(150); /* Let completion to be generated */

    /* Make sure all messages delivered */
    while ((test_uct_wakeup::am_handler_count    < msg_count) ||
           (test_uct_wakeup_ib::bcopy_pack_count < msg_count)) {
        progress();
    }

    /* make sure the file descriptor is signaled */
    ASSERT_EQ(poll(&wakeup_fd, 1, 1), 1);

    status = uct_wakeup_wait(wakeup_handle);
    ASSERT_EQ(status, UCS_OK);

    /* make sure [send|recv]_cq handled properly */
    check_send_cq(m_e1->iface(), 1);
    check_recv_cq(m_e1->iface(), 1);

}


UCT_INSTANTIATE_IB_TEST_CASE(test_uct_wakeup_ib);
