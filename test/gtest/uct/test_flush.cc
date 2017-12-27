/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"
extern "C" {
#include <ucs/arch/atomic.h>
}

class uct_flush_test : public uct_test {
public:
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    static const uint64_t SEED3 = 0x3333333333333333lu;
    static const int      AM_ID         = 1;
    static const int      AM_ID_CANCEL  = 2;

    typedef void (uct_flush_test::* flush_func_t)();

    struct test_req_t {
        uct_pending_req_t  uct;
        uct_completion_t   comp;
        mapped_buffer      *sendbuf;
        uct_flush_test     *test;
    };

    void init() {
        if (UCT_DEVICE_TYPE_SELF == GetParam()->dev_type) {
            entity *e = uct_test::create_entity(0);
            m_entities.push_back(e);

            e->connect(0, *e, 0);
        } else {
            entity *m_sender = uct_test::create_entity(0);
            m_entities.push_back(m_sender);

            entity *m_receiver = uct_test::create_entity(0);
            m_entities.push_back(m_receiver);

            m_sender->connect(0, *m_receiver, 0);
        }
        am_rx_count   = 0;
        m_flush_flags = 0;
    }

    static size_t pack_cb(void *dest, void *arg)
    {
        const mapped_buffer *sendbuf = (const mapped_buffer *)arg;
        memcpy(dest, sendbuf->ptr(), sendbuf->length());
        return sendbuf->length();
    }

    void blocking_put_bcopy(const mapped_buffer &sendbuf,
                            const mapped_buffer &recvbuf)
    {
        ssize_t status;
         for (;;) {
             status = uct_ep_put_bcopy(sender().ep(0), pack_cb, (void*)&sendbuf,
                                       recvbuf.addr(), recvbuf.rkey());
             if (status >= 0) {
                 return;
             } else if (status == UCS_ERR_NO_RESOURCE) {
                 progress();
                 continue;
             } else {
                 ASSERT_UCS_OK((ucs_status_t)status);
             }
         }
    }

    void blocking_am_bcopy(const mapped_buffer &sendbuf)
    {
         ssize_t status;
         for (;;) {
             status = uct_ep_am_bcopy(sender().ep(0), get_am_id(), pack_cb,
                                      (void*)&sendbuf, 0);
             if (status >= 0) {
                 return;
             } else if (status == UCS_ERR_NO_RESOURCE) {
                 progress();
                 continue;
             } else {
                 ASSERT_UCS_OK((ucs_status_t)status);
             }
         }
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags)
    {
        if (arg == NULL) {
            /* This is not completely canceled message, drop it */
            return UCS_OK;
        }
        const mapped_buffer *recvbuf = (const mapped_buffer *)arg;
        memcpy(recvbuf->ptr(), data, ucs_min(length, recvbuf->length()));
        ucs_atomic_add32(&am_rx_count, 1);
        return UCS_OK;
    }

    ucs_status_t am_send_pending(test_req_t *am_req)
    {
        ssize_t status;

        status = uct_ep_am_bcopy(sender().ep(0), get_am_id(), pack_cb,
                                 (void*)am_req->sendbuf, 0);
        if (status >= 0) {
            --am_req->comp.count;
            return UCS_OK;
        } else {
            return (ucs_status_t)status;
        }
    }

    static ucs_status_t am_progress(uct_pending_req_t *req)
    {
        test_req_t *am_req = ucs_container_of(req, test_req_t, uct);
        return am_req->test->am_send_pending(am_req);
    }

    static ucs_status_t flush_progress(uct_pending_req_t *req)
    {
        test_req_t *flush_req = ucs_container_of(req, test_req_t, uct);
        ucs_status_t status;

        status = uct_ep_flush(flush_req->test->sender().ep(0), 0,
                              &flush_req->comp);
        if (status == UCS_OK) {
            --flush_req->comp.count;
            return UCS_OK;
        } else if (status == UCS_INPROGRESS) {
            return UCS_OK;
        } else if (status == UCS_ERR_NO_RESOURCE) {
            return UCS_ERR_NO_RESOURCE;
        } else {
            UCS_TEST_ABORT("Error: " << ucs_status_string(status));
        }
    }

    void test_flush_put_bcopy(flush_func_t flush) {
        const size_t length = 8;
        check_caps(UCT_IFACE_FLAG_PUT_BCOPY);
        mapped_buffer sendbuf(length, SEED1, sender());
        mapped_buffer recvbuf(length, SEED2, receiver());
        sendbuf.pattern_fill(SEED3);
        blocking_put_bcopy(sendbuf, recvbuf);
        (this->*flush)();

        if (is_flush_cancel()) {
            return;
        }

        recvbuf.pattern_check(SEED3);
    }

    void wait_am(unsigned count) {
        while (am_rx_count < count) {
            progress();
            sched_yield();
        }
    }

    void test_flush_am_zcopy(flush_func_t flush, bool destroy_ep) {
        const size_t length = 8;
        check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
        mapped_buffer sendbuf(length, SEED1, sender());
        mapped_buffer recvbuf(length, SEED2, receiver());
        sendbuf.pattern_fill(SEED3);

        uct_iface_set_am_handler(receiver().iface(), get_am_id(), am_handler,
                                 is_flush_cancel() ? NULL : &recvbuf,
                                 UCT_CB_FLAG_ASYNC);

        uct_completion_t zcomp;
        zcomp.count = 2;
        zcomp.func  = NULL;

        ucs_status_t status;
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                                sendbuf.memh(),
                                sender().iface_attr().cap.am.max_iov);
        do {
            status = uct_ep_am_zcopy(sender().ep(0), get_am_id(), NULL, 0, iov,
                                     iovcnt, 0, &zcomp);
        } while (status == UCS_ERR_NO_RESOURCE);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
        if (status == UCS_OK) {
            --zcomp.count;
        }

        (this->*flush)();

        EXPECT_EQ(1, zcomp.count); /* Zero copy op should be already completed
                                      since flush returned */

        if (destroy_ep) {
            sender().destroy_ep(0);
        }

        if (is_flush_cancel()) {
            return;
        }

        wait_am(1);

        uct_iface_set_am_handler(receiver().iface(), get_am_id(), NULL, NULL, 0);

        recvbuf.pattern_check(SEED3);
    }

    void test_flush_am_disconnect(flush_func_t flush, bool destroy_ep) {
        const size_t length = 8;
        check_caps(UCT_IFACE_FLAG_AM_BCOPY);
        mapped_buffer sendbuf(length, SEED1, sender());
        mapped_buffer recvbuf(length, SEED2, receiver());
        sendbuf.pattern_fill(SEED3);

        uct_iface_set_am_handler(receiver().iface(), get_am_id(), am_handler,
                                 is_flush_cancel() ? NULL : &recvbuf,
                                 UCT_CB_FLAG_ASYNC);
        blocking_am_bcopy(sendbuf);
        (this->*flush)();

        if (destroy_ep) {
            sender().destroy_ep(0);
        }

        if (is_flush_cancel()) {
            return;
        }

        wait_am(1);
        uct_iface_set_am_handler(receiver().iface(), get_am_id(), NULL, NULL, 0);

        recvbuf.pattern_check(SEED3);
    }

    void flush_ep_no_comp() {
        ucs_status_t status;
        do {
            progress();
            status = uct_ep_flush(sender().ep(0), m_flush_flags, NULL);
        } while ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS));
        ASSERT_UCS_OK(status);
    }

    void flush_iface_no_comp() {
        ucs_status_t status;
        do {
            progress();
            status = uct_iface_flush(sender().iface(), m_flush_flags, NULL);
        } while ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS));
        ASSERT_UCS_OK(status);
    }

    void flush_ep_nb() {
        uct_completion_t comp;
        ucs_status_t status;
        comp.count = 2;
        comp.func  = NULL;
        do {
            progress();
            status = uct_ep_flush(sender().ep(0), m_flush_flags, &comp);
        } while (status == UCS_ERR_NO_RESOURCE);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
        if (status == UCS_OK) {
            return;
        }
        /* coverity[loop_condition] */
        while (comp.count != 1) {
            progress();
        }
    }

    void test_flush_am_pending(flush_func_t flush, bool destroy_ep);

protected:
    uct_test::entity& sender() {
        return **m_entities.begin();
    }

    uct_test::entity& receiver() {
        return **(m_entities.end() - 1);
    }

    bool is_flush_cancel() const {
        return (m_flush_flags & UCT_FLUSH_FLAG_CANCEL);
    }

    uint8_t get_am_id() const {
        return is_flush_cancel() ? AM_ID_CANCEL : AM_ID;
    }

    static uint32_t am_rx_count;
    unsigned        m_flush_flags;
};

uint32_t uct_flush_test::am_rx_count = 0;

void uct_flush_test::test_flush_am_pending(flush_func_t flush, bool destroy_ep)
{
     const size_t length = 8;
     check_caps(UCT_IFACE_FLAG_AM_BCOPY | UCT_IFACE_FLAG_PENDING);
     mapped_buffer sendbuf(length, SEED1, sender());
     mapped_buffer recvbuf(length, SEED2, receiver());
     sendbuf.pattern_fill(SEED3);

     uct_iface_set_am_handler(receiver().iface(), get_am_id(), am_handler,
                              is_flush_cancel() ? NULL : &recvbuf,
                              UCT_CB_FLAG_ASYNC);

     /* Send until resources are exhausted or timeout in 1sec*/
     unsigned count = 0;
     ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(1.0);
     ssize_t packed_len;
     for (;;) {
         packed_len = uct_ep_am_bcopy(sender().ep(0), get_am_id(), pack_cb,
                                      (void*)&sendbuf, 0);
         if (packed_len == UCS_ERR_NO_RESOURCE) {
             break;
         }
         if (ucs_get_time() > loop_end_limit) {
             ++count;
             break;
         }

         if (packed_len >= 0) {
             ++count;
         } else {
             ASSERT_UCS_OK((ucs_status_t)packed_len);
         }
     }

     /* Queue some pending AMs */
     ucs_status_t status;
     std::vector<test_req_t> reqs;
     reqs.resize(10);
     for (std::vector<test_req_t>::iterator it = reqs.begin(); it != reqs.end();) {
         it->sendbuf    = &sendbuf;
         it->test       = this;
         it->uct.func   = am_progress;
         it->comp.count = 2;
         it->comp.func  = NULL;
         status = uct_ep_pending_add(sender().ep(0), &it->uct);
         if (UCS_ERR_BUSY == status) {
             /* User advised to retry the send. It means no requests added
              * to the queue
              */
             it = reqs.erase(it);
             status = UCS_OK;
         } else {
             ++count;
             ++it;
         }
         ASSERT_UCS_OK(status);
     }

     /* Try to start a flush */
     test_req_t flush_req;
     flush_req.comp.count = 2;
     flush_req.comp.func  = NULL;

     for (;;) {
         status = uct_ep_flush(sender().ep(0), m_flush_flags, &flush_req.comp);
         if (status == UCS_OK) {
             --flush_req.comp.count;
         } else if (status == UCS_ERR_NO_RESOURCE) {
             if (is_flush_cancel()) {
                 continue;
             }
             /* If flush returned NO_RESOURCE, add to pending must succeed */
             flush_req.test      = this;
             flush_req.uct.func  = flush_progress;
             status = uct_ep_pending_add(sender().ep(0), &flush_req.uct);
             if (status == UCS_ERR_BUSY) {
                 continue;
             }
             EXPECT_EQ(UCS_OK, status);
         } else if (status == UCS_INPROGRESS) {
         } else {
             UCS_TEST_ABORT("failed to flush ep: " << ucs_status_string(status));
         }
         break;
     }

     /* timeout used to prevent test hung */
     wait_for_value(&flush_req.comp.count, 1, true, 60.0);
     EXPECT_EQ(1, flush_req.comp.count);

     while (!reqs.empty()) {
         if (is_flush_cancel()) {
            EXPECT_EQ(2, reqs.back().comp.count);
         } else {
            EXPECT_EQ(1, reqs.back().comp.count);
         }
         reqs.pop_back();
     }

     if (!is_flush_cancel()) {
        wait_am(count);
     }

     if (destroy_ep) {
        sender().destroy_ep(0);
     }

     if (is_flush_cancel()) {
         return;
     }

     uct_iface_set_am_handler(receiver().iface(), get_am_id(), NULL, NULL, 0);

     recvbuf.pattern_check(SEED3);
}

UCS_TEST_P(uct_flush_test, put_bcopy_flush_ep_no_comp) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;
    test_flush_put_bcopy(&uct_flush_test::flush_ep_no_comp);

    if (!is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        return;
    }

    am_rx_count   = 0;
    m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
    test_flush_put_bcopy(&uct_flush_test::flush_ep_no_comp);

    am_rx_count   = 0;
    m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    test_flush_put_bcopy(&uct_flush_test::flush_ep_no_comp);
}

UCS_TEST_P(uct_flush_test, put_bcopy_flush_iface_no_comp) {
    test_flush_put_bcopy(&uct_flush_test::flush_iface_no_comp);
}

UCS_TEST_P(uct_flush_test, put_bcopy_flush_ep_nb) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;
    test_flush_put_bcopy(&uct_flush_test::flush_ep_nb);

    if (!is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        return;
    }

    am_rx_count   = 0;
    m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
    test_flush_put_bcopy(&uct_flush_test::flush_ep_nb);

    am_rx_count   = 0;
    m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    test_flush_put_bcopy(&uct_flush_test::flush_ep_nb);
}

UCS_TEST_P(uct_flush_test, am_zcopy_flush_ep_no_comp) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {

        test_flush_am_zcopy(&uct_flush_test::flush_ep_no_comp, false);

        am_rx_count   = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_zcopy(&uct_flush_test::flush_ep_no_comp, false);

        am_rx_count   = 0;
        m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    }

    test_flush_am_zcopy(&uct_flush_test::flush_ep_no_comp, true);
}

UCS_TEST_P(uct_flush_test, am_zcopy_flush_iface_no_comp) {
    test_flush_am_zcopy(&uct_flush_test::flush_iface_no_comp, true);
}

UCS_TEST_P(uct_flush_test, am_zcopy_flush_ep_nb) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        test_flush_am_zcopy(&uct_flush_test::flush_ep_nb, false);

        am_rx_count   = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_zcopy(&uct_flush_test::flush_ep_nb, false);

        am_rx_count   = 0;
        m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    }

    test_flush_am_zcopy(&uct_flush_test::flush_ep_nb, true);
}

UCS_TEST_P(uct_flush_test, am_flush_ep_no_comp) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        test_flush_am_disconnect(&uct_flush_test::flush_ep_no_comp, false);

        am_rx_count   = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_disconnect(&uct_flush_test::flush_ep_no_comp, false);

        am_rx_count   = 0;
        m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    }

    test_flush_am_disconnect(&uct_flush_test::flush_ep_no_comp, true);
}

UCS_TEST_P(uct_flush_test, am_flush_iface_no_comp) {
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;
    test_flush_am_disconnect(&uct_flush_test::flush_iface_no_comp, true);
}

UCS_TEST_P(uct_flush_test, am_flush_ep_nb) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;
    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        test_flush_am_disconnect(&uct_flush_test::flush_ep_nb, false);

        am_rx_count   = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_disconnect(&uct_flush_test::flush_ep_nb, false);

        am_rx_count   = 0;
        m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    }

    test_flush_am_disconnect(&uct_flush_test::flush_ep_nb, true);
}

UCS_TEST_P(uct_flush_test, am_pending_flush_nb) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        test_flush_am_pending(&uct_flush_test::flush_ep_nb, false);

        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_pending(&uct_flush_test::flush_ep_nb, false);

        am_rx_count    = 0;
        m_flush_flags &= ~UCT_FLUSH_FLAG_CANCEL;
    }

    test_flush_am_pending(&uct_flush_test::flush_ep_nb, false);
}

UCT_INSTANTIATE_TEST_CASE(uct_flush_test)
