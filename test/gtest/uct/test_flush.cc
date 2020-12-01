/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"
extern "C" {
#include <ucs/arch/atomic.h>
}
#include <list>

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
        uct_test::init();

        entity *m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        check_skip_test();

        if (UCT_DEVICE_TYPE_SELF == GetParam()->dev_type) {
            m_sender->connect(0, *m_sender, 0);
        } else {
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

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        test_req_t *req = ucs_container_of(self, test_req_t, uct);
        --req->comp.count;
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
        if (is_flush_cancel()) {
            ASSERT_TRUE(destroy_ep);
        }
        mapped_buffer sendbuf(length, SEED1, sender());
        mapped_buffer recvbuf(length, SEED2, receiver());
        sendbuf.pattern_fill(SEED3);

        uct_iface_set_am_handler(receiver().iface(), get_am_id(), am_handler,
                                 is_flush_cancel() ? NULL : &recvbuf,
                                 UCT_CB_FLAG_ASYNC);

        uct_completion_t zcomp;
        zcomp.count  = 2;
        zcomp.status = UCS_OK;
        zcomp.func   = NULL;

        ucs_status_t status;
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                                sendbuf.memh(),
                                sender().iface_attr().cap.am.max_iov);
        do {
            status = uct_ep_am_zcopy(sender().ep(0), get_am_id(), NULL, 0, iov,
                                     iovcnt, 0, &zcomp);
            progress();
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
        if (is_flush_cancel()) {
            ASSERT_TRUE(destroy_ep);
        }
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
        comp.count  = 2;
        comp.status = UCS_OK;
        comp.func   = NULL;
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
     if (is_flush_cancel()) {
         ASSERT_TRUE(destroy_ep);
     }
     const size_t length = 8;
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
         it->sendbuf     = &sendbuf;
         it->test        = this;
         it->uct.func    = am_progress;
         it->comp.count  = 2;
         it->comp.func   = NULL;
         it->comp.status = UCS_OK;
         status = uct_ep_pending_add(sender().ep(0), &it->uct, 0);
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

     if (is_flush_cancel()) {
         uct_ep_pending_purge(sender().ep(0), purge_cb, NULL);
     }

     /* Try to start a flush */
     test_req_t flush_req;
     flush_req.comp.count  = 2;
     flush_req.comp.status = UCS_OK;
     flush_req.comp.func   = NULL;

     for (;;) {
         status = uct_ep_flush(sender().ep(0), m_flush_flags, &flush_req.comp);
         if (status == UCS_OK) {
             --flush_req.comp.count;
         } else if (status == UCS_ERR_NO_RESOURCE) {
             /* If flush returned NO_RESOURCE, add to pending must succeed */
             flush_req.test      = this;
             flush_req.uct.func  = flush_progress;
             status = uct_ep_pending_add(sender().ep(0), &flush_req.uct, 0);
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
         EXPECT_EQ(1, reqs.back().comp.count);
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

UCS_TEST_SKIP_COND_P(uct_flush_test, put_bcopy_flush_ep_no_comp,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY)) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_put_bcopy(&uct_flush_test::flush_ep_no_comp);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_put_bcopy(&uct_flush_test::flush_ep_no_comp);
    }
}

UCS_TEST_SKIP_COND_P(uct_flush_test, put_bcopy_flush_iface_no_comp,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY)) {
    test_flush_put_bcopy(&uct_flush_test::flush_iface_no_comp);
}

UCS_TEST_SKIP_COND_P(uct_flush_test, put_bcopy_flush_ep_nb,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY)) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_put_bcopy(&uct_flush_test::flush_ep_nb);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_put_bcopy(&uct_flush_test::flush_ep_nb);
    }
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_zcopy_flush_ep_no_comp,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY),
                     "UD_TIMER_TICK?=100ms") {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_am_zcopy(&uct_flush_test::flush_ep_no_comp, false);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_zcopy(&uct_flush_test::flush_ep_no_comp, true);
    }
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_zcopy_flush_iface_no_comp,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY),
                     "UD_TIMER_TICK?=100ms") {
    test_flush_am_zcopy(&uct_flush_test::flush_iface_no_comp, true);
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_zcopy_flush_ep_nb,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY),
                     "UD_TIMER_TICK?=100ms") {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_am_zcopy(&uct_flush_test::flush_ep_nb, false);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_zcopy(&uct_flush_test::flush_ep_nb, true);
    }
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_flush_ep_no_comp,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY)) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_am_disconnect(&uct_flush_test::flush_ep_no_comp, false);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_disconnect(&uct_flush_test::flush_ep_no_comp, true);
    }
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_flush_iface_no_comp,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY)) {
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;
    test_flush_am_disconnect(&uct_flush_test::flush_iface_no_comp, true);
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_flush_ep_nb,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY)) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_am_disconnect(&uct_flush_test::flush_ep_nb, false);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_disconnect(&uct_flush_test::flush_ep_nb, true);
    }
}

UCS_TEST_SKIP_COND_P(uct_flush_test, am_pending_flush_nb,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY |
                                 UCT_IFACE_FLAG_PENDING)) {
    am_rx_count   = 0;
    m_flush_flags = UCT_FLUSH_FLAG_LOCAL;

    test_flush_am_pending(&uct_flush_test::flush_ep_nb, false);

    if (is_caps_supported(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)) {
        am_rx_count    = 0;
        m_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        test_flush_am_pending(&uct_flush_test::flush_ep_nb, true);
    }
}

UCT_INSTANTIATE_TEST_CASE(uct_flush_test)

class uct_cancel_test : public uct_test {
public:
    static const size_t BUF_SIZE = 8 * 1024;

    class peer {
    public:
        peer(uct_cancel_test &test) :
            m_e(NULL), m_buf(NULL), m_buf32(NULL), m_peer(NULL), m_test(test)
        {
            m_e = m_test.uct_test::create_entity(0, error_handler_cb);
            m_test.m_entities.push_back(m_e);

            m_buf.reset(new mapped_buffer(BUF_SIZE, 0, *m_e));
            m_buf32.reset(new mapped_buffer(32, 0, *m_e));
            uct_iface_set_am_handler(m_e->iface(), 0, am_cb, &m_test,
                                     UCT_CB_FLAG_ASYNC);
        }

        void connect() {
            m_e->connect(0, *m_peer->m_e, 0);
            m_peer->m_e->connect(0, *m_e, 0);
        }

        entity                       *m_e;
        ucs::auto_ptr<mapped_buffer> m_buf;
        ucs::auto_ptr<mapped_buffer> m_buf32;
        peer                         *m_peer;

    private:
        uct_cancel_test &m_test;
    };

    uct_cancel_test() :
        uct_test(), m_s0(NULL), m_s1(NULL), m_err_count(0)
    {
    }

    ucs_status_t am_bcopy(peer *s) {
        mapped_buffer &sendbuf = *s->m_buf32;
        ssize_t packed_len;

        packed_len = uct_ep_am_bcopy(s->m_e->ep(0), 0, mapped_buffer::pack,
                                     (void*)&sendbuf, 0);
        if (packed_len >= 0) {
            EXPECT_EQ(sendbuf.length(), (size_t)packed_len);
            return UCS_OK;
        } else {
            return (ucs_status_t)packed_len;
        }
    }

    ucs_status_t am_zcopy(peer *s) {
        size_t size = ucs_min(BUF_SIZE, s->m_e->iface_attr().cap.am.max_zcopy);
        mapped_buffer &sendbuf = *s->m_buf;
        size_t header_length = 0;
        uct_iov_t iov;

        iov.buffer = (char*)sendbuf.ptr() + header_length;
        iov.count  = 1;
        iov.length = size - header_length;
        iov.memh   = sendbuf.memh();
        return uct_ep_am_zcopy(s->m_e->ep(0), 0, sendbuf.ptr(), header_length,
                               &iov, 1, 0, NULL);
    }

    ucs_status_t get_zcopy(peer *s) {
        size_t size = ucs_min(BUF_SIZE, s->m_e->iface_attr().cap.get.max_zcopy);
        mapped_buffer &sendbuf = *s->m_buf;
        mapped_buffer &recvbuf = *s->m_peer->m_buf;

        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), size,
                                sendbuf.memh(), s->m_e->iface_attr().cap.get.max_iov);

        return uct_ep_get_zcopy(s->m_e->ep(0), iov, iovcnt, recvbuf.addr(),
                                recvbuf.rkey(), NULL);
    }

    ucs_status_t get_bcopy(peer *s) {
        mapped_buffer &sendbuf = *s->m_buf32;
        mapped_buffer &recvbuf = *s->m_peer->m_buf32;

        return uct_ep_get_bcopy(s->m_e->ep(0), (uct_unpack_callback_t)memcpy,
                                sendbuf.ptr(), sendbuf.length(),
                                recvbuf.addr(), recvbuf.rkey(), NULL);
    }

    void flush_and_reconnect() {
        std::list<entity *> flushing;
        ucs_status_t status = UCS_OK;
        uct_completion_t done;

        m_err_count = 0;
        flushing.push_back(m_s0->m_e);
        flushing.push_back(m_s1->m_e);
        done.count  = flushing.size() + 1;
        done.status = UCS_OK;
        done.func   = NULL;
        ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(50.0);
        while (!flushing.empty() && (ucs_get_time() < loop_end_limit)) {
            std::list<entity *>::iterator iter = flushing.begin();
            while (iter != flushing.end()) {
                status = uct_ep_flush((*iter)->ep(0), UCT_FLUSH_FLAG_CANCEL, &done);
                if (status == UCS_ERR_NO_RESOURCE) {
                    iter++;
                } else {
                    ASSERT_UCS_OK_OR_INPROGRESS(status);
                    iter = flushing.erase(iter);
                    if (status == UCS_OK) {
                        done.count--;
                    }
                }
            }

            short_progress_loop();
        }
        ASSERT_UCS_OK_OR_INPROGRESS(status);

        /* coverity[loop_condition] */
        while (done.count != 1) {
            progress();
        }

        m_s1->m_e->destroy_eps();
        m_s1->m_e->connect(0, *m_s0->m_e, 0);

        /* there is a chance that one side getting disconect error before
         * calling flush(CANCEL) */
        EXPECT_LE(m_err_count, 1);
    }

    typedef ucs_status_t (uct_cancel_test::* send_func_t)(peer *s);

    void fill(send_func_t send) {
        ucs_status_t status;
        std::list<peer *> filling;

        filling.push_back(m_s0);
        filling.push_back(m_s1);
        while (!filling.empty()) {
            std::list<peer *>::iterator iter = filling.begin();
            while (iter != filling.end()) {
                status = (this->*send)(*iter);
                if (status == UCS_ERR_NO_RESOURCE) {
                    iter = filling.erase(iter);
                } else {
                    ASSERT_UCS_OK_OR_INPROGRESS(status);
                    iter++;
                }
            }
        }
    }

    int count() {
        return 100;
    }

    void do_test(send_func_t send) {
        for (int i = 0; i < count(); ++i) {
            fill(send);
            flush_and_reconnect();
        }
    }

protected:

    ucs::auto_ptr<peer> m_s0;
    ucs::auto_ptr<peer> m_s1;
    int m_err_count;

    virtual void init() {
        uct_test::init();

        m_s0.reset(new peer(*this));
        check_skip_test_tl();
        m_s1.reset(new peer(*this));

        m_s0->m_peer = m_s1;
        m_s1->m_peer = m_s0;

        m_s0->connect();
        flush();
    }

    virtual void cleanup() {
        flush();

        m_s0.reset();
        m_s1.reset();

        uct_test::cleanup();
    }

    static ucs_status_t
    error_handler_cb(void *arg, uct_ep_h ep, ucs_status_t status) {
        uct_cancel_test *test = reinterpret_cast<uct_cancel_test*>(arg);
        return test->error_handler(ep, status);
    }

    static ucs_status_t am_cb(void *arg, void *data, size_t length, unsigned flags) {
        uct_cancel_test *test = reinterpret_cast<uct_cancel_test*>(arg);
        return test->am(data, length, flags);
    }

    ucs_status_t error_handler(uct_ep_h ep, ucs_status_t status) {
        EXPECT_EQ(UCS_ERR_ENDPOINT_TIMEOUT, status);
        m_err_count++;
        return UCS_OK;
    }

    ucs_status_t am(void *data, size_t length, unsigned flags) {
        return UCS_OK;
    }

    void check_skip_test_tl() {
        const resource *r = dynamic_cast<const resource*>(GetParam());

        if ((r->tl_name != "rc_mlx5") && (r->tl_name != "rc_verbs")) {
            UCS_TEST_SKIP_R("not supported yet");
        }

        check_skip_test();
    }

};

UCS_TEST_SKIP_COND_P(uct_cancel_test, am_zcopy,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY)) {
    do_test(&uct_cancel_test::am_zcopy);
}

UCS_TEST_SKIP_COND_P(uct_cancel_test, am_bcopy,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY)) {
    do_test(&uct_cancel_test::am_bcopy);
}

UCS_TEST_SKIP_COND_P(uct_cancel_test, get_bcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY)) {
    do_test(&uct_cancel_test::get_bcopy);
}

UCS_TEST_SKIP_COND_P(uct_cancel_test, get_zcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY)) {
    do_test(&uct_cancel_test::get_zcopy);
}

UCT_INSTANTIATE_TEST_CASE(uct_cancel_test)
