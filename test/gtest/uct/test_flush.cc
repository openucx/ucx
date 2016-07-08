/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"


class uct_flush_test : public uct_test {
public:
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    static const uint64_t SEED3 = 0x3333333333333333lu;
    static const int      AM_ID = 1;

    typedef void (uct_flush_test::* flush_func_t)();

    struct test_req_t {
        uct_pending_req_t  uct;
        uct_completion_t   comp;
        mapped_buffer      *sendbuf;
        uct_flush_test     *test;
    };

    void init() {
        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender->connect(0, *m_receiver, 0);
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
             status = uct_ep_put_bcopy(m_sender->ep(0), pack_cb, (void*)&sendbuf,
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
             status = uct_ep_am_bcopy(m_sender->ep(0), AM_ID, pack_cb,
                                      (void*)&sendbuf);
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
                                   void *desc)
    {
        const mapped_buffer *recvbuf = (const mapped_buffer *)arg;
        memcpy(recvbuf->ptr(), data, ucs_min(length, recvbuf->length()));
        return UCS_OK;
    }

    ucs_status_t am_send_pending(test_req_t *am_req)
    {
        ssize_t status;

        status = uct_ep_am_bcopy(m_sender->ep(0), AM_ID, pack_cb,
                                 (void*)am_req->sendbuf);
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

        status = uct_ep_flush(flush_req->test->m_sender->ep(0), 0,
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

    virtual void test_flush_put_bcopy(flush_func_t flush) {
        const size_t length = 8;
        check_caps(UCT_IFACE_FLAG_PUT_SHORT);
        mapped_buffer sendbuf(length, SEED1, *m_sender);
        mapped_buffer recvbuf(length, SEED2, *m_receiver);
        sendbuf.pattern_fill(SEED3);
        blocking_put_bcopy(sendbuf, recvbuf);
        (this->*flush)();
        recvbuf.pattern_check(SEED3);
    }

    virtual void test_flush_am_zcopy(flush_func_t flush) {
        const size_t length = 8;
        check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
        mapped_buffer sendbuf(length, SEED1, *m_sender);
        mapped_buffer recvbuf(length, SEED2, *m_receiver);
        sendbuf.pattern_fill(SEED3);

        uct_iface_set_am_handler(m_receiver->iface(), AM_ID, am_handler, &recvbuf,
                                 UCT_AM_CB_FLAG_ASYNC);

        uct_completion_t zcomp;
        zcomp.count = 2;
        zcomp.func  = NULL;

        ucs_status_t status;
        do {
            status = uct_ep_am_zcopy(m_sender->ep(0), AM_ID, NULL, 0, sendbuf.ptr(),
                                     sendbuf.length(), sendbuf.memh(), &zcomp);
        } while (status == UCS_ERR_NO_RESOURCE);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
        if (status == UCS_OK) {
            --zcomp.count;
        }

        (this->*flush)();

        EXPECT_EQ(1, zcomp.count); /* Zero copy op should be already completed
                                      since flush returned */

        m_sender->destroy_ep(0);

        short_progress_loop();

        uct_iface_set_am_handler(m_receiver->iface(), AM_ID, NULL, NULL, 0);

        recvbuf.pattern_check(SEED3);
    }

    virtual void test_flush_am_disconnect(flush_func_t flush) {
        const size_t length = 8;
        check_caps(UCT_IFACE_FLAG_AM_BCOPY);
        mapped_buffer sendbuf(length, SEED1, *m_sender);
        mapped_buffer recvbuf(length, SEED2, *m_receiver);
        sendbuf.pattern_fill(SEED3);

        uct_iface_set_am_handler(m_receiver->iface(), AM_ID, am_handler, &recvbuf,
                                 UCT_AM_CB_FLAG_ASYNC);
        blocking_am_bcopy(sendbuf);
        (this->*flush)();
        m_sender->destroy_ep(0);

        short_progress_loop();

        uct_iface_set_am_handler(m_receiver->iface(), AM_ID, NULL, NULL, 0);

        recvbuf.pattern_check(SEED3);
    }

    void flush_ep_no_comp() {
        ucs_status_t status;
        do {
            progress();
            status = uct_ep_flush(m_sender->ep(0), 0, NULL);
        } while ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS));
        ASSERT_UCS_OK(status);
    }

    void flush_iface_no_comp() {
        ucs_status_t status;
        do {
            progress();
            status = uct_ep_flush(m_sender->ep(0), 0, NULL);
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
            status = uct_ep_flush(m_sender->ep(0), 0, &comp);
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

protected:
    entity *m_sender;
    entity *m_receiver;
};

UCS_TEST_P(uct_flush_test, put_bcopy_flush_ep_no_comp) {
    test_flush_put_bcopy(&uct_flush_test::flush_ep_no_comp);
}

UCS_TEST_P(uct_flush_test, put_bcopy_flush_iface_no_comp) {
    test_flush_put_bcopy(&uct_flush_test::flush_iface_no_comp);
}

UCS_TEST_P(uct_flush_test, put_bcopy_flush_ep_nb) {
    test_flush_put_bcopy(&uct_flush_test::flush_ep_nb);
}

UCS_TEST_P(uct_flush_test, am_zcopy_flush_ep_no_comp) {
    test_flush_am_zcopy(&uct_flush_test::flush_ep_no_comp);
}

UCS_TEST_P(uct_flush_test, am_zcopy_flush_iface_no_comp) {
    test_flush_am_zcopy(&uct_flush_test::flush_iface_no_comp);
}

UCS_TEST_P(uct_flush_test, am_zcopy_flush_ep_nb) {
    test_flush_am_zcopy(&uct_flush_test::flush_ep_nb);
}

UCS_TEST_P(uct_flush_test, am_flush_ep_no_comp) {
    test_flush_am_disconnect(&uct_flush_test::flush_ep_no_comp);
}

UCS_TEST_P(uct_flush_test, am_flush_iface_no_comp) {
    test_flush_am_disconnect(&uct_flush_test::flush_iface_no_comp);
}

UCS_TEST_P(uct_flush_test, am_flush_ep_nb) {
    test_flush_am_disconnect(&uct_flush_test::flush_ep_nb);
}

UCS_TEST_P(uct_flush_test, am_pending_flush_nb) {
     const size_t length = 8;
     check_caps(UCT_IFACE_FLAG_AM_BCOPY);
     mapped_buffer sendbuf(length, SEED1, *m_sender);
     mapped_buffer recvbuf(length, SEED2, *m_receiver);
     sendbuf.pattern_fill(SEED3);

     uct_iface_set_am_handler(m_receiver->iface(), AM_ID, am_handler, &recvbuf,
                              UCT_AM_CB_FLAG_ASYNC);

     /* Send until resources are exhausted */
     unsigned count = 0;
     ssize_t packed_len;
     for (;;) {
         packed_len = uct_ep_am_bcopy(m_sender->ep(0), AM_ID, pack_cb,
                                      (void*)&sendbuf);
         if (packed_len == UCS_ERR_NO_RESOURCE) {
             break;
         }

         if (packed_len >= 0) {
             ++count;
         } else {
             ASSERT_UCS_OK((ucs_status_t)packed_len);
         }
     }

     /* Queue some pending AMs */
     std::vector<test_req_t> reqs;
     reqs.resize(10);
     for (size_t i = 0; i < reqs.size(); ++i) {
         reqs[i].sendbuf    = &sendbuf;
         reqs[i].test       = this;
         reqs[i].uct.func   = am_progress;
         reqs[i].comp.count = 2;
         reqs[i].comp.func  = NULL;
         ucs_status_t status =
                         uct_ep_pending_add(m_sender->ep(0), &reqs[i].uct);
         ASSERT_UCS_OK(status);
     }

     /* Try to start a flush */
     test_req_t flush_req;
     ucs_status_t status;
     flush_req.comp.count = 2;
     flush_req.comp.func  = NULL;

     status = uct_ep_flush(m_sender->ep(0), 0, &flush_req.comp);
     if (status == UCS_OK) {
         --flush_req.comp.count;
     } else if (status == UCS_ERR_NO_RESOURCE) {
         /* If flush returned NO_RESOURCE, add to pending must succeed */
         flush_req.test      = this;
         flush_req.uct.func  = flush_progress;
         status = uct_ep_pending_add(m_sender->ep(0), &flush_req.uct);
         EXPECT_EQ(UCS_OK, status);
     } else if (status == UCS_INPROGRESS) {
     } else {
         ASSERT_UCS_OK(status);
     }

     /* coverity[loop_condition] */
     while (flush_req.comp.count != 1) {
         progress();
     }

     while (!reqs.empty()) {
         EXPECT_EQ(1, reqs.back().comp.count);
         reqs.pop_back();
     }

     m_sender->destroy_ep(0);

     short_progress_loop();

     uct_iface_set_am_handler(m_receiver->iface(), AM_ID, NULL, NULL, 0);

     recvbuf.pattern_check(SEED3);
}

UCT_INSTANTIATE_TEST_CASE(uct_flush_test)
