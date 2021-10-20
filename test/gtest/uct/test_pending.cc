/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
#include <ucs/arch/atomic.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_uct_pending : public uct_test {
public:
    test_uct_pending() : uct_test() {
        m_e1 = NULL;
        m_e2 = NULL;

        reduce_tl_send_queues();
    }

    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        check_skip_test();
    }

    void initialize() {

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
        flush();
    }

    typedef struct pending_send_request {
        uct_pending_req_t uct;
        uct_ep_h          ep;
        uint64_t          data;
        int               countdown;  /* Actually send after X calls */
        int               send_count; /* Used by fairness test */
        bool              pending;
        bool              delete_me;
    } pending_send_request_t;

    struct am_completion_t {
        uct_completion_t uct;
        uct_ep_h         ep;
    };

    bool send_am_or_add_pending(uint64_t *send_data, uint64_t header,
                                unsigned idx, pending_send_request_t *preq) {
        ucs_time_t loop_end_limit = ucs::get_deadline();
        ucs_status_t status, status_pend;

        do {
            status = uct_ep_am_short(m_e1->ep(idx), AM_ID, header, send_data,
                                     sizeof(*send_data));
            if (status != UCS_OK) {
                EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
                pending_send_request_t *req = (preq != NULL) ? preq :
                                              pending_alloc(*send_data, idx);
                status_pend                 = uct_ep_pending_add(m_e1->ep(idx),
                                                                 &req->uct, 0);
                if (status_pend == UCS_ERR_BUSY) { /* retry */
                    if (preq == NULL) {
                        pending_delete(req);
                    }
                    continue;
                }
                ASSERT_UCS_OK(status_pend);
                ++n_pending;
                req->pending = true;
                /* coverity[leaked_storage] */
            } else if (preq != NULL) {
                ++preq->send_count; /* used by fairness test */
            }
            ++(*send_data);
            return true;
        } while (ucs_get_time() < loop_end_limit);

        return false;
    }

    unsigned send_ams_and_add_pending(uint64_t *send_data,
                                      uint64_t header      = PENDING_HDR,
                                      bool add_single_pend = true,
                                      bool change_ep       = false,
                                      unsigned ep_idx      = 0,
                                      unsigned iters       = 10000) {
        ucs_time_t loop_end_limit   = ucs_get_time() + ucs_time_from_sec(3);
        unsigned i                  = 0;
        int init_pending            = n_pending;
        int added_pending           = 0;
        unsigned idx;

        do {
            idx = change_ep ? i : ep_idx;
            if (!send_am_or_add_pending(send_data, header, idx, NULL)) {
                break;
            }
            ++i;
            added_pending = n_pending - init_pending;
            if ((added_pending != 0) && add_single_pend) {
                EXPECT_EQ(1, added_pending);
                break;
            }
        } while ((i < iters) && (ucs_get_time() < loop_end_limit));

        if (added_pending == 0) {
            UCS_TEST_SKIP_R("Can't fill UCT resources in the given time.");
        }

        return i;
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {

        volatile unsigned *counter = (volatile unsigned*) arg;
        uint64_t test_hdr          = *(uint64_t *) data;
        uint64_t actual_data       = *(unsigned*)((char*)data + sizeof(test_hdr));

        if ((test_hdr == PENDING_HDR) &&
            (actual_data == (0xdeadbeef + *counter))) {
            ucs_atomic_add32(counter, 1);
        } else {
            UCS_TEST_ABORT("Error in comparison in pending_am_handler. Counter: "
                           << counter << ", header: " << test_hdr
                           << ", data: " << actual_data);
        }

        return UCS_OK;
    }

    static ucs_status_t am_handler_count(void *arg, void *data, size_t length,
                                         unsigned flags) {
        volatile unsigned *counter = (volatile unsigned*) arg;
        ucs_atomic_add32(counter, 1);
        return UCS_OK;
    }

    static ucs_status_t am_handler_simple(void *arg, void *data, size_t length,
                                          unsigned flags) {
        return UCS_OK;
    }

    static ucs_status_t am_handler_check_rx_order(void *arg, void *data,
                                                  size_t length, unsigned flags) {
        volatile bool *comp_received = (volatile bool*)arg;
        uint64_t hdr                 = *(uint64_t*)data;

        /* We expect that message sent from pending callback will arrive
         * before the one sent from the completion callback. */
        if (hdr == PENDING_HDR) {
            pend_received = true;
            EXPECT_FALSE(*comp_received);
        } else if (hdr == COMP_HDR) {
            *comp_received = true;
            EXPECT_TRUE(pend_received);
        } else {
            EXPECT_EQ(AM_HDR, hdr);
        }

        return UCS_OK;
    }

    static void completion_cb(uct_completion_t *self) {
        am_completion_t *comp = ucs_container_of(self, am_completion_t, uct);

        EXPECT_UCS_OK(self->status);

        ucs_status_t status = uct_ep_am_short(comp->ep, AM_ID, COMP_HDR,
                                              NULL, 0);
        EXPECT_TRUE(!UCS_STATUS_IS_ERR(status) ||
                    (status == UCS_ERR_NO_RESOURCE));
    }

    static ucs_status_t pending_send_op(uct_pending_req_t *self) {

        pending_send_request_t *req = ucs_container_of(self,
                                                       pending_send_request_t,
                                                       uct);
        if (req->countdown > 0) {
            --req->countdown;
            return UCS_INPROGRESS;
        }

        ucs_status_t status = uct_ep_am_short(req->ep, AM_ID, PENDING_HDR,
                                              &req->data, sizeof(req->data));
        if (status == UCS_OK) {
            req->pending = false;
            req->send_count++;
            n_pending--;
            if (req->delete_me) {
                pending_delete(req);
            }
        }

        return status;
    }

    static ucs_status_t pending_send_op_add_pending(uct_pending_req_t *self) {
        ucs_status_t status = pending_send_op(self);
        if (status == UCS_ERR_NO_RESOURCE) {
            pending_send_request_t *req = ucs_container_of(self,
                                                           pending_send_request_t,
                                                           uct);
            /* replace with the callback that just do sends and return
             * `UCS_ERR_NO_RESOURCE` in case of no resources on the given EP */
            req->uct.func = pending_send_op;

            status = uct_ep_pending_add(req->ep, &req->uct, 0);
            ASSERT_UCS_OK(status);
            return UCS_OK;
        }

        return status;
    }

    static void purge_cb(uct_pending_req_t *self, void *arg)
    {
        pending_send_request_t *req = ucs_container_of(self,
                                                       pending_send_request_t,
                                                       uct);
        pending_delete(req);
        ++n_purge;
    }

    pending_send_request_t* pending_alloc(uint64_t send_data, int ep_idx = 0,
                                          int count = 5, bool delete_me = true,
                                          uct_pending_callback_t cb = pending_send_op) {
        pending_send_request_t *req = new pending_send_request_t();
        req->ep                     = m_e1->ep(ep_idx);
        req->data                   = send_data;
        req->pending                = false;
        req->countdown              = count;
        req->uct.func               = cb;
        req->delete_me              = delete_me;
        req->send_count             = 0;

        return req;
    }

    static void pending_delete(pending_send_request_t *req) {
        delete req;
    }

protected:
    static const uint64_t AM_HDR;
    static const uint64_t PENDING_HDR;
    static const uint64_t COMP_HDR;
    static const uint8_t  AM_ID;
    entity *m_e1, *m_e2;
    static int n_pending;
    static int n_purge;
    static bool pend_received;
};

int test_uct_pending::n_pending              = 0;
int test_uct_pending::n_purge                = 0;
bool test_uct_pending::pend_received         = false;
const uint64_t test_uct_pending::AM_HDR      = 0x0ul;
const uint64_t test_uct_pending::PENDING_HDR = 0x1ul;
const uint64_t test_uct_pending::COMP_HDR    = 0x2ul;
const uint8_t  test_uct_pending::AM_ID       = 0;

void install_handler_sync_or_async(uct_iface_t *iface, uint8_t id,
                                   uct_am_callback_t cb, void *arg)
{
    ucs_status_t status;
    uct_iface_attr_t attr;

    status = uct_iface_query(iface, &attr);
    ASSERT_UCS_OK(status);

    if (attr.cap.flags & UCT_IFACE_FLAG_CB_SYNC) {
        uct_iface_set_am_handler(iface, id, cb, arg, 0);
    } else {
        ASSERT_TRUE(attr.cap.flags & UCT_IFACE_FLAG_CB_ASYNC);
        uct_iface_set_am_handler(iface, id, cb, arg, UCT_CB_FLAG_ASYNC);
    }
}

UCS_TEST_SKIP_COND_P(test_uct_pending, pending_op,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_PENDING))
{
    uint64_t send_data = 0xdeadbeef;
    unsigned counter   = 0;

    initialize();

    /* set a callback for the uct to invoke for receiving the data */
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler, &counter);

    /* send the data until the resources run out */
    unsigned n_sends = send_ams_and_add_pending(&send_data, PENDING_HDR, false);

    /* coverity[loop_condition] */
    while (counter != n_sends) {
        progress();
    }

    flush();

    ASSERT_EQ(counter, n_sends);
}

UCS_TEST_SKIP_COND_P(test_uct_pending, send_ooo_with_pending,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_PENDING))
{
    uint64_t send_data = 0xdeadbeef;
    unsigned counter   = 0;
    ucs_status_t status;

    initialize();

    /* set a callback for the uct to invoke when receiving the data */
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler, &counter);

    unsigned n_sends = send_ams_and_add_pending(&send_data);

    /* progress the receiver a bit to release resources */
    for (unsigned i = 0; i < 1000; i++) {
        m_e2->progress();
    }

    /* send a new message. the transport should make sure that this new message
     * isn't sent before the one in pending, thus preventing out-of-order in
     * sending. */
    do {
        status = uct_ep_am_short(m_e1->ep(0), AM_ID, PENDING_HDR, &send_data,
                                 sizeof(send_data));
        short_progress_loop();
    } while (status == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status);
    ++n_sends;

    /* the receive side checks that the messages were received in order.
     * check the last message here. (counter was raised by one for next iteration) */
    wait_for_value(&counter, n_sends, true);
    EXPECT_EQ(n_sends, counter);
}

UCS_TEST_SKIP_COND_P(test_uct_pending, send_ooo_with_pending_another_ep,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_PENDING))
{
    const int num_eps  = 2;
    uint64_t send_data = 0xdeadbeefUL;
    unsigned counter   = 0;
    unsigned n_sends   = 0;
    bool ep_pending_idx[num_eps];

     /* set a callback for the uct to invoke when receiving the data */
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler_count,
                                  &counter);

    for (unsigned idx = 0; idx < num_eps; ++idx) {
        m_e1->connect(idx, *m_e2, idx);
        ep_pending_idx[idx] = false;
    }

    ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(3);
    unsigned n_iters          = 10000;
    unsigned i                = 0;
    unsigned num_ep_pending   = 0;

    n_pending = 0;

    do {
        ucs_status_t status;

        for (unsigned idx = 0; idx < num_eps; ++idx) {
            if (ep_pending_idx[idx]) {
                continue;
            }

            /* try to user all transport's resources */
            status = uct_ep_am_short(m_e1->ep(idx), AM_ID, PENDING_HDR,
                                     &send_data, sizeof(send_data));
            if (status != UCS_OK) {
                ASSERT_EQ(UCS_ERR_NO_RESOURCE, status);
                ep_pending_idx[idx] = true;
                num_ep_pending++;

                /* schedule pending req to send data on the another EP */
                pending_send_request_t *preq =
                    pending_alloc(send_data, num_eps - idx - 1,
                                  0, true, pending_send_op_add_pending);
                status = uct_ep_pending_add(m_e1->ep(idx), &preq->uct, 0);
                ASSERT_UCS_OK(status);
                ++n_pending;
                preq->pending = true;
                /* coverity[leaked_storage] */
            }
            ++n_sends;
        }

        ++i;
    } while ((num_ep_pending < num_eps) &&
             (i < n_iters) && (ucs_get_time() < loop_end_limit));

    UCS_TEST_MESSAGE << "eps with pending: " << num_ep_pending << "/" << num_eps
                     << ", current pending: " << n_pending;

    flush();

    wait_for_value(&n_pending, 0, true);
    EXPECT_EQ(0, n_pending);

    wait_for_value(&counter, n_sends, true);
    EXPECT_EQ(n_sends, counter);
}

UCS_TEST_SKIP_COND_P(test_uct_pending, pending_purge,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_PENDING))
{
    const int num_eps  = 5;
    uint64_t send_data = 0xdeadbeefUL;

     /* set a callback for the uct to invoke when receiving the data */
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler_simple, NULL);

    for (int i = 0; i < num_eps; ++i) {
        m_e1->connect(i, *m_e2, i);
        send_ams_and_add_pending(&send_data, PENDING_HDR, true, false, i);
    }

    for (int i = 0; i < num_eps; ++i) {
        n_purge = 0;
        uct_ep_pending_purge(m_e1->ep(i), purge_cb, NULL);
        EXPECT_EQ(1, n_purge);
    }
}

/*
 * test that the pending op callback is only called from the progress()
 */
UCS_TEST_SKIP_COND_P(test_uct_pending, pending_async,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_AM_BCOPY |
                                 UCT_IFACE_FLAG_PENDING  |
                                 UCT_IFACE_FLAG_CB_ASYNC))
{
    initialize();

    /* set a callback for the uct to invoke when receiving the data */
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler_simple, 0);

    /* send while resources are available */
    uint64_t send_data = 0xABC;
    n_pending          = 0;
    send_ams_and_add_pending(&send_data);

    /* pending op must not be called either asynchronously or from the
     * uct_ep_am_bcopy/short() */
    twait(300);
    EXPECT_EQ(1, n_pending);

    /* send should fail, because we have pending op */
    mapped_buffer sbuf(ucs_min(64ul, m_e1->iface_attr().cap.am.max_bcopy),
                       0, *m_e1);
    ssize_t packed_len = uct_ep_am_bcopy(m_e1->ep(0), AM_ID,
                                         mapped_buffer::pack, &sbuf, 0);
    EXPECT_EQ(1, n_pending);
    EXPECT_EQ((ssize_t)UCS_ERR_NO_RESOURCE, packed_len);

    wait_for_value(&n_pending, 0, true);
    EXPECT_EQ(0, n_pending);
}

/*
 * test that arbiter does not block when ucs_ok is returned
 * The issue is a dc transport specific but test may be also useful
 * for other transports
 */
UCS_TEST_SKIP_COND_P(test_uct_pending, pending_ucs_ok_dc_arbiter_bug,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_PENDING))
{
    int N, max_listen_conn;

    initialize();

    mapped_buffer sbuf(ucs_min(64ul, m_e1->iface_attr().cap.am.max_bcopy), 0,
                       *m_e1);

    /* set a callback for the uct to invoke when receiving the data */
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler_simple, 0);

    if (RUNNING_ON_VALGRIND) {
        N = 64;
    } else if (m_e1->iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        N = 2048;
    } else {
        N = 128;
    }

    N = ucs_min(N, max_connections());

    /* idx 0 is setup in initialize(). only need to alloc request */
    for (int j, i = 1; i < N; i += j) {
        max_listen_conn = ucs_min(max_connect_batch(), N - i);

        for (j = 0; j < max_listen_conn; j++) {
            int idx = i + j;
            m_e1->connect(idx, *m_e2, idx);
        }
        /* give a chance to finish connection for some transports (ib/ud, tcp) */
        flush();
    }

    n_pending = 0;

    /* try to exhaust global resources and create a pending queue */
    uint64_t send_data = 0xBEEBEE;
    send_ams_and_add_pending(&send_data, PENDING_HDR, false, true,0, N);

    UCS_TEST_MESSAGE << "pending queue len: " << n_pending;

    wait_for_value(&n_pending, 0, true);
    EXPECT_EQ(0, n_pending);
}

UCS_TEST_SKIP_COND_P(test_uct_pending, pending_fairness,
                     (RUNNING_ON_VALGRIND ||
                      !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                  UCT_IFACE_FLAG_PENDING)))
{
    int N              = 16;
    uint64_t send_data = 0xdeadbeef;
    int i, iters;

    initialize();

    if (m_e1->iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        N = ucs_min(128, max_connect_batch());
    }
    pending_send_request_t *reqs[N];
    install_handler_sync_or_async(m_e2->iface(), AM_ID, am_handler_simple, 0);

    /* idx 0 is setup in initialize(). only need to alloc request */
    reqs[0] = pending_alloc(send_data, 0, 0, false);
    for (i = 1; i < N; i++) {
        m_e1->connect(i, *m_e2, i);
        reqs[i] = pending_alloc(send_data, i, 0, false);
    }

    /* give a chance to finish connection for some transports (ib/ud, tcp) */
    flush();

    n_pending = 0;
    for (iters = 0; iters < 10000; iters++) {
        /* send until resources of all eps are exhausted */
        while (n_pending < N) {
            for (i = 0; i < N; ++i) { /* TODO: change to list */
                if (reqs[i]->pending) {
                    continue;
                }
                if (!send_am_or_add_pending(&send_data, PENDING_HDR, i, reqs[i])) {
                    UCS_TEST_SKIP_R("Can't fill UCT resources in the given time.");
                }
            }
        }
        /* progress until it is possible to send more */
        while(n_pending == N) {
            progress();
        }
        /* repeat the cycle.
         * it is expected that every ep will send about
         * the same number of messages.
         */
    }

    /* check fairness:  */
    int min_sends = INT_MAX;
    int max_sends = 0;
    for (i = 0; i < N; i++) {
        min_sends = ucs_min(min_sends, reqs[i]->send_count);
        max_sends = ucs_max(max_sends, reqs[i]->send_count);
    }
    UCS_TEST_MESSAGE << " min_sends: " << min_sends
                     << " max_sends: " << max_sends
                     << " still pending: " << n_pending;

    while(n_pending > 0) {
        progress();
    }

    flush();

    for (i = 0; i < N; i++) {
        pending_delete(reqs[i]);
    }

    /* there must be no starvation */
    EXPECT_LT(0, min_sends);
    /* TODO: add stricter fairness criteria */
    if (min_sends < max_sends /2) {
        UCS_TEST_MESSAGE << " CHECK: pending queue is not fair";
    }
}

/* Check that pending requests are processed before the sends from
 * completion callbacks */
UCS_TEST_SKIP_COND_P(test_uct_pending, send_ooo_with_comp,
                     !check_caps(UCT_IFACE_FLAG_AM_SHORT |
                                 UCT_IFACE_FLAG_AM_ZCOPY |
                                 UCT_IFACE_FLAG_PENDING))
{
    initialize();

    bool comp_received = false;
    pend_received      = false;

    uct_iface_set_am_handler(m_e2->iface(), AM_ID, am_handler_check_rx_order,
                             &comp_received, 0);

    mapped_buffer sendbuf(32, 0, *m_e1);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(), 1);
    am_completion_t comp;
    comp.uct.func       = completion_cb;
    comp.uct.count      = 1;
    comp.uct.status     = UCS_OK;
    comp.ep             = m_e1->ep(0);
    ucs_status_t status = uct_ep_am_zcopy(m_e1->ep(0), AM_ID, &AM_HDR,
                                           sizeof(AM_HDR), iov, iovcnt, 0,
                                           &comp.uct);
    ASSERT_FALSE(UCS_STATUS_IS_ERR(status));

    uint64_t send_data = 0xFAFAul;
    send_ams_and_add_pending(&send_data, AM_HDR);

    wait_for_flag(&pend_received);
    EXPECT_TRUE(pend_received);

    flush();
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_pending);
