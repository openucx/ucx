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
    void initialize() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
    }

    typedef struct pending_send_request {
        uct_ep_h          ep;
        uint64_t          data;
        int               countdown;  /* Actually send after X calls */
        uct_pending_req_t uct;
        int               active;
        int               id;
    } pending_send_request_t;

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {

        volatile unsigned *counter = (volatile unsigned*) arg;
        uint64_t test_hdr = *(uint64_t *) data;
        uint64_t actual_data = *(unsigned*)((char*)data + sizeof(test_hdr));

        if ((test_hdr == 0xabcd) && (actual_data == (0xdeadbeef + *counter))) {
            ucs_atomic_add32(counter, 1);
        } else {
            UCS_TEST_ABORT("Error in comparison in pending_am_handler. Counter: " << counter);
        }

        return UCS_OK;
    }

    static ucs_status_t am_handler_simple(void *arg, void *data, size_t length,
                                          unsigned flags) {
        return UCS_OK;
    }

    static ucs_status_t pending_send_op(uct_pending_req_t *self) {

        pending_send_request_t *req = ucs_container_of(self, pending_send_request_t, uct);
        ucs_status_t status;

        if (req->countdown > 0) {
            --req->countdown;
            return UCS_INPROGRESS;
        }

        status = uct_ep_am_short(req->ep, 0, test_pending_hdr, &req->data,
                                 sizeof(req->data));
        if (status == UCS_OK) {
            delete req;
        }
        return status;
    }

    static ucs_status_t pending_send_op_simple(uct_pending_req_t *self) {

        pending_send_request_t *req = ucs_container_of(self, pending_send_request_t, uct);
        ucs_status_t status;

        status = uct_ep_am_short(req->ep, 0, test_pending_hdr, &req->data,
                                 sizeof(req->data));
        if (status == UCS_OK) {
            req->countdown ++;
            n_pending--;
            req->active = 0;
            //ucs_warn("dispatched %p idx %d total %d", req->ep, req->id, req->countdown);
        }
        return status;
    }

    pending_send_request_t* pending_alloc(uint64_t send_data) {
        pending_send_request_t *req =  new pending_send_request_t();
        req->ep        = m_e1->ep(0);
        req->data      = send_data;
        req->countdown = 5;
        req->uct.func  = pending_send_op;
        return req;
    }

    pending_send_request_t* pending_alloc_simple(uint64_t send_data, int idx) {
        pending_send_request_t *req =  new pending_send_request_t();
        req->ep        = m_e1->ep(idx);
        req->data      = send_data;
        req->countdown = 0;
        req->uct.func  = pending_send_op_simple;
        req->active    = 0;
        req->id        = idx;
        return req;
    }


protected:
    static const uint64_t test_pending_hdr = 0xabcd;
    entity *m_e1, *m_e2;
    static int n_pending;
};

int test_uct_pending::n_pending = 0;

void install_handler_sync_or_async(uct_iface_t *iface, uint8_t id, uct_am_callback_t cb, void *arg)
{
    ucs_status_t status;
    uct_iface_attr_t attr;

    status = uct_iface_query(iface, &attr);
    ASSERT_UCS_OK(status);

    if (attr.cap.flags & UCT_IFACE_FLAG_CB_SYNC) {
        uct_iface_set_am_handler(iface, id, cb, arg, UCT_CB_FLAG_SYNC);
    } else {
        uct_iface_set_am_handler(iface, id, cb, arg, UCT_CB_FLAG_ASYNC);
    }
}

UCS_TEST_P(test_uct_pending, pending_op)
{
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;
    unsigned i, iters, counter = 0;

    initialize();
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    iters = 1000000/ucs::test_time_multiplier();

    /* set a callback for the uct to invoke for receiving the data */
    install_handler_sync_or_async(m_e2->iface(), 0, am_handler, &counter);

    /* send the data until the resources run out */
    i = 0;
    while (i < iters) {
        status = uct_ep_am_short(m_e1->ep(0), 0, test_pending_hdr, &send_data,
                                 sizeof(send_data));
        if (status != UCS_OK) {
            if (status == UCS_ERR_NO_RESOURCE) {

                pending_send_request_t *req = pending_alloc(send_data);

                status = uct_ep_pending_add(m_e1->ep(0), &req->uct);
                if (status != UCS_OK) {
                    /* the request wasn't added to the pending data structure
                     * since resources became available. retry sending this message */
                    delete req;
                } else {
                    /* the request was added to the pending data structure */
                    send_data += 1;
                    i++;
                }
                /* coverity[leaked_storage] */
            } else {
                UCS_TEST_ABORT("Error: " << ucs_status_string(status));
            }
        } else {
            send_data += 1;
            i++;
        }
    }
    /* coverity[loop_condition] */
    while (counter != iters) {
        progress();
    }
    flush();

    ASSERT_EQ(counter, iters);
}

UCS_TEST_P(test_uct_pending, send_ooo_with_pending)
{
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status_send, status_pend = UCS_ERR_LAST;
    ucs_time_t loop_end_limit;
    unsigned i, counter = 0;

    initialize();
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    /* set a callback for the uct to invoke when receiving the data */
    install_handler_sync_or_async(m_e2->iface(), 0, am_handler, &counter);

    loop_end_limit = ucs_get_time() + ucs_time_from_sec(2);
    /* send while resources are available. try to add a request to pending */
    do {
        status_send = uct_ep_am_short(m_e1->ep(0), 0, test_pending_hdr, &send_data,
                                      sizeof(send_data));
        if (status_send == UCS_ERR_NO_RESOURCE) {

            pending_send_request_t *req = pending_alloc(send_data);

            status_pend = uct_ep_pending_add(m_e1->ep(0), &req->uct);
            if (status_pend == UCS_ERR_BUSY) {
                delete req;
            } else {
                /* coverity[leaked_storage] */
                ++send_data;
                break;
            }
        } else {
            ASSERT_UCS_OK(status_send);
            ++send_data;
        }
    } while (ucs_get_time() < loop_end_limit);

    if ((status_send == UCS_OK) || (status_pend == UCS_ERR_BUSY)) {
        /* got here due to reaching the time limit in the above loop.
         * couldn't add a request to pending. all sends were successful. */
        UCS_TEST_MESSAGE << "Can't create out-of-order in the given time.";
        return;
    }
    /* there is one pending request */
    EXPECT_EQ(UCS_OK, status_pend);

    /* progress the receiver a bit to release resources */
    for (i = 0; i < 1000; i++) {
        m_e2->progress();
    }

    /* send a new message. the transport should make sure that this new message
     * isn't sent before the one in pending, thus preventing out-of-order in sending. */
    do {
        status_send = uct_ep_am_short(m_e1->ep(0), 0, test_pending_hdr,
                                      &send_data, sizeof(send_data));
        short_progress_loop();
    } while (status_send == UCS_ERR_NO_RESOURCE);
    ASSERT_UCS_OK(status_send);
    ++send_data;

    /* the receive side checks that the messages were received in order.
     * check the last message here. (counter was raised by one for next iteration) */
    unsigned exp_counter = send_data - 0xdeadbeefUL;
    wait_for_value(&counter, exp_counter, true);
    EXPECT_EQ(exp_counter, counter);
}

UCS_TEST_P(test_uct_pending, pending_fairness)
{
    int N=16;
    int i, iters;
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;
    
    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }

    initialize();
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);
    if (m_e1->iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        N = 128;
    }
    pending_send_request_t *reqs[N];
    install_handler_sync_or_async(m_e2->iface(), 0, am_handler_simple, 0);

    /* idx 0 is setup in initialize(). only need to alloc request */
    reqs[0] = pending_alloc_simple(send_data, 0);
    for (i = 1; i < N; i++) {
        m_e1->connect(i, *m_e2, i);
        reqs[i] = pending_alloc_simple(send_data, i);
    }

    /* give a chance to finish connection for some transports (ud) */
    short_progress_loop();

    n_pending = 0;
    for (iters = 0; iters < 10000; iters++) { 
        /* send until resources of all eps are exausted */
        while (n_pending < N) {
            for (i = 0; i < N; ++i) { /* TODO: change to list */
                if (reqs[i]->active) {
                    continue;
                }
                for (;;) {
                    status = uct_ep_am_short(m_e1->ep(i), 0, test_pending_hdr,
                                             &send_data, sizeof(send_data));
                    if (status == UCS_ERR_NO_RESOURCE) {
                        /* schedule pending */
                        status = uct_ep_pending_add(m_e1->ep(i), &reqs[i]->uct);
                        if (status == UCS_ERR_BUSY) {
                            continue; /* retry */
                        }
                        ASSERT_UCS_OK(status);

                        n_pending++;
                        reqs[i]->active = 1;
                        break;
                    } else {
                        ASSERT_UCS_OK(status);
                        /* sent */
                        reqs[i]->countdown++;
                        break;
                    }
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
        min_sends = ucs_min(min_sends, reqs[i]->countdown);
        max_sends = ucs_max(max_sends, reqs[i]->countdown);
        //printf("%d: send %d\n", i, reqs[i]->countdown);
    }
    UCS_TEST_MESSAGE << " min_sends: " << min_sends 
                     << " max_sends: " << max_sends 
                     << " still pending: " << n_pending;

    while(n_pending > 0) {
       progress();
    }

    flush();

    for (i = 0; i < N; i++) {
        delete reqs[i];
    }

    /* there must be no starvation */
    EXPECT_LT(0, min_sends);
    /* TODO: add stricter fairness criteria */
    if (min_sends < max_sends /2) {
        UCS_TEST_MESSAGE << " CHECK: pending queue is not fair";
    }
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_pending);
