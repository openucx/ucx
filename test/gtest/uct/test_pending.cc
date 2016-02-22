/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_uct_pending : public uct_test {
public:
    void initialize() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_e2 = uct_test::create_entity(0);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);
    }

    typedef struct pending_send_request {
        uct_ep_h          ep;
        uint64_t          data;
        int               countdown;  /* Actually send after X calls */
        uct_pending_req_t uct;
    } pending_send_request_t;

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   void *desc) {

        unsigned *counter = (unsigned *) arg;
        uint64_t test_hdr = *(uint64_t *) data;
        uint64_t actual_data = *(unsigned*)((char*)data + sizeof(test_hdr));

        if ((test_hdr == 0xabcd) && (actual_data == (0xdeadbeef + *counter))) {
            (*counter)++;
        } else {
            UCS_TEST_ABORT("Error in comparison in pending_am_handler. Counter: " << counter);
        }

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

    pending_send_request_t* pending_alloc(uint64_t send_data) {
        pending_send_request_t *req =  new pending_send_request_t();
        req->ep        = m_e1->ep(0);
        req->data      = send_data;
        req->countdown = 5;
        req->uct.func  = pending_send_op;
        return req;
    }

protected:
    static const uint64_t test_pending_hdr = 0xabcd;
    entity *m_e1, *m_e2;
};


UCS_TEST_P(test_uct_pending, pending_op)
{
    uint64_t send_data = 0xdeadbeef;
    ucs_status_t status;
    unsigned i, iters, counter = 0;

    initialize();
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    iters = 1000000/ucs::test_time_multiplier();
    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler , &counter, UCT_AM_CB_FLAG_SYNC);

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
                    free(req);
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
    uct_iface_set_am_handler(m_e2->iface(), 0, am_handler , &counter, UCT_AM_CB_FLAG_SYNC);

    loop_end_limit = ucs_get_time() + ucs_time_from_sec(2);
    /* send while resources are available. try to add a request to pending */
    do {
        status_send = uct_ep_am_short(m_e1->ep(0), 0, test_pending_hdr, &send_data,
                                      sizeof(send_data));
        if (status_send == UCS_ERR_NO_RESOURCE) {

            pending_send_request_t *req = pending_alloc(send_data);

            status_pend = uct_ep_pending_add(m_e1->ep(0), &req->uct);
            if (status_pend == UCS_ERR_BUSY) {
                free(req);
            } else {
                /* coverity[leaked_storage] */
                break;
            }
        } else {
            send_data += 1;
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
    send_data += 1;
    do {
        status_send = uct_ep_am_short(m_e1->ep(0), 0, test_pending_hdr,
                                      &send_data, sizeof(send_data));
        short_progress_loop();
    } while (status_send == UCS_ERR_NO_RESOURCE);

    /* the receive side checks that the messages were received in order.
     * check the last message here. (counter was raised by one for next iteration) */
    EXPECT_EQ(send_data, 0xdeadbeef + counter - 1);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_pending);
