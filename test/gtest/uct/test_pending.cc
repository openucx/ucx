/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <ucs/gtest/test.h>
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
        uint64_t          hdr;
        unsigned          buffer;     /* Send buffer */
        size_t            length;     /* Total length, in bytes */
        uct_pending_req_t uct;
    } pending_send_request_t;

    static ucs_status_t pending_am_handler(void *arg, void *data, size_t length,
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

        status = uct_ep_am_short(req->ep, 0, req->hdr, &req->buffer, req->length);
        if (status == UCS_OK) {
            free(req);
        }
        return status;
    }

protected:
    entity *m_e1, *m_e2;
};

UCS_TEST_P(test_uct_pending, pending_op)
{
    uint64_t send_data = 0xdeadbeef;
    uint64_t test_pending_hdr = 0xabcd;
    ucs_status_t status;
    pending_send_request_t *req;
    unsigned i, iters, counter = 0;

    initialize();
    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING);

    iters = 100000/ucs::test_time_multiplier();
    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, pending_am_handler , &counter);

    /* send the data until the resources run out */
    i = 0;
    while (i < iters) {
        status = uct_ep_am_short(m_e1->ep(0), 0, test_pending_hdr, &send_data,
                                 sizeof(send_data));
        if (status != UCS_OK) {
            if (status == UCS_ERR_NO_RESOURCE) {

                req = (pending_send_request_t *) malloc(sizeof(*req));
                req->buffer   = send_data;
                req->hdr      = test_pending_hdr;
                req->length   = sizeof(send_data);
                req->ep       = m_e1->ep(0);
                req->uct.func = pending_send_op;

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

            } else {
                UCS_TEST_ABORT("Error: " << ucs_status_string(status));
            }
        } else {
            send_data += 1;
            i++;
        }
    }

    while (counter != iters) {
        progress();
    }

    ASSERT_EQ(counter, iters);
}

UCT_INSTANTIATE_TEST_CASE(test_uct_pending);
