/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
 *
 */
#include <list>
#include <numeric>
#include <set>
#include <vector>
#include <math.h>

#include "ucp_datatype.h"
#include "ucp_test.h"

#define NUM_MESSAGES 17

#define UCP_REALLOC_ID 1000
#define UCP_SEND_ID 0
#define UCP_REPLY_ID 1
#define UCP_RELEASE 1

class test_ucp_am_base : public ucp_test {
public:
    int sent_ams;
    int replies;
    int recv_ams;
    void *reply;
    void *for_release[NUM_MESSAGES];
    int release;

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_AM;
        return params;
    }

    static void ucp_send_am_cb(void *request, ucs_status_t status);

    static ucs_status_t ucp_process_am_cb(void *arg, void *data,
                                          size_t length,
                                          ucp_ep_h reply_ep,
                                          unsigned flags);

    static ucs_status_t ucp_process_reply_cb(void *arg, void *data,
                                             size_t length,
                                             ucp_ep_h reply_ep,
                                             unsigned flags);

    ucs_status_t am_handler(test_ucp_am_base *me, void *data,
                            size_t  length, unsigned flags);
};

ucs_status_t test_ucp_am_base::ucp_process_reply_cb(void *arg, void *data,
                                                    size_t length,
                                                    ucp_ep_h reply_ep,
                                                    unsigned flags)
{
    test_ucp_am_base *self = reinterpret_cast<test_ucp_am_base*>(arg);
    self->replies++;
    return UCS_OK;
}

ucs_status_t test_ucp_am_base::ucp_process_am_cb(void *arg, void *data,
                                                 size_t length,
                                                 ucp_ep_h reply_ep,
                                                 unsigned flags)
{
    test_ucp_am_base *self = reinterpret_cast<test_ucp_am_base*>(arg);
    
    if (reply_ep) {
        self->reply = ucp_am_send_nb(reply_ep, UCP_REPLY_ID, NULL, 1,
                                     ucp_dt_make_contig(0),
                                     (ucp_send_callback_t) ucs_empty_function,
                                     0);
        EXPECT_FALSE(UCS_PTR_IS_ERR(self->reply));
    }
    
    return self->am_handler(self, data, length, flags);
}

ucs_status_t test_ucp_am_base::am_handler(test_ucp_am_base *me, void *data,
                                          size_t length, unsigned flags)
{
    ucs_status_t status;
    std::vector<char> cmp(length, (char)length);
    std::vector<char> databuf(length, 'r');

    memcpy(&databuf[0], data, length);

    EXPECT_EQ(cmp, databuf);
    if (me->release) {
        me->for_release[me->recv_ams] = data;
        status                        = UCS_INPROGRESS;
    } else {
        status = UCS_OK;
    }
    
    me->recv_ams++;
    return status;
}

class test_ucp_am : public test_ucp_am_base {
public:
    ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params = test_ucp_am_base::get_ep_params();
        params.field_mask     |= UCP_EP_PARAM_FIELD_FLAGS;
        params.flags          |= UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
        return params;
    }

    virtual void init() {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

protected:
    void do_set_am_handler_realloc_test();
    void do_send_process_data_test(int test_release, uint16_t am_id,
                                   int send_reply);
    void do_send_process_data_iov_test();
    void set_handlers(uint16_t am_id);
    void set_reply_handlers();
};

void test_ucp_am::set_reply_handlers()
{
    ucp_worker_set_am_handler(sender().worker(), UCP_REPLY_ID,
                              ucp_process_reply_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
    ucp_worker_set_am_handler(receiver().worker(), UCP_REPLY_ID,
                              ucp_process_reply_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
}

void test_ucp_am::set_handlers(uint16_t am_id)
{
    ucp_worker_set_am_handler(sender().worker(), am_id,
                              ucp_process_am_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
    ucp_worker_set_am_handler(receiver().worker(), am_id,
                              ucp_process_am_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
}

void test_ucp_am::do_send_process_data_test(int test_release, uint16_t am_id,
                                            int send_reply)
{
    size_t buf_size          = pow(2, NUM_MESSAGES - 2);
    ucs_status_ptr_t sstatus = NULL;
    std::vector<char> buf(buf_size);

    recv_ams      = 0;
    sent_ams      = 0;
    replies       = 0;
    this->release = test_release;

    for (size_t i = 0; i < buf_size + 1; i = i ? (i * 2) : 1) {
        for (size_t j = 0; j < i; j++) {
            buf[j] = i;
        }

        reply   = NULL;
        sstatus = ucp_am_send_nb(receiver().ep(), am_id,
                                 buf.data(), 1, ucp_dt_make_contig(i),
                                 (ucp_send_callback_t) ucs_empty_function,
                                 send_reply);

        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        sent_ams++;
        
        if (send_reply) {
            while (sent_ams != replies) {
                progress();
            }

            if (reply != NULL) {
                ucp_request_release(reply);
            }
        }
    }

    while (sent_ams != recv_ams) {
        progress();
    }

    if (send_reply) {
        while (sent_ams != replies) {
            progress();
        }
    }

    if (test_release) {
        for(int i = 0; i < recv_ams; i++) {
            if (for_release[i] != NULL) {
                ucp_am_data_release(receiver().worker(), for_release[i]);
            }
        }
    }
}

void test_ucp_am::do_send_process_data_iov_test()
{
    ucs_status_ptr_t sstatus;
    size_t iovcnt = 2;
    size_t size   = 8192;
    size_t index;
    size_t i;

    recv_ams = 0;
    sent_ams = 0;
    release  = 0;

    std::vector<char> b1(size);
    std::vector<char> b2(size);
    ucp_dt_iov_t iovec[iovcnt];

    set_handlers(0);

    for (i = 1; i < size; i *= 2) {
        for (index = 0; index < i; index++) {
            b1[index] = i * 2;
            b2[index] = i * 2;
        }

        iovec[0].buffer = b1.data();
        iovec[1].buffer = b2.data();

        iovec[0].length = i;
        iovec[1].length = i;

        sstatus = ucp_am_send_nb(receiver().ep(), 0,
                                 iovec, 2, ucp_dt_make_iov(),
                                 (ucp_send_callback_t) ucs_empty_function, 0);
        wait(sstatus);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        sent_ams++;
    }

    while (sent_ams != recv_ams) {
        progress();
    }
}

void test_ucp_am::do_set_am_handler_realloc_test()
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(0, UCP_SEND_ID, 0);

    set_handlers(UCP_REALLOC_ID);
    do_send_process_data_test(0, UCP_REALLOC_ID, 0);

    set_handlers(UCP_SEND_ID + 1);
    do_send_process_data_test(0, UCP_SEND_ID + 1, 0);
}

UCS_TEST_P(test_ucp_am, send_process_am) 
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(0, UCP_SEND_ID, 0);

    set_reply_handlers();
    do_send_process_data_test(0, UCP_SEND_ID, UCP_AM_SEND_REPLY);
}

UCS_TEST_P(test_ucp_am, send_process_am_release)
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(UCP_RELEASE, 0, 0);
}

UCS_TEST_P(test_ucp_am, send_process_iov_am) 
{
    do_send_process_data_iov_test();
}

UCS_TEST_P(test_ucp_am, set_am_handler_realloc)
{
    do_set_am_handler_realloc_test();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am)
