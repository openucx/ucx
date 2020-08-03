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

#include <common/test.h>

#include "ucp_datatype.h"
#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_ep.inl>
}

#define NUM_MESSAGES 17

#define UCP_REALLOC_ID 1000
#define UCP_SEND_ID 0
#define UCP_REPLY_ID 1
#define UCP_RELEASE 1

class test_ucp_am_base : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_AM;
        return params;
    }

    virtual void init() {
        modify_config("MAX_EAGER_LANES", "2");

        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }
};

class test_ucp_am : public test_ucp_am_base {
public:
    int sent_ams;
    int replies;
    int recv_ams;
    void *reply;
    void *for_release[NUM_MESSAGES];
    int release;

    static ucs_status_t ucp_process_am_cb(void *arg, void *data,
                                          size_t length,
                                          ucp_ep_h reply_ep,
                                          unsigned flags);

    static ucs_status_t ucp_process_reply_cb(void *arg, void *data,
                                             size_t length,
                                             ucp_ep_h reply_ep,
                                             unsigned flags);

    ucs_status_t am_handler(test_ucp_am *me, void *data,
                            size_t  length, unsigned flags);

protected:
    void do_set_am_handler_realloc_test();
    void do_send_process_data_test(int test_release, uint16_t am_id,
                                   int send_reply);
    void do_send_process_data_iov_test(size_t size);
    void set_handlers(uint16_t am_id);
    void set_reply_handlers();
};

ucs_status_t test_ucp_am::ucp_process_reply_cb(void *arg, void *data,
                                               size_t length,
                                               ucp_ep_h reply_ep,
                                               unsigned flags)
{
    test_ucp_am *self = reinterpret_cast<test_ucp_am*>(arg);
    self->replies++;
    return UCS_OK;
}

ucs_status_t test_ucp_am::ucp_process_am_cb(void *arg, void *data,
                                            size_t length,
                                            ucp_ep_h reply_ep,
                                            unsigned flags)
{
    test_ucp_am *self = reinterpret_cast<test_ucp_am*>(arg);

    if (reply_ep) {
        self->reply = ucp_am_send_nb(reply_ep, UCP_REPLY_ID, NULL, 1,
                                     ucp_dt_make_contig(0),
                                     (ucp_send_callback_t) ucs_empty_function,
                                     0);
        EXPECT_FALSE(UCS_PTR_IS_ERR(self->reply));
    }

    return self->am_handler(self, data, length, flags);
}

ucs_status_t test_ucp_am::am_handler(test_ucp_am *me, void *data,
                                     size_t length, unsigned flags)
{
    ucs_status_t status;
    std::vector<char> cmp(length, (char)length);
    std::vector<char> databuf(length, 'r');

    memcpy(&databuf[0], data, length);

    EXPECT_EQ(cmp, databuf);

    bool has_desc = flags & UCP_CB_PARAM_FLAG_DATA;
    if (me->release) {
        me->for_release[me->recv_ams] = has_desc ? data : NULL;
        status                        = has_desc ? UCS_INPROGRESS : UCS_OK;
    } else {
        status                        = UCS_OK;
    }

    me->recv_ams++;
    return status;
}


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
        sstatus = ucp_am_send_nb(sender().ep(), am_id,
                                 buf.data(), 1, ucp_dt_make_contig(i),
                                 (ucp_send_callback_t) ucs_empty_function,
                                 send_reply);

        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        request_wait(sstatus);
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

void test_ucp_am::do_send_process_data_iov_test(size_t size)
{
    ucs_status_ptr_t sstatus;
    size_t index;
    size_t i;

    recv_ams = 0;
    sent_ams = 0;
    release  = 0;

    const size_t iovcnt = 2;
    std::vector<char> sendbuf(size * iovcnt, 0);

    ucs::fill_random(sendbuf);

    set_handlers(0);

    for (i = 1; i < size; i *= 2) {
        for (size_t iov_it = 0; iov_it < iovcnt; iov_it++) {
            for (index = 0; index < i; index++) {
                sendbuf[(iov_it * i) + index] = i * 2;
            }
        }

        ucp::data_type_desc_t send_dt_desc(DATATYPE_IOV, sendbuf.data(),
                                           i * iovcnt, iovcnt);

        sstatus = ucp_am_send_nb(sender().ep(), 0,
                                 send_dt_desc.buf(), iovcnt, DATATYPE_IOV,
                                 (ucp_send_callback_t) ucs_empty_function, 0);
        request_wait(sstatus);
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
    ucs::detail::message_stream ms("INFO");

    for (unsigned i = 1; i <= 7; ++i) {
        size_t max = (long)pow(10.0, i);
        long count = ucs_max((long)(5000.0 / sqrt(max) /
                                    ucs::test_time_multiplier()), 3);
        ms << count << "x10^" << i << " " << std::flush;
        for (long j = 0; j < count; ++j) {
            size_t size = ucs::rand() % max + 1;
            do_send_process_data_iov_test(size);
        }
    }
}

UCS_TEST_P(test_ucp_am, set_am_handler_realloc)
{
    do_set_am_handler_realloc_test();
}

UCS_TEST_P(test_ucp_am, max_am_header)
{
    size_t min_am_bcopy       = std::numeric_limits<size_t>::max();
    bool has_tl_with_am_bcopy = false;

    for (ucp_rsc_index_t idx = 0; idx < sender().ucph()->num_tls; ++idx) {
        uct_iface_attr_t *attr = ucp_worker_iface_get_attr(sender().worker(), idx);
        if (attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
            min_am_bcopy = ucs_min(min_am_bcopy, attr->cap.am.max_bcopy);
            has_tl_with_am_bcopy = true;
        }
    }

    EXPECT_TRUE(has_tl_with_am_bcopy);

    ucp_worker_attr_t attr;
    attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

    ASSERT_UCS_OK(ucp_worker_query(sender().worker(), &attr));

    EXPECT_GE(attr.max_am_header, 64ul);
    EXPECT_LT(attr.max_am_header, min_am_bcopy);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am)


class test_ucp_am_nbx : public test_ucp_am_base {
    // To be extended in the following PRs
};

UCS_TEST_P(test_ucp_am_nbx, set_invalid_handler)
{
    ucp_am_handler_param_t params;

    params.id           = 0;
    params.cb           = NULL;
    params.field_mask   = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_CB;
    ucs_status_t status = ucp_worker_set_am_recv_handler(sender().worker(),
                                                         &params);
    EXPECT_UCS_OK(status);

    // Check that error is returned if id or callback is not set
    scoped_log_handler wrap_err(wrap_errors_logger);

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_CB;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    params.field_mask = 0ul;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    // Check that error is returned if private flag is requested by the user
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    params.flags      = UCP_AM_CB_PRIV_FLAG_NBX;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx)
