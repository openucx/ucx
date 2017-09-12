/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"


class test_ucp_stream : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_STREAM;
        return params;
    }

    virtual void init() {
        ucp_test::init();

        sender().connect(&receiver());
        if (!is_loopback()) {
            receiver().connect(&sender());
        }
    }

    static void ucp_send_cb(void *request, ucs_status_t status) {}
};

UCS_TEST_P(test_ucp_stream, send_recv_data) {
    std::vector<char> sbuf(1024, 's');
    size_t            ssize = 1;
    std::vector<char> rbuf(1024, 'r');
    size_t            roffset = 0;

    ASSERT_LE(ssize, sbuf.size());
    ASSERT_LE(ssize, rbuf.size());

    ucs_status_ptr_t sstatus = ucp_stream_send_nb(sender().ep(), sbuf.data(),
                                                  ssize, ucp_dt_make_contig(1),
                                                  ucp_send_cb, 0);
    EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
    wait(sstatus);

    ucs_status_ptr_t rdata;
    size_t length;
    do {
        progress();
        rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
        if (UCS_PTR_STATUS(rdata) == UCS_OK) {
            continue;
        }

        memcpy(&rbuf[roffset], rdata, length);
        roffset += length;
        ucp_stream_data_release(receiver().ep(), rdata);
    } while (roffset < ssize);

    EXPECT_EQ(roffset, ssize);
    EXPECT_EQ(std::vector<char>(sbuf.begin(), sbuf.begin() + ssize),
              std::vector<char>(rbuf.begin(), rbuf.begin() + roffset));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream)
