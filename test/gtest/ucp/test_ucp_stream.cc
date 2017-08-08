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

        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
    }

    static void ucp_send_cb(void *request, ucs_status_t status) {}
};

UCS_TEST_P(test_ucp_stream, send_recv_data) {
    std::vector<char> sbuf(size_t(16)*1024*1024, 's');
    size_t            ssize = 0; /* total send size */

    /* send all msg sizes*/
    for (size_t i = 3; i < sbuf.size(); i *= 2) {
        ucs_status_ptr_t sstatus = ucp_stream_send_nb(sender().ep(), sbuf.data(),
                                                      i, ucp_dt_make_contig(1),
                                                      ucp_send_cb, 0);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        ssize += i;
    }

    std::vector<char> rbuf(ssize, 'r');
    size_t            roffset = 0;
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
    sbuf.resize(ssize, 's');
    EXPECT_EQ(sbuf, rbuf);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream)
