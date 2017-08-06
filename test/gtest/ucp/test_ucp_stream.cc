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
    }
    
    static void ucp_send_cb(void *request, ucs_status_t status) {}
};

UCS_TEST_P(test_ucp_stream, send_recv_data) {
    std::vector<char> buf(1024, '\0');
    ucs_status_ptr_t status = ucp_stream_send_nb(sender().ep(), buf.data(), 1, UCP_DATATYPE_CONTIG, ucp_send_cb, 0);
    ASSERT_FALSE(UCS_PTR_IS_PTR(status));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream)
