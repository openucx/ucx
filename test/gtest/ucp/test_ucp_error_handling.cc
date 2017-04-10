/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"


class test_ucp_error_handling : public test_ucp_tag {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_tag::get_ctx_params();
        params.features |= UCP_FEATURE_FAULT_TOLERANCE;
        return params;
    }
};

UCS_TEST_P(test_ucp_error_handling, disable_sync_send) {

    const size_t        max_size = 1024 * 1024 * 1024;
    std::vector<char>   buf(max_size, 0);
    request             *req;

    /* Make sure API is disabled for any size and data type */
    for (size_t size = 1; size <= max_size; size *= 2) {
        req = send_sync_nb(buf.data(), size, DATATYPE, 0x111337);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));

        UCS_TEST_GET_BUFFER_DT_IOV(iov_, iov_cnt_, buf.data(), size, 40ul);
        req = send_sync_nb(iov_, iov_cnt_, DATATYPE_IOV, 0x111337);
        EXPECT_FALSE(UCS_PTR_IS_PTR(req));
        EXPECT_EQ(UCS_ERR_UNSUPPORTED, UCS_PTR_STATUS(req));
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_error_handling)
