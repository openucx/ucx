/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"


class test_ucp_peer_failure : public test_ucp_tag {
public:
    static ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params = test_ucp_tag::get_ep_params();
        params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        params.err_mode    = UCP_ERR_HANDLING_MODE_PEER;
        return params;
    }
};

UCS_TEST_P(test_ucp_peer_failure, disable_sync_send) {
    /* 1GB memory markup takes too long time with valgrind, reduce to 1MB */
    const size_t        max_size = RUNNING_ON_VALGRIND ? (1024 * 1024) :
                                   (1024 * 1024 * 1024);
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

UCP_INSTANTIATE_TEST_CASE(test_ucp_peer_failure)
