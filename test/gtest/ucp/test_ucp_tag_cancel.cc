/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>


class test_ucp_tag_cancel : public test_ucp_tag {
};

UCS_TEST_P(test_ucp_tag_cancel, cancel_exp) {
    uint64_t recv_data = 0;
    request *req;

    req = recv_nb(&recv_data, sizeof(recv_data), DATATYPE, 1, 1);
    if (UCS_PTR_IS_ERR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    } else if (req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    }

    ucp_request_cancel(receiver().worker(), req);
    wait(req);

    EXPECT_EQ(UCS_ERR_CANCELED, req->status);
    EXPECT_EQ(0ul, recv_data);
    request_release(req);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_cancel)
