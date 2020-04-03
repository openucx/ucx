/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>

extern "C" {
#include <ucp/tag/tag_match.h>
}

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

// Test that cancelling already matched (but not yet completed) request does
// not produce any error. GH bug #4490.
UCS_TEST_P(test_ucp_tag_cancel, cancel_matched, "RNDV_THRESH=32K") {
    uint64_t small_data = 0;
    ucp_tag_t tag       = 0xfafa;
    size_t size         = 50000;

    std::vector<char> sbuf(size, 0);
    std::vector<char> rbuf(size, 0);

    request *rreq1 = recv_nb(&rbuf[0], rbuf.size(), DATATYPE, tag,
                             UCP_TAG_MASK_FULL);
    request *rreq2 = recv_nb(&small_data, sizeof(small_data), DATATYPE, tag,
                             UCP_TAG_MASK_FULL);

    request *sreq1 = send_nb(&sbuf[0], sbuf.size(), DATATYPE, tag);
    request *sreq2 = send_nb(&small_data, sizeof(small_data), DATATYPE, tag);

    wait_and_validate(rreq2);

    if (!rreq1->completed) {
        ucp_request_cancel(receiver().worker(), rreq1);
    } else {
        UCS_TEST_MESSAGE << "nothing to cancel";
    }

    wait_and_validate(rreq1);
    wait_and_validate(sreq1);
    wait_and_validate(sreq2);
}

UCS_TEST_P(test_ucp_tag_cancel, ucp_worker_tag_cancel) {
    uint64_t recv_data = 0;
    std::vector<std::vector<request *> > reqs(10);
    std::vector<ucp_tag_t> tags(reqs.size());

    /* post 10x10 recvs with 10 different tags */
    for (size_t i = 0; i < reqs.size(); ++i) {
        tags[i] = i;
        reqs[i].push_back(recv_nb(&recv_data, sizeof(recv_data), DATATYPE, i,
                                  std::numeric_limits<ucp_tag_t>::max()));
    }

    /* shuffle tags to cancel them in random order */
    std::random_shuffle(tags.begin(), tags.end());

    for (size_t i = 0; i < tags.size(); ++i) {
        ucp_tag_t cancel_tag = tags[i];

        ucp_tag_recv_cancel_params_t params;
        params.field_mask = UCP_TAG_RECV_CANCEL_PARAMS_FIELD_TAG;
        params.tag        = cancel_tag;

        void *status_ptr  = ucp_worker_tag_recv_cancel_all(receiver().worker(),
                                                           &params);
        ASSERT_EQ(NULL,   status_ptr);
        ASSERT_EQ(UCS_OK, UCS_PTR_STATUS(status_ptr));
        for (size_t j = 0; j < reqs.size(); ++j) {
            for (size_t k = reqs[j].size(); k > 0; --k) {
                const size_t last_idx = k - 1;
                if (j == cancel_tag) {
                    wait(reqs[j][last_idx]);
                    EXPECT_EQ(UCS_ERR_CANCELED, reqs[j][last_idx]->status);
                    EXPECT_EQ(UCS_ERR_CANCELED,
                              ucp_request_check_status(reqs[j][last_idx]));
                    ucp_request_free(reqs[j][last_idx]);
                    reqs[j].erase(reqs[j].begin() + last_idx);
                } else {
                    EXPECT_EQ(UCS_INPROGRESS,
                              ucp_request_check_status(reqs[j][last_idx]));
                }
            }

            if (j == cancel_tag) {
                EXPECT_TRUE(reqs[j].empty());
            }
        }
    }

    for (size_t j = 0; j < reqs.size(); ++j) {
        EXPECT_TRUE(reqs[j].empty());
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_cancel)
