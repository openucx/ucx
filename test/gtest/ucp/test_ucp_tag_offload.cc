/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>

extern "C" {
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/tag/tag_match.h>
}

class test_ucp_tag_offload : public test_ucp_tag {
public:

    void init() {
        test_ucp_tag::init();
        if (!(sender().ep()->flags & UCP_EP_FLAG_TAG_OFFLOAD_ENABLED)) {
            test_ucp_tag::cleanup();
            UCS_TEST_SKIP_R("no tag offload");
        }
    }

    request* recv_nb_and_check(void *buffer, size_t count, ucp_datatype_t dt,
                               ucp_tag_t tag, ucp_tag_t tag_mask)
    {
        request *req = recv_nb(buffer, count, dt, tag, tag_mask);
        EXPECT_TRUE(!UCS_PTR_IS_ERR(req));
        EXPECT_TRUE(req != NULL);
        return req;
    }

    void req_cancel(entity &e, request *req)
    {
        ucp_request_cancel(e.worker(), req);
        request_free(req);
    }
};

UCS_TEST_P(test_ucp_tag_offload, post_after_cancel, "TM_OFFLOAD=y",
                                                    "TM_THRESH=1024")
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag      = 0x11;
    std::vector<char> recvbuf(2048, 0);

    request *req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                     tag, UCP_TAG_MASK_FULL);

    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag,
                            UCP_TAG_MASK_FULL);

    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);
}

UCS_TEST_P(test_ucp_tag_offload, post_after_comp, "TM_OFFLOAD=y",
                                                  "TM_THRESH=1024")
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag      = 0x11;
    std::vector<char> recvbuf(2048, 0);

    request *req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                     tag, UCP_TAG_MASK_FULL);

    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);

    send_b(&small_val, sizeof(small_val), DATATYPE, 0x11);
    wait(req);
    request_release(req);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag,
                            UCP_TAG_MASK_FULL);

    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);
}

UCS_TEST_P(test_ucp_tag_offload, post_wild, "TM_OFFLOAD=y",
                                            "TM_THRESH=1024")
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag1     = 0x11; // these two tags should go to different
    ucp_tag_t tag2     = 0x13; // hash buckets in the TM expected queue
    std::vector<char> recvbuf(2048, 0);

    request *req1 = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                      tag1, 0);
    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);

    request *req2 = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag2,
                                      UCP_TAG_MASK_FULL);
    // Second request should not be posted as well. Even though it has another
    // tag, the first request is a wildcard, which needs to be handled in SW, 
    // so it blocks all other requests
    EXPECT_EQ(2u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req1);
    req_cancel(receiver(), req2);
}

UCS_TEST_P(test_ucp_tag_offload, post_dif_buckets, "TM_OFFLOAD=y",
                                                   "TM_THRESH=1024")
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag1     = 0x11; // these two tags should go to different
    ucp_tag_t tag2     = 0x13; // hash buckets in the TM expected queue
    std::vector<request*> reqs;
    request *req;

    std::vector<char> recvbuf(2048, 0);

    req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE, tag1,
                            UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag1,
                            UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // The first request was not offloaded due to small size and the second
    // is blocked by the first one.
    EXPECT_EQ(2u, receiver().worker()->tm.expected.sw_all_count);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag2,
                            UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // Check that another request with different tag is offloaded.
    EXPECT_EQ(2u, receiver().worker()->tm.expected.sw_all_count);

    for (std::vector<request*>::const_iterator iter = reqs.begin();
         iter != reqs.end(); ++iter)
    {
        req_cancel(receiver(), *iter);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_offload)
