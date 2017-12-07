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

    void init()
    {
        // TODO: Remove this when DC TM is enabled by default
        m_env.push_back(new ucs::scoped_setenv("UCX_DC_VERBS_TM_ENABLE", "y"));
        modify_config("TM_OFFLOAD", "y");
        modify_config("TM_THRESH" , "1024");

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

    void send_b_from(entity &se, const void *buffer, size_t count,
                     ucp_datatype_t datatype, ucp_tag_t tag)
    {

        request *req = (request*)ucp_tag_send_nb(se.ep(), buffer, count,
                                                 datatype, tag, send_callback);
        if (UCS_PTR_IS_ERR(req)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(req));
        } else if (req != NULL) {
            wait(req);
            request_free(req);
        }
    }

    void activate_offload(entity &se, ucp_tag_t tag = 0x11)
    {
        uint64_t small_val = 0xFAFA;

        request *req = recv_nb_and_check(&small_val, sizeof(small_val),
                                         DATATYPE, tag, UCP_TAG_MASK_FULL);

        send_b_from(se, &small_val, sizeof(small_val), DATATYPE, tag);
        wait(req);
        request_free(req);
    }

    void req_cancel(entity &e, request *req)
    {
        ucp_request_cancel(e.worker(), req);
        wait(req);
        request_free(req);
    }

protected:
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

UCS_TEST_P(test_ucp_tag_offload, post_after_cancel)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag      = 0x11;
    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

    request *req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                     tag, UCP_TAG_MASK_FULL);

    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag,
                            UCP_TAG_MASK_FULL);

    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);
}

UCS_TEST_P(test_ucp_tag_offload, post_after_comp)
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

UCS_TEST_P(test_ucp_tag_offload, post_wild)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag1     = 0x11; // these two tags should go to different
    ucp_tag_t tag2     = 0x13; // hash buckets in the TM expected queue
    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

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

UCS_TEST_P(test_ucp_tag_offload, post_dif_buckets)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag1     = 0x11; // these two tags should go to different
    ucp_tag_t tag2     = 0x13; // hash buckets in the TM expected queue
    std::vector<request*> reqs;
    request *req;

    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

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


class test_ucp_tag_offload_multi : public test_ucp_tag_offload {
public:

    static ucp_params_t get_ctx_params()
    {
        ucp_params_t params    = test_ucp_tag::get_ctx_params();
        params.field_mask     |= UCP_PARAM_FIELD_TAG_SENDER_MASK;
        params.tag_sender_mask = TAG_SENDER;
        return params;
    }

    void init()
    {
        test_ucp_tag_offload::init();

        // TODO: add more tls which support tag offloading
        std::vector<std::string> tls;
        tls.push_back("rc");
        tls.push_back("dc");
        ucp_test_param params = GetParam();

        for (std::vector<std::string>::const_iterator i = tls.begin();
             i != tls.end(); ++i) {
            params.transports.clear();
            params.transports.push_back(*i);
            create_entity(true, params);
            e(0).connect(&receiver(), get_ep_params());
        }
    }

    ucp_tag_t make_tag(entity &e, ucp_tag_t t)
    {
        uint64_t i;

        for (i = 0; i < m_entities.size(); ++i) {
             if (&m_entities.at(i) == &e) {
                 break;
             }
        }
        return (i << 48) | t;
    }

    void post_recv_and_check(entity &e, unsigned sw_count, ucp_tag_t tag,
                             ucp_tag_t tag_mask)
    {
        std::vector<char> recvbuf(2048, 0);
        request *req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE,
                                         make_tag(e, tag), UCP_TAG_MASK_FULL);

        EXPECT_EQ(sw_count, receiver().worker()->tm.expected.sw_all_count);
        req_cancel(receiver(), req);
    }


protected:
    static const uint64_t TAG_SENDER = 0xFFFFFFFFFFFF0000;
};


UCS_TEST_P(test_ucp_tag_offload_multi, recv_from_multi)
{
    ucp_tag_t tag = 0x11;

    // Activate first offload iface. Tag hashing is not done yet, since we
    // have only one iface so far.
    activate_offload(e(0), make_tag(e(0), tag));
    EXPECT_EQ(0u, kh_size(&receiver().worker()->tm.offload.tag_hash));

    // Activate second offload iface. The tag has been added to the hash.
    // From now requests will be offloaded only for those tags which are
    // in the hash.
    activate_offload(e(1), make_tag(e(1), tag));
    EXPECT_EQ(1u, kh_size(&receiver().worker()->tm.offload.tag_hash));

    // Need to send a message on the first iface again, for its 'tag_sender'
    // part of the tag to be added to the hash.
    activate_offload(e(0), make_tag(e(0), tag));

    // Now requests from first two senders should be always offloaded regardless
    // of the tag value. Tag does not matter, because hashing is done with
    // 'tag & tag_sender_mask' as a key.
    for (int i = 0; i < 2; ++i) {
        post_recv_and_check(e(i), 0u, tag + i, UCP_TAG_MASK_FULL);
    }

    // This request should not be offloaded, because it is sent by the new
    // sender and its 'tag_sender_mask' is not added to the hash yet.
    post_recv_and_check(e(2), 1u, tag, UCP_TAG_MASK_FULL);

    activate_offload(e(2), make_tag(e(2), tag));
    // Check that this sender was added as well
    post_recv_and_check(e(2), 0u, tag + 1, UCP_TAG_MASK_FULL);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tag_offload_multi, rcdc, "\\rc,\\dc,\\ud")
