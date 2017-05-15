/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>


class test_ucp_tag_perf : public test_ucp_tag {
public:
    virtual void init() {
        if (RUNNING_ON_VALGRIND) {
            UCS_TEST_SKIP_R("valgrind");
        }
        test_ucp_tag::init();
    }

protected:
    static const size_t    COUNT    = 10000;
    static const ucp_tag_t TAG_MASK = 0xffffffffffffffffUL;

    void check_perf(ucs_time_t start_time, double exp_time_ns);
    void do_sends();
};

void test_ucp_tag_perf::check_perf(ucs_time_t start_time, double exp_time_ns)
{
    ucs_time_t elapsed = ucs_get_time() - start_time;
    double nsec_per_req = (ucs_time_to_nsec(elapsed) / COUNT);
    UCS_TEST_MESSAGE << nsec_per_req << " ns per request";
    EXPECT_LT(nsec_per_req, exp_time_ns) << "Tag matching is not scalable";
}

void test_ucp_tag_perf::do_sends()
{
    for (int i = (int)COUNT - 1; i >= 0; --i) {
        send_b(NULL, 0, DATATYPE, i);
    }
}

UCS_TEST_P(test_ucp_tag_perf, multi_exp) {
    std::vector<request*> rreqs;

    for (size_t i = 0; i < COUNT; ++i) {
        request *rreq = recv_nb(NULL, 0, DATATYPE, i, TAG_MASK);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq));
        EXPECT_FALSE(rreq->completed);
        rreqs.push_back(rreq);
    }

    ucs_time_t start_time = ucs_get_time();
    do_sends();

    for (size_t i = 0; i < COUNT; ++i) {
        request *rreq = rreqs.back();
        rreqs.pop_back();
        wait_and_validate(rreq);
    }
    check_perf(start_time, 1e5);
}

UCS_TEST_P(test_ucp_tag_perf, multi_unexp) {
    ucp_tag_recv_info_t info;


    send_b(NULL, 0, DATATYPE, 0xdeadbeef);
    do_sends();
    recv_b(NULL, 0, DATATYPE, 0xdeadbeef, TAG_MASK, &info);

    ucs_time_t start_time = ucs_get_time();
    for (size_t i = 0; i < COUNT; ++i) {
        recv_b(NULL, 0, DATATYPE, i, TAG_MASK, &info);
    }
    check_perf(start_time, 1e7);
}

/**
 * Instantiate performance test only on 'self' transport, so network time would
 * not affect the performance. We only care to check SW matching performance.
 */
UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, self,  "\\self")
