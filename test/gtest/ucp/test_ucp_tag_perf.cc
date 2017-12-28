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
    static const size_t    COUNT    = 8192;
    static const ucp_tag_t TAG_MASK = 0xffffffffffffffffUL;

    double check_perf(size_t count, bool is_exp);
    void check_scalability(double max_growth, bool is_exp);
    void do_sends(size_t count);
};

double test_ucp_tag_perf::check_perf(size_t count, bool is_exp)
{
    ucs_time_t start_time;

    if (is_exp) {
        std::vector<request*> rreqs;

        for (size_t i = 0; i < count; ++i) {
            request *rreq = recv_nb(NULL, 0, DATATYPE, i, TAG_MASK);
            ucs_assert(!UCS_PTR_IS_ERR(rreq));
            EXPECT_FALSE(rreq->completed);
            rreqs.push_back(rreq);
        }

        start_time = ucs_get_time();
        do_sends(count);
        while (!rreqs.empty()) {
            request *rreq = rreqs.back();
            rreqs.pop_back();
            wait_and_validate(rreq);
        }
    } else {
        ucp_tag_recv_info_t info;

        send_b(NULL, 0, DATATYPE, 0xdeadbeef);
        do_sends(count);
        recv_b(NULL, 0, DATATYPE, 0xdeadbeef, TAG_MASK, &info);

        start_time = ucs_get_time();
        for (size_t i = 0; i < count; ++i) {
            recv_b(NULL, 0, DATATYPE, i, TAG_MASK, &info);
        }
    }

    return ucs_time_to_sec(ucs_get_time() - start_time) / count;
}

void test_ucp_tag_perf::do_sends(size_t count)
{
    size_t i = count;
    while (i > 0) {
        --i;
        send_b(NULL, 0, DATATYPE, i);
    }
}

void test_ucp_tag_perf::check_scalability(double max_growth, bool is_exp)
{
    double prev_time = 0.0, total_growth = 0.0, avg_growth;
    size_t n = 0;

    for (int i = 0; i < (ucs::perf_retry_count + 1); ++i) {

        /* Estimate by how much the tag matching time grows when the matching queue
         * length grows by 2x. A result close to 1.0 means O(1) scalability (which
         * is good), while a result of 2.0 or higher means O(n) or higher.
         */
        for (size_t count = 1; count <= COUNT; count *= 2) {
            size_t iters = 10 * ucs_max(1ul, COUNT / count);
            double total_time = 0;
            for (size_t i = 0; i < iters; ++i) {
                total_time += check_perf(count, is_exp);
            }

            double time = total_time / iters;
            if (count >= 16) {
                /* don't measure first few iterations - warmup */
                total_growth += (time / prev_time);
                ++n;
            }
            prev_time = time;
        }

        avg_growth = total_growth / n;
        UCS_TEST_MESSAGE << "Average growth: " << avg_growth;

        if (!ucs::perf_retry_count) {
            UCS_TEST_MESSAGE << "not validating performance";
            return; /* Skip */
        } else if (avg_growth < max_growth) {
            return; /* Success */
        } else {
            ucs::safe_sleep(ucs::perf_retry_interval);
        }
    }

    ADD_FAILURE() << "Tag matching is not scalable";
}

UCS_TEST_P(test_ucp_tag_perf, multi_exp) {
    check_scalability(1.5, true);
}

UCS_TEST_P(test_ucp_tag_perf, multi_unexp) {
    check_scalability(1.5, false);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_perf)
