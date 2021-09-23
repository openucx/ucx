/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>

#if _OPENMP
#include "omp.h"
#endif

using namespace ucs; /* For vector<char> serialization */


class test_ucp_tag_mt : public test_ucp_tag {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant_with_value(variants, get_ctx_params(), RECV_REQ_INTERNAL,
                               "req_int,mt_context", MULTI_THREAD_CONTEXT);
        add_variant_with_value(variants, get_ctx_params(), RECV_REQ_EXTERNAL,
                               "req_ext,mt_context", MULTI_THREAD_CONTEXT);

        add_variant_with_value(variants, get_ctx_params(), RECV_REQ_INTERNAL,
                               "req_int,mt_worker", MULTI_THREAD_WORKER);
        add_variant_with_value(variants, get_ctx_params(), RECV_REQ_EXTERNAL,
                               "req_ext,mt_worker", MULTI_THREAD_WORKER);
    }

    virtual bool is_external_request()
    {
        return get_variant_value() == RECV_REQ_EXTERNAL;
    }
};

UCS_TEST_P(test_ucp_tag_mt, send_recv) {
    const unsigned num_threads = mt_num_threads();
    uint64_t send_data[num_threads] GTEST_ATTRIBUTE_UNUSED_;
    uint64_t recv_data[num_threads] GTEST_ATTRIBUTE_UNUSED_;
    ucp_tag_recv_info_t info[num_threads] GTEST_ATTRIBUTE_UNUSED_;

    for (int i = 0; i < num_threads; i++) {
        send_data[i] = 0xdeadbeefdeadbeef + 10 * i;
        recv_data[i] = 0;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ucs_status_t status;
        int worker_index = 0;

        if (get_variant_thread_type() == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        send_b(&(send_data[i]), sizeof(send_data[i]), DATATYPE, 0x111337+i,
               NULL, i);

        short_progress_loop(worker_index); /* Receive messages as unexpected */

        status = recv_b(&(recv_data[i]), sizeof(recv_data[i]), DATATYPE, 0x1337+i,
                        0xffff, &(info[i]), NULL, i);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(sizeof(send_data[i]),   info[i].length);
        EXPECT_EQ((ucp_tag_t)(0x111337+i), info[i].sender_tag);
        EXPECT_EQ(send_data[i], recv_data[i]);
    }
#endif
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_mt)
