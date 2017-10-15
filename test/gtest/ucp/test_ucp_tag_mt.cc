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
    virtual void init()
    {
        test_ucp_tag::init();
        ucp_test_param param = GetParam();
    }

    static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls)
    {
        std::vector<ucp_test_param> result;

        generate_test_params_variant(ctx_params, name,
                                     test_case_name, tls, RECV_REQ_INTERNAL,
                                     result, MULTI_THREAD_CONTEXT);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name, tls, RECV_REQ_EXTERNAL,
                                     result, MULTI_THREAD_CONTEXT);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name, tls, RECV_REQ_INTERNAL,
                                     result, MULTI_THREAD_WORKER);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name, tls, RECV_REQ_EXTERNAL,
                                     result, MULTI_THREAD_WORKER);
        return result;
    }

    virtual bool is_external_request()
    {
        return GetParam().variant == RECV_REQ_EXTERNAL;
    }
};

UCS_TEST_P(test_ucp_tag_mt, send_recv) {
    int i;
    uint64_t            send_data[MT_TEST_NUM_THREADS] GTEST_ATTRIBUTE_UNUSED_;
    uint64_t            recv_data[MT_TEST_NUM_THREADS] GTEST_ATTRIBUTE_UNUSED_;
    ucp_tag_recv_info_t info[MT_TEST_NUM_THREADS] GTEST_ATTRIBUTE_UNUSED_;

    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        send_data[i] = 0xdeadbeefdeadbeef + 10 * i;
        recv_data[i] = 0;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        ucs_status_t status;
        int worker_index = 0;

        if (GetParam().thread_type == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        send_b(&(send_data[i]), sizeof(send_data[i]), DATATYPE, 0x111337+i, i);

        short_progress_loop(worker_index); /* Receive messages as unexpected */

        status = recv_b(&(recv_data[i]), sizeof(recv_data[i]), DATATYPE, 0x1337+i,
                        0xffff, &(info[i]), i);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(sizeof(send_data[i]),   info[i].length);
        EXPECT_EQ((ucp_tag_t)(0x111337+i), info[i].sender_tag);
        EXPECT_EQ(send_data[i], recv_data[i]);
    }
#endif
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_mt)
