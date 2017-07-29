/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <gtest/common/test_perf.h>


#define MB   pow(1024.0, -2)
#define UCP_ARM_PERF_TEST_MULTIPLIER 2
class test_ucp_perf : public ucp_test, public test_perf {
public:
    using ucp_test::get_ctx_params;
protected:
    virtual void init() {
        test_base::init(); /* Skip entities creation in ucp_test */
    }
    static test_spec tests[];
};


test_perf::test_spec test_ucp_perf::tests[] =
{
  { "tag latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "tag iov latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCP_PERF_DATATYPE_IOV, 8192, 3, { 1024, 1024, 1024 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "tag mr", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0,
    0 },

  { "tag sync mr", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0,
    UCX_PERF_TEST_FLAG_TAG_SYNC },

  { "tag wild mr", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0,
    UCX_PERF_TEST_FLAG_TAG_WILDCARD },

  { "put latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "put rate", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0,
    0 },

  { "put bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 2048 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0,
    0 },

  { "get latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "get bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 16384 }, 1, 10000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0,
    0 },

  { "atomic add rate", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 1000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0,
    0 },

  { "atomic fadd latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "atomic swap latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "atomic cswap latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { NULL }
};


UCS_TEST_P(test_ucp_perf, envelope) {
    /* Run all tests */
    std::stringstream ss;
    ss << GetParam();
    bool check_perf = true;
    if (ss.str().find("tcp") != std::string::npos) {
        check_perf = false;
    }

    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv tls("UCX_TLS", ss.str().c_str());
    for (test_spec *test = tests; test->title != NULL; ++test) {
        unsigned flags = (test->command == UCX_PERF_CMD_TAG) ? 0 :
                                 UCX_PERF_TEST_FLAG_ONE_SIDED;
        if (ucs_arch_get_cpu_model() == UCS_CPU_MODEL_ARM_AARCH64) {
            test->max *= UCP_ARM_PERF_TEST_MULTIPLIER;
            test->min /= UCP_ARM_PERF_TEST_MULTIPLIER;
        }
        run_test(*test, flags, check_perf, "", "");
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_perf)
