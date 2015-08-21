/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <gtest/base/test_perf.h>


#define MB   pow(1024.0, -2)

class test_ucp_perf : public ucp_test, public test_perf {
protected:
    virtual uint64_t features() const {
        return 0; /* Unused */
    }

    static test_spec tests[];

};


test_perf::test_spec test_ucp_perf::tests[] =
{
  { "put latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
    UCT_PERF_DATA_LAYOUT_LAST, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 15.0 },

  { "put rate", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_LAST, 8, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0 },

  { "put bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_LAST, 2048, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0 },

  { "get latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_LAST, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 15.0 },

  { "get bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_LAST, 16384, 1, 10000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0 },

  { "atomic add rate", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 1000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0 },

  { "atomic fadd latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 15.0 },

  { "atomic swap latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 15.0 },

  { "atomic cswap latency", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 15.0 },

  { NULL }
};


UCS_TEST_F(test_ucp_perf, envelope) {
    /* Run all tests */
    for (test_spec *test = tests; test->title != NULL; ++test) {
        run_test(*test, UCX_PERF_TEST_FLAG_ONE_SIDED, test->min, test->max, "", "");
    }
}
