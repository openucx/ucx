/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "uct_test.h"

#include <gtest/common/test_perf.h>
extern "C" {
#include <ucs/arch/cpu.h>
}


#define MB   pow(1024.0, -2)

class test_uct_perf : public uct_test, public test_perf {
protected:
    static test_spec tests[];
};


test_perf::test_spec test_uct_perf::tests[] =
{
  { "am latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 2.5},

  { "am rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 1.1, 80.0 },

  { "am rate64", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 64, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 1.1, 80.0 },

  { "am bcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_BCOPY, 2048, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 50000.0 },

  { "am zcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 2048, 32, 100000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 50000.0 },

  { "put latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 1.5 },

  { "put rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 1.5, 80.0 },

  { "put bcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_BCOPY, 2048, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 50000.0 },

  { "put zcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 2048, 32, 100000l,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 50000.0 },

  { "get latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 2.5 },

  { "atomic add latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_PINGPONG,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5 },

  { "atomic add rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 50.0 },

  { "atomic fadd latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5 },

  { "atomic cswap latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5 },

  { "atomic swap latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCT_PERF_DATA_LAYOUT_SHORT, 8, 1, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5 },

  { NULL }
};


UCS_TEST_P(test_uct_perf, envelope) {
    bool check_perf;

    if (GetParam()->tl_name == "cm" || GetParam()->tl_name == "ugni_udt") {
        /* TODO calibrate expected performance and iterations based on transport */
        UCS_TEST_SKIP;
    }

    /* For SandyBridge CPUs, don't check performance of far-socket devices */
    std::vector<int> cpus = get_affinity();
    check_perf = true;
    if (ucs_arch_get_cpu_model() == UCS_CPU_MODEL_INTEL_SANDYBRIDGE) {
        for (std::vector<int>::iterator iter = cpus.begin(); iter != cpus.end(); ++iter) {
            if (!CPU_ISSET(*iter, &GetParam()->local_cpus)) {
                UCS_TEST_MESSAGE << "Not enforcing performance on SandyBridge far socket";
                check_perf = false;
                break;
            }
        }
    }

    /* Run all tests */
    for (test_spec *test = tests; test->title != NULL; ++test) {
        double min, max;
        if (check_perf) {
            min = test->min;
            max = test->max;
        } else {
            min = 0;
            max = INT_MAX;
        }
        run_test(*test, 0, min, max, GetParam()->tl_name, GetParam()->dev_name);
    }
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_perf);
