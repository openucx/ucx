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

#define MB                        pow(1024, -2)
#define UCT_PERF_TEST_MULTIPLIER  5
#define UCT_ARM_PERF_TEST_MULTIPLIER  15
#define UCT_CUDA_PERF_TEST_MULTIPLIER  5

class test_uct_perf : public uct_test, public test_perf {
public:
    void test_execute(unsigned flags, ucs_memory_type_t send_mem_type,
                      ucs_memory_type_t recv_mem_type);
protected:
    const static test_spec tests[];
};

void test_uct_perf::test_execute(unsigned flags = 0,
                                 ucs_memory_type_t send_mem_type =
                                 UCS_MEMORY_TYPE_HOST,
                                 ucs_memory_type_t recv_mem_type =
                                 UCS_MEMORY_TYPE_HOST) {
    if (has_transport("ugni_udt")) {
        UCS_TEST_SKIP;
    }

    /* For SandyBridge CPUs, don't check performance of far-socket devices */
    std::vector<int> cpus = get_affinity();
    bool check_perf       = true;
    size_t max_iter       = std::numeric_limits<size_t>::max();

    if (ucs_arch_get_cpu_model() == UCS_CPU_MODEL_INTEL_SANDYBRIDGE) {
        for (std::vector<int>::iterator iter = cpus.begin();
             iter != cpus.end(); ++iter) {
            if (!ucs_cpu_is_set(*iter, &GetParam()->local_cpus)) {
                UCS_TEST_MESSAGE << "Not enforcing performance on "
                                    "SandyBridge far socket";
                check_perf = false;
                break;
            }
        }
    }

    if (has_transport("tcp")) {
        check_perf = false; /* TODO calibrate expected performance based on transport */
        max_iter   = 1000lu;
    }

    /* Run all tests */
    for (const test_spec *test_iter = tests; test_iter->title != NULL;
         ++test_iter) {
        test_spec test = *test_iter;

        test.send_mem_type = send_mem_type;
        test.recv_mem_type = recv_mem_type;

        if (has_transport("cuda_copy")) {
            test.max *= UCT_CUDA_PERF_TEST_MULTIPLIER;
            test.min /= UCT_CUDA_PERF_TEST_MULTIPLIER;
        }

        if (ucs_arch_get_cpu_model() == UCS_CPU_MODEL_ARM_AARCH64) {
            test.max *= UCT_ARM_PERF_TEST_MULTIPLIER;
            test.min /= UCT_ARM_PERF_TEST_MULTIPLIER;
        } else {
            test.max *= UCT_PERF_TEST_MULTIPLIER;
            test.min /= UCT_PERF_TEST_MULTIPLIER;
        }

        test.iters = ucs_min(test.iters, max_iter);

        run_test(test, flags, check_perf, GetParam()->tl_name,
                 GetParam()->dev_name);
    }
}


const test_perf::test_spec test_uct_perf::tests[] =
{
  { "am short latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 2.5,
    0 },

  { "am short rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.8, 80.0,
    0 },

  { "am short rate64", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 64 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.8, 80.0,
    0 },

  { "am short iov latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT_IOV, 0, 2, { 4, 4 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 2.5,
    0 },

  { "am short iov rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT_IOV, 0, 2, { 4, 4 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.8, 80.0,
    0 },

  { "am short iov rate64", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT_IOV, 0, 2, { 32, 32 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.8, 80.0,
    0 },

  { "am bcopy latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_BCOPY, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 2.5},

  { "am bcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_BCOPY, 0, 1, { 1000 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 15000.0,
    0 },

  { "am zcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 0, 1, { 1000 }, 32, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 600.0, 15000.0,
    0 },

  { "am zcopy bw flush ep", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 0, 1, { 1000 }, 32, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 600.0, 15000.0,
    UCX_PERF_TEST_FLAG_FLUSH_EP },

  { "put latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 1.5,
    0 },

  { "put rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.8, 80.0,
    0 },

  { "put bcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_BCOPY, 0, 1, { 2048 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 50000.0,
    0 },

  { "put zcopy bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 0, 1, { 2048 }, 32, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 620.0, 50000.0,
    0 },

  { "get latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5,
    0 },

  { "atomic add latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5,
    0 },

  { "atomic add rate", "Mpps",
    UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 50.0,
    0 },

  { "atomic fadd latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5,
    0 },

  { "atomic cswap latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5,
    0 },

  { "atomic swap latency", "usec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_SHORT, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.01, 3.5,
    0 },

  { "am iov bw", "MB/sec",
    UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_ZCOPY, 8192, 3, { 256, 256, 512 }, 32, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 600.0, 15000.0,
    0 },

  { NULL }
};


UCS_TEST_P(test_uct_perf, envelope) {
    test_execute();
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_perf);

class test_uct_loopback : public test_uct_perf {
};

UCS_TEST_P(test_uct_loopback, envelope)
{
    test_execute(UCX_PERF_TEST_FLAG_LOOPBACK);
}

UCT_INSTANTIATE_NO_GPU_TEST_CASE(test_uct_loopback);

class test_uct_loopback_cuda : public test_uct_perf {
public:
    const std::vector<std::vector<ucs_memory_type_t>> mem_type_pairs() {
        std::vector<std::vector<ucs_memory_type_t>> result;
        std::vector<ucs_memory_type_t> input = {UCS_MEMORY_TYPE_HOST,
                                                UCS_MEMORY_TYPE_CUDA};

        /* gdr_copy test supports from host to GPU mem case only */
        if (has_transport("gdr_copy")) {
            result.push_back(input);
        } else {
            result = ucs::make_pairs(input);
        }

        return result;
    }
};

UCS_TEST_P(test_uct_loopback_cuda, envelope)
{
    std::vector<std::vector<ucs_memory_type_t>> pairs = mem_type_pairs();

    for (auto pair : pairs) {
        UCS_TEST_MESSAGE << "send mem type: "
                         << ucs_memory_type_names[pair[0]] << " / "
                         << "recv mem type: "
                         << ucs_memory_type_names[pair[1]];
        test_execute(UCX_PERF_TEST_FLAG_LOOPBACK, pair[0], pair[1]);
    }
}

UCT_INSTANTIATE_CUDA_TEST_CASE(test_uct_loopback_cuda);
