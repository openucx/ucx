/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_perf.h>
#include <ucp/core/ucp_types.h>


#define MB   pow(1024.0, -2)
#define UCT_PERF_TEST_MULTIPLIER  5
#define UCT_ARM_PERF_TEST_MULTIPLIER  15

class test_ucp_perf : public ucp_test, public test_perf {
public:
    enum {
        VARIANT_TEST_TYPE,
        VARIANT_ATOMIC_MODE
    };

    enum {
        ATOMIC_CPU = 1,
        ATOMIC_DEVICE
    };

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        ucp_test_variant* variant;

        for (int i = 0; i < tests_num; i++) {
            const test_spec *test = &tests[i];

            if ((test->command == UCX_PERF_CMD_ADD) ||
                (test->command == UCX_PERF_CMD_FADD) ||
                (test->command == UCX_PERF_CMD_SWAP) ||
                (test->command == UCX_PERF_CMD_CSWAP)) {
                variant = &add_variant(variants, 0);
                add_variant_value(variant->values, i, test->title);
                add_variant_value(variant->values, ATOMIC_CPU, "cpu");

                variant = &add_variant(variants, 0);
                add_variant_value(variant->values, i, test->title);
                add_variant_value(variant->values, ATOMIC_DEVICE, "device");
            } else {
                add_variant_with_value(variants, 0, i, test->title);
            }
        }
    }

protected:
    virtual void init() {
        test_base::init(); /* Skip entities creation in ucp_test */
        ucs_log_push_handler(log_handler);
    }

    virtual void cleanup() {
        ucs_log_pop_handler();
        test_base::cleanup();
    }

    static ucs_log_func_rc_t
    log_handler(const char *file, unsigned line, const char *function,
                ucs_log_level_t level,
                const ucs_log_component_config_t *comp_conf,
                const char *message, va_list ap) {
        // Ignore errors that transport cannot reach peer
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);
            if (strstr(err_str.c_str(), ucs_status_string(UCS_ERR_UNREACHABLE)) ||
                strstr(err_str.c_str(), ucs_status_string(UCS_ERR_UNSUPPORTED)) ||
                strstr(err_str.c_str(), "no peer failure handler")) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    const static test_spec tests[];
    const static size_t tests_num;
};


const test_perf::test_spec test_ucp_perf::tests[] =
{
  { "tag0_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 0 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "tag_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "tag_lat_errh", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    UCX_PERF_TEST_FLAG_ERR_HANDLING },

  { "tag_lat_b", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_SLEEP,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "tag_lat_iov", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_IOV, 8192, 3, { 1024, 1024, 1024 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "tag_mr", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.1, 100.0,
    0 },

  { "tag_mr_b", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_SLEEP,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.1, 100.0,
    0 },

  { "tag_mr_sync", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 200000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.05, 100.0, 0},

  { "tag_mr_wild", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.1, 100.0,
    UCX_PERF_TEST_FLAG_TAG_WILDCARD },

  { "tag_bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_LAST, 0, 1, { 2048 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 100.0, 100000.0 },

  { "tag_bw_b", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_SLEEP,
    UCT_PERF_DATA_LAYOUT_LAST, 0, 1, { 2048 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 100.0, 100000.0 },

  { "tag_bw_zcopy", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_LAST, 0, 1, { 2048 }, 16, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 100.0, 100000.0 },

  { "put_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "put_rate", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.5, 100.0,
    0 },

  { "put_bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 2048 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0,
    0 },

  { "get_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "get_bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 16384 }, 1, 10000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0,
    0 },

  { "str_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0, 0 },

  { "str_bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 16384 }, 1, 10000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0, 0 },

  { "str_recv_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    UCX_PERF_TEST_FLAG_STREAM_RECV_DATA },

  { "str_recv_lat", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 16384 }, 1, 10000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 200.0, 100000.0,
    UCX_PERF_TEST_FLAG_STREAM_RECV_DATA },

  { "amo_add", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 1000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.1, 500.0,
    0 },

  { "amo_fadd", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "amo_swap", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "amo_cswap", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
    0 },

  { "am0_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 0 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "am_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "am_lat_b", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_SLEEP,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "am_iov_lat", "usec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_IOV, 8192, 3, { 1024, 1024, 1024 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 60.0,
    0 },

  { "am_mr", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.1, 100.0,
    0 },

  { "am_mr_b", "Mpps",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_SLEEP,
    UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 2000000lu,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6, 0.1, 100.0,
    0 },

  { "am_bw", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_POLL,
    UCT_PERF_DATA_LAYOUT_LAST, 0, 1, { 2048 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 100.0, 100000.0 },

  { "am_bw_b", "MB/sec",
    UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_WAIT_MODE_SLEEP,
    UCT_PERF_DATA_LAYOUT_LAST, 0, 1, { 2048 }, 1, 100000lu,
    ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), MB, 100.0, 100000.0 },
};

const size_t test_ucp_perf::tests_num = ucs_static_array_size(test_ucp_perf::tests);


UCS_TEST_SKIP_COND_P(test_ucp_perf, envelope, has_transport("self"))
{
    bool check_perf = true;
    size_t max_iter = std::numeric_limits<size_t>::max();

    if (has_transport("tcp")) {
        check_perf = false;
        max_iter   = 1000lu;
    }

    std::stringstream ss;
    ss << GetParam().transports;
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv tls("UCX_TLS", ss.str().c_str());
    ucs::scoped_setenv warn_invalid("UCX_WARN_INVALID_CONFIG", "no");
    const char* atomic_mode_str = "guess";

    if (get_variant_value(VARIANT_ATOMIC_MODE) == ATOMIC_CPU) {
        atomic_mode_str = "cpu";
    } else if (get_variant_value(VARIANT_ATOMIC_MODE) == ATOMIC_DEVICE) {
        atomic_mode_str = "device";
    }

    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv atomic_mode("UCX_ATOMIC_MODE", atomic_mode_str);

    test_spec test = tests[get_variant_value(VARIANT_TEST_TYPE)];

    if (ucs_arch_get_cpu_model() == UCS_CPU_MODEL_ARM_AARCH64) {
        test.max *= UCT_ARM_PERF_TEST_MULTIPLIER;
        test.min /= UCT_ARM_PERF_TEST_MULTIPLIER;
    } else {
        test.max *= UCT_PERF_TEST_MULTIPLIER;
        test.min /= UCT_PERF_TEST_MULTIPLIER;
    }
    test.iters = ucs_min(test.iters, max_iter);

    test.send_mem_type = UCS_MEMORY_TYPE_HOST;
    test.recv_mem_type = UCS_MEMORY_TYPE_HOST;

    run_test(test, 0, check_perf, "", "");
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_perf)

class test_ucp_loopback : public test_ucp_perf {};

UCS_TEST_P(test_ucp_loopback, envelope)
{
    test_spec test = tests[get_variant_value(VARIANT_TEST_TYPE)];

    test.send_mem_type = UCS_MEMORY_TYPE_HOST;
    test.recv_mem_type = UCS_MEMORY_TYPE_HOST;

    run_test(test, UCX_PERF_TEST_FLAG_LOOPBACK, true, "", "");
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_loopback)


class test_ucp_wait_mem : public test_ucp_perf {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        // Add test instance only if wait_mem is non-trivial for this arch
#ifdef __aarch64__
        add_variant(variants, 0);
#else
        ucs_assert(ucs_arch_wait_mem == ucs_arch_generic_wait_mem);
#endif
    }
};

UCS_TEST_P(test_ucp_wait_mem, envelope) {
    double perf_avg    = 0;
    double perf_min    = std::numeric_limits<double>::max();
    double perf_iter   = 0;
    const int max_iter = ucs_max(ucs::perf_retry_count, 1);
    int i;

    /* Run ping-pong with no WFE and get latency reference values */
    const test_spec test1 = { "put latency reference", "usec",
                              UCX_PERF_API_UCP, UCX_PERF_CMD_PUT,
                              UCX_PERF_TEST_TYPE_PINGPONG,
                              UCX_PERF_WAIT_MODE_POLL,
                              UCP_PERF_DATATYPE_CONTIG,
                              0, 1, { 8 }, 1, 1000lu,
                              ucs_offsetof(ucx_perf_result_t,
                                           latency.total_average),
                              1e6, 0.001, 30.0, 0,
                              UCS_MEMORY_TYPE_HOST,
                              UCS_MEMORY_TYPE_HOST };
    for (i = 0; i < max_iter; i++) {
        perf_iter = run_test(test1, 0, false, "", "");
        perf_avg += perf_iter;
        perf_min  = std::min(perf_min, perf_iter);
    }
    perf_avg /= max_iter;

    /* Run ping-pong with WFE while re-using previous run numbers as
     * a min/max boundary. The latency of the WFE run should stay nearly
     * identical with 200 percent margin. When WFE does not work as expected
     * the slow down is typically 10x-100x */
    const test_spec test2 = { "put latency with ucp_worker_wait_mem()",
                              "usec", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT,
                              UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM,
                              UCX_PERF_WAIT_MODE_POLL,
                              UCP_PERF_DATATYPE_CONTIG,
                              0, 1, { 8 }, 1, 1000lu,
                              ucs_offsetof(ucx_perf_result_t,
                                           latency.total_average),
                              1e6, perf_min * 0.3, perf_avg * 3, 0,
                              UCS_MEMORY_TYPE_HOST,
                              UCS_MEMORY_TYPE_HOST };
    run_test(test2, 0, true, "", "");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wait_mem, shm, "shm")
