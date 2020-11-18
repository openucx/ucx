/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <gtest/common/test_perf.h>


class test_ucp_wfe : public ucp_test, public test_perf {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, 0);
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
            if ((err_str.find(ucs_status_string(UCS_ERR_UNREACHABLE)) != std::string::npos) ||
                (err_str.find(ucs_status_string(UCS_ERR_UNSUPPORTED)) != std::string::npos)) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    const static test_spec tests[];
};


enum {
    UCX_PERF_TEST_LAT_NO_WFE,
    UCX_PERF_TEST_LAT_WITH_WFE
};


const test_perf::test_spec test_ucp_wfe::tests[] =
{
    { "put latency", "usec",
      UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
      UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 1000lu,
      ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
      0 },

    { "put latency with WFE", "usec",
      UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG_WFE,
      UCP_PERF_DATATYPE_CONTIG, 0, 1, { 8 }, 1, 1000lu,
      ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6, 0.001, 30.0,
      0 }
};


UCS_TEST_P(test_ucp_wfe, envelope) {
    double perf_value = 0;
    test_spec test;

    /* Run ping-pong with no WFE and get latency reference values */
    test = tests[UCX_PERF_TEST_LAT_NO_WFE];
    test_adjust(test);
    run_test(test, 0, false, "", "", &perf_value);

    /* Run ping-pong with WFE while re-using previous run numbers as a min/max
     * boundary. The latency of the WFE run should stay nearly identical with 250
     * percent margin. When WFE does not work as expected the slow down is
     * typically 10x-100x */
    test     = tests[UCX_PERF_TEST_LAT_WITH_WFE];
    test.max = perf_value * 2.5;
    test.min = perf_value * 0.7;
    run_test(test, 0, true, "", "");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wfe, shm, "shm")
