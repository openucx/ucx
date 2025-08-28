/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#pragma once

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_tests.h"
#include <tools/perf/gdaki/gdaki_mem.h>

template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, unsigned FLAGS>
class ucp_perf_test_gdaki_runner: public ucp_perf_test_runner_base_psn<uint64_t> {
public:
    using psn_t = uint64_t;

    ucp_perf_test_gdaki_runner(ucx_perf_context_t &perf) :
        ucp_perf_test_runner_base_psn<uint64_t>(perf)
    {
        ucs_status_t status;

        status = gdaki_mem_create(&m_gdaki_mem, sizeof(ucx_perf_context_t));
        if (status != UCS_OK) {
            ucs_fatal("Failed to create GDAKI memory: %s",
                      ucs_status_string(status));
        }

        memcpy(m_gdaki_mem.cpu_ptr, &perf, sizeof(ucx_perf_context_t));
    }

    ucs_status_t run()
    {
        /* coverity[switch_selector_expr_is_constant] */
        switch (TYPE) {
        case UCX_PERF_TEST_TYPE_PINGPONG:
            switch (CMD) {
            case UCX_PERF_CMD_PUT_MULTI:
                return run_pingpong_batch_gdaki();
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_TEST_TYPE_STREAM_UNI:
            switch (CMD) {
            case UCX_PERF_CMD_PUT_MULTI:
                return run_stream_req_uni_batch_gdaki();
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

private:
    gdaki_mem_t m_gdaki_mem;

    ucs_status_t run_pingpong_batch_gdaki()
    {
        return UCS_OK;
    }

    ucs_status_t run_stream_req_uni_batch_gdaki()
    {
        return UCS_OK;
    }
};
