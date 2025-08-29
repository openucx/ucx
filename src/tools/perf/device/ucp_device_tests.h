/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#pragma once

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/ucp_tests.h>
#include "device_mem.h"

template <ucx_perf_cmd_t CMD, ucx_perf_test_type_t TYPE, unsigned FLAGS>
class ucp_perf_test_device_runner: public ucp_perf_test_runner_base_psn<uint64_t> {
public:
    using psn_t = uint64_t;

    ucp_perf_test_device_runner(ucx_perf_context_t &perf) :
        ucp_perf_test_runner_base_psn<uint64_t>(perf)
    {
        ucs_status_t status;

        status = device_mem_create(&m_device_mem, sizeof(ucx_perf_context_t));
        if (status != UCS_OK) {
            ucs_fatal("Failed to create GDAKI memory: %s",
                      ucs_status_string(status));
        }

        m_gpu_ctx = static_cast<ucx_perf_context_t*>(m_device_mem.gpu_ptr);
        m_cpu_ctx = static_cast<ucx_perf_context_t*>(m_device_mem.cpu_ptr);
        memcpy(m_cpu_ctx, &perf, sizeof(ucx_perf_context_t));
    }

    ~ucp_perf_test_device_runner()
    {
        device_mem_destroy(&m_device_mem);
    }

    ucs_status_t run()
    {
        /* coverity[switch_selector_expr_is_constant] */
        switch (TYPE) {
        case UCX_PERF_TEST_TYPE_PINGPONG:
            /* coverity[switch_selector_expr_is_constant] */
            switch (CMD) {
            case UCX_PERF_CMD_PUT_MULTI:
                return run_pingpong_batch_device();
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        case UCX_PERF_TEST_TYPE_STREAM_UNI:
            /* coverity[switch_selector_expr_is_constant] */
            switch (CMD) {
            case UCX_PERF_CMD_PUT_MULTI:
                return run_stream_req_uni_batch_device();
            default:
                return UCS_ERR_INVALID_PARAM;
            }
        default:
            return UCS_ERR_INVALID_PARAM;
        }
    }

private:
    device_mem_t       m_device_mem;
    ucx_perf_context_t *m_cpu_ctx;
    ucx_perf_context_t *m_gpu_ctx;

    ucs_status_t run_pingpong_batch_device()
    {
        return UCS_OK;
    }

    ucs_status_t run_stream_req_uni_batch_device()
    {
        return UCS_OK;
    }
};
