/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>
#include <tools/perf/lib/ucp_tests.h>

#include <cuda_runtime.h>


class cuda_ucp_test_runner: public ucp_perf_test_runner_base<uint64_t> {
public:
    using psn_t = uint64_t;

    cuda_ucp_test_runner(ucx_perf_context_t &perf) :
        ucp_perf_test_runner_base<uint64_t>(perf)
    {
    }

    ucs_status_t run_pingpong()
    {
        return UCS_OK;
    }

    ucs_status_t run_stream_uni()
    {
        return UCS_OK;
    }
};

static ucs_status_t
ucp_perf_cuda_dispatch(ucx_perf_context_t *perf)
{
    cuda_ucp_test_runner runner(*perf);
    if (perf->params.command == UCX_PERF_CMD_PUT_MULTI) {
        if (perf->params.test_type == UCX_PERF_TEST_TYPE_PINGPONG) {
            return runner.run_pingpong();
        } else if (perf->params.test_type == UCX_PERF_TEST_TYPE_STREAM_UNI) {
            return runner.run_stream_uni();
        }
    }
    return UCS_ERR_INVALID_PARAM;
}

UCS_STATIC_INIT {
    static ucx_perf_device_dispatcher_t cuda_dispatcher = {
        .ucp_dispatch = ucp_perf_cuda_dispatch,
    };

    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = &cuda_dispatcher;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = &cuda_dispatcher;
}

UCS_STATIC_CLEANUP {
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA]         = NULL;
    ucx_perf_mem_type_device_dispatchers[UCS_MEMORY_TYPE_CUDA_MANAGED] = NULL;
}
