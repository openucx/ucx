/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_CUDA_IMPL_H_
#define UCP_CUDA_IMPL_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_context.h"
#include <tools/perf/lib/libperf_int.h>
#include <ucp/api/device/ucp_host.h>

typedef struct ucp_perf_cuda_params {
    ucp_device_local_mem_list_h  local_mem_list;
    ucp_device_remote_mem_list_h remote_mem_list;
    size_t                       length;
    uint64_t                     *counter_send;
    uint64_t                     *counter_recv;
} ucp_perf_cuda_params_t;

BEGIN_C_DECLS

ucs_status_t ucp_perf_cuda_params_init(const ucx_perf_context_t *perf,
                                        ucp_perf_cuda_params_t *params);
void ucp_perf_cuda_params_cleanup(ucp_perf_cuda_params_t *params);

ucs_status_t ucp_perf_cuda_run_pingpong(ucx_perf_context_t *perf,
                                         ucx_perf_cuda_context_t *cpu_ctx,
                                         ucx_perf_cuda_context_t *gpu_ctx);
ucs_status_t ucp_perf_cuda_run_stream_uni(ucx_perf_context_t *perf,
                                           ucx_perf_cuda_context_t *cpu_ctx,
                                           ucx_perf_cuda_context_t *gpu_ctx);

END_C_DECLS

#endif /* UCP_CUDA_IMPL_H_ */
