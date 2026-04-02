/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Host-only API: launch pingpong/latency kernel.
 */

#ifndef UCP_CUDA_KERNEL_LATENCY_CUH_
#define UCP_CUDA_KERNEL_LATENCY_CUH_

#include "ucp_cuda_impl.h"

BEGIN_C_DECLS

void ucp_perf_cuda_launch_pingpong(const ucx_perf_context_t *perf,
                                   ucx_perf_cuda_context_t *gpu_ctx,
                                   const ucp_perf_cuda_params_t *params,
                                   unsigned my_index);

void ucp_perf_cuda_launch_pingpong_thread(const ucx_perf_context_t *perf,
                                          ucx_perf_cuda_context_t *gpu_ctx,
                                          const ucp_perf_cuda_params_t *params,
                                          unsigned my_index);
void ucp_perf_cuda_launch_pingpong_warp(const ucx_perf_context_t *perf,
                                        ucx_perf_cuda_context_t *gpu_ctx,
                                        const ucp_perf_cuda_params_t *params,
                                        unsigned my_index);

END_C_DECLS

#endif /* UCP_CUDA_KERNEL_LATENCY_CUH_ */
