/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 *
 * Host-only API: launch wait kernel.
 */

#ifndef UCP_CUDA_KERNEL_WAIT_CUH_
#define UCP_CUDA_KERNEL_WAIT_CUH_

#include "ucp_cuda_impl.h"

BEGIN_C_DECLS

void ucp_perf_cuda_launch_wait(ucx_perf_cuda_context_t *gpu_ctx,
                               const ucp_perf_cuda_params_t *params);

END_C_DECLS

#endif /* UCP_CUDA_KERNEL_WAIT_CUH_ */
