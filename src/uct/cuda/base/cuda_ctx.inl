/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_CTX_INL
#define UCT_CUDA_CTX_INL

#include <uct/cuda/base/cuda_ctx.h>
#include <ucs/sys/compiler_def.h>


static UCS_F_ALWAYS_INLINE int uct_cuda_ctx_is_active()
{
    CUcontext ctx;

    return (CUDA_SUCCESS == cuCtxGetCurrent(&ctx)) && (ctx != NULL);
}


static UCS_F_ALWAYS_INLINE CUresult
uct_cuda_ctx_get_id(CUcontext ctx, unsigned long long *ctx_id_p)
{
#if CUDA_VERSION >= 12000
    return cuCtxGetId(ctx, ctx_id_p);
#else
    *ctx_id_p = 0;
    return CUDA_SUCCESS;
#endif
}


static UCS_F_ALWAYS_INLINE void
uct_cuda_ctx_pop_and_release(CUdevice cuda_device, CUcontext cuda_context)
{
    if ((cuda_device == CU_DEVICE_INVALID) && (cuda_context == NULL)) {
        return;
    }

    UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    if (cuda_device == CU_DEVICE_INVALID) {
        return;
    }

    UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(cuda_device));
}

#endif
