/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_INL
#define UCT_CUDA_COPY_IFACE_INL

#include "cuda_copy_iface.h"

KHASH_IMPL(cuda_copy_iface_ctx_rscs, khint64_t, uct_cuda_copy_iface_ctx_rsc_t,
           1, kh_int64_hash_func, kh_int64_hash_equal);

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_iface_get_ctx_id(unsigned long long *ctx_id_p)
{
#if CUDA_VERSION >= 12000
    CUcontext current_ctx;
    ucs_status_t status;
    unsigned long long ctx_id;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&current_ctx));
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    } else if (ucs_unlikely(current_ctx == NULL)) {
        ucs_error("no context bound to calling thread");
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetId(current_ctx, &ctx_id));
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *ctx_id_p = ctx_id;
#else
    *ctx_id_p = 0;
#endif

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_iface_ctx_rsc_get(uct_cuda_copy_iface_t *iface,
                                uct_cuda_copy_iface_ctx_rsc_t **ctx_rsc_p)
{
    unsigned long long ctx_id;
    ucs_status_t status;
    khiter_t iter;

    status = uct_cuda_copy_iface_get_ctx_id(&ctx_id);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    iter = kh_get(cuda_copy_iface_ctx_rscs, &iface->ctx_rscs, ctx_id);
    if (ucs_likely(iter != kh_end(&iface->ctx_rscs))) {
        *ctx_rsc_p = &kh_value(&iface->ctx_rscs, iter);
        return UCS_OK;
    }

    return uct_cuda_copy_iface_ctx_rsc_create(iface, ctx_id, ctx_rsc_p);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_iface_init_stream(CUstream *stream)
{
    if (ucs_likely(*stream != NULL)) {
        return UCS_OK;
    }

    return UCT_CUDADRV_FUNC_LOG_ERR(
            cuStreamCreate(stream, CU_STREAM_NON_BLOCKING));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_iface_get_stream(uct_cuda_copy_iface_ctx_rsc_t *ctx_rsc,
                               ucs_memory_type_t src_type,
                               ucs_memory_type_t dst_type,
                               CUstream *stream_p)
{
    CUstream *stream;
    ucs_status_t status;

    ucs_assert((src_type < UCS_MEMORY_TYPE_LAST) &&
               (dst_type < UCS_MEMORY_TYPE_LAST));

    stream = &ctx_rsc->queue_desc[src_type][dst_type].stream;
    status = uct_cuda_copy_iface_init_stream(stream);
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_error("stream for src %s dst %s not available",
                   ucs_memory_type_names[src_type],
                   ucs_memory_type_names[dst_type]);
        return status;
    }

    *stream_p = *stream;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_iface_get_short_stream(uct_cuda_copy_iface_t *iface,
                                     CUstream *stream_p)
{
    uct_cuda_copy_iface_ctx_rsc_t *ctx_rsc;
    CUstream *stream;
    ucs_status_t status;

    status = uct_cuda_copy_iface_ctx_rsc_get(iface, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    stream = &ctx_rsc->short_stream;
    status = uct_cuda_copy_iface_init_stream(stream);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *stream_p = *stream;
    return UCS_OK;
}

#endif
