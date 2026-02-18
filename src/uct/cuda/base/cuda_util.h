/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_UTIL_H
#define UCT_CUDA_UTIL_H

#include <ucs/debug/log.h>
#include <cuda.h>


const char *uct_cuda_cu_get_error_string(CUresult result);


#define UCT_CUDADRV_LOG(_func, _log_level, _result) \
    ucs_log((_log_level), "%s failed: %s", UCS_PP_MAKE_STRING(_func), \
            uct_cuda_cu_get_error_string(_result))


#define UCT_CUDADRV_FUNC(_func, _log_level) \
    ({ \
        CUresult _result = (_func); \
        ucs_status_t _status; \
        if (ucs_likely(_result == CUDA_SUCCESS)) { \
            _status = UCS_OK; \
        } else { \
            UCT_CUDADRV_LOG(_func, _log_level, _result); \
            _status = UCS_ERR_IO_ERROR; \
        } \
        _status; \
    })


#define UCT_CUDADRV_FUNC_LOG_ERR(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_ERROR)


#define UCT_CUDADRV_FUNC_LOG_WARN(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_WARN)


#define UCT_CUDADRV_FUNC_LOG_DEBUG(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_DEBUG)


/**
 * Get the system device from the CUDA device.
 *
 * @param [in]  cuda_device CUDA device.
 * @param [out] sys_dev_p   Returned system device.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
void uct_cuda_get_sys_dev(CUdevice cuda_device, ucs_sys_device_t *sys_dev_p);


/**
 * Get the CUDA device from the system device.
 *
 * @param [in]  sys_dev     System device.
 * @param [out] device_p    Returned CUDA device.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
ucs_status_t uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev, CUdevice *device);


/**
 * Retain the primary context on the given CUDA device.
 *
 * @param [in]  cuda_device Device for which primary context is requested.
 * @param [in]  force       Retain the primary context regardless of its state.
 * @param [out] cuda_ctx_p  Returned context handle of the retained context.
 *
 * @return UCS_OK if the method completes successfully. UCS_ERR_NO_DEVICE if the
 *         primary device context is inactive on the given CUDA device and
 *         retaining is not forced. UCS_ERR_IO_ERROR if the CUDA driver API
 *         methods called inside failed with an error.
 */
ucs_status_t uct_cuda_primary_ctx_retain(CUdevice cuda_device, int force,
                                         CUcontext *cuda_ctx_p);


/**
 * Find the first active primary context and push it as the current context.
 *
 * @param [out] cuda_device_p Returned CUDA device that has the active primary
 *                            context.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
ucs_status_t uct_cuda_primary_ctx_push_first_active(CUdevice *cuda_device_p);


/**
 * Push the primary context on the given CUDA device.
 *
 * @param [in]  device          CUDA device.
 * @param [in]  retain_inactive Retain the primary context regardless of its state.
 * @param [in]  log_level       Log level.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
ucs_status_t uct_cuda_push_ctx(CUdevice device, int retain_inactive,
                               ucs_log_level_t log_level);


/**
 * With a valid sys_dev, the function pushes on the current thread the
 * corresponding CUDA context.
 * When sys_dev was specified as unknown, the function leaves the current CUDA
 * context untouched. If no context is set, it tries to push the first
 * available context among all CUDA GPUs found.
 *
 * @param [in]  retain_inactive   Retain the primary context regardless of its state.
 * @param [in]  sys_dev           System device.
 * @param [in]  cu_device_p       Returned CUDA device.
 * @param [in]  alloc_cu_device_p Returned CUDA device that has the active primary
 *                                context.
 * @param [in]  log_level         Log level.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
ucs_status_t uct_cuda_push_alloc_ctx(int retain_inactive,
                                     const ucs_sys_device_t sys_dev,
                                     CUdevice *cu_device_p,
                                     CUdevice *alloc_cu_device_p,
                                     ucs_log_level_t log_level);


/**
 * Pop the allocated CUDA context.
 *
 * @param [in]  cu_device         CUDA device.
 */
void uct_cuda_pop_alloc_ctx(CUdevice cu_device);


static UCS_F_ALWAYS_INLINE int uct_cuda_is_context_active()
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
uct_cuda_copy_ctx_pop_and_release(CUdevice cuda_device, CUcontext cuda_context)
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
