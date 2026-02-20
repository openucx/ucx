/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_CTX_H
#define UCT_CUDA_CTX_H

#include <uct/cuda/base/cuda_util.h>


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
ucs_status_t uct_cuda_ctx_primary_retain(CUdevice cuda_device, int force,
                                         CUcontext *cuda_ctx_p);


/**
 * Find the first active primary context and push it as the current context.
 *
 * @param [out] cuda_device_p Returned CUDA device that has the active primary
 *                            context.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
ucs_status_t uct_cuda_ctx_primary_push_first_active(CUdevice *cuda_device_p);


/**
 * Push the primary context on the given CUDA device.
 *
 * @param [in]  device          CUDA device.
 * @param [in]  retain_inactive Retain the primary context regardless of its state.
 * @param [in]  log_level       Log level.
 *
 * @return UCS_OK if completes successfully, error code otherwise.
 */
ucs_status_t uct_cuda_ctx_push(CUdevice device, int retain_inactive,
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
ucs_status_t uct_cuda_ctx_push_alloc(int retain_inactive,
                                     const ucs_sys_device_t sys_dev,
                                     CUdevice *cu_device_p,
                                     CUdevice *alloc_cu_device_p,
                                     ucs_log_level_t log_level);


/**
 * Pop the allocated CUDA context.
 *
 * @param [in]  cu_device         CUDA device.
 */
void uct_cuda_ctx_pop_alloc(CUdevice cu_device);

#endif
