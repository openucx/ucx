/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_UTIL_H
#define UCT_CUDA_UTIL_H

#include <ucs/sys/topo/base/topo.h>
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

#endif
