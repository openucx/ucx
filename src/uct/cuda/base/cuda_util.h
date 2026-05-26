/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_UTIL_H
#define UCT_CUDA_UTIL_H

#include <ucs/debug/log.h>
#include <ucs/profile/profile_defs.h>
#include <ucs/sys/topo/base/topo.h>

#include <cuda.h>


const char *uct_cuda_cu_get_error_string(CUresult result);


#define UCT_CUDADRV_LOG(_log_level, _func, _result) \
    ucs_log((_log_level), "%s failed: %s", UCS_PP_MAKE_STRING(_func), \
            uct_cuda_cu_get_error_string(_result))


#define UCT_CUDADRV_FUNC(_log_level, _func, ...) \
    ({ \
        CUresult _result = UCS_PROFILE_CALL_ALWAYS(_func, ##__VA_ARGS__); \
        ucs_status_t _status; \
        if (ucs_likely(_result == CUDA_SUCCESS)) { \
            _status = UCS_OK; \
        } else { \
            UCT_CUDADRV_LOG(_log_level, _func, _result); \
            _status = UCS_ERR_IO_ERROR; \
        } \
        _status; \
    })


#define UCT_CUDADRV_FUNC_LOG_ERR(_func, ...) \
    UCT_CUDADRV_FUNC(UCS_LOG_LEVEL_ERROR, _func, ##__VA_ARGS__)


#define UCT_CUDADRV_FUNC_LOG_WARN(_func, ...) \
    UCT_CUDADRV_FUNC(UCS_LOG_LEVEL_WARN, _func, ##__VA_ARGS__)


#define UCT_CUDADRV_FUNC_LOG_DIAG(_func, ...) \
    UCT_CUDADRV_FUNC(UCS_LOG_LEVEL_DIAG, _func, ##__VA_ARGS__)


#define UCT_CUDADRV_FUNC_LOG_DEBUG(_func, ...) \
    UCT_CUDADRV_FUNC(UCS_LOG_LEVEL_DEBUG, _func, ##__VA_ARGS__)


/**
 * Get the system device from the CUDA device.
 *
 * @param [in]  cuda_device CUDA device.
 *
 * @return System device corresponding to the CUDA device.
 */
ucs_sys_device_t uct_cuda_get_sys_dev(CUdevice cuda_device);


/**
 * Get the CUDA device from the system device.
 *
 * @param [in]  sys_dev     System device.
 *
 * @return CUDA device corresponding to the system device.
 */
CUdevice uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev);

#endif
