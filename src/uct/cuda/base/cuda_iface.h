/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/sys/preprocessor.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define UCT_CUDA_DEV_NAME       "cuda"


#define UCT_CUDA_FUNC(_func, _log_level)                        \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            cudaError_t _result = (_func);                      \
            if (cudaSuccess != _result) {                       \
                ucs_log((_log_level), "%s() failed: %s",        \
                        UCS_PP_MAKE_STRING(_func),              \
                        cudaGetErrorString(_result));           \
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_CUDA_FUNC_LOG_ERR(_func) \
    UCT_CUDA_FUNC(_func, UCS_LOG_LEVEL_ERROR)


#define UCT_CUDADRV_FUNC(_func, _log_level)                     \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            CUresult _result = (_func);                         \
            const char *cu_err_str;                             \
            if (CUDA_ERROR_NOT_READY == _result) {              \
                _status = UCS_INPROGRESS;                       \
            } else if (CUDA_SUCCESS != _result) {               \
                cuGetErrorString(_result, &cu_err_str);         \
                ucs_log((_log_level), "%s() failed: %s",        \
                        UCS_PP_MAKE_STRING(_func), cu_err_str); \
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_CUDADRV_FUNC_LOG_ERR(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_ERROR)


#define UCT_CUDADRV_CTX_ACTIVE(_state)                                       \
    {                                                                        \
        CUcontext cur_ctx;                                                   \
        CUdevice dev;                                                        \
        unsigned flags;                                                      \
                                                                             \
        _state = 0;                                                          \
        /* avoid active state check if no cuda activity */                   \
        if ((CUDA_SUCCESS == cuCtxGetCurrent(&cur_ctx)) &&                   \
            (NULL != cur_ctx)) {                                             \
            UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetDevice(&dev));                  \
            UCT_CUDADRV_FUNC_LOG_ERR(cuDevicePrimaryCtxGetState(dev, &flags, \
                                                                &_state));   \
        }                                                                    \
    }


typedef enum uct_cuda_base_gen {
    UCT_CUDA_BASE_GEN_P100 = 6,
    UCT_CUDA_BASE_GEN_V100 = 7,
    UCT_CUDA_BASE_GEN_A100 = 8
} uct_cuda_base_gen_t;


ucs_status_t
uct_cuda_base_query_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p);

ucs_status_t
uct_cuda_base_get_sys_dev(CUdevice cuda_device, ucs_sys_device_t *sys_dev_p);

#endif
