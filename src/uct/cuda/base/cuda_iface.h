/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/profile/profile.h>
#include <ucs/async/eventfd.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>


#define UCT_CUDA_DEV_NAME       "cuda"

#define UCT_CUDAR_CALL(_log_level, _func, ...) \
    ({ \
        cudaError_t _result = UCS_PROFILE_CALL(_func, __VA_ARGS__); \
        ucs_status_t _status; \
        \
        if (cudaSuccess != _result) { \
            ucs_log((_log_level), "%s() failed: %s", \
                    UCS_PP_MAKE_STRING(_func), cudaGetErrorString(_result)); \
            _status = UCS_ERR_IO_ERROR; \
        } else { \
            _status = UCS_OK; \
        } \
        _status; \
    })


#define UCT_CUDAR_CALL_LOG_ERR(_func, ...) \
    UCT_CUDAR_CALL(UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)


#if CUDART_VERSION >= 11010
#define UCT_CUDA_FUNC_PTX_ERR(_result, _func, _err_str)         \
    do {                                                        \
        if (_result == cudaErrorUnsupportedPtxVersion) {        \
            ucs_error("%s() failed: %s",                        \
                      UCS_PP_MAKE_STRING(_func), _err_str);     \
        }                                                       \
    } while (0);
#else
#define UCT_CUDA_FUNC_PTX_ERR(_result, _func, _err_str)         \
    do {                                                        \
    } while (0);
#endif


#define UCT_CUDA_FUNC(_func, _log_level)                        \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            cudaError_t _result = (_func);                      \
            if (cudaSuccess != _result) {                       \
                if (_log_level != UCS_LOG_LEVEL_ERROR) {        \
                    UCT_CUDA_FUNC_PTX_ERR(_result,  _func,      \
                        cudaGetErrorString(_result));           \
                }                                               \
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


#define UCT_NVML_FUNC(_func, _log_level)                        \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            nvmlReturn_t _err = (_func);                        \
            if (NVML_SUCCESS != _err) {                         \
                ucs_log((_log_level), "%s failed: %s",          \
                        UCS_PP_MAKE_STRING(_func),              \
                        nvmlErrorString(_err));                 \
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_NVML_FUNC_LOG_ERR(_func) \
    UCT_NVML_FUNC(_func, UCS_LOG_LEVEL_ERROR)


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


#define UCT_CUDADRV_FUNC_LOG_DEBUG(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_DEBUG)


static UCS_F_ALWAYS_INLINE int uct_cuda_base_is_context_active()
{
    CUcontext ctx;

    return (CUDA_SUCCESS == cuCtxGetCurrent(&ctx)) && (ctx != NULL);
}


static UCS_F_ALWAYS_INLINE int uct_cuda_base_context_match(CUcontext ctx1,
                                                           CUcontext ctx2)
{
    return ((ctx1 != NULL) && (ctx1 == ctx2));
}


typedef enum uct_cuda_base_gen {
    UCT_CUDA_BASE_GEN_P100 = 6,
    UCT_CUDA_BASE_GEN_V100 = 7,
    UCT_CUDA_BASE_GEN_A100 = 8,
    UCT_CUDA_BASE_GEN_H100 = 9
} uct_cuda_base_gen_t;


typedef struct uct_cuda_iface {
    uct_base_iface_t super;
    int              eventfd;
} uct_cuda_iface_t;

ucs_status_t
uct_cuda_base_query_devices_common(
        uct_md_h md, uct_device_type_t dev_type,
        uct_tl_device_resource_t **tl_devices_p, unsigned *num_tl_devices_p);

ucs_status_t
uct_cuda_base_query_devices(
        uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
        unsigned *num_tl_devices_p);

ucs_status_t
uct_cuda_base_get_sys_dev(CUdevice cuda_device,
                          ucs_sys_device_t *sys_dev_p);

ucs_status_t uct_cuda_base_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p);

#if (__CUDACC_VER_MAJOR__ >= 100000)
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(void *arg);
#else
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(CUstream hStream,  CUresult status,
                                               void *arg);
#endif

UCS_CLASS_INIT_FUNC(uct_cuda_iface_t, uct_iface_ops_t *tl_ops,
                    uct_iface_internal_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config, const char *dev_name);

#endif
