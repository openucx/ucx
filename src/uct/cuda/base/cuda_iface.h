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
#include <cuda.h>
#include <nvml.h>


const char *uct_cuda_base_cu_get_error_string(CUresult result);


#define UCT_NVML_FUNC(_func, _log_level) \
    ({ \
        ucs_status_t _status = UCS_OK; \
        do { \
            nvmlReturn_t _err = (_func); \
            if (NVML_SUCCESS != _err) { \
                ucs_log((_log_level), "%s failed: %s", \
                        UCS_PP_MAKE_STRING(_func), \
                        (NVML_ERROR_DRIVER_NOT_LOADED != _err) ? \
                                nvmlErrorString(_err) : \
                                "nvml is a stub library"); \
                _status = UCS_ERR_IO_ERROR; \
            } \
        } while (0); \
        _status; \
    })


#define UCT_NVML_FUNC_LOG_ERR(_func) \
    UCT_NVML_FUNC(_func, UCS_LOG_LEVEL_ERROR)

#define UCT_CUDADRV_FUNC(_func, _log_level) \
    ({ \
        ucs_status_t _status = UCS_OK; \
        do { \
            CUresult _result = (_func); \
            if (CUDA_ERROR_NOT_READY == _result) { \
                _status = UCS_INPROGRESS; \
            } else if (CUDA_SUCCESS != _result) { \
                ucs_log((_log_level), "%s failed: %s", \
                        UCS_PP_MAKE_STRING(_func), \
                        uct_cuda_base_cu_get_error_string(_result)); \
                _status = UCS_ERR_IO_ERROR; \
            } \
        } while (0); \
        _status; \
    })


#define UCT_CUDADRV_FUNC_LOG_ERR(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_ERROR)


#define UCT_CUDADRV_FUNC_LOG_WARN(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_WARN)


#define UCT_CUDADRV_FUNC_LOG_DEBUG(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_DEBUG)


static UCS_F_ALWAYS_INLINE int uct_cuda_base_is_context_active()
{
    CUcontext ctx;

    return (CUDA_SUCCESS == cuCtxGetCurrent(&ctx)) && (ctx != NULL);
}


static UCS_F_ALWAYS_INLINE int uct_cuda_base_is_context_valid(CUcontext ctx)
{
    unsigned version;
    ucs_status_t status;

    /* Check if CUDA context is valid by running a dummy operation on it */
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetApiVersion(ctx, &version));
    return (status == UCS_OK);
}


static UCS_F_ALWAYS_INLINE int uct_cuda_base_context_match(CUcontext ctx1,
                                                           CUcontext ctx2)
{
    return ((ctx1 != NULL) && (ctx1 == ctx2) &&
            uct_cuda_base_is_context_valid(ctx1));
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

void
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
