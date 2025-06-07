/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_nvml.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log_def.h>
#include <ucs/sys/preprocessor.h>

#include <stddef.h>
#include <dlfcn.h>
#include <pthread.h>

#define UCT_CUDA_NVML_LIB_NAME "libnvidia-ml.so.1"
#define UCT_CUDA_NVML_LIB_MODE RTLD_NOW

#define UCT_CUDA_NVML_FOR_EACH_EXT(_macro) \
    _macro(nvmlInit_v2); \
    _macro(nvmlShutdown); \
    UCT_CUDA_NVML_FOR_EACH(_macro)

#if HAVE_NVML_FABRIC_INFO
#define UCT_CUDA_NVML_FOR_EACH_GET_FABRIC_INFO(_macro) \
    _macro(nvmlDeviceGetGpuFabricInfoV, nvmlDevice_t, nvmlGpuFabricInfoV_t*); \
    _macro(nvmlDeviceGetGpuFabricInfo, nvmlDevice_t, nvmlGpuFabricInfo_t*)
#endif

#define UCT_CUDA_NVML_FPTR(_func) UCS_PP_TOKENPASTE(uct_cuda_fptr_, _func)

#define UCT_CUDA_NVML_FPTR_DECL(_func, ...) \
    static nvmlReturn_t (*UCT_CUDA_NVML_FPTR(_func))(__VA_ARGS__)

#define UCT_CUDA_NVML_FPTR_LOAD(_func, _fail_action) \
    do { \
        UCT_CUDA_NVML_FPTR( \
                _func) = uct_cuda_nvml_dlsym(lib_handle, \
                                             UCS_PP_MAKE_STRING(_func)); \
        if (UCT_CUDA_NVML_FPTR(_func) == NULL) { \
            _fail_action; \
        } \
    } while (0)

#define UCT_CUDA_NVML_FOR_EACH_ACTION(_macro, _fail_action) \
    _macro(nvmlInit_v2, _fail_action); \
    _macro(nvmlDeviceGetHandleByIndex, _fail_action); \
    _macro(nvmlDeviceGetFieldValues, _fail_action); \
    _macro(nvmlDeviceGetNvLinkRemotePciInfo, _fail_action); \
    _macro(nvmlShutdown, _fail_action)

#define UCT_CUDA_NVML_FPTR_CALL(_func, ...) \
    ({ \
        nvmlReturn_t _nvml_fptr_ret; \
        ucs_assertv(UCT_CUDA_NVML_FPTR(_func) != NULL, "%s is not loaded", \
                    UCS_PP_MAKE_STRING(_func)); \
        _nvml_fptr_ret = UCT_CUDA_NVML_FPTR(_func)(__VA_ARGS__); \
        _nvml_fptr_ret; \
    })

#define UCT_CUDA_NVML_CALL_DEBUG(_func, ...) \
    ({ \
        ucs_status_t _nvml_call_status; \
        nvmlReturn_t _nvml_fptr_call_ret; \
        _nvml_fptr_call_ret = UCT_CUDA_NVML_FPTR_CALL(_func, __VA_ARGS__); \
        if (_nvml_fptr_call_ret == NVML_SUCCESS) { \
            _nvml_call_status = UCS_OK; \
        } else { \
            ucs_debug("%s failed: %s", UCS_PP_MAKE_STRING(_func), \
                      (_nvml_fptr_call_ret != NVML_ERROR_DRIVER_NOT_LOADED) ? \
                              nvmlErrorString(_nvml_fptr_call_ret) : \
                              "nvml is a stub library"); \
            _nvml_call_status = UCS_ERR_IO_ERROR; \
        } \
        _nvml_call_status; \
    })

#define UCT_CUDA_NVML_WRAP_IMPL(_func, ...) \
    UCT_CUDA_NVML_WRAP_DECL(_func, UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        ucs_status_t _status; \
        pthread_mutex_lock(&uct_cuda_nvml_mutex); \
        _status = uct_cuda_nvml_init(); \
        if (_status != UCS_OK) { \
            goto _out; \
        } \
        _status = UCT_CUDA_NVML_CALL_DEBUG(_func, \
                                           UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
    _out: \
        pthread_mutex_unlock(&uct_cuda_nvml_mutex); \
        return _status; \
    }


static pthread_mutex_t uct_cuda_nvml_mutex    = PTHREAD_MUTEX_INITIALIZER;
static ucs_status_t uct_cuda_nvml_init_status = UCS_ERR_LAST;
static void *lib_handle                       = NULL;

UCT_CUDA_NVML_FOR_EACH_EXT(UCT_CUDA_NVML_FPTR_DECL);

#if HAVE_NVML_FABRIC_INFO
UCT_CUDA_NVML_FOR_EACH_GET_FABRIC_INFO(UCT_CUDA_NVML_FPTR_DECL);

static nvmlReturn_t
nvmlDeviceGetGpuFabricInfoV_proxy(nvmlDevice_t device,
                                  nvmlGpuFabricInfoV_t *gpu_fabric_info_v)
{
    nvmlGpuFabricInfo_t gpu_fabric_info;
    nvmlReturn_t ret;

    ret = UCT_CUDA_NVML_FPTR_CALL(nvmlDeviceGetGpuFabricInfo, device,
                                  &gpu_fabric_info);
    if (ret != NVML_SUCCESS) {
        return ret;
    }

    gpu_fabric_info_v->cliqueId   = gpu_fabric_info.cliqueId;
    gpu_fabric_info_v->healthMask = 0;
    gpu_fabric_info_v->state      = gpu_fabric_info.state;
    gpu_fabric_info_v->status     = gpu_fabric_info.status;
    memcpy(gpu_fabric_info_v->clusterUuid, gpu_fabric_info.clusterUuid,
           NVML_GPU_FABRIC_UUID_LEN);

    return NVML_SUCCESS;
}
#endif

static void *uct_cuda_nvml_dlsym(void *handle, const char *name)
{
    void *symbol;
    const char *error;

    symbol = dlsym(handle, name);
    if (symbol == NULL) {
        error = dlerror();
        ucs_debug("dlsym('%s') failed: %s", name,
                  error ? error : "unknown error");
    }

    return symbol;
}

static ucs_status_t uct_cuda_nvml_init()
{
    const char *error;

    if (uct_cuda_nvml_init_status != UCS_ERR_LAST) {
        goto out;
    }

    lib_handle = dlopen(UCT_CUDA_NVML_LIB_NAME, UCT_CUDA_NVML_LIB_MODE);
    if (lib_handle == NULL) {
        error = dlerror();
        ucs_debug("dlopen('%s', mode=0x%x) failed: %s", UCT_CUDA_NVML_LIB_NAME,
                  UCT_CUDA_NVML_LIB_MODE, error ? error : "unknown error");
        goto err;
    }

    UCT_CUDA_NVML_FOR_EACH_ACTION(UCT_CUDA_NVML_FPTR_LOAD, goto err_dlclose);

#if HAVE_NVML_FABRIC_INFO
    UCT_CUDA_NVML_FPTR_LOAD(nvmlDeviceGetGpuFabricInfoV, {
        /**
         * nvmlDeviceGetGpuFabricInfo was deprecated in CUDA 12.4.0. The old API
         * is used to ensure compatibility of the UCX. Thus, UCX built with CUDA
         * 12.4.0 or newer can be used on a system with older CUDA 12 versions.
         */
        UCT_CUDA_NVML_FPTR_LOAD(nvmlDeviceGetGpuFabricInfo, goto err_dlclose);

        UCT_CUDA_NVML_FPTR(nvmlDeviceGetGpuFabricInfoV) =
                nvmlDeviceGetGpuFabricInfoV_proxy;
    });
#endif

    if (UCT_CUDA_NVML_CALL_DEBUG(nvmlInit_v2) == UCS_OK) {
        uct_cuda_nvml_init_status = UCS_OK;
        goto out;
    }

err_dlclose:
    dlclose(lib_handle);
err:
    uct_cuda_nvml_init_status = UCS_ERR_IO_ERROR;
    ucs_diag("failed to initialize nvml");
out:
    return uct_cuda_nvml_init_status;
}

UCT_CUDA_NVML_FOR_EACH(UCT_CUDA_NVML_WRAP_IMPL)

#if HAVE_NVML_FABRIC_INFO
UCT_CUDA_NVML_WRAP_IMPL(nvmlDeviceGetGpuFabricInfoV, nvmlDevice_t,
                        nvmlGpuFabricInfoV_t*)
#endif

UCS_STATIC_CLEANUP
{
    if (uct_cuda_nvml_init_status == UCS_OK) {
        UCT_CUDA_NVML_CALL_DEBUG(nvmlShutdown);
        dlclose(lib_handle);
    }
}
