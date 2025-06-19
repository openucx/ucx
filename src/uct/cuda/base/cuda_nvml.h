/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_NVML_H
#define UCT_CUDA_NVML_H

#include <ucs/sys/preprocessor.h>
#include <ucs/type/status.h>

#include <nvml.h>


/**
  * Wrapper for NVML functions.
  *
  * @param _func NVML function name.
  * @param ...   NVML function arguments.
  *
  * @return UCS_OK if the function call was successful, UCS_ERR_IO_ERROR
  *         otherwise.
  */
#define UCT_CUDA_NVML_WRAP_CALL(_func, ...) \
    UCT_CUDA_NVML_WRAP_NAME(_func)(__VA_ARGS__)


#define UCT_CUDA_NVML_WRAP_NAME(_func) UCS_PP_TOKENPASTE(uct_cuda_, _func)


#define UCT_CUDA_NVML_WRAP_DECL(_func, ...) \
    ucs_status_t UCT_CUDA_NVML_WRAP_NAME(_func)(__VA_ARGS__)


#define UCT_CUDA_NVML_FOR_EACH(_macro) \
    _macro(nvmlDeviceGetHandleByIndex, unsigned, nvmlDevice_t*); \
    _macro(nvmlDeviceGetFieldValues, nvmlDevice_t, int, nvmlFieldValue_t*); \
    _macro(nvmlDeviceGetNvLinkRemotePciInfo, nvmlDevice_t, unsigned int, \
           nvmlPciInfo_t*)


UCT_CUDA_NVML_FOR_EACH(UCT_CUDA_NVML_WRAP_DECL);


#if HAVE_NVML_FABRIC_INFO
UCT_CUDA_NVML_WRAP_DECL(nvmlDeviceGetGpuFabricInfoV, nvmlDevice_t,
                        nvmlGpuFabricInfoV_t*);
#endif

#endif
