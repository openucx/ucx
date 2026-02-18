/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_util.h"
#include <ucs/sys/string.h>


const char *uct_cuda_cu_get_error_string(CUresult result)
{
    static __thread char buf[64];
    const char *error_str;

    if (cuGetErrorString(result, &error_str) != CUDA_SUCCESS) {
        ucs_snprintf_safe(buf, sizeof(buf), "unrecognized error code %d",
                          result);
        error_str = buf;
    }

    return error_str;
}

void uct_cuda_get_sys_dev(CUdevice cuda_device, ucs_sys_device_t *sys_dev_p)
{
    ucs_sys_bus_id_t bus_id;
    CUresult cu_err;
    int attrib;
    ucs_status_t status;

    /* PCI domain id */
    cu_err = cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                  cuda_device);
    if (cu_err != CUDA_SUCCESS) {
        goto err;
    }
    bus_id.domain = (uint16_t)attrib;

    /* PCI bus id */
    cu_err = cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                  cuda_device);
    if (cu_err != CUDA_SUCCESS) {
        goto err;
    }
    bus_id.bus = (uint8_t)attrib;

    /* PCI slot id */
    cu_err = cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                  cuda_device);
    if (cu_err != CUDA_SUCCESS) {
        goto err;
    }
    bus_id.slot = (uint8_t)attrib;

    /* Function - always 0 */
    bus_id.function = 0;

    status = ucs_topo_find_device_by_bus_id(&bus_id, sys_dev_p);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_sys_device_set_user_value(*sys_dev_p, cuda_device);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_sys_device_enable_aux_path(*sys_dev_p);
    if (status != UCS_OK) {
        goto err;
    }

    return;

err:
    *sys_dev_p = UCS_SYS_DEVICE_ID_UNKNOWN;
}

ucs_status_t uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev, CUdevice *device)
{
    uintptr_t user_value;

    user_value = ucs_topo_sys_device_get_user_value(sys_dev);
    if (user_value == UINTPTR_MAX) {
        return UCS_ERR_NO_DEVICE;
    }

    *device = user_value;
    if (*device == CU_DEVICE_INVALID) {
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

ucs_status_t uct_cuda_primary_ctx_retain(CUdevice cuda_device, int force,
                                         CUcontext *cuda_ctx_p)
{
    unsigned int flags;
    int active;
    ucs_status_t status;
    CUcontext cuda_ctx;

    if (!force) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuDevicePrimaryCtxGetState(cuda_device, &flags, &active));
        if (status != UCS_OK) {
            return status;
        }

        if (!active) {
            ucs_debug("cuda primary context is inactive on device %d",
                      cuda_device);
            return UCS_ERR_NO_DEVICE;
        }
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDevicePrimaryCtxRetain(&cuda_ctx, cuda_device));
    if (status != UCS_OK) {
        return status;
    }

    *cuda_ctx_p = cuda_ctx;
    return UCS_OK;
}

ucs_status_t uct_cuda_primary_ctx_push_first_active(CUdevice *cuda_device_p)
{
    int num_devices, device_index;
    ucs_status_t status;
    CUdevice cuda_device;
    CUcontext cuda_ctx;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_devices));
    if (status != UCS_OK) {
        return status;
    }

    for (device_index = 0; device_index < num_devices; ++device_index) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuDeviceGet(&cuda_device, device_index));
        if (status != UCS_OK) {
            return status;
        }

        status = uct_cuda_primary_ctx_retain(cuda_device, 0, &cuda_ctx);
        if (status == UCS_OK) {
            /* Found active primary context */
            status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(cuda_ctx));
            if (status != UCS_OK) {
                UCT_CUDADRV_FUNC_LOG_WARN(
                        cuDevicePrimaryCtxRelease(cuda_device));
                return status;
            }

            *cuda_device_p = cuda_device;
            return UCS_OK;
        } else if (status != UCS_ERR_NO_DEVICE) {
            return status;
        }
    }

    return UCS_ERR_NO_DEVICE;
}

ucs_status_t uct_cuda_push_ctx(CUdevice device, int retain_inactive,
                               ucs_log_level_t log_level)
{
    ucs_status_t status;
    CUcontext primary_ctx;

    status = uct_cuda_primary_ctx_retain(device, retain_inactive, &primary_ctx);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC(cuCtxPushCurrent(primary_ctx), log_level);
    if (status != UCS_OK) {
        (void)UCT_CUDADRV_FUNC(cuDevicePrimaryCtxRelease(device), log_level);
    }

    return status;
}

ucs_status_t uct_cuda_push_alloc_ctx(int retain_inactive,
                                     const ucs_sys_device_t sys_dev,
                                     CUdevice *cu_device_p,
                                     CUdevice *alloc_cu_device_p,
                                     ucs_log_level_t log_level)
{
    ucs_status_t status;
    int dev_ordinal, num_devices;

    status = UCT_CUDADRV_FUNC_LOG_DEBUG(cuCtxGetDevice(cu_device_p));
    if (status != UCS_OK) {
        *cu_device_p = CU_DEVICE_INVALID;
    }

    if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        status = uct_cuda_get_cuda_device(sys_dev, alloc_cu_device_p);
        if (status != UCS_OK) {
            ucs_log(log_level,
                    "failed to get cuda device for system device %u",
                    sys_dev);
            return UCS_ERR_INVALID_PARAM;
        }

        /* sys_dev is the active cuda device */
        if (*cu_device_p == *alloc_cu_device_p) {
            return UCS_OK;
        }

        /* Make sys_dev the active cuda device */
        status = uct_cuda_push_ctx(*alloc_cu_device_p, retain_inactive, log_level);
        if (status != UCS_OK) {
            ucs_log(log_level, "failed to set cuda context for system device %u "
                    "(cu_device=%d)",
                    sys_dev, *alloc_cu_device_p);
        }

        return status;
    }

    /* Use the active cuda device */
    if (*cu_device_p != CU_DEVICE_INVALID) {
        *alloc_cu_device_p = *cu_device_p;
        return UCS_OK;
    }

    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&num_devices),
                              UCS_LOG_LEVEL_DIAG);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    /* Use the first active cuda device for allocation */
    for (dev_ordinal = 0; dev_ordinal < num_devices; dev_ordinal++) {
        if (UCT_CUDADRV_FUNC_LOG_DEBUG(cuDeviceGet(alloc_cu_device_p, dev_ordinal)) !=
            UCS_OK) {
            continue;
        }

        status = uct_cuda_push_ctx(*alloc_cu_device_p, retain_inactive, log_level);
        if (status == UCS_OK) {
            break;
        }
    }

    if (status != UCS_OK) {
        ucs_log(log_level,
                "no active cuda primary context for memory allocation");
    }

    return status;
}

void uct_cuda_pop_alloc_ctx(CUdevice cu_device)
{
    (void)UCT_CUDADRV_FUNC(cuCtxPopCurrent(NULL), UCS_LOG_LEVEL_WARN);
    (void)UCT_CUDADRV_FUNC(cuDevicePrimaryCtxRelease(cu_device),
                           UCS_LOG_LEVEL_WARN);
}
