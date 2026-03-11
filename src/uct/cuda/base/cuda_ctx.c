/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_ctx.h"


ucs_status_t uct_cuda_ctx_primary_retain(CUdevice cuda_device, int force,
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

ucs_status_t uct_cuda_ctx_primary_push_first_active(CUdevice *cuda_device_p)
{
    int num_devices, device_index;
    ucs_status_t status;
    CUdevice cuda_device;

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

        status = uct_cuda_ctx_primary_push(cuda_device, 0, UCS_LOG_LEVEL_ERROR);
        if (status == UCS_OK) {
            *cuda_device_p = cuda_device;
            return UCS_OK;
        } else if (status != UCS_ERR_NO_DEVICE) {
            return status;
        }
    }

    return UCS_ERR_NO_DEVICE;
}

ucs_status_t uct_cuda_ctx_primary_push(CUdevice cuda_device, int retain_inactive,
                                       ucs_log_level_t log_level)
{
    ucs_status_t status;
    CUcontext primary_ctx;

    status = uct_cuda_ctx_primary_retain(cuda_device, retain_inactive,
                                         &primary_ctx);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC(cuCtxPushCurrent(primary_ctx), log_level);
    if (status != UCS_OK) {
        (void)UCT_CUDADRV_FUNC(cuDevicePrimaryCtxRelease(cuda_device), log_level);
    }

    return status;
}

ucs_status_t uct_cuda_ctx_primary_push_avail(int retain_inactive,
                                             const ucs_sys_device_t sys_dev,
                                             CUdevice *cuda_device_p,
                                             CUdevice *avail_cuda_device_p,
                                             ucs_log_level_t log_level)
{
    ucs_status_t status;
    int dev_ordinal, num_devices;

    status = UCT_CUDADRV_FUNC_LOG_DEBUG(cuCtxGetDevice(cuda_device_p));
    if (status != UCS_OK) {
        *cuda_device_p = CU_DEVICE_INVALID;
    }

    if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        *avail_cuda_device_p = uct_cuda_get_cuda_device(sys_dev);
        if (*avail_cuda_device_p == CU_DEVICE_INVALID) {
            ucs_log(log_level, "failed to get cuda device for system device %u",
                    sys_dev);
            return UCS_ERR_INVALID_PARAM;
        }

        /* sys_dev is the active cuda device */
        if (*cuda_device_p == *avail_cuda_device_p) {
            return UCS_OK;
        }

        /* Make sys_dev the active cuda device */
        status = uct_cuda_ctx_primary_push(*avail_cuda_device_p, retain_inactive,
                                           log_level);
        if (status != UCS_OK) {
            ucs_log(log_level, "failed to set cuda context for system device %u "
                    "(cu_device=%d)", sys_dev, *avail_cuda_device_p);
        }

        return status;
    }

    /* Use the active cuda device */
    if (*cuda_device_p != CU_DEVICE_INVALID) {
        *avail_cuda_device_p = *cuda_device_p;
        return UCS_OK;
    }

    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&num_devices), UCS_LOG_LEVEL_DIAG);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    /* Use the first active cuda device for allocation */
    for (dev_ordinal = 0; dev_ordinal < num_devices; dev_ordinal++) {
        if (UCT_CUDADRV_FUNC_LOG_DEBUG(cuDeviceGet(avail_cuda_device_p,
                                                   dev_ordinal)) != UCS_OK) {
            continue;
        }

        status = uct_cuda_ctx_primary_push(*avail_cuda_device_p, retain_inactive,
                                           log_level);
        if (status == UCS_OK) {
            break;
        }
    }

    if (status != UCS_OK) {
        ucs_log(log_level, "no active cuda primary context for memory allocation");
    }

    return status;
}

void uct_cuda_ctx_primary_pop_and_release(CUdevice cuda_device)
{
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(cuda_device));
}
