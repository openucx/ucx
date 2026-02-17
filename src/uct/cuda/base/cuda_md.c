/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_md.h"
#include "cuda_iface.h"

#include <ucs/sys/module.h>
#include <ucs/sys/string.h>


void uct_cuda_base_get_sys_dev(CUdevice cuda_device,
                               ucs_sys_device_t *sys_dev_p)
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

ucs_status_t
uct_cuda_base_get_cuda_device(ucs_sys_device_t sys_dev, CUdevice *device)
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

ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    const unsigned sys_device_priority = 10;
    ucs_sys_device_t sys_dev;
    CUdevice cuda_device;
    ucs_status_t status;
    char device_name[10];
    int i, num_gpus;

    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&num_gpus), UCS_LOG_LEVEL_DIAG);
    if ((status != UCS_OK) || (num_gpus == 0)) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    for (i = 0; i < num_gpus; ++i) {
        status = UCT_CUDADRV_FUNC(cuDeviceGet(&cuda_device, i),
                                  UCS_LOG_LEVEL_DIAG);
        if (status != UCS_OK) {
            continue;
        }

        uct_cuda_base_get_sys_dev(cuda_device, &sys_dev);
        if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
            continue;
        }

        ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%d",
                          cuda_device);
        status = ucs_topo_sys_device_set_name(sys_dev, device_name,
                                              sys_device_priority);
        ucs_assert_always(status == UCS_OK);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
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

ucs_status_t uct_cuda_base_push_ctx(CUdevice device, int retain_inactive,
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

/*
 * With a valid sys_dev, the function pushes on the current thread the
 * corresponding CUDA context.
 * When sys_dev was specified as unknown, the function leaves the current CUDA
 * context untouched. If no context is set, it tries to push the first
 * available context among all CUDA GPUs found.
 */
ucs_status_t uct_cuda_base_push_alloc_ctx(int retain_inactive,
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
        status = uct_cuda_base_get_cuda_device(sys_dev, alloc_cu_device_p);
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
        status = uct_cuda_base_push_ctx(*alloc_cu_device_p, retain_inactive,
                                        log_level);
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

        status = uct_cuda_base_push_ctx(*alloc_cu_device_p, retain_inactive,
                                        log_level);
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

void uct_cuda_base_pop_alloc_ctx(CUdevice cu_device)
{
    (void)UCT_CUDADRV_FUNC(cuCtxPopCurrent(NULL), UCS_LOG_LEVEL_WARN);
    (void)UCT_CUDADRV_FUNC(cuDevicePrimaryCtxRelease(cu_device),
                           UCS_LOG_LEVEL_WARN);
}

UCS_STATIC_INIT
{
    UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));
}

UCS_STATIC_CLEANUP
{
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
