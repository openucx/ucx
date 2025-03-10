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
#include <cuda.h>


UCS_ARRAY_DECLARE_TYPE(uct_cuda_retained_devices_t, unsigned, CUdevice);
uct_cuda_retained_devices_t uct_cuda_retained_devices;


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

    return;

err:
    *sys_dev_p = UCS_SYS_DEVICE_ID_UNKNOWN;
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

ucs_status_t
uct_cuda_primary_ctx_retain(CUdevice cuda_device, CUcontext *cuda_ctx_p)
{
    CUcontext cuda_ctx;
    ucs_status_t status;
    CUdevice *cuda_device_retained;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDevicePrimaryCtxRetain(&cuda_ctx,
                                                               cuda_device));
    if (status != UCS_OK) {
        return status;
    }

    cuda_device_retained  = ucs_array_append(&uct_cuda_retained_devices,
                                             return UCS_ERR_NO_MEMORY);
    *cuda_device_retained = cuda_device;
    *cuda_ctx_p           = cuda_ctx;
    return UCS_OK;
}

UCS_STATIC_INIT
{
    UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));
    ucs_array_init_dynamic(&uct_cuda_retained_devices);
}

UCS_STATIC_CLEANUP
{
    CUdevice *device;

    ucs_array_for_each(device, &uct_cuda_retained_devices) {
        UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(*device));
    }
    ucs_array_cleanup_dynamic(&uct_cuda_retained_devices);
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
