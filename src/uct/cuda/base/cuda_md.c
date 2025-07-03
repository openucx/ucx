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


/* Assume uniformity of the GPU devices here */
int uct_cuda_base_has_c2c(void)
{
#if CUDA_VERSION >= 12080
    ucs_status_t status;
    nvmlDevice_t device;
    nvmlFieldValue_t fv[2];
    unsigned i, links;

    status = UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetHandleByIndex(0, &device));
    if (status != UCS_OK) {
        return 0;
    }

    /* Check if any C2C between GPU to CPU */
    fv->fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
    status = UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device, 1, fv));
    if ((status != UCS_OK) || (fv->nvmlReturn != NVML_SUCCESS)) {
        return 0;
    }

    /* Check availability of a C2C */
    links = fv->value.uiVal;
    for (i = 0; i < links; i++) {
        fv[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
        fv[0].scopeId = i;
        fv[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
        fv[1].scopeId = i;
        status = UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetFieldValues(device, 2, fv));
        if ((status == UCS_OK) &&
            (fv[0].nvmlReturn == NVML_SUCCESS) &&
            (fv[0].value.uiVal == 1) &&
            (fv[1].nvmlReturn == NVML_SUCCESS)) {

            ucs_debug("GPUs have C2C link UP links=%d", links);
            return 1;
        }
    }
#endif
    return 0;
}

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


    if (uct_cuda_base_has_c2c()) {
        status = ucs_topo_sys_device_enable_aux_path(*sys_dev_p);
        if (status != UCS_OK) {
            goto err;
        }
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

UCS_STATIC_INIT
{
    ucs_status_t status;

    UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));

    status = UCT_NVML_FUNC(nvmlInit_v2(), UCS_LOG_LEVEL_DIAG);
    if (status != UCS_OK) {
        ucs_fatal("Could not initialize NVML: %s", ucs_status_string(status));
    }
}

UCS_STATIC_CLEANUP
{
    nvmlShutdown();
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
