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
