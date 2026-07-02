/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_util.h"
#include "cuda_nvml.h"
#include <ucs/sys/string.h>
#include <ucs/debug/assert.h>
#include <ucs/type/init_once.h>


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

ucs_status_t uct_cuda_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev)
{
    ucs_status_t status;

    status = ucs_topo_find_device_by_bus_id(bus_id, sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_set_class(*sys_dev, UCS_TOPO_DEVICE_CLASS_ACC);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

ucs_sys_device_t uct_cuda_get_sys_dev(CUdevice cuda_device)
{
    ucs_sys_device_t sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
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

    status = uct_cuda_find_device_by_bus_id(&bus_id, &sys_dev);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_sys_device_set_user_value(sys_dev, cuda_device);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_topo_sys_device_enable_aux_path(sys_dev);
    if (status != UCS_OK) {
        goto err;
    }

    return sys_dev;

err:
    return UCS_SYS_DEVICE_ID_UNKNOWN;
}

CUdevice uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev)
{
    uintptr_t user_value;

    user_value = ucs_topo_sys_device_get_user_value(sys_dev);
    if (user_value == UINTPTR_MAX) {
        return CU_DEVICE_INVALID;
    }

    return (CUdevice)user_value;
}

static ucs_status_t
uct_cuda_enum_gpus_internal(ucs_sys_device_t *sys_devs, unsigned *count_p)
{
    unsigned nvml_dev_count, nvml_idx;
    int cuda_dev_count, cuda_idx;
    nvmlDevice_t nvml_dev;
    nvmlPciInfo_t nvml_pci;
    ucs_sys_bus_id_t bus_id;
    ucs_sys_device_t sys_dev;
    CUdevice cuda_dev;
    ucs_status_t status;

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetCount_v2, &nvml_dev_count);
    if (status != UCS_OK) {
        /* NVML unavailable: fall back to CUDA-visible devices only. This path
         * cannot enumerate GPUs hidden by CUDA_VISIBLE_DEVICES. */
        ucs_diag("nvml unavailable: using cuda-only gpu enumeration");

        status = UCT_CUDADRV_FUNC_LOG_DEBUG(cuDeviceGetCount(&cuda_dev_count));
        if (status != UCS_OK) {
            return status;
        }

        ucs_assert_always(cuda_dev_count <= UCT_CUDA_MAX_DEVICES);

        for (cuda_idx = 0; cuda_idx < cuda_dev_count; cuda_idx++) {
            status = UCT_CUDADRV_FUNC_LOG_DEBUG(
                    cuDeviceGet(&cuda_dev, cuda_idx));
            if (status != UCS_OK) {
                return status;
            }

            sys_dev = uct_cuda_get_sys_dev(cuda_dev);
            if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
                return UCS_ERR_NO_DEVICE;
            }

            sys_devs[cuda_idx] = sys_dev;
        }

        *count_p = (unsigned)cuda_dev_count;
        return UCS_OK;
    }

    ucs_assert_always(nvml_dev_count <= UCT_CUDA_MAX_DEVICES);

    for (nvml_idx = 0; nvml_idx < nvml_dev_count; nvml_idx++) {
        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, nvml_idx,
                                         &nvml_dev);
        if (status != UCS_OK) {
            return status;
        }

        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetPciInfo_v3, nvml_dev,
                                         &nvml_pci);
        if (status != UCS_OK) {
            return status;
        }

        bus_id.domain   = nvml_pci.domain;
        bus_id.bus      = nvml_pci.bus;
        bus_id.slot     = nvml_pci.device;
        bus_id.function = 0;

        status = uct_cuda_find_device_by_bus_id(&bus_id, &sys_dev);
        if (status != UCS_OK) {
            return status;
        }

        sys_devs[nvml_idx] = sys_dev;
    }

    *count_p = nvml_dev_count;
    return UCS_OK;
}

ucs_status_t
uct_cuda_enum_gpus(const ucs_sys_device_t **sys_devs_p, unsigned *count_p)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    static ucs_sys_device_t sys_devs[UCT_CUDA_MAX_DEVICES];
    static unsigned count;
    static ucs_status_t status;

    /* Enumerate the GPUs once and cache the result. */
    UCS_INIT_ONCE(&init_once) {
        status = uct_cuda_enum_gpus_internal(sys_devs, &count);
    }

    if (status != UCS_OK) {
        return status;
    }

    if (sys_devs_p != NULL) {
        *sys_devs_p = sys_devs;
    }

    if (count_p != NULL) {
        *count_p = count;
    }

    return UCS_OK;
}
