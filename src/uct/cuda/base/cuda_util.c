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

ucs_status_t uct_cuda_get_sys_dev_and_bus_id(CUdevice cuda_device,
                                             ucs_sys_device_t *sys_dev_p,
                                             ucs_sys_bus_id_t *bus_id_p)
{
    ucs_sys_device_t sys_dev;
    ucs_sys_bus_id_t bus_id;
    int attrib;
    ucs_status_t status;

    /* PCI domain id */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                 cuda_device));
    if (status != UCS_OK) {
        return status;
    }
    bus_id.domain = (uint16_t)attrib;

    /* PCI bus id */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                 cuda_device));
    if (status != UCS_OK) {
        return status;
    }
    bus_id.bus = (uint8_t)attrib;

    /* PCI slot id */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuDeviceGetAttribute(&attrib, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                 cuda_device));
    if (status != UCS_OK) {
        return status;
    }
    bus_id.slot = (uint8_t)attrib;

    /* Function - always 0 */
    bus_id.function = 0;

    status = ucs_topo_find_device_by_bus_id_and_user_value(
            &bus_id, (uintptr_t)cuda_device, &sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_set_class(sys_dev, UCS_TOPO_DEVICE_CLASS_ACC);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_enable_aux_path(sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    *sys_dev_p = sys_dev;
    if (bus_id_p != NULL) {
        *bus_id_p = bus_id;
    }

    return UCS_OK;
}

ucs_status_t
uct_cuda_get_sys_dev(CUdevice cuda_device, ucs_sys_device_t *sys_dev_p)
{
    return uct_cuda_get_sys_dev_and_bus_id(cuda_device, sys_dev_p, NULL);
}

CUdevice uct_cuda_get_cuda_device(ucs_sys_device_t sys_dev)
{
    uintptr_t user_value;

    user_value = ucs_topo_sys_device_get_user_value(sys_dev);
    if (user_value == UCS_SYS_DEVICE_USER_VALUE_EMPTY) {
        return CU_DEVICE_INVALID;
    }

    return (CUdevice)user_value;
}

static int uct_cuda_bus_id_is_visible(const ucs_sys_bus_id_t *visible_bus_ids,
                                      unsigned num_visible_gpus,
                                      const ucs_sys_bus_id_t *bus_id)
{
    ucs_bus_id_bit_rep_t bus_id_key = ucs_topo_get_bus_id_bit_repr(bus_id);
    unsigned i;

    for (i = 0; i < num_visible_gpus; ++i) {
        if (bus_id_key == ucs_topo_get_bus_id_bit_repr(&visible_bus_ids[i])) {
            return 1;
        }
    }

    return 0;
}

static ucs_status_t uct_cuda_init_devices_cu(ucs_sys_bus_id_t *visible_bus_ids,
                                             int *num_visible_gpus_p)
{
    const unsigned sys_device_priority = 10;
    ucs_sys_device_t sys_dev;
    int num_visible_gpus, i;
    char device_name[10];
    ucs_status_t status;
    CUdevice cuda_dev;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_visible_gpus));
    if (status != UCS_OK) {
        goto out;
    }

    ucs_assert_always(num_visible_gpus <= UCT_CUDA_MAX_DEVICES);

    for (i = 0; i < num_visible_gpus; ++i) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&cuda_dev, i));
        if (status != UCS_OK) {
            goto out;
        }

        status = uct_cuda_get_sys_dev_and_bus_id(cuda_dev, &sys_dev,
                                                 &visible_bus_ids[i]);
        if (status != UCS_OK) {
            goto out;
        }

        ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%d", cuda_dev);
        status = ucs_topo_sys_device_set_name(sys_dev, device_name,
                                              sys_device_priority);
        if (status != UCS_OK) {
            goto out;
        }
    }

    *num_visible_gpus_p = num_visible_gpus;

out:
    return status;
}

static ucs_status_t
uct_cuda_init_devices_nvml(const ucs_sys_bus_id_t *visible_bus_ids,
                           int num_visible_gpus)
{
    ucs_status_t status = UCS_OK;
    unsigned nvml_dev_count, i;
    nvmlDevice_t nvml_dev;
    nvmlPciInfo_t nvml_pci;
    ucs_sys_bus_id_t bus_id;
    ucs_sys_device_t sys_dev;

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetCount_v2, &nvml_dev_count);
    if (status != UCS_OK) {
        ucs_warn("nvml unavailable: using cuda-only gpu enumeration");
        return UCS_OK;
    }

    ucs_assert_always(nvml_dev_count <= UCT_CUDA_MAX_DEVICES);

    for (i = 0; i < nvml_dev_count; ++i) {
        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, i,
                                         &nvml_dev);
        if (status != UCS_OK) {
            goto out;
        }

        status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetPciInfo_v3, nvml_dev,
                                         &nvml_pci);
        if (status != UCS_OK) {
            goto out;
        }

        bus_id.domain   = nvml_pci.domain;
        bus_id.bus      = nvml_pci.bus;
        bus_id.slot     = nvml_pci.device;
        bus_id.function = 0;

        if (uct_cuda_bus_id_is_visible(visible_bus_ids, num_visible_gpus,
                                       &bus_id)) {
            continue;
        }

        status = uct_cuda_find_device_by_bus_id(&bus_id, &sys_dev);
        if (status != UCS_OK) {
            goto out;
        }
    }

out:
    return status;
}

static ucs_status_t uct_cuda_init_devices_internal(int *num_visible_gpus_p)
{
    ucs_sys_bus_id_t visible_bus_ids[UCT_CUDA_MAX_DEVICES];
    int num_visible_gpus;
    ucs_status_t status;

    /* Init visible devices first using the CUDA driver */
    status = uct_cuda_init_devices_cu(visible_bus_ids, &num_visible_gpus);
    if (status != UCS_OK) {
        ucs_error("failed to initialize CUDA devices: %s",
                  ucs_status_string(status));
        goto out;
    }

    /* Init the remaining non-visible devices using NVML */
    status = uct_cuda_init_devices_nvml(visible_bus_ids, num_visible_gpus);
    if (status != UCS_OK) {
        ucs_error("failed to initialize NVML devices: %s",
                  ucs_status_string(status));
        goto out;
    }

    *num_visible_gpus_p = num_visible_gpus;

out:
    return status;
}

ucs_status_t uct_cuda_init_devices(int *num_visible_gpus_p)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    static int num_visible_gpus;
    static ucs_status_t status;

    UCS_INIT_ONCE(&init_once) {
        status = uct_cuda_init_devices_internal(&num_visible_gpus);
    }

    if (status == UCS_OK) {
        *num_visible_gpus_p = num_visible_gpus;
    }

    return status;
}
