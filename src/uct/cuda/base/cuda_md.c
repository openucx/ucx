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

    return;

err:
    *sys_dev_p = UCS_SYS_DEVICE_ID_UNKNOWN;
}

static ucs_status_t uct_cuda_base_get_sys_dev_nvml(unsigned device_index,
                                                   ucs_sys_device_t *sys_dev_p)
{
    ucs_status_t status;
    nvmlDevice_t nvml_device;
    nvmlPciInfo_t nvml_pci_info;
    ucs_sys_bus_id_t bus_id;
    ucs_sys_device_t sys_dev;

    status = UCT_NVML_FUNC_LOG_ERR(
            nvmlDeviceGetHandleByIndex_v2(device_index, &nvml_device));

    if (status != UCS_OK) {
        return status;
    }

    status = UCT_NVML_FUNC_LOG_ERR(
            nvmlDeviceGetPciInfo_v3(nvml_device, &nvml_pci_info));
    if (status != UCS_OK) {
        return status;
    }

    bus_id.domain   = (uint16_t)nvml_pci_info.domain;
    bus_id.bus      = (uint8_t)nvml_pci_info.bus;
    bus_id.slot     = (uint8_t)nvml_pci_info.device;
    bus_id.function = 0;

    status = ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    *sys_dev_p = sys_dev;
    return UCS_OK;
}

ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    const unsigned sys_device_priority = 10;
    unsigned num_devices, device_index;
    ucs_sys_device_t sys_dev;
    char device_name[10];
    ucs_status_t status;

    if (UCT_NVML_FUNC_LOG_ERR(nvmlInit_v2()) != UCS_OK) {
        goto err;
    }

    if ((UCT_NVML_FUNC_LOG_ERR(nvmlDeviceGetCount_v2(&num_devices)) !=
         UCS_OK) ||
        (num_devices == 0)) {
        goto cleanup;
    }

    for (device_index = 0; device_index < num_devices; ++device_index) {
        if (uct_cuda_base_get_sys_dev_nvml(device_index, &sys_dev) != UCS_OK) {
            continue;
        }

        ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%u",
                          device_index);
        status = ucs_topo_sys_device_set_name(sys_dev, device_name,
                                              sys_device_priority);
        ucs_assert_always(status == UCS_OK);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);

cleanup:
    UCT_NVML_FUNC_LOG_ERR(nvmlShutdown());
err:
    return uct_md_query_empty_md_resource(resources_p, num_resources_p);
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
