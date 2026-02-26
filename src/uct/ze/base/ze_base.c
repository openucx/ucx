/*
 * Copyright (C) Intel Corporation, 2023-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_base.h"

#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>


#define UCT_ZE_LOG_LEVEL   UCS_LOG_LEVEL_DEBUG
#define UCT_ZE_MAX_DEVICES 32 /* Max root devices (GPUs) */


/* Global state */
typedef struct {
    ze_driver_handle_t driver;
    uct_ze_device_t    devices[UCT_ZE_MAX_DEVICES];
    int                num_devices;
    int                num_subdevices;  /* Total sub-devices across all devices */
    ucs_init_once_t    init_once;
    ze_result_t        init_status;  /* Store init result for later calls */
} uct_ze_base_state_t;


static uct_ze_base_state_t uct_ze_base = {
    .init_once   = UCS_INIT_ONCE_INITIALIZER,
    .init_status = ZE_RESULT_ERROR_UNINITIALIZED
};


/* Static sub-device array - populated during init, read-only after */
static uct_ze_subdevice_t
        ze_subdevices[UCT_ZE_MAX_DEVICES * UCT_ZE_MAX_SUBDEVICES];
static int ze_num_subdevices = 0;

ze_driver_handle_t uct_ze_base_get_driver(void)
{
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        return NULL;
    }

    return uct_ze_base.driver;
}

/**
 * Get PCI properties from device using separate API call
 * Compatible with older Level Zero versions
 */
static ze_result_t
uct_ze_get_pci_properties(ze_device_handle_t device, ucs_sys_bus_id_t *bus_id)
{
    ze_pci_ext_properties_t pci_props = {
        .stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES,
        .pNext = NULL
    };
    ze_result_t ret;

    ret = zeDevicePciGetPropertiesExt(device, &pci_props);
    if (ret != ZE_RESULT_SUCCESS) {
        return ret;
    }

    bus_id->domain   = (uint16_t)pci_props.address.domain;
    bus_id->bus      = (uint8_t)pci_props.address.bus;
    bus_id->slot     = (uint8_t)pci_props.address.device;
    bus_id->function = (uint8_t)pci_props.address.function;
    return ZE_RESULT_SUCCESS;
}

/**
 * Initialize ZE driver and enumerate all devices and sub-devices
 * Thread-safe via UCS_INIT_ONCE
 */
ze_result_t uct_ze_base_init(void)
{
    /* Temporary array to track PCI addresses for sys_dev sharing */
    typedef struct {
        ucs_sys_bus_id_t bus_id;
        ucs_sys_device_t sys_dev;
        int              valid;
    } uct_ze_pci_info_t;

    ze_result_t ret         = ZE_RESULT_SUCCESS;
    uint32_t driver_count   = 1;
    uint32_t root_dev_count = 0;
    uint32_t subdev_count   = 0;
    int global_subdevice_id = 0;
    ze_device_handle_t root_devices[UCT_ZE_MAX_DEVICES];
    uct_ze_pci_info_t pci_info[UCT_ZE_MAX_DEVICES];
    ucs_sys_bus_id_t bus_id;
    uct_ze_device_t *device;
    uct_ze_subdevice_t *subdevice;
    ze_result_t subret;
    ucs_status_t status;
    int device_idx;
    int found_idx;
    int i, j;
    char name[16];

    UCS_INIT_ONCE(&uct_ze_base.init_once) {
        uct_ze_base.num_subdevices = 0;

        /* Initialize Level Zero */
        ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
        if (ret != ZE_RESULT_SUCCESS) {
            ucs_debug("failure to initialize ze library: 0x%x", ret);
            uct_ze_base.init_status = ret;
            continue;
        }

        /* Get driver */
        ret = zeDriverGet(&driver_count, &uct_ze_base.driver);
        if ((ret != ZE_RESULT_SUCCESS) || (driver_count == 0)) {
            ucs_debug("failure to get ze driver: 0x%x, count=%u", ret,
                      driver_count);
            uct_ze_base.init_status = (ret != ZE_RESULT_SUCCESS) ?
                                              ret :
                                              ZE_RESULT_ERROR_UNKNOWN;
            continue;
        }

        /* Enumerate root devices (physical devices) */
        root_dev_count = UCT_ZE_MAX_DEVICES;
        ret = zeDeviceGet(uct_ze_base.driver, &root_dev_count, root_devices);
        if (ret != ZE_RESULT_SUCCESS) {
            ucs_debug("failure to get ze devices: 0x%x", ret);
            uct_ze_base.init_status = ret;
            continue;
        }

        if (root_dev_count > UCT_ZE_MAX_DEVICES) {
            ucs_warn("ze returned %u devices, limiting to %u",
                     root_dev_count, UCT_ZE_MAX_DEVICES);
            root_dev_count = UCT_ZE_MAX_DEVICES;
        }

        uct_ze_base.num_devices = (int)root_dev_count;
        ucs_debug("found %d ze root devices", uct_ze_base.num_devices);

        /* Initialize PCI info array */
        for (i = 0; i < UCT_ZE_MAX_DEVICES; i++) {
            pci_info[i].valid = 0;
        }

        /* Process each root device */
        for (device_idx = 0; device_idx < uct_ze_base.num_devices;
             device_idx++) {
            device               = &uct_ze_base.devices[device_idx];
            device->root_device  = root_devices[device_idx];
            device->device_index = device_idx;

            /* Get standard device properties */
            device->device_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            device->device_props.pNext = NULL;
            ret = zeDeviceGetProperties(device->root_device,
                                        &device->device_props);
            if (ret != ZE_RESULT_SUCCESS) {
                ucs_debug("failure to get ze device properties "
                          "for device %d: 0x%x",
                          device_idx, ret);
                continue;
            }

            /* Get PCI properties */
            ret = uct_ze_get_pci_properties(device->root_device, &bus_id);
            if (ret != ZE_RESULT_SUCCESS) {
                ucs_debug("failure to get pci properties for device %d: 0x%x",
                          device_idx, ret);
                continue;
            }

            /* Check if we already have a sys_dev for this PCI address */
            found_idx = -1;
            for (i = 0; i < device_idx; i++) {
                if (pci_info[i].valid &&
                    pci_info[i].bus_id.domain == bus_id.domain &&
                    pci_info[i].bus_id.bus == bus_id.bus &&
                    pci_info[i].bus_id.slot == bus_id.slot &&
                    pci_info[i].bus_id.function == bus_id.function) {
                    found_idx = i;
                    break;
                }
            }

            if (found_idx >= 0) {
                /* Duplicate PCI - skip topology registration, share sys_dev */
                device->sys_dev = pci_info[found_idx].sys_dev;
                ucs_debug("device %d shares sys_dev %u with device %d (same pci)",
                          device_idx, device->sys_dev, found_idx);
            } else {
                /* Register new PCI device with topology */
                status = ucs_topo_find_device_by_bus_id(&bus_id, &device->sys_dev);
                if (status != UCS_OK) {
                    ucs_debug("ucs_topo_find_device_by_bus_id failed for device %d",
                              device_idx);
                    device->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
                } else {
                    /* Set device name: "GPU0", "GPU1", etc. */
                    ucs_snprintf_safe(name, sizeof(name), "GPU%d", device_idx);
                    ucs_topo_sys_device_set_name(device->sys_dev, name, 10);

                    /* Store device index as user value for reverse lookup */
                    status = ucs_topo_sys_device_set_user_value(device->sys_dev,
                                                                device_idx);
                    if (status == UCS_OK) {
                        /* Enable auxiliary path for multi-path routing */
                        status = ucs_topo_sys_device_enable_aux_path(
                                device->sys_dev);
                        if (status != UCS_OK) {
                            ucs_debug("ucs_topo_sys_device_enable_aux_path failed "
                                      "for device %d",
                                      device_idx);
                        }
                    }
                }

                /* Remember this PCI -> sys_dev mapping */
                pci_info[device_idx].bus_id = bus_id;
                pci_info[device_idx].sys_dev = device->sys_dev;
                pci_info[device_idx].valid = 1;
            }

            /* Check for sub-devices (hierarchical) vs flat tiles */
            subdev_count = 0;
            subret = zeDeviceGetSubDevices(device->root_device, &subdev_count, NULL);

            if (subret == ZE_RESULT_SUCCESS && subdev_count > 0) {
                /* Hierarchical: Use Level Zero sub-devices */
                if (subdev_count > UCT_ZE_MAX_SUBDEVICES) {
                    ucs_warn("device %d has %u sub-devices, limiting to %d",
                             device_idx, subdev_count, UCT_ZE_MAX_SUBDEVICES);
                    subdev_count = UCT_ZE_MAX_SUBDEVICES;
                }

                ret = zeDeviceGetSubDevices(device->root_device, &subdev_count,
                                            device->subdevices);
                if (ret != ZE_RESULT_SUCCESS) {
                    ucs_debug("failure to get ze sub-devices for device %d: 0x%x",
                              device_idx, ret);
                    device->num_subdevices = 1;
                    device->subdevices[0] = device->root_device;
                } else {
                    device->num_subdevices = (int)subdev_count;
                    ucs_debug("device %d has %d sub-devices (hierarchical)",
                              device_idx, device->num_subdevices);
                }
            } else {
                /* Flat: Root device is the sub-device */
                device->num_subdevices = 1;
                device->subdevices[0] = device->root_device;
                ucs_debug("device %d is single sub-device (flat)", device_idx);
            }

            /* Populate global sub-device array */
            for (j = 0; j < device->num_subdevices; j++) {
                if (global_subdevice_id >= ucs_static_array_size(ze_subdevices)) {
                    ucs_error("too many sub-devices! max %zu",
                              ucs_static_array_size(ze_subdevices));
                    device->num_subdevices = j; /* only what was actually added */
                    break;
                }

                subdevice = &ze_subdevices[global_subdevice_id];
                subdevice->device = device;
                subdevice->subdevice_idx = j;
                subdevice->global_id = global_subdevice_id;
                global_subdevice_id++;
            }

            uct_ze_base.num_subdevices += device->num_subdevices;
        }

        ze_num_subdevices = global_subdevice_id;
        uct_ze_base.init_status = ZE_RESULT_SUCCESS;

        ucs_debug("ze init complete: %d devices, %d total sub-devices",
                  uct_ze_base.num_devices, ze_num_subdevices);
    }

    return uct_ze_base.init_status;
}

const uct_ze_subdevice_t *uct_ze_base_get_subdevice_by_global_id(int global_id)
{
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        return NULL;
    }

    if ((global_id < 0) || (global_id >= ze_num_subdevices)) {
        return NULL;
    }

    return &ze_subdevices[global_id];
}

ze_device_handle_t
uct_ze_base_get_device_handle_from_subdevice(const uct_ze_subdevice_t *subdevice)
{
    if ((subdevice == NULL) || (subdevice->device == NULL)) {
        return NULL;
    }

    if ((subdevice->subdevice_idx < 0) ||
        (subdevice->subdevice_idx >= subdevice->device->num_subdevices)) {
        return NULL;
    }

    return subdevice->device->subdevices[subdevice->subdevice_idx];
}

/**
 * Query MD resources - returns one MD per sub-device
 * This is correct because each sub-device has separate memory
 */
ucs_status_t
uct_ze_base_query_md_resources(uct_component_h component,
                               uct_md_resource_desc_t **resources_p,
                               unsigned *num_resources_p)
{
    if ((uct_ze_base_init() != ZE_RESULT_SUCCESS) || (ze_num_subdevices == 0)) {
        ucs_debug("ze initialization failed or no sub-devices, returning empty "
                  "resources");
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    /* Return single MD resource - actual sub-device selection happens in md_open */
    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

/**
 * Query devices for transport layer - returns one device per sub-device
 * Each sub-device is a separate memory domain but shares sys_dev with siblings
 */
ucs_status_t uct_ze_base_query_devices(uct_md_h md,
                                       uct_tl_device_resource_t **tl_devices_p,
                                       unsigned *num_tl_devices_p)
{
    uct_tl_device_resource_t *resources;
    const uct_ze_subdevice_t *subdevice;
    const uct_ze_device_t *device;
    int i;

    if ((uct_ze_base_init() != ZE_RESULT_SUCCESS) || (ze_num_subdevices == 0)) {
        *tl_devices_p     = NULL;
        *num_tl_devices_p = 0;
        return UCS_OK;
    }

    resources = ucs_calloc(ze_num_subdevices, sizeof(*resources),
                           "ze_tl_devices");
    if (resources == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* Return one entry per sub-device */
    for (i = 0; i < ze_num_subdevices; i++) {
        subdevice = &ze_subdevices[i];
        device    = subdevice->device;

        /* Name format: "GPU0" for single sub-device, "GPU0.0" for multi sub-device */
        if (device->num_subdevices == 1) {
            ucs_snprintf_safe(resources[i].name, UCT_DEVICE_NAME_MAX, "GPU%d",
                              device->device_index);
        } else {
            ucs_snprintf_safe(resources[i].name, UCT_DEVICE_NAME_MAX,
                              "GPU%d.%d", device->device_index,
                              subdevice->subdevice_idx);
        }

        resources[i].type = UCT_DEVICE_TYPE_ACC;

        /* CRITICAL: All sub-devices on same device share the same sys_dev */
        /* This enables correct IB affinity for all sub-devices */
        resources[i].sys_device = device->sys_dev;

        ucs_debug("sub-device %d: name=%s sys_dev=%u (device %d, sub-device "
                  "%d/%d)",
                  i, resources[i].name, resources[i].sys_device,
                  device->device_index, subdevice->subdevice_idx,
                  device->num_subdevices);
    }

    *tl_devices_p     = resources;
    *num_tl_devices_p = ze_num_subdevices;

    return UCS_OK;
}

/* Cleanup function - called on module unload */
static void UCS_F_DTOR uct_ze_base_cleanup(void)
{
    /* Note: ze_subdevices array is static, no cleanup needed */
    /* ZE driver cleanup happens automatically on process exit */
}

UCS_MODULE_INIT()
{
    return UCS_OK;
}
