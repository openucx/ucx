/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "vr.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


#define UCS_TOPO_VR_PCI_VENDOR_ID 0x15b3
#define UCS_TOPO_VR_CX9_DEVICE_ID 0x1025


static int
ucs_topo_vr_sys_dev_cmp(const void *elem1, const void *elem2, void *arg)
{
    const ucs_topo_sys_device_info_t *devices = arg;
    ucs_sys_device_t sys_dev1 = *(const ucs_sys_device_t*)elem1;
    ucs_sys_device_t sys_dev2 = *(const ucs_sys_device_t*)elem2;
    ucs_bus_id_bit_rep_t bus_id1, bus_id2;
    uintptr_t user_value1, user_value2;

    bus_id1 = ucs_topo_get_bus_id_bit_repr(&devices[sys_dev1].bus_id);
    bus_id2 = ucs_topo_get_bus_id_bit_repr(&devices[sys_dev2].bus_id);

    if (bus_id1 != bus_id2) {
        return (bus_id1 > bus_id2) - (bus_id1 < bus_id2);
    }

    user_value1 = devices[sys_dev1].user_value;
    user_value2 = devices[sys_dev2].user_value;

    return (user_value1 > user_value2) - (user_value1 < user_value2);
}

static void ucs_topo_vr_sys_dev_sort(ucs_topo_vr_sys_dev_array_t *sys_devs,
                                     const ucs_topo_sys_device_info_t *devices)
{
    if (ucs_array_is_empty(sys_devs)) {
        return;
    }

    ucs_qsort_r(ucs_array_begin(sys_devs), ucs_array_length(sys_devs),
                sizeof(*ucs_array_begin(sys_devs)), ucs_topo_vr_sys_dev_cmp,
                (void*)devices);
}


/* Compact the array by removing unknown devices. */
static void ucs_topo_vr_sys_dev_compact(ucs_topo_vr_sys_dev_array_t *sys_devs)
{
    size_t dst = 0;
    size_t src;

    for (src = 0; src < ucs_array_length(sys_devs); ++src) {
        if (ucs_array_elem(sys_devs, src) != UCS_SYS_DEVICE_ID_UNKNOWN) {
            ucs_array_elem(sys_devs, dst++) = ucs_array_elem(sys_devs, src);
        }
    }

    for (src = dst; src < ucs_array_length(sys_devs); ++src) {
        ucs_array_elem(sys_devs, src) = UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    ucs_array_set_length(sys_devs, dst);
}


/* This filter is required because currently CUDA gpus may have duplicates in
 * the devices array due to duplicate insertion by NVML and the CUDA driver. */
static void
ucs_topo_vr_gpu_aliases_filter(ucs_topo_vr_sys_dev_array_t *gpus,
                               const ucs_topo_sys_device_info_t *devices)
{
    ucs_bus_id_bit_rep_t bus_id1, bus_id2;
    ucs_sys_device_t sys_dev1, sys_dev2;
    size_t i;

    if (ucs_array_length(gpus) < 2) {
        return;
    }

    i = 0;
    while (i < ucs_array_length(gpus) - 1) {
        sys_dev1 = ucs_array_elem(gpus, i);
        sys_dev2 = ucs_array_elem(gpus, i + 1);

        bus_id1 = ucs_topo_get_bus_id_bit_repr(&devices[sys_dev1].bus_id);
        bus_id2 = ucs_topo_get_bus_id_bit_repr(&devices[sys_dev2].bus_id);

        if ((bus_id1 == bus_id2) &&
            (devices[sys_dev2].user_value == UCS_SYS_DEVICE_USER_VALUE_EMPTY)) {
            ucs_array_elem(gpus, i + 1) = UCS_SYS_DEVICE_ID_UNKNOWN;

            /* Promised by sorting. */
            ucs_assert(devices[sys_dev1].user_value !=
                       UCS_SYS_DEVICE_USER_VALUE_EMPTY);

            i += 2;
        } else {
            i++;
        }
    }

    ucs_topo_vr_sys_dev_compact(gpus);
}


static void
ucs_topo_vr_cx9_ports_filter(const ucs_topo_sys_device_info_t *devices,
                             ucs_topo_vr_sys_dev_array_t *cx9_ports)
{
    uint16_t vendor_id, device_id;
    ucs_sys_device_t sys_dev;
    size_t i;

    for (i = 0; i < ucs_array_length(cx9_ports); ++i) {
        sys_dev = ucs_array_elem(cx9_ports, i);

        /* TODO: Read directly from devices[sys_dev].pci_id when available. */
        vendor_id = UCS_TOPO_VR_PCI_VENDOR_ID;
        device_id = UCS_TOPO_VR_CX9_DEVICE_ID;

        if ((vendor_id != UCS_TOPO_VR_PCI_VENDOR_ID) ||
            (device_id != UCS_TOPO_VR_CX9_DEVICE_ID)) {
            ucs_trace("ignoring network device " UCS_SYS_BUS_ID_FMT
                      " with pci id %04x:%04x",
                      UCS_SYS_BUS_ID_ARG(&devices[sys_dev].bus_id), vendor_id,
                      device_id);
            ucs_array_elem(cx9_ports, i) = UCS_SYS_DEVICE_ID_UNKNOWN;
        }
    }

    ucs_topo_vr_sys_dev_compact(cx9_ports);
}


static ucs_status_t
ucs_topo_vr_devices_collect(const ucs_topo_sys_device_info_t *devices,
                            unsigned num_devices,
                            ucs_topo_vr_sys_dev_array_t *acc_devices,
                            ucs_topo_vr_sys_dev_array_t *net_devices)
{
    ucs_topo_vr_sys_dev_array_t *target_array;
    unsigned i;

    for (i = 0; i < num_devices; ++i) {
        if (devices[i].device_class == UCS_TOPO_DEVICE_CLASS_ACC) {
            target_array = acc_devices;
        } else if (devices[i].device_class == UCS_TOPO_DEVICE_CLASS_NET) {
            target_array = net_devices;
        } else {
            continue;
        }

        *ucs_array_append(target_array,
                          return UCS_ERR_NO_MEMORY) = (ucs_sys_device_t)i;
    }

    return UCS_OK;
}


static ucs_status_t
ucs_topo_vr_inventory_build(const ucs_topo_sys_device_info_t *devices,
                            unsigned num_devices,
                            ucs_topo_vr_inventory_t *inventory_p)
{
    ucs_topo_vr_sys_dev_array_t acc_devices = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_topo_vr_sys_dev_array_t net_devices = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_status_t status;

    if (num_devices == 0) {
        goto out_success;
    }

    status = ucs_topo_vr_devices_collect(devices, num_devices, &acc_devices,
                                         &net_devices);
    if (status != UCS_OK) {
        goto err_free_arrays;
    }

    ucs_topo_vr_sys_dev_sort(&acc_devices, devices);
    ucs_topo_vr_sys_dev_sort(&net_devices, devices);

    ucs_topo_vr_gpu_aliases_filter(&acc_devices, devices);
    ucs_topo_vr_cx9_ports_filter(devices, &net_devices);

out_success:
    inventory_p->gpus      = acc_devices;
    inventory_p->cx9_ports = net_devices;
    return UCS_OK;

err_free_arrays:
    ucs_array_cleanup_dynamic(&net_devices);
    ucs_array_cleanup_dynamic(&acc_devices);
    return status;
}


static void ucs_topo_vr_inventory_cleanup(ucs_topo_vr_inventory_t *inventory)
{
    ucs_array_cleanup_dynamic(&inventory->cx9_ports);
    ucs_array_cleanup_dynamic(&inventory->gpus);
}


ucs_status_t ucs_topo_vr_init_groups(const ucs_topo_sys_device_info_t *devices,
                                     unsigned num_devices,
                                     ucs_topo_group_t **groups_p,
                                     size_t *num_groups_p)
{
    ucs_topo_vr_inventory_t inventory;
    ucs_status_t status;

    status = ucs_topo_vr_inventory_build(devices, num_devices, &inventory);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("built vera-rubin inventory with %zu gpus and %zu cx9 functions",
              (size_t)ucs_array_length(&inventory.gpus),
              (size_t)ucs_array_length(&inventory.cx9_ports));

    /* TODO: Build groups from inventory. */

    ucs_topo_vr_inventory_cleanup(&inventory);

    *groups_p     = NULL;
    *num_groups_p = 0;

    return UCS_OK;
}
