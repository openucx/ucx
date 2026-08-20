/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "vr_inventory.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>

#include <dirent.h>
#include <string.h>


#define UCS_TOPO_VR_PCI_VENDOR_ID     0x15b3
#define UCS_TOPO_VR_CX9_DEVICE_ID     0x1025
#define UCS_TOPO_VR_MLX5_VF_DEVICE_ID 0x101e
#define UCS_TOPO_VR_FW_VER_MAX        64


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


static ucs_status_t ucs_topo_vr_read_fw_ver(const ucs_sys_bus_id_t *bus_id,
                                            char *fw_ver, size_t max)
{
    char *sysfs_path;
    struct dirent *entry;
    ucs_status_t status;
    size_t path_len;
    DIR *dir;

    status = ucs_string_alloc_path_buffer(&sysfs_path, "sysfs_path");
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_bus_id_to_sysfs_path(bus_id, sysfs_path, PATH_MAX);
    if (status != UCS_OK) {
        goto out_free_sysfs_path;
    }

    path_len = strlen(sysfs_path);
    ucs_strncpy_safe(sysfs_path + path_len, "/infiniband", PATH_MAX - path_len);

    dir = opendir(sysfs_path);
    if (dir == NULL) {
        status = UCS_ERR_NO_ELEM;
        goto out_free_sysfs_path;
    }

    /* Find the device name directory (e.g. mlx5_0) */
    do {
        entry = readdir(dir);
    } while ((entry != NULL) && (entry->d_name[0] == '.'));

    if (entry == NULL) {
        status = UCS_ERR_NO_ELEM;
        goto out_close_dir;
    }

    if (ucs_read_file_str(fw_ver, max, 1, "%s/%s/fw_ver", sysfs_path,
                          entry->d_name) < 0) {
        status = UCS_ERR_IO_ERROR;
        goto out_close_dir;
    }

    ucs_strtrim(fw_ver);
    status = UCS_OK;

out_close_dir:
    closedir(dir);
out_free_sysfs_path:
    ucs_free(sysfs_path);
    return status;
}


static void
ucs_topo_vr_cx9_ports_filter(const ucs_topo_sys_device_info_t *devices,
                             ucs_topo_vr_sys_dev_array_t *cx9_ports)
{
    char fw_ver[UCS_TOPO_VR_FW_VER_MAX];
    uint16_t vendor_id, device_id;
    ucs_sys_device_t sys_dev;
    ucs_status_t status;
    size_t i;

    ucs_log_indent(1);

    for (i = 0; i < ucs_array_length(cx9_ports); ++i) {
        sys_dev = ucs_array_elem(cx9_ports, i);

        ucs_log_indent(-1);

        ucs_trace("cx9_ports_filter: processing network "
                  "device " UCS_SYS_BUS_ID_FMT,
                  UCS_SYS_BUS_ID_ARG(&devices[sys_dev].bus_id));

        ucs_log_indent(1);

        /* TODO: Read directly from devices[sys_dev].pci_id when available. */
        vendor_id = UCS_TOPO_VR_PCI_VENDOR_ID;
        device_id = UCS_TOPO_VR_MLX5_VF_DEVICE_ID;

        if (vendor_id == UCS_TOPO_VR_PCI_VENDOR_ID) {
            if (device_id == UCS_TOPO_VR_CX9_DEVICE_ID) {
                ucs_trace("cx9 device found (device id)");
                continue;
            } else if (device_id == UCS_TOPO_VR_MLX5_VF_DEVICE_ID) {
                ucs_trace("mlx5 VF device found");
                status = ucs_topo_vr_read_fw_ver(&devices[sys_dev].bus_id,
                                                 fw_ver, sizeof(fw_ver));
                if (status == UCS_OK) {
                    if (strncmp(fw_ver, "82.", 3) == 0) {
                        ucs_trace("cx9 device found (firmware version)");
                        continue;
                    } else {
                        ucs_trace("firmware version mismatch: %s", fw_ver);
                    }
                } else {
                    ucs_trace("could not read firmware version (error: %s)",
                              ucs_status_string(status));
                }
            }
        }

        ucs_trace("ignoring network device " UCS_SYS_BUS_ID_FMT
                  " (pci id %04x:%04x)",
                  UCS_SYS_BUS_ID_ARG(&devices[sys_dev].bus_id), vendor_id,
                  device_id);
        ucs_array_elem(cx9_ports, i) = UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    ucs_log_indent(-1);

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


ucs_status_t
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


void ucs_topo_vr_inventory_cleanup(ucs_topo_vr_inventory_t *inventory)
{
    ucs_array_cleanup_dynamic(&inventory->cx9_ports);
    ucs_array_cleanup_dynamic(&inventory->gpus);
}
