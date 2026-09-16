/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "topo_groups.h"
#include "topo_int.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucs/datastruct/array.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>

#include <dirent.h>
#include <string.h>


#define UCS_TOPO_GROUPS_MELLANOX_VENDOR_ID 0x15b3
#define UCS_TOPO_GROUPS_CX9_DEVICE_ID      0x1025
#define UCS_TOPO_GROUPS_MLX5_VF_DEVICE_ID  0x101e
#define UCS_TOPO_GROUPS_FW_VER_MAX         64


UCS_ARRAY_DECLARE_TYPE(ucs_topo_groups_sys_dev_array_t, size_t,
                       ucs_sys_device_t);


static int ucs_topo_groups_sys_dev_cmp(const void *elem1, const void *elem2,
                                       void *UCS_V_UNUSED arg)
{
    ucs_sys_device_t sys_dev1 = *(const ucs_sys_device_t*)elem1;
    ucs_sys_device_t sys_dev2 = *(const ucs_sys_device_t*)elem2;

    return ucs_topo_sys_device_cmp_nolock(sys_dev1, sys_dev2);
}

static void
ucs_topo_groups_sys_dev_sort(ucs_topo_groups_sys_dev_array_t *sys_devs)
{
    if (ucs_array_is_empty(sys_devs)) {
        return;
    }

    ucs_assert(ucs_array_begin(sys_devs) != NULL);

    ucs_qsort_r(ucs_array_begin(sys_devs), ucs_array_length(sys_devs),
                sizeof(*ucs_array_begin(sys_devs)), ucs_topo_groups_sys_dev_cmp,
                NULL);
}

/* Compare two bus ids up to their slot (excluding the function). */
static int ucs_topo_groups_bus_id_same_slot(const ucs_sys_bus_id_t *bus_id1,
                                            const ucs_sys_bus_id_t *bus_id2)
{
    return (bus_id1->domain == bus_id2->domain) &&
           (bus_id1->bus == bus_id2->bus) && (bus_id1->slot == bus_id2->slot);
}

/* Compare two bus ids. */
static int ucs_topo_groups_bus_id_equal(const ucs_sys_bus_id_t *bus_id1,
                                        const ucs_sys_bus_id_t *bus_id2)
{
    return ucs_topo_groups_bus_id_same_slot(bus_id1, bus_id2) &&
           (bus_id1->function == bus_id2->function);
}

/* This filter is required because currently CUDA gpus may have duplicates in
 * the devices array due to duplicate insertion by NVML and the CUDA driver. */
static void
ucs_topo_groups_gpu_aliases_filter(const ucs_topo_sys_device_info_t *devices,
                                   ucs_topo_groups_sys_dev_array_t *gpus)
{
    ucs_sys_device_t sys_dev1, sys_dev2;
    size_t src, dst, i;

    if (ucs_array_length(gpus) < 2) {
        return;
    }

    i = 0;
    while (i < ucs_array_length(gpus) - 1) {
        sys_dev1 = ucs_array_elem(gpus, i);
        sys_dev2 = ucs_array_elem(gpus, i + 1);

        if (ucs_topo_groups_bus_id_equal(&devices[sys_dev1].bus_id,
                                         &devices[sys_dev2].bus_id) &&
            (devices[sys_dev2].user_value == UCS_SYS_DEVICE_USER_VALUE_EMPTY)) {
            /* Mark the device as unknown to be removed later. */
            ucs_array_elem(gpus, i + 1) = UCS_SYS_DEVICE_ID_UNKNOWN;

            /* Promised by sorting. */
            ucs_assert(devices[sys_dev1].user_value !=
                       UCS_SYS_DEVICE_USER_VALUE_EMPTY);

            i += 2;
        } else {
            i++;
        }
    }

    /* Compact the array by removing unknown devices. */
    dst = 0;
    for (src = 0; src < ucs_array_length(gpus); ++src) {
        if (ucs_array_elem(gpus, src) == UCS_SYS_DEVICE_ID_UNKNOWN) {
            continue;
        }

        ucs_array_elem(gpus, dst++) = ucs_array_elem(gpus, src);
    }

    ucs_array_set_length(gpus, dst);
}

static ucs_status_t
ucs_topo_groups_read_ib_fw_ver(const ucs_sys_bus_id_t *bus_id, char *fw_ver,
                               size_t max)
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

static int ucs_topo_groups_is_nic_cx9(const ucs_topo_sys_device_info_t *device,
                                      const char **reason_p)
{
    const ucs_sys_pci_id_t *pci_id = &device->pci_id;
    const ucs_sys_bus_id_t *bus_id = &device->bus_id;
    char fw_ver[UCS_TOPO_GROUPS_FW_VER_MAX];
    ucs_status_t status;

    if (pci_id->vendor != UCS_TOPO_GROUPS_MELLANOX_VENDOR_ID) {
        *reason_p = "not a mellanox device";
        return 0;
    }

    if (pci_id->device == UCS_TOPO_GROUPS_CX9_DEVICE_ID) {
        *reason_p = "cx9 device by device id";
        return 1;
    }

    if (pci_id->device != UCS_TOPO_GROUPS_MLX5_VF_DEVICE_ID) {
        *reason_p = "not a cx9 device by device id";
        return 0;
    }

    /* PCI device ID is not indicative when the device is a VF, instead use
     * the fact that fw version is 82.XX.XXXX for CX9 */
    status = ucs_topo_groups_read_ib_fw_ver(bus_id, fw_ver, sizeof(fw_ver));
    if (status != UCS_OK) {
        ucs_debug("could not read firmware version (error: %s)",
                  ucs_status_string(status));
        *reason_p = "vf device, could not read firmware version";
        return 0;
    }

    if (strncmp(fw_ver, "82.", 3) != 0) {
        *reason_p = "vf device, firmware version mismatch";
        return 0;
    }

    *reason_p = "cx9 device by firmware version";
    return 1;
}

static void
ucs_topo_groups_nics_cx9_filter(const ucs_topo_sys_device_info_t *devices,
                                ucs_topo_groups_sys_dev_array_t *nics)
{
    size_t dst = 0;
    const ucs_topo_sys_device_info_t *device;
    ucs_sys_device_t sys_dev;
    const char *reason;
    size_t src;
    int is_cx9;

    if (ucs_array_is_empty(nics)) {
        return;
    }

    ucs_assert(ucs_array_begin(nics) != NULL);

    for (src = 0; src < ucs_array_length(nics); ++src) {
        sys_dev = ucs_array_elem(nics, src);
        device  = &devices[sys_dev];

        /* TODO: Refactor to have this provided by UCT */
        is_cx9 = ucs_topo_groups_is_nic_cx9(device, &reason);
        if (is_cx9) {
            ucs_array_elem(nics, dst++) = sys_dev;
        }

        ucs_debug("cx9_filter: network device sys_dev=%u, "
                  "bus_id=" UCS_SYS_BUS_ID_FMT ", pci_id=" UCS_SYS_PCI_ID_FMT
                  " %s (%s)",
                  sys_dev, UCS_SYS_BUS_ID_ARG(&device->bus_id),
                  UCS_SYS_PCI_ID_ARG(&device->pci_id),
                  is_cx9 ? "added" : "skipped",
                  reason);
    }

    ucs_array_set_length(nics, dst);
}

static ucs_status_t
ucs_topo_groups_devices_collect(const ucs_topo_sys_device_info_t *devices,
                                unsigned num_devices,
                                ucs_topo_groups_sys_dev_array_t *acc_devices,
                                ucs_topo_groups_sys_dev_array_t *net_devices)
{
    ucs_topo_groups_sys_dev_array_t *target_array;
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

static ucs_status_t ucs_topo_groups_elements_build(
        const ucs_topo_sys_device_info_t *devices,
        const ucs_topo_groups_sys_dev_array_t *sys_devices,
        int (*bus_id_match)(const ucs_sys_bus_id_t *bus_id1,
                            const ucs_sys_bus_id_t *bus_id2),
        ucs_topo_group_element_array_t *elements)
{
    const ucs_sys_bus_id_t *bus_id, *prev_bus_id;
    const ucs_sys_device_t *sys_dev;
    ucs_topo_group_element_t *element;

    if (ucs_array_is_empty(sys_devices)) {
        return UCS_OK;
    }

    ucs_assert(ucs_array_begin(sys_devices) != NULL);

    prev_bus_id = NULL;
    ucs_array_for_each(sys_dev, sys_devices) {
        bus_id = &devices[*sys_dev].bus_id;

        if ((prev_bus_id == NULL) || !bus_id_match(prev_bus_id, bus_id)) {
            /* No match, add a new element. */
            element = ucs_array_append(elements, return UCS_ERR_NO_MEMORY);
            memset(element, 0, sizeof(*element));
            prev_bus_id = bus_id;
        }

        if (element->num_sys_devs >= UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT) {
            ucs_error("too many devices (%zu) with bus id " UCS_SYS_BUS_ID_FMT,
                      element->num_sys_devs, UCS_SYS_BUS_ID_ARG(bus_id));
            return UCS_ERR_EXCEEDS_LIMIT;
        }

        element->sys_devs[element->num_sys_devs++] = *sys_dev;
    }

    return UCS_OK;
}

static void ucs_topo_init_groups(ucs_topo_groups_t *groups)
{
    ucs_array_init_dynamic(groups);
}

void ucs_topo_release_groups(ucs_topo_groups_t *groups)
{
    size_t i;

    ucs_assert(groups != NULL);

    for (i = 0; i < ucs_array_length(groups); ++i) {
        ucs_topo_release_group(&ucs_array_elem(groups, i));
    }

    ucs_array_cleanup_dynamic(groups);
}

void ucs_topo_init_group(ucs_topo_group_t *group)
{
    ucs_array_init_dynamic(&group->gpus);
    ucs_array_init_dynamic(&group->nics);
}

void ucs_topo_release_group(ucs_topo_group_t *group)
{
    ucs_assert(group != NULL);
    ucs_array_cleanup_dynamic(&group->nics);
    ucs_array_cleanup_dynamic(&group->gpus);
}

static ucs_status_t
ucs_topo_groups_inventory_build(const ucs_topo_sys_device_info_t *devices,
                                unsigned num_devices, int is_vera_rubin,
                                ucs_topo_group_t *inventory_p)
{
    ucs_topo_groups_sys_dev_array_t acc_devices = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_topo_groups_sys_dev_array_t net_devices = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_topo_group_t inventory;
    ucs_status_t status;

    ucs_topo_init_group(&inventory);

    status = ucs_topo_groups_devices_collect(devices, num_devices, &acc_devices,
                                             &net_devices);
    if (status != UCS_OK) {
        goto err_free_arrays;
    }

    ucs_topo_groups_sys_dev_sort(&acc_devices);
    ucs_topo_groups_sys_dev_sort(&net_devices);

    /* TODO: Remove this filter when NVML duplicates issue is fixed. */
    ucs_topo_groups_gpu_aliases_filter(devices, &acc_devices);

    if (is_vera_rubin) {
        ucs_topo_groups_nics_cx9_filter(devices, &net_devices);
    }

    /* Accelerator devices (GPUs) are grouped by full bus id equality. */
    status = ucs_topo_groups_elements_build(devices, &acc_devices,
                                            ucs_topo_groups_bus_id_equal,
                                            &inventory.gpus);
    if (status != UCS_OK) {
        goto err_free_arrays;
    }

    /* Network devices (NICs) are grouped by bdf equality excl. the function. */
    status = ucs_topo_groups_elements_build(devices, &net_devices,
                                            ucs_topo_groups_bus_id_same_slot,
                                            &inventory.nics);
    if (status != UCS_OK) {
        goto err_free_arrays;
    }

    ucs_debug("built inventory with %zu physical gpus (%zu devices) and %zu "
              "physical nics (%zu devices)",
              ucs_array_length(&inventory.gpus), ucs_array_length(&acc_devices),
              ucs_array_length(&inventory.nics),
              ucs_array_length(&net_devices));

    ucs_array_cleanup_dynamic(&net_devices);
    ucs_array_cleanup_dynamic(&acc_devices);

    *inventory_p = inventory;
    return UCS_OK;

err_free_arrays:
    ucs_topo_release_group(&inventory);
    ucs_array_cleanup_dynamic(&net_devices);
    ucs_array_cleanup_dynamic(&acc_devices);
    return status;
}

static ucs_status_t
ucs_topo_groups_build_groups(const ucs_topo_group_t *inventory,
                             ucs_topo_groups_t *groups)
{
    (void)inventory;
    (void)groups;

    /* TODO: Build groups from inventory. */

    return UCS_OK;
}

ucs_status_t
ucs_topo_build_groups_inner(const ucs_topo_sys_device_info_t *devices,
                            unsigned num_devices, ucs_topo_groups_t *groups_p)
{
    ucs_topo_group_t inventory;
    ucs_topo_groups_t groups;
    ucs_status_t status;

    ucs_topo_init_groups(&groups);

    status = ucs_topo_groups_inventory_build(devices, num_devices, 1,
                                             &inventory);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_groups_build_groups(&inventory, &groups);
    if (status != UCS_OK) {
        goto err_cleanup_inventory;
    }

    ucs_topo_release_group(&inventory);

    ucs_debug("initialized topo groups with %zu groups",
              ucs_array_length(&groups));

    *groups_p = groups;
    return UCS_OK;

err_cleanup_inventory:
    ucs_topo_release_group(&inventory);
    ucs_topo_release_groups(&groups);
    return status;
}
