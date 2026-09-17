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
#include <ucs/datastruct/string_buffer.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>

#include <dirent.h>
#include <string.h>


#define UCS_TOPO_GROUPS_FW_VER_MAX 64


UCS_ARRAY_DECLARE_TYPE(ucs_topo_groups_sys_dev_array_t, size_t,
                       ucs_sys_device_t);
UCS_ARRAY_DECLARE_TYPE(ucs_topo_groups_numa_node_array_t, size_t,
                       ucs_numa_node_t);


static int
ucs_topo_groups_sys_dev_cmp(const void *elem1, const void *elem2, void *arg)
{
    const ucs_topo_sys_device_info_t *devices = arg;
    ucs_sys_device_t sys_dev1 = *(const ucs_sys_device_t*)elem1;
    ucs_sys_device_t sys_dev2 = *(const ucs_sys_device_t*)elem2;

    return ucs_topo_sys_device_info_cmp(&devices[sys_dev1], &devices[sys_dev2]);
}

static void
ucs_topo_groups_sys_dev_sort(ucs_topo_groups_sys_dev_array_t *sys_devs,
                             const ucs_topo_sys_device_info_t *devices)
{
    if (ucs_array_is_empty(sys_devs)) {
        return;
    }

    ucs_assert(ucs_array_begin(sys_devs) != NULL);

    ucs_qsort_r(ucs_array_begin(sys_devs), ucs_array_length(sys_devs),
                sizeof(*ucs_array_begin(sys_devs)), ucs_topo_groups_sys_dev_cmp,
                (void*)devices);
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
                  is_cx9 ? "added" : "skipped", reason);
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

    ucs_topo_groups_sys_dev_sort(&acc_devices, devices);
    ucs_topo_groups_sys_dev_sort(&net_devices, devices);

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

static ucs_status_t ucs_topo_groups_get_or_add_group_by_numa_node(
        ucs_numa_node_t numa_node,
        ucs_topo_groups_numa_node_array_t *numa_nodes,
        ucs_topo_groups_t *groups, ucs_topo_group_t **group_p)
{
    ucs_topo_group_t *group;
    size_t i;

    /* Check if the group already exists for the given NUMA node. */
    for (i = 0; i < ucs_array_length(numa_nodes); ++i) {
        if (ucs_array_elem(numa_nodes, i) == numa_node) {
            *group_p = &ucs_array_elem(groups, i);
            return UCS_OK;
        }
    }

    *ucs_array_append(numa_nodes, return UCS_ERR_NO_MEMORY) = numa_node;

    group = ucs_array_append(groups, return UCS_ERR_NO_MEMORY);
    ucs_topo_init_group(group);

    *group_p = group;
    return UCS_OK;
}

static ucs_status_t ucs_topo_groups_add_elements_by_numa_node(
        const ucs_topo_sys_device_info_t *devices,
        const ucs_topo_group_element_array_t *elements,
        ucs_topo_groups_numa_node_array_t *numa_nodes,
        ucs_topo_device_class_t device_class, ucs_topo_groups_t *groups)
{
    ucs_topo_group_element_array_t *group_elements;
    const ucs_topo_group_element_t *element;
    ucs_topo_group_t *group;
    ucs_numa_node_t numa_node;
    ucs_status_t status;

    ucs_assert((device_class == UCS_TOPO_DEVICE_CLASS_ACC) ||
               (device_class == UCS_TOPO_DEVICE_CLASS_NET));

    ucs_array_for_each(element, elements) {
        numa_node = devices[element->sys_devs[0]].numa_node;
        if (numa_node == UCS_NUMA_NODE_UNDEFINED) {
            ucs_debug("skipping topology element for system device %u with "
                      "undefined numa node",
                      element->sys_devs[0]);
            continue;
        }

        status = ucs_topo_groups_get_or_add_group_by_numa_node(numa_node,
                                                               numa_nodes,
                                                               groups, &group);
        if (status != UCS_OK) {
            return status;
        }

        group_elements = (device_class == UCS_TOPO_DEVICE_CLASS_ACC) ?
                                 &group->gpus :
                                 &group->nics;
        *ucs_array_append(group_elements, return UCS_ERR_NO_MEMORY) = *element;
    }

    return UCS_OK;
}

static ucs_status_t
ucs_topo_groups_build_groups(const ucs_topo_sys_device_info_t *devices,
                             const ucs_topo_group_t *inventory,
                             ucs_topo_groups_t *groups)
{
    ucs_topo_groups_numa_node_array_t numa_nodes = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_status_t status;

    status = ucs_topo_groups_add_elements_by_numa_node(
            devices, &inventory->gpus, &numa_nodes,
            UCS_TOPO_DEVICE_CLASS_ACC, groups);
    if (status != UCS_OK) {
        goto out_cleanup_numa_nodes;
    }

    status = ucs_topo_groups_add_elements_by_numa_node(
            devices, &inventory->nics, &numa_nodes,
            UCS_TOPO_DEVICE_CLASS_NET, groups);

out_cleanup_numa_nodes:
    ucs_array_cleanup_dynamic(&numa_nodes);
    return status;
}

static void
ucs_topo_groups_log_element(const ucs_topo_sys_device_info_t *devices,
                            const ucs_topo_group_element_t *element,
                            size_t group_idx,
                            ucs_topo_device_class_t device_class)
{
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    const ucs_topo_sys_device_info_t *first_device, *device;
    size_t i;

    ucs_assert((device_class == UCS_TOPO_DEVICE_CLASS_ACC) ||
               (device_class == UCS_TOPO_DEVICE_CLASS_NET));

    if (element->num_sys_devs == 0) {
        ucs_string_buffer_appendf(&strb, " <empty>");
        goto out;
    }

    first_device = &devices[element->sys_devs[0]];

    /* With priority 0 the device name is its bdf, which is already printed */
    if (first_device->name_priority > 0) {
        ucs_string_buffer_appendf(&strb, " ");
        for (i = 0; i < element->num_sys_devs; ++i) {
            device = &devices[element->sys_devs[i]];
            ucs_string_buffer_appendf(&strb, "%s/", device->name);
        }
        ucs_string_buffer_rtrim(&strb, "/");
    }

    ucs_string_buffer_appendf(&strb, " bdf " UCS_SYS_BUS_ID_FMT,
                              UCS_SYS_BUS_ID_ARG(&first_device->bus_id));
    if (device_class == UCS_TOPO_DEVICE_CLASS_NET) {
        for (i = 1; i < element->num_sys_devs; ++i) {
            device = &devices[element->sys_devs[i]];
            ucs_string_buffer_appendf(&strb, "/%u",
                                      (unsigned)device->bus_id.function);
        }
    }

    ucs_string_buffer_appendf(&strb, " sys_dev ");
    ucs_string_buffer_append_array(&strb, "/", "%hhu", element->sys_devs,
                                   element->num_sys_devs);

out:
    ucs_debug("topology group %zu element: %s%s", group_idx,
              (device_class == UCS_TOPO_DEVICE_CLASS_ACC) ? "gpu" : "nic",
              ucs_string_buffer_cstr(&strb));
}

static void ucs_topo_groups_log(const ucs_topo_sys_device_info_t *devices,
                                const ucs_topo_groups_t *groups)
{
    const ucs_topo_group_element_t *element;
    const ucs_topo_group_t *group;
    size_t group_idx;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        return;
    }

    ucs_array_for_each_index(group, group_idx, groups) {
        ucs_debug("topology group %zu: %zu gpus, %zu nics", group_idx,
                  ucs_array_length(&group->gpus),
                  ucs_array_length(&group->nics));

        ucs_log_indent(1);

        ucs_array_for_each(element, &group->gpus) {
            ucs_topo_groups_log_element(devices, element, group_idx,
                                        UCS_TOPO_DEVICE_CLASS_ACC);
        }

        ucs_array_for_each(element, &group->nics) {
            ucs_topo_groups_log_element(devices, element, group_idx,
                                        UCS_TOPO_DEVICE_CLASS_NET);
        }

        ucs_log_indent(-1);
    }
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

    status = ucs_topo_groups_build_groups(devices, &inventory, &groups);
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    ucs_topo_release_group(&inventory);

    ucs_topo_groups_log(devices, &groups);

    ucs_debug("initialized topo groups with %zu groups",
              ucs_array_length(&groups));

    *groups_p = groups;
    return UCS_OK;

err_cleanup:
    ucs_topo_release_group(&inventory);
    ucs_topo_release_groups(&groups);
    return status;
}
