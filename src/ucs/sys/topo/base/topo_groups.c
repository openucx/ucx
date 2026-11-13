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
#include <ucs/sys/math.h>

#include <stdint.h>
#include <string.h>


const char *ucs_topo_group_by_names[] = {
    [UCS_TOPO_GROUP_BY_AUTO]       = "auto",
    [UCS_TOPO_GROUP_BY_NUMA]       = "numa",
    [UCS_TOPO_GROUP_BY_PCI_SWITCH] = "pci_switch",
    [UCS_TOPO_GROUP_BY_LAST]       = NULL
};

UCS_ARRAY_DECLARE_TYPE(ucs_topo_groups_sys_dev_array_t, size_t,
                       ucs_sys_device_t);

typedef struct {
    ucs_numa_node_t numa_node;
    size_t          group_idx;
} ucs_topo_groups_numa_node_t;

UCS_ARRAY_DECLARE_TYPE(ucs_topo_groups_numa_node_array_t, size_t,
                       ucs_topo_groups_numa_node_t);


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

static ucs_status_t ucs_topo_groups_devices_collect(
        const ucs_topo_sys_device_info_t *devices, unsigned num_devices,
        const ucs_sys_device_bitmap_t *net_device_filter,
        ucs_topo_groups_sys_dev_array_t *acc_devices,
        ucs_topo_groups_sys_dev_array_t *net_devices)
{
    ucs_topo_groups_sys_dev_array_t *target_array;
    unsigned i;

    for (i = 0; i < num_devices; ++i) {
        if (devices[i].device_class == UCS_TOPO_DEVICE_CLASS_ACC) {
            target_array = acc_devices;
        } else if (devices[i].device_class == UCS_TOPO_DEVICE_CLASS_NET) {
            if ((net_device_filter != NULL) &&
                !UCS_STATIC_BITMAP_GET(*net_device_filter, i)) {
                ucs_debug("skipping network device %s (sys_dev=%u): not in "
                          "the device filter",
                          devices[i].name, i);
                continue;
            }

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

static ucs_status_t ucs_topo_groups_inventory_build(
        const ucs_topo_sys_device_info_t *devices, unsigned num_devices,
        const ucs_sys_device_bitmap_t *net_device_filter,
        ucs_topo_group_t *inventory_p)
{
    ucs_topo_groups_sys_dev_array_t acc_devices = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_topo_groups_sys_dev_array_t net_devices = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_topo_group_t inventory;
    ucs_status_t status;

    ucs_topo_init_group(&inventory);

    status = ucs_topo_groups_devices_collect(devices, num_devices,
                                             net_device_filter, &acc_devices,
                                             &net_devices);
    if (status != UCS_OK) {
        goto err_free_arrays;
    }

    ucs_topo_groups_sys_dev_sort(&acc_devices, devices);
    ucs_topo_groups_sys_dev_sort(&net_devices, devices);

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
    ucs_topo_groups_numa_node_t *entry;
    ucs_topo_group_t *group;
    size_t i;

    /* Check if the group already exists for the given NUMA node. */
    for (i = 0; i < ucs_array_length(numa_nodes); ++i) {
        entry = &ucs_array_elem(numa_nodes, i);
        if (entry->numa_node == numa_node) {
            *group_p = &ucs_array_elem(groups, entry->group_idx);
            return UCS_OK;
        }
    }

    entry            = ucs_array_append(numa_nodes, return UCS_ERR_NO_MEMORY);
    entry->numa_node = numa_node;
    entry->group_idx = ucs_array_length(groups);

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
ucs_topo_groups_build_groups_by_numa(const ucs_topo_sys_device_info_t *devices,
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

/* Decide the PCIe switch of every GPU, as the deepest PCIe bridge it shares
 * with any NIC */
static void ucs_topo_groups_get_pci_bridge_depths(
        const ucs_topo_group_t *inventory,
        ucs_topo_pci_bridge_depth_func_t pci_bridge_depth, unsigned *depths,
        unsigned *switch_depths)
{
    const size_t num_gpus = ucs_array_length(&inventory->gpus);
    const ucs_topo_group_element_t *nic, *gpu;
    size_t gpu_idx, nic_idx;
    unsigned depth;

    ucs_array_for_each_index(nic, nic_idx, &inventory->nics) {
        ucs_array_for_each_index(gpu, gpu_idx, &inventory->gpus) {
            depth = pci_bridge_depth(gpu->sys_devs[0], nic->sys_devs[0]);
            depths[(nic_idx * num_gpus) + gpu_idx] = depth;
            switch_depths[gpu_idx] = ucs_max(switch_depths[gpu_idx], depth);
        }
    }
}

/* Create the groups in the order of their first GPU */
static ucs_status_t ucs_topo_groups_add_gpus_by_pci_switch(
        const ucs_topo_group_t *inventory,
        ucs_topo_pci_bridge_depth_func_t pci_bridge_depth,
        const unsigned *switch_depths, size_t *gpu_groups,
        ucs_topo_groups_t *groups)
{
    const ucs_topo_group_element_t *gpu, *prev_gpu;
    size_t gpu_idx, prev_idx;
    ucs_topo_group_t *group;

    ucs_array_for_each_index(gpu, gpu_idx, &inventory->gpus) {
        gpu_groups[gpu_idx] = SIZE_MAX;
        if (switch_depths[gpu_idx] == 0) {
            continue;
        }

        for (prev_idx = 0; prev_idx < gpu_idx; ++prev_idx) {
            prev_gpu = &ucs_array_elem(&inventory->gpus, prev_idx);
            if ((switch_depths[prev_idx] == switch_depths[gpu_idx]) &&
                (pci_bridge_depth(prev_gpu->sys_devs[0], gpu->sys_devs[0]) >=
                 switch_depths[gpu_idx])) {
                gpu_groups[gpu_idx] = gpu_groups[prev_idx];
                break;
            }
        }

        if (gpu_groups[gpu_idx] == SIZE_MAX) {
            gpu_groups[gpu_idx] = ucs_array_length(groups);
            group = ucs_array_append(groups, return UCS_ERR_NO_MEMORY);
            ucs_topo_init_group(group);
        }

        group = &ucs_array_elem(groups, gpu_groups[gpu_idx]);
        *ucs_array_append(&group->gpus, return UCS_ERR_NO_MEMORY) = *gpu;
    }

    return UCS_OK;
}

/* The PCIe switches decided above a NIC are all its ancestors, so the deepest
 * one is unique */
static ucs_status_t ucs_topo_groups_add_nics_by_pci_switch(
        const ucs_topo_group_t *inventory, const unsigned *depths,
        const unsigned *switch_depths, const size_t *gpu_groups,
        ucs_topo_groups_t *groups, ucs_topo_group_t *unmatched)
{
    const size_t num_gpus = ucs_array_length(&inventory->gpus);
    const ucs_topo_group_element_t *nic;
    size_t gpu_idx, nic_idx, group_idx;
    ucs_topo_group_t *group;
    unsigned best_depth;

    ucs_array_for_each_index(nic, nic_idx, &inventory->nics) {
        best_depth = 0;
        group_idx  = SIZE_MAX;
        for (gpu_idx = 0; gpu_idx < num_gpus; ++gpu_idx) {
            if ((switch_depths[gpu_idx] > best_depth) &&
                (depths[(nic_idx * num_gpus) + gpu_idx] ==
                 switch_depths[gpu_idx])) {
                best_depth = switch_depths[gpu_idx];
                group_idx  = gpu_groups[gpu_idx];
            }
        }

        group = (group_idx == SIZE_MAX) ? unmatched :
                                          &ucs_array_elem(groups, group_idx);
        *ucs_array_append(&group->nics, return UCS_ERR_NO_MEMORY) = *nic;
    }

    return UCS_OK;
}

/* A PCIe switch group is left without NICs when all its NICs joined deeper
 * PCIe switches, so its GPUs are unmatched as well. They are added in the
 * inventory order, as in any other group. */
static ucs_status_t ucs_topo_groups_add_unmatched_gpus(
        const ucs_topo_group_t *inventory, const size_t *gpu_groups,
        const ucs_topo_groups_t *groups, ucs_topo_group_t *unmatched)
{
    const ucs_topo_group_element_t *gpu;
    size_t gpu_idx;

    ucs_array_for_each_index(gpu, gpu_idx, &inventory->gpus) {
        if ((gpu_groups[gpu_idx] == SIZE_MAX) ||
            ucs_array_is_empty(
                    &ucs_array_elem(groups, gpu_groups[gpu_idx]).nics)) {
            *ucs_array_append(&unmatched->gpus,
                              return UCS_ERR_NO_MEMORY) = *gpu;
        }
    }

    return UCS_OK;
}

static void
ucs_topo_groups_remove_groups_without_nics(ucs_topo_groups_t *groups)
{
    size_t group_idx, num_groups;
    ucs_topo_group_t *group;

    num_groups = 0;
    ucs_array_for_each_index(group, group_idx, groups) {
        if (ucs_array_is_empty(&group->nics)) {
            ucs_topo_release_group(group);
        } else {
            ucs_array_elem(groups, num_groups++) = *group;
        }
    }

    ucs_array_set_length(groups, num_groups);
}

/* See UCS_TOPO_GROUP_BY_PCI_SWITCH */
static ucs_status_t ucs_topo_groups_build_groups_by_pci_switch(
        const ucs_topo_sys_device_info_t *devices,
        const ucs_topo_group_t *inventory,
        ucs_topo_pci_bridge_depth_func_t pci_bridge_depth,
        ucs_topo_groups_t *groups)
{
    const size_t num_gpus = ucs_array_length(&inventory->gpus);
    const size_t num_nics = ucs_array_length(&inventory->nics);
    unsigned *depths      = NULL, *switch_depths = NULL;
    size_t *gpu_groups    = NULL;
    ucs_topo_group_t unmatched;
    ucs_status_t status;

    ucs_assert(pci_bridge_depth != NULL);

    if ((num_gpus == 0) || (num_nics == 0)) {
        /* No GPU and NIC share a PCIe switch */
        return ucs_topo_groups_build_groups_by_numa(devices, inventory,
                                                    groups);
    }

    ucs_topo_init_group(&unmatched);

    switch_depths = ucs_calloc(num_gpus, sizeof(*switch_depths),
                               "topo switch depths");
    gpu_groups    = ucs_malloc(num_gpus * sizeof(*gpu_groups),
                               "topo gpu groups");
    depths        = ucs_malloc(num_gpus * num_nics * sizeof(*depths),
                               "topo depths");
    if ((switch_depths == NULL) || (gpu_groups == NULL) || (depths == NULL)) {
        ucs_error("failed to allocate pci switch grouping state for %zu gpus "
                  "and %zu nics",
                  num_gpus, num_nics);
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    ucs_topo_groups_get_pci_bridge_depths(inventory, pci_bridge_depth, depths,
                                          switch_depths);

    status = ucs_topo_groups_add_gpus_by_pci_switch(inventory,
                                                    pci_bridge_depth,
                                                    switch_depths, gpu_groups,
                                                    groups);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_topo_groups_add_nics_by_pci_switch(inventory, depths,
                                                    switch_depths, gpu_groups,
                                                    groups, &unmatched);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_topo_groups_add_unmatched_gpus(inventory, gpu_groups, groups,
                                                &unmatched);
    if (status != UCS_OK) {
        goto out;
    }

    ucs_topo_groups_remove_groups_without_nics(groups);

    status = ucs_topo_groups_build_groups_by_numa(devices, &unmatched, groups);

out:
    ucs_topo_release_group(&unmatched);
    ucs_free(depths);
    ucs_free(gpu_groups);
    ucs_free(switch_depths);
    return status;
}

static ucs_status_t
ucs_topo_groups_build_groups(const ucs_topo_sys_device_info_t *devices,
                             const ucs_topo_group_t *inventory,
                             ucs_topo_group_by_t group_by,
                             ucs_topo_pci_bridge_depth_func_t pci_bridge_depth,
                             ucs_topo_groups_t *groups)
{
    if (group_by == UCS_TOPO_GROUP_BY_AUTO) {
#if defined(__x86_64__)
        group_by = UCS_TOPO_GROUP_BY_PCI_SWITCH;
#else
        group_by = UCS_TOPO_GROUP_BY_NUMA;
#endif
    }

    if (group_by == UCS_TOPO_GROUP_BY_PCI_SWITCH) {
        ucs_debug("grouping gpus and nics by pci switch");
        return ucs_topo_groups_build_groups_by_pci_switch(devices, inventory,
                                                          pci_bridge_depth,
                                                          groups);
    }

    ucs_assertv(group_by == UCS_TOPO_GROUP_BY_NUMA, "group_by=%d",
                (int)group_by);
    ucs_debug("grouping gpus and nics by numa node");
    return ucs_topo_groups_build_groups_by_numa(devices, inventory, groups);
}

static void
ucs_topo_groups_append_elements(const ucs_topo_sys_device_info_t *devices,
                                const ucs_topo_group_element_array_t *elements,
                                ucs_string_buffer_t *strb)
{
    const ucs_topo_group_element_t *element;
    size_t sys_dev_idx;

    ucs_string_buffer_appendf(strb, "[");
    ucs_array_for_each(element, elements) {
        for (sys_dev_idx = 0; sys_dev_idx < element->num_sys_devs;
             ++sys_dev_idx) {
            ucs_string_buffer_appendf(
                    strb, "%s/", devices[element->sys_devs[sys_dev_idx]].name);
        }

        ucs_string_buffer_rtrim(strb, "/");
        ucs_string_buffer_appendf(strb, ", ");
    }
    ucs_string_buffer_rtrim(strb, ", ");
    ucs_string_buffer_appendf(strb, "]");
}

static void ucs_topo_groups_log(const ucs_topo_sys_device_info_t *devices,
                                const ucs_topo_groups_t *groups)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    const ucs_topo_group_t *group;
    size_t group_idx;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        return;
    }

    ucs_array_for_each_index(group, group_idx, groups) {
        ucs_string_buffer_appendf(&strb, "%zu gpus ",
                                  ucs_array_length(&group->gpus));
        ucs_topo_groups_append_elements(devices, &group->gpus, &strb);
        ucs_string_buffer_appendf(&strb, ", %zu nics ",
                                  ucs_array_length(&group->nics));
        ucs_topo_groups_append_elements(devices, &group->nics, &strb);

        ucs_debug("topology group %zu: %s", group_idx,
                  ucs_string_buffer_cstr(&strb));
        ucs_string_buffer_reset(&strb);
    }

    ucs_string_buffer_cleanup(&strb);
}

ucs_status_t
ucs_topo_build_groups_inner(const ucs_topo_sys_device_info_t *devices,
                            unsigned num_devices,
                            const ucs_sys_device_bitmap_t *net_device_filter,
                            ucs_topo_group_by_t group_by,
                            ucs_topo_pci_bridge_depth_func_t pci_bridge_depth,
                            ucs_topo_groups_t *groups_p)
{
    ucs_topo_group_t inventory;
    ucs_topo_groups_t groups;
    ucs_status_t status;

    ucs_topo_init_groups(&groups);

    status = ucs_topo_groups_inventory_build(devices, num_devices,
                                             net_device_filter, &inventory);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_groups_build_groups(devices, &inventory, group_by,
                                          pci_bridge_depth, &groups);
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
