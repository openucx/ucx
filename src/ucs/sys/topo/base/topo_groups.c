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

#include <string.h>


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
