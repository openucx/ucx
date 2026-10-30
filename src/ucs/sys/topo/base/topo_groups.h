/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TOPO_GROUPS_H
#define UCS_TOPO_GROUPS_H

#include <ucs/sys/topo/base/topo_int.h>
#include <ucs/datastruct/array.h>
#include <ucs/datastruct/static_bitmap.h>

BEGIN_C_DECLS

#define UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT 8


/**
 * @ingroup UCS_RESOURCE
 * Physical device represented in a topology group.
 */
typedef struct {
    ucs_sys_device_t sys_devs[UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT];
    size_t           num_sys_devs;
} ucs_topo_group_element_t;


UCS_ARRAY_DECLARE_TYPE(ucs_topo_group_element_array_t, size_t,
                       ucs_topo_group_element_t);


/**
 * @ingroup UCS_RESOURCE
 * Group of GPUs and NICs sharing a topology locality.
 */
typedef struct {
    ucs_topo_group_element_array_t gpus;
    ucs_topo_group_element_array_t nics;
} ucs_topo_group_t;


UCS_ARRAY_DECLARE_TYPE(ucs_topo_group_array_t, size_t, ucs_topo_group_t);


/**
 * @ingroup UCS_RESOURCE
 * Collection of system topology groups.
 */
typedef ucs_topo_group_array_t ucs_topo_groups_t;


/**
 * @ingroup UCS_RESOURCE
 * Bitmap of system devices.
 */
typedef ucs_static_bitmap_s(UCS_SYS_DEVICE_ID_COUNT) ucs_sys_device_bitmap_t;


/**
 * @ingroup UCS_RESOURCE
 * Criterion used to place GPUs and NICs into topology groups.
 */
typedef enum {
    /** One group per NUMA node */
    UCS_TOPO_GROUP_BY_NUMA,

    /**
     * Every GPU decides its PCIe switch, the deepest PCIe switch it shares with
     * any NIC, and GPUs which decided the same PCIe switch form one group. A
     * NIC joins the group with the deepest decided PCIe switch above it, so it
     * is never placed in more than one group. A GPU sharing no PCIe switch
     * with any NIC forms a group without NICs, and NICs below no decided PCIe
     * switch are placed in a single group without GPUs.
     */
    UCS_TOPO_GROUP_BY_PCI_SWITCH,

    UCS_TOPO_GROUP_BY_LAST
} ucs_topo_group_by_t;


/**
 * Get the depth of the deepest PCIe switch (bridge) shared by two system
 * devices, below the PCI root.
 *
 * @param [in] sys_dev1  First system device.
 * @param [in] sys_dev2  Second system device.
 *
 * @return Number of PCIe hierarchy levels shared below the PCI root, or 0 if
 *         the devices do not share a PCIe switch.
 */
typedef unsigned (*ucs_topo_pci_switch_depth_func_t)(ucs_sys_device_t sys_dev1,
                                                     ucs_sys_device_t sys_dev2);


/**
 * Build system topology groups.
 *
 * The caller takes ownership of the returned group data and must release it
 * with @ref ucs_topo_release_groups.
 *
 * @param [in]  net_device_filter  Network devices to include in the groups,
 *                                 or NULL to include all network devices.
 * @param [in]  group_by           Grouping criterion.
 * @param [out] groups_p           Initialized topology groups.
 *
 * @return UCS_OK on success, or an error status if topology group
 *         initialization failed.
 */
ucs_status_t
ucs_topo_build_groups(const ucs_sys_device_bitmap_t *net_device_filter,
                      ucs_topo_group_by_t group_by,
                      ucs_topo_groups_t *groups_p);


/**
 * Release topology groups returned by @ref ucs_topo_build_groups.
 *
 * @param [in] groups  Topology groups to release.
 */
void ucs_topo_release_groups(ucs_topo_groups_t *groups);


/**
 * Initialize a topology group.
 *
 * @param [out] group  Group to initialize.
 */
void ucs_topo_init_group(ucs_topo_group_t *group);


/**
 * Release resources allocated by a topology group.
 *
 * @param [in] group  Group to release.
 */
void ucs_topo_release_group(ucs_topo_group_t *group);


/**
 * Build system topology groups (internal function).
 *
 * @param [in]  devices            Array of registered system devices.
 * @param [in]  num_devices        Number of elements in @a devices.
 * @param [in]  net_device_filter  Network devices to include in the groups,
 *                                 or NULL to include all network devices.
 * @param [in]  group_by           Grouping criterion.
 * @param [in]  pci_switch_depth   Get the depth of the PCIe switch shared by
 *                                 two devices, used by
 *                                 @ref UCS_TOPO_GROUP_BY_PCI_SWITCH.
 * @param [out] groups_p           Initialized topology groups.
 *
 * @return UCS_OK on success, or an error status if topology group
 *         initialization failed.
 */
ucs_status_t
ucs_topo_build_groups_inner(const ucs_topo_sys_device_info_t *devices,
                            unsigned num_devices,
                            const ucs_sys_device_bitmap_t *net_device_filter,
                            ucs_topo_group_by_t group_by,
                            ucs_topo_pci_switch_depth_func_t pci_switch_depth,
                            ucs_topo_groups_t *groups_p);

END_C_DECLS

#endif
