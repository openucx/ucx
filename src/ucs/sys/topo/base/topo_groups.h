/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TOPO_GROUPS_H
#define UCS_TOPO_GROUPS_H

#include <ucs/sys/topo/base/topo_int.h>
#include <ucs/datastruct/array.h>

BEGIN_C_DECLS

#define UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT 8

/* CX-9 PCI identifiers */
#define UCS_TOPO_GROUPS_MELLANOX_VENDOR_ID 0x15b3
#define UCS_TOPO_GROUPS_CX9_DEVICE_ID      0x1025
#define UCS_TOPO_GROUPS_MLX5_VF_DEVICE_ID  0x101e


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
 * Build system topology groups.
 *
 * The caller takes ownership of the returned group data and must release it
 * with @ref ucs_topo_release_groups.
 *
 * @param [out] groups_p  Initialized topology groups.
 *
 * @return UCS_OK on success, or an error status if topology group
 *         initialization failed.
 */
ucs_status_t ucs_topo_build_groups(ucs_topo_groups_t *groups_p);


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
 * @param [in]  devices      Array of registered system devices.
 * @param [in]  num_devices  Number of elements in @a devices.
 * @param [out] groups_p     Initialized topology groups.
 *
 * @return UCS_OK on success, or an error status if topology group
 *         initialization failed.
 */
ucs_status_t
ucs_topo_build_groups_inner(const ucs_topo_sys_device_info_t *devices,
                            unsigned num_devices, ucs_topo_groups_t *groups_p);

END_C_DECLS

#endif
