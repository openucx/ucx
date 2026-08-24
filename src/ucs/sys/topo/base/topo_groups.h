/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TOPO_GROUPS_H
#define UCS_TOPO_GROUPS_H

#include "topo_int.h"

#include <ucs/datastruct/array.h>

BEGIN_C_DECLS

/* Maximal number of ports per NIC */
#define UCS_TOPO_MAX_PORTS_PER_NIC 2

/* Maximal number of devices (uGPUs) per physical GPU */
#define UCS_TOPO_MAX_DEVICES_PER_GPU 2


/**
 * @ingroup UCS_RESOURCE
 * Type of system topology represented by topology groups.
 */
typedef enum {
    UCS_TOPO_GROUPS_TYPE_UNKNOWN,
    UCS_TOPO_GROUPS_TYPE_VERA_RUBIN
} ucs_topo_groups_type_t;


/**
 * @ingroup UCS_RESOURCE
 * Physical GPU represented in a topology group.
 * When MPS MLOParts is enabled, the list contains the uGPUs under the same GPU.
 * When MPS MLOParts is disabled, the list contains only one device.
 */
typedef struct {
    ucs_sys_device_t devices[UCS_TOPO_MAX_DEVICES_PER_GPU];
    size_t           num_devices;
} ucs_topo_gpu_t;


/**
 * @ingroup UCS_RESOURCE
 * Physical NIC represented in a topology group.
 */
typedef struct {
    ucs_sys_device_t ports[UCS_TOPO_MAX_PORTS_PER_NIC];
    size_t           num_ports;
} ucs_topo_nic_t;


UCS_ARRAY_DECLARE_TYPE(ucs_topo_gpu_array_t, size_t, ucs_topo_gpu_t);
UCS_ARRAY_DECLARE_TYPE(ucs_topo_nic_array_t, size_t, ucs_topo_nic_t);


/**
 * @ingroup UCS_RESOURCE
 * Group of GPUs and NICs sharing a topology locality.
 */
typedef struct {
    ucs_topo_gpu_array_t gpus;
    ucs_topo_nic_array_t nics;
} ucs_topo_group_t;

UCS_ARRAY_DECLARE_TYPE(ucs_topo_group_array_t, size_t, ucs_topo_group_t);

/**
 * @ingroup UCS_RESOURCE
 * Collection of system topology groups.
 */
typedef struct {
    ucs_topo_groups_type_t type;
    ucs_topo_group_array_t groups;
} ucs_topo_groups_t;


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
 * Initialize topology groups.
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
