/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TOPO_INT_H
#define UCS_TOPO_INT_H

#include "topo.h"

#include <ucs/config/types.h>
#include <ucs/datastruct/static_bitmap.h>

BEGIN_C_DECLS

/* Root port of a device which is not behind a PCIe root port */
#define UCS_TOPO_ROOT_PORT_NONE ((ucs_bus_id_bit_rep_t)-1)

/* Possible role of a current device wrt its sibling */
typedef enum {
    /* No sibling capability */
    UCS_TOPO_SIBLING_ROLE_NONE,

    /* Memory device, a sibling device could access its memory */
    UCS_TOPO_SIBLING_ROLE_MEM,

    /* Device that could access memory from its sibling */
    UCS_TOPO_SIBLING_ROLE_DEV
} ucs_topo_sibling_role_t;


/**
 * @ingroup UCS_RESOURCE
 * Bitmap of system devices.
 */
typedef ucs_static_bitmap_s(UCS_SYS_DEVICE_ID_COUNT) ucs_sys_device_bitmap_t;


typedef struct {
    ucs_sys_bus_id_t        bus_id;
    char                    *name;
    unsigned                name_priority;
    ucs_numa_node_t         numa_node;
    ucs_sys_pci_id_t        pci_id;
    uintptr_t               user_value;
    ucs_topo_device_class_t device_class;
    unsigned                flags; /* ucs_topo_device_flags_t */

    /* Cached rank of the device's BDF within its class, or
     * UCS_SYS_DEVICE_ORDINAL_INVALID if not yet computed.
     * Invalidated when any device's class changes. */
    unsigned                class_ordinal;

    /* Secondary device for the current device */
    ucs_sys_device_t        sys_dev_aux;

    ucs_topo_sibling_role_t sibling_role; /* Role of the current device */
    /* Matched siblings: all DEVs for the MEM role, all MEMs for the DEV role.
     * Several DEVs can share one auxiliary device (e.g. planes of a multiplane
     * NIC), and several MEMs can sit behind it (e.g. MPS MLOPart aliases). */
    ucs_sys_device_bitmap_t siblings;

    /* Whether peer-to-peer traffic between PCI devices under the same root
     * port goes through the root complex, UCS_AUTO until detected */
    ucs_ternary_auto_value_t p2p_redirect;

    /* PCIe root port of the device as returned by
     * ucs_topo_get_bus_id_bit_repr(), or UCS_TOPO_ROOT_PORT_NONE */
    ucs_bus_id_bit_rep_t     root_port;
} ucs_topo_sys_device_info_t;


/**
 * Compare two system device information entries by their topology identity.
 */
int ucs_topo_sys_device_info_cmp(const ucs_topo_sys_device_info_t *device1,
                                 const ucs_topo_sys_device_info_t *device2);


/**
 * Convert a PCI bus id to its canonical sysfs path.
 *
 * @param [in]  bus_id  Device PCI bus id.
 * @param [out] path    Filled with the resolved sysfs path. Must have room
 *                      for at least PATH_MAX bytes.
 * @param [in]  max     Size of @a path in bytes.
 *
 * @return UCS_OK on success, or an error status otherwise.
 */
ucs_status_t ucs_topo_bus_id_to_sysfs_path(const ucs_sys_bus_id_t *bus_id,
                                           char *path, size_t max);


/**
 * Check in sysfs whether peer-to-peer traffic between a PCI device and the
 * devices with memory (e.g. GPU) under its PCIe root port goes through the root
 * complex. The peers are the display controllers and processing accelerators
 * under the root port, except the functions of the device itself.
 *
 * @param [in] dev_path  Canonical sysfs path of the device.
 *
 * @return Nonzero if the traffic between the device and a peer under its root
 *         port goes through the root complex, 0 otherwise.
 */
int ucs_topo_sysfs_is_p2p_redirected(const char *dev_path);


/**
 * Get the PCIe root port of a PCI device.
 *
 * @param [in] path  Canonical sysfs path of the device.
 *
 * @return Bus ID of the root port as returned by
 *         ucs_topo_get_bus_id_bit_repr(), or UCS_TOPO_ROOT_PORT_NONE if the
 *         device is not behind a root port.
 */
ucs_bus_id_bit_rep_t ucs_topo_sysfs_get_root_port(const char *path);


END_C_DECLS

#endif
