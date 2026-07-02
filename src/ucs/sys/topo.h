/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TOPO_H
#define UCS_TOPO_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/type/status.h>
#include <limits.h>

BEGIN_C_DECLS

/** @file topo.h */

/**
 * System device type
 */
typedef enum {
    UCS_SYS_DEVICE_IB = 0, /* Infiniband Device */
    UCS_SYS_DEVICE_CUDA,   /* NVIDIA GPU Device */
    UCS_SYS_DEVICE_NUMA,   /* NUMA Memory */
    UCS_SYS_DEVICE_LAST
} ucs_sys_device_enum_t;


/**
 * @ingroup UCS_RESOURCE
 * System Device abstraction
 */
typedef struct ucs_sys_device {
    ucs_sys_device_enum_t sys_dev_type;             /**< Type of system device*/
    unsigned int          id;                       /**< Index of the unit */
    unsigned int          bus_id;                   /**< bus ID of of the device if applicable*/
} ucs_sys_device_t;


/**
 * Detect system devices such as HCAs, GPUs, and other PCIe devices
 * and return a list of such devices
 *
 * @param [out] sys_devices     Array of system device abstractions
 * @param [out] num_sys_devices Number of system deivces detected in the system
 * @return UCS_OK or error in case of failure.
 */
ucs_status_t ucs_topo_get_sys_devices(ucs_sys_device_t **sys_devices, int *num_sys_devices);


/**
 * Release resources allocated for system devices
 *
 * @param [out] sys_devices Array of system device abstractions
 * @return UCS_OK or error in case of failure.
 */
ucs_status_t ucs_topo_free_sys_devices(ucs_sys_device_t *sys_devices);

END_C_DECLS

#endif
