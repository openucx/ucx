/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TOPO_H
#define UCS_TOPO_H

#include <ucs/type/status.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file topo.h */

typedef struct ucs_sys_bus_id {
    uint16_t domain;   /* range: 0 to ffff */
    uint8_t  bus;      /* range: 0 to ff */
    uint8_t  slot;     /* range: 0 to 1f */
    uint8_t  function; /* range: 0 to 7 */
} ucs_sys_bus_id_t;


/**
 * @ingroup UCS_RESOURCE
 * System Device abstraction
 */
typedef struct ucs_sys_device {
    unsigned         id;        /**< Index of the device */
    int              numa_node; /**< NUMA node assoicated with the device*/
    ucs_sys_bus_id_t bus_id;    /**< bus ID of of the device if applicable.
                                   eg: 0000:06:00.0 {domain:bus:slot.function}*/
} ucs_sys_device_t;


/*
 * Capture the estimated latency, bandwidth between two system devices
 * referred by ucs_sys_device_t handle
 */
typedef struct ucs_sys_dev_distance {
    double latency;   /**< in seconds */
    double bandwidth; /**< in bytes/second */
} ucs_sys_dev_distance_t;


/**
 * Find system device by pci bus id
 *
 * @param [in]     bus_id  bus id of the device of interest
 * @param [in/out] sys_dev pointer to ucs_sys_device_t
 *                         populated with device associated with the bus_id
 *
 * @return UCS_OK or error in case device cannot be found
 */
ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            const ucs_sys_device_t **sys_dev);


/**
 * Find the distance between two system devices (in terms of latency,
 * bandwidth, hops, etc)
 *
 * @param [in]     device1 ucs_sys_device_t pointing to the first device
 * @param [in]     device2 ucs_sys_device_t pointing to the second device
 * @param [in/out] result  pointer to ucs_sys_dev_distance_t
 *                         populated with distance details between the two
 *                         devices
 */
ucs_status_t ucs_topo_get_distance(const ucs_sys_device_t *device1,
                                   const ucs_sys_device_t *device2,
                                   ucs_sys_dev_distance_t *distance);


/**
 * Print a map indicating the topology information between system
 * devices discovered
 */
void ucs_topo_print_info(FILE *stream);

END_C_DECLS

#endif
