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

#define UCS_SYS_DEVICE_ID_UNKNOWN UINT_MAX /* Indicate that the ucs_sys_device_t
                                              for the device has no real bus_id
                                              E.g. virtual devices like CMA/knem
                                            */


/** @file topo.h */

typedef struct ucs_sys_bus_id {
    uint16_t domain;   /* range: 0 to ffff */
    uint8_t  bus;      /* range: 0 to ff */
    uint8_t  slot;     /* range: 0 to 1f */
    uint8_t  function; /* range: 0 to 7 */
} ucs_sys_bus_id_t;


/**
 * @ingroup UCS_RESOURCE
 * System Device Index
 * Obtained from a translation of the device bus id into an unsigned int
 * Refer ucs_topo_find_device_by_bus_id()
 */
typedef unsigned ucs_sys_device_t;


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
 * @param [in]  bus_id  pointer to bus id of the device of interest
 * @param [out] sys_dev system device index associated with the bus_id
 *
 * @return UCS_OK or error in case device cannot be found
 */
ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev);


/**
 * Find the distance between two system devices (in terms of latency,
 * bandwidth, hops, etc)
 *
 * @param [in]  device1  system device index of the first device
 * @param [in]  device2  system device index of the second device
 * @param [out] distance result populated with distance details between the two
 *                       devices
 *
 * @return UCS_OK or error in case distance cannot be determined
 */
ucs_status_t ucs_topo_get_distance(ucs_sys_device_t device1,
                                   ucs_sys_device_t device2,
                                   ucs_sys_dev_distance_t *distance);


/**
 * Print a map indicating the topology information between system
 * devices discovered
 */
void ucs_topo_print_info(FILE *stream);

/**
 * Initialize UCS topology subsystem.
 */
void ucs_topo_init();

/**
 * Cleanup UCS topology subsystem.
 */
void ucs_topo_cleanup();

END_C_DECLS

#endif
