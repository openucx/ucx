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


/* Upper limit on system device id */
#define UCS_SYS_DEVICE_ID_MAX UINT8_MAX

/* Indicate that the ucs_sys_device_t for the device has no real bus_id
 * e.g. virtual devices like CMA/knem */
#define UCS_SYS_DEVICE_ID_UNKNOWN UINT8_MAX


typedef struct ucs_sys_bus_id {
    uint16_t domain;   /* range: 0 to ffff */
    uint8_t  bus;      /* range: 0 to ff */
    uint8_t  slot;     /* range: 0 to 1f */
    uint8_t  function; /* range: 0 to 7 */
} ucs_sys_bus_id_t;


/**
 * @ingroup UCS_RESOURCE
 * System Device Index
 * Obtained from a translation of the device bus id into a short integer
 * Refer ucs_topo_find_device_by_bus_id()
 */
typedef uint8_t ucs_sys_device_t;


/*
 * Captures the estimated latency and bandwidth between two system devices
 * referred by ucs_sys_device_t handle.
 */
typedef struct ucs_sys_dev_distance {
    double latency;   /**< in seconds */
    double bandwidth; /**< in bytes/second */
} ucs_sys_dev_distance_t;


extern const ucs_sys_dev_distance_t ucs_topo_default_distance;


/**
 * Find system device by pci bus id.
 *
 * @param [in]  bus_id  pointer to bus id of the device of interest.
 * @param [out] sys_dev system device index associated with the bus_id.
 *
 * @return UCS_OK or error in case device cannot be found.
 */
ucs_status_t ucs_topo_find_device_by_bus_id(const ucs_sys_bus_id_t *bus_id,
                                            ucs_sys_device_t *sys_dev);


/**
 * Find the distance between two system devices (in terms of latency,
 * bandwidth, hops, etc).
 *
 * @param [in]  device1   System device index of the first device.
 * @param [in]  device2   System device index of the second device.
 * @param [out] distance  Result populated with distance details between the two
*                         devices.
 *
 * @return UCS_OK or error in case distance cannot be determined.
 */
ucs_status_t ucs_topo_get_distance(ucs_sys_device_t device1,
                                   ucs_sys_device_t device2,
                                   ucs_sys_dev_distance_t *distance);


/**
 * Convert the distance to a human-readable string.
 *
 * @param [in]  distance   Distance between two devices.
 * @param [out] buffer     String buffer to fill with distance string.
 * @param [in]  max        Maximal size of the string buffer.
 *
 * @return Pointer to the distance string.
 */
const char *ucs_topo_distance_str(const ucs_sys_dev_distance_t *distance,
                                  char *buffer, size_t max);


/**
 * Return system device name in BDF format: "<domain>:<bus>:<device>.<function>".
 *
 * @param [in]  sys_dev  System device id, as returned from
 *                       @ref ucs_topo_find_device_by_bus_id.
 * @param [out] buffer   String buffer, filled the device name.
 * @param [in]  max      Maximal size of @a buffer.
 */
const char *
ucs_topo_sys_device_bdf_name(ucs_sys_device_t sys_dev, char *buffer, size_t max);


/**
 * Find a system device by its BDF name: "[<domain>:]<bus>:<device>.<function>".
 *
 * @param [in]  name     BDF name to search for.
 * @param [out] sys_dev  Filled with system device id, if found.
 *
 * @return UCS_OK if the device was found, error otherwise.
 */
ucs_status_t
ucs_topo_find_device_by_bdf_name(const char *name, ucs_sys_device_t *sys_dev);


/**
 * Get the number of registered system devices.
 *
 * @return Number of system devices.
 */
unsigned ucs_topo_num_devices();


/**
 * Print a map indicating the topology information between system devices
 * discovered.
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
