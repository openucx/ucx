/**
* Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TOPO_H
#define UCS_TOPO_H

#include <ucs/type/status.h>
#include <ucs/datastruct/list.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>

BEGIN_C_DECLS


/* Upper limit on system device id */
#define UCS_SYS_DEVICE_ID_MAX UINT8_MAX

/* Indicate that the ucs_sys_device_t for the device has no real bus_id
 * e.g. virtual devices like CMA/knem */
#define UCS_SYS_DEVICE_ID_UNKNOWN UINT8_MAX

/* Maximal size of BDF string */
#define UCS_SYS_BDF_NAME_MAX 16


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


/*
 * Function pointer used to refer to specific implementations of
 * ucs_topo_get_distance function by topology modules
 */
typedef ucs_status_t
(*ucs_topo_get_distance_func_t)(ucs_sys_device_t device1,
                                ucs_sys_device_t device2,
                                ucs_sys_dev_distance_t *distance);


/*
 * Structure needed to define a topology module implementation
 */
typedef struct {

    /* Name of the topology module */
    const char                   *name;

    /* Points to the module's ucs_topo_get_distance implementation */
    ucs_topo_get_distance_func_t get_distance;

    ucs_list_link_t              list;
} ucs_sys_topo_method_t;


/* Global list of topology detection methods */
extern ucs_list_link_t ucs_sys_topo_methods_list;


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
 * Find pci bus id of the given system device.
 *
 * @param [in]  sys_dev system device index.
 * @param [out] bus_id  pointer to bus id to be populated.
 *
 * @return UCS_OK or error in case system device or its bus id cannot be found.
 */
ucs_status_t ucs_topo_get_device_bus_id(ucs_sys_device_t sys_dev,
                                        ucs_sys_bus_id_t *bus_id);


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
 * Set a name for a given system device. If the name was set previously, the new
 * name will replace the old one.
 *
 * @param [in]  sys_dev  System device to set the name for.
 * @param [in]  name     Name to set for this system device. Note: the name can
 *                       be released after this call.
 *
 * @return UCS_OK if the name was set, error otherwise.
 */
ucs_status_t
ucs_topo_sys_device_set_name(ucs_sys_device_t sys_dev, const char *name);


/**
 * Get the name of a given system device. If the name was never set, it defaults
 * to the BDF representation of the system device bus id.
 *
 * @param [in]  sys_dev  System device to set the name for.
 *
 * @return The name of the system device, or NULL if the system device is
 *         invalid.
 */
const char *ucs_topo_sys_device_get_name(ucs_sys_device_t sys_dev);


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
