/**
* Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "topo.h"

#include <ucs/config/global_opts.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/spinlock.h>
#include <ucs/type/status.h>

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define COMPARE(a, b)                      ((a) < (b) ? -1 : (a) > (b) ? 1 : 0)
#define UCS_GAUDI_TOPO_ACCEL_PATH          "/sys/class/accel/"
#define UCS_GAUDI_TOPO_INFINIBAND_PORT_FMT "/sys/class/infiniband/%s/ports/1/"
#define UCS_GAUDI_TOPO_VENDOR_ID           0x1da3 /* Habana Labs Vendor ID */
#define UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID  0x15b3
#define UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID  0x14e4
#define GAUDI_DEVICE_NAME_LEN              10

static pthread_mutex_t gaudi_init_mutex = PTHREAD_MUTEX_INITIALIZER;
/* File-scope one-time control and status */
static pthread_once_t gaudi_spinlock_once_flag = PTHREAD_ONCE_INIT;
static ucs_status_t gaudi_spinlock_init_status = UCS_OK;

static const ucs_sys_dev_distance_t gaudi_fallback_node_distance =
        {.latency = 100e-9, .bandwidth = 17e9}; /* 100ns, 17 GB/s */
static const ucs_sys_dev_distance_t gaudi_fallback_sys_distance =
        {.latency = 300e-9, .bandwidth = 220e6}; /* 300ns, 220 MB/s */

/* Structure to hold Gaudi and HNIC mappings */
typedef struct {
    ucs_sys_device_t gaudi_device;
    ucs_sys_device_t hnic_device;
    uint16_t hnic_vendor_id;
    ucs_sys_dev_distance_t distance;
    ucs_numa_node_t common_numa_node;
    char gaudi_dev_name[GAUDI_DEVICE_NAME_LEN];
} ucs_gaudi_connection_t;

/* Static context for Gaudi topology */
typedef struct {
    ucs_spinlock_t lock;
    unsigned initialized;
    unsigned provider_added;
    ucs_sys_device_t *gaudi_devices;
    char (*gaudi_devices_names)[GAUDI_DEVICE_NAME_LEN];
    unsigned num_gaudi_devices;
    ucs_sys_device_t *hnic_devices;
    uint16_t *hnic_vendor_ids;
    unsigned num_hnic_devices;
    ucs_gaudi_connection_t *connections;
    unsigned num_connections;
    ucs_sys_device_t *assigned_hnic_for_gaudi;
    int *assigned_port_for_gaudi;
    unsigned have_assignment;
} ucs_gaudi_topo_ctx_t;

static ucs_gaudi_topo_ctx_t ucs_gaudi_topo_ctx = {0};

/* Compatible definition of ucs_sys_topo_ops_t (from topo.c layout) */
typedef struct {
    ucs_status_t (*get_distance)(ucs_sys_device_t device1,
                                 ucs_sys_device_t device2,
                                 ucs_sys_dev_distance_t *distance);
    void (*get_memory_distance)(ucs_sys_device_t device,
                                ucs_sys_dev_distance_t *distance);
} compatible_topo_ops_t;

/* Compatible definition of ucs_sys_topo_provider_t (from topo.c layout) */
typedef struct {
    const char *name;
    compatible_topo_ops_t ops;
    ucs_list_link_t list;
} compatible_topo_provider_t;

/* Forward declarations */
static ucs_status_t ucs_gaudi_get_distance(ucs_sys_device_t device1,
                                           ucs_sys_device_t device2,
                                           ucs_sys_dev_distance_t *distance);

static void ucs_gaudi_get_memory_distance(ucs_sys_device_t device,
                                          ucs_sys_dev_distance_t *distance);

static ucs_status_t ucs_gaudi_lazy_init();

static int
ucs_gaudi_is_hnic_active(ucs_sys_device_t hnic_device, uint16_t hnic_vendor_id);

/* Gaudi topology provider (compatible structure) */
static compatible_topo_provider_t ucs_gaudi_topo_provider = {
        .name = "gaudi",
        .ops =
                {
                        .get_distance        = ucs_gaudi_get_distance,
                        .get_memory_distance = ucs_gaudi_get_memory_distance,
                },
        .list = {NULL, NULL}};


/* Helper function to construct sysfs path from ucs_sys_device_t */
static ucs_status_t ucs_gaudi_sys_dev_to_sysfs_path(ucs_sys_device_t sys_dev,
                                                    char *path, size_t max)
{
    ucs_sys_bus_id_t bus_id;
    ucs_status_t status;
    char *link_path;
    const char *prefix = "/sys/bus/pci/devices/";
    size_t prefix_len;

    status = ucs_topo_get_device_bus_id(sys_dev, &bus_id);
    if (status != UCS_OK) {
        ucs_error("Failed to get bus ID for device %d", sys_dev);
        return status;
    }

    if (max < PATH_MAX) {
        ucs_error("Output buffer too small (%zu < %d)", max, PATH_MAX);
        return UCS_ERR_BUFFER_TOO_SMALL;
    }

    /* Build path directly in output buffer */
    prefix_len = strlen(prefix);
    ucs_strncpy_safe(path, prefix, max);
    ucs_snprintf_safe(path + prefix_len, max - prefix_len, "%04x:%02x:%02x.%x",
                      bus_id.domain, bus_id.bus, bus_id.slot, bus_id.function);

    link_path = realpath(path, NULL);
    if (link_path == NULL) {
        ucs_error("Failed to resolve realpath for %s: %s", path,
                  strerror(errno));
        return UCS_ERR_IO_ERROR;
    }

    ucs_strncpy_safe(path, link_path, max);
    ucs_free(link_path);

    return UCS_OK;
}

/* Helper function to read PCI vendor ID from sysfs */
static ucs_status_t
ucs_gaudi_read_vendor_id(ucs_sys_device_t sys_dev, uint16_t *vendor_id)
{
    char *path;
    char vendor_str[16];
    ucs_status_t status;
    FILE *file;
    char *endptr;
    unsigned long val;
    size_t path_len;

    status = ucs_string_alloc_path_buffer(&path, "gaudi_vendor_path");
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_gaudi_sys_dev_to_sysfs_path(sys_dev, path, PATH_MAX);
    if (status != UCS_OK) {
        ucs_debug("Failed to get sysfs path for device %d: %s", sys_dev,
                  ucs_status_string(status));
        ucs_free(path);
        return status;
    }

    path_len = strlen(path);
    if ((PATH_MAX - path_len) < sizeof("/vendor")) {
        ucs_free(path);
        return UCS_ERR_BUFFER_TOO_SMALL;
    }
    ucs_snprintf_safe(path + path_len, PATH_MAX - path_len, "/vendor");

    file = fopen(path, "r");
    if (!file) {
        ucs_debug("Failed to open %s", path);
        ucs_free(path);
        return UCS_ERR_IO_ERROR;
    }

    if (!fgets(vendor_str, sizeof(vendor_str), file)) {
        ucs_debug("Failed to read vendor ID from %s", path);
        fclose(file);
        ucs_free(path);
        return UCS_ERR_IO_ERROR;
    }
    fclose(file);

    errno = 0;
    val   = strtoul(vendor_str, &endptr, 0);

    /* Skip trailing whitespace */
    while (isspace((unsigned char)*endptr)) {
        endptr++;
    }

    if (errno != 0 || endptr == vendor_str || *endptr != '\0') {
        ucs_debug("Invalid vendor ID '%s' in %s", vendor_str, path);
        ucs_free(path);
        return UCS_ERR_INVALID_PARAM;
    }

    if (val > UINT16_MAX) {
        ucs_debug("Vendor ID 0x%lx exceeds maximum value in %s", val, path);
        ucs_free(path);
        return UCS_ERR_INVALID_PARAM;
    }
    ucs_free(path);

    *vendor_id = (uint16_t)val;
    return UCS_OK;
}

/* Helper function to read PCI address from sysfs */
static ucs_status_t
ucs_gaudi_read_pci_addr(const char *accel_name, char *pci_addr, size_t max)
{
    char path[PATH_MAX];
    FILE *file;
    ucs_status_t status;
    size_t len;

    ucs_snprintf_safe(path, PATH_MAX, "%s%s/device/pci_addr",
                      UCS_GAUDI_TOPO_ACCEL_PATH, accel_name);

    file = fopen(path, "r");
    if (!file) {
        ucs_debug("Failed to open %s", path);
        return UCS_ERR_IO_ERROR;
    }

    if (fgets(pci_addr, max, file) == NULL) {
        ucs_debug("Failed to read PCI address from %s", path);
        status = UCS_ERR_IO_ERROR;
    } else {
        /* Remove trailing newline */
        len = strlen(pci_addr);
        if (len > 0 && pci_addr[len - 1] == '\n') {
            pci_addr[len - 1] = '\0';
        }
        status = UCS_OK;
    }

    fclose(file);
    return status;
}

/* Helper function to read module ID from sysfs */
static ucs_status_t
ucs_gaudi_read_module_id(const char *accel_name, uint32_t *module_id)
{
    char path[PATH_MAX];
    FILE *file;
    char buffer[16];
    char *endptr;
    unsigned long val;

    ucs_snprintf_safe(path, PATH_MAX, "%s%s/device/module_id",
                      UCS_GAUDI_TOPO_ACCEL_PATH, accel_name);

    file = fopen(path, "r");
    if (!file) {
        ucs_debug("Failed to open %s", path);
        return UCS_ERR_IO_ERROR;
    }

    if (fgets(buffer, sizeof(buffer), file) == NULL) {
        ucs_debug("Failed to read module ID from %s", path);
        fclose(file);
        return UCS_ERR_IO_ERROR;
    }
    fclose(file);

    errno = 0;
    val   = strtoul(buffer, &endptr, 10);
    while (isspace((unsigned char)*endptr)) {
        endptr++;
    }
    if (errno != 0 || endptr == buffer || *endptr != '\0') {
        ucs_debug("Invalid module ID in %s: '%s'", path, buffer);
        return UCS_ERR_INVALID_PARAM;
    }

    *module_id = (uint32_t)val;
    return UCS_OK;
}

/* Get Gaudi device index from a given module ID. */
int ucs_gaudi_get_index_from_module_id(uint32_t module_id)
{
    DIR *dir;
    struct dirent *entry;
    ucs_status_t status;
    uint32_t read_module_id;
    int device_id;

    dir = opendir(UCS_GAUDI_TOPO_ACCEL_PATH);
    if (!dir) {
        ucs_error("Failed to open directory %s: %s", UCS_GAUDI_TOPO_ACCEL_PATH,
                  strerror(errno));
        return -1;
    }

    /* Default: not found */
    device_id = -1;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "accel", 5) != 0 ||
            strncmp(entry->d_name, "accel_", 6) == 0) {
            continue;
        }

        status = ucs_gaudi_read_module_id(entry->d_name, &read_module_id);
        if (status != UCS_OK) {
            continue;
        }

        if (read_module_id == module_id) {
            device_id = (int)strtol(entry->d_name + 5, NULL, 10);
            break;
        }
    }

    closedir(dir);

    if (device_id < 0) {
        ucs_error("no Gaudi accelerator with module_id %u found", module_id);
    }

    return device_id;
}

/* Enumerate Gaudi devices and HNICs */
static ucs_status_t ucs_gaudi_enumerate_devices()
{
    ucs_sys_device_t sys_dev;
    ucs_sys_bus_id_t bus_id;
    ucs_status_t status;
    struct stat statbuf;
    DIR *dir;
    struct dirent *entry;
    char *accel_path;
    char pci_addr[32];
    uint16_t vendor_id;
    unsigned i;
    unsigned gaudi_idx;
    unsigned hnic_idx;

    dir = opendir(UCS_GAUDI_TOPO_ACCEL_PATH);
    if (!dir) {
        ucs_error("Failed to open directory %s", UCS_GAUDI_TOPO_ACCEL_PATH);
        return UCS_ERR_IO_ERROR;
    }

    status = ucs_string_alloc_path_buffer(&accel_path, "accel_path");
    if (status != UCS_OK) {
        closedir(dir);
        return status;
    }

    /* Count Gaudi devices and HNICs */
    ucs_gaudi_topo_ctx.num_gaudi_devices = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "accel", 5) != 0 ||
            strncmp(entry->d_name, "accel_", 6) == 0) {
            continue;
        }

        ucs_snprintf_safe(accel_path, PATH_MAX, "%s%s",
                          UCS_GAUDI_TOPO_ACCEL_PATH, entry->d_name);

        if (stat(accel_path, &statbuf) == 0 && S_ISDIR(statbuf.st_mode)) {
            ucs_debug("Found Gaudi device: %s", entry->d_name);
            ucs_gaudi_topo_ctx.num_gaudi_devices++;
        }
    }

    if (ucs_gaudi_topo_ctx.num_gaudi_devices == 0) {
        ucs_error("No Gaudi devices found under %s — aborting enumeration",
                  UCS_GAUDI_TOPO_ACCEL_PATH);
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    /* Enumerate HNICs using UCX topo */
    ucs_gaudi_topo_ctx.num_hnic_devices = 0;
    for (i = 0; i < ucs_topo_num_devices(); i++) {
        if (ucs_topo_get_device_bus_id(i, &bus_id) == UCS_OK) {
            if (ucs_gaudi_read_vendor_id(i, &vendor_id) == UCS_OK) {
                /* Assume Mellanox and Broadcom devices are HNICs */
                if (vendor_id == UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID ||
                    vendor_id == UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID) {
                    ucs_debug("Found HNIC device: %d (%s, active: %d)", i,
                              ucs_topo_sys_device_get_name(i),
                              ucs_gaudi_is_hnic_active(i, vendor_id));
                    ucs_gaudi_topo_ctx.num_hnic_devices++;
                }
            }
        }
    }

    if (ucs_gaudi_topo_ctx.num_hnic_devices == 0) {
        ucs_error("No HNIC devices found (no Mellanox/Broadcom NICs) — "
                  "aborting enumeration");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    /* Allocate arrays */
    ucs_gaudi_topo_ctx.gaudi_devices =
            ucs_calloc(ucs_gaudi_topo_ctx.num_gaudi_devices,
                       sizeof(ucs_sys_device_t), "gaudi_devices");
    ucs_gaudi_topo_ctx.gaudi_devices_names =
            ucs_calloc(ucs_gaudi_topo_ctx.num_gaudi_devices,
                       GAUDI_DEVICE_NAME_LEN, "gaudi_devices_names");
    ucs_gaudi_topo_ctx.hnic_devices =
            ucs_calloc(ucs_gaudi_topo_ctx.num_hnic_devices,
                       sizeof(ucs_sys_device_t), "hnic_devices");
    ucs_gaudi_topo_ctx.hnic_vendor_ids =
            ucs_calloc(ucs_gaudi_topo_ctx.num_hnic_devices, sizeof(uint16_t),
                       "hnic_vendor_ids");
    if (!ucs_gaudi_topo_ctx.gaudi_devices ||
        !ucs_gaudi_topo_ctx.gaudi_devices_names ||
        !ucs_gaudi_topo_ctx.hnic_devices ||
        !ucs_gaudi_topo_ctx.hnic_vendor_ids) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    /* Populate Gaudi devices */
    gaudi_idx = 0;
    rewinddir(dir);
    while ((entry = readdir(dir)) != NULL &&
           gaudi_idx < ucs_gaudi_topo_ctx.num_gaudi_devices) {
        if (strncmp(entry->d_name, "accel", 5) != 0 ||
            strncmp(entry->d_name, "accel_", 6) == 0) {
            continue;
        }

        ucs_snprintf_safe(accel_path, PATH_MAX, "%s%s",
                          UCS_GAUDI_TOPO_ACCEL_PATH, entry->d_name);

        if (stat(accel_path, &statbuf) == 0 && S_ISDIR(statbuf.st_mode)) {
            status = ucs_gaudi_read_pci_addr(entry->d_name, pci_addr,
                                             sizeof(pci_addr));
            if (status != UCS_OK) {
                ucs_debug("Skipping device %s due to PCI address read "
                          "failure",
                          entry->d_name);
                continue;
            }

            status = ucs_topo_find_device_by_bdf_name(pci_addr, &sys_dev);
            if (status == UCS_OK) {
                ucs_gaudi_topo_ctx.gaudi_devices[gaudi_idx] = sys_dev;
                ucs_topo_sys_device_set_name(sys_dev, entry->d_name, 1);
                ucs_strncpy_safe(
                        ucs_gaudi_topo_ctx.gaudi_devices_names[gaudi_idx],
                        entry->d_name, GAUDI_DEVICE_NAME_LEN);
                gaudi_idx++;
            } else {
                ucs_debug("Failed to find device by BDF %s for %s", pci_addr,
                          entry->d_name);
            }
        }
    }

    /* Populate HNIC devices */
    hnic_idx = 0;
    for (i = 0; i < ucs_topo_num_devices() &&
                hnic_idx < ucs_gaudi_topo_ctx.num_hnic_devices;
         i++) {
        if (ucs_topo_get_device_bus_id(i, &bus_id) == UCS_OK) {
            if (ucs_gaudi_read_vendor_id(i, &vendor_id) == UCS_OK) {
                if (vendor_id == UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID ||
                    vendor_id == UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID) {
                    ucs_gaudi_topo_ctx.hnic_devices[hnic_idx]      = i;
                    ucs_gaudi_topo_ctx.hnic_vendor_ids[hnic_idx++] = vendor_id;
                }
            }
        }
    }

    status = UCS_OK;

out:
    /*
     * Memory is freed ONLY on allocation failure (UCS_ERR_NO_MEMORY).
     * For other errors, no allocations were made, so nothing to free.
     */
    if (status == UCS_ERR_NO_MEMORY) {
        ucs_free(ucs_gaudi_topo_ctx.gaudi_devices);
        ucs_free(ucs_gaudi_topo_ctx.gaudi_devices_names);
        ucs_free(ucs_gaudi_topo_ctx.hnic_devices);
        ucs_free(ucs_gaudi_topo_ctx.hnic_vendor_ids);

        ucs_gaudi_topo_ctx.gaudi_devices       = NULL;
        ucs_gaudi_topo_ctx.gaudi_devices_names = NULL;
        ucs_gaudi_topo_ctx.hnic_devices        = NULL;
        ucs_gaudi_topo_ctx.hnic_vendor_ids     = NULL;

        ucs_gaudi_topo_ctx.num_gaudi_devices = 0;
        ucs_gaudi_topo_ctx.num_hnic_devices  = 0;
    }

    ucs_free(accel_path);

    closedir(dir);
    return status;
}

/*
 * Check if HNIC is active (RoCE)
 *
 * Both Mellanox and Broadcom vendors are expected to expose
 * InfiniBand sysfs state files at:
 *   /sys/class/infiniband/<dev_name>/ports/1/{state,phys_state}
 * 
 * No fallback to generic Port1State is provided because:
 * 1. Mellanox: Always uses IB sysfs in production
 * 2. Broadcom: bnxt_re driver creates IB sysfs for RoCE
 */
static int
ucs_gaudi_is_hnic_active(ucs_sys_device_t hnic_device, uint16_t hnic_vendor_id)
{
    ucs_status_t status;
    char *path;
    char state[64];
    const char *dev_name;
    size_t len;

    status = ucs_string_alloc_path_buffer(&path, "gaudi_hnic_path");
    if (status != UCS_OK) {
        /* Out of memory, assume inactive */
        return 0;
    }

    /* RDMA device name for MLX/BNXT, e.g. "mlx5_3" or "bnxt_re0". Never NULL. */
    dev_name = ucs_topo_sys_device_get_name(hnic_device);

    /* Mellanox/Broadcom devices: prefer InfiniBand state file (port 1) */
    ucs_snprintf_safe(path, PATH_MAX, UCS_GAUDI_TOPO_INFINIBAND_PORT_FMT,
                      dev_name);

    /* ports/1/state first (typically "4: ACTIVE\n" when up) */
    status = ucs_sys_read_sysfs_file(dev_name, path, "state", state,
                                     sizeof(state), UCS_LOG_LEVEL_DEBUG);
    if (status == UCS_OK) {
        ucs_free(path);
        state[sizeof(state) - 1] = '\0';
        len                      = strlen(state);
        while (len > 0 && (state[len - 1] == '\n' || state[len - 1] == '\r')) {
            state[--len] = '\0';
        }
        return ((len > 0 && state[0] == '4') ||
                strstr(state, "ACTIVE") != NULL) &&
               !strstr(state, "INACTIVE");
    }

    /* ports/1/phys_state */
    status = ucs_sys_read_sysfs_file(dev_name, path, "phys_state", state,
                                     sizeof(state), UCS_LOG_LEVEL_DEBUG);
    if (status == UCS_OK) {
        ucs_free(path);
        state[sizeof(state) - 1] = '\0';
        len                      = strlen(state);
        while (len > 0 && (state[len - 1] == '\n' || state[len - 1] == '\r')) {
            state[--len] = '\0';
        }
        return (strstr(state, "LinkUp") != NULL);
    }

    /* Fallback: assume inactive */
    if (hnic_vendor_id == UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID) {
        ucs_debug("Mellanox HNIC %s: IB state files absent, assuming inactive",
                  dev_name);
    } else {
        ucs_debug("Broadcom HNIC %s: IB state files absent, assuming inactive",
                  dev_name);
    }

    ucs_free(path);
    return 0;
}

/* 
 * Return PCIe hop count between two sysfs paths and the common ancestor path; 
 * 255 if they live in separate root complexes. 
 */
static inline unsigned ucs_gaudi_count_pcie_hops(const char *path1,
                                                 const char *path2,
                                                 char *common_path)
{
    /* compute common parent */
    ucs_path_get_common_parent(path1, path2, common_path);
    if (common_path[0] == '\0') {
        return 255; /* fallback */
    }

    return ucs_path_calc_distance(path1, path2);
}

/* 
 * Estimate device-to-device distance based on NUMA and path depth 
 * - If either device is unknown or identical -> node distance.
 * - If both NUMA nodes are valid and differ -> system distance.
 * - Otherwise (same NUMA or at least one unknown) -> refine with 
 *   PCIe hop count via sysfs paths.
 */
static ucs_status_t
ucs_gaudi_estimate_distance(ucs_sys_device_t device1, ucs_sys_device_t device2,
                            ucs_sys_dev_distance_t *distance)
{
    ucs_numa_node_t numa1, numa2;
    ucs_status_t status;
    const double hop_latency_ns = 10e-9; /* ~10 ns per hop */
    const double hop_bw_penalty = 0.95; /* 5 % loss per hop */
    unsigned hops;
    char *path1, *path2, *common_path;

    /* If either device is unknown or they are identical, assume node distance */
    if ((device1 == UCS_SYS_DEVICE_ID_UNKNOWN) ||
        (device2 == UCS_SYS_DEVICE_ID_UNKNOWN) || (device1 == device2)) {
        *distance = (ucs_global_opts.dist.node.bandwidth > 0) ?
                            ucs_global_opts.dist.node :
                            gaudi_fallback_node_distance;
        return UCS_OK;
    }

    /* Default distance */
    *distance = ucs_topo_default_distance;

    /* Step 1: Check NUMA nodes */
    numa1 = ucs_topo_sys_device_get_numa_node(device1);
    numa2 = ucs_topo_sys_device_get_numa_node(device2);

    if (numa1 != UCS_NUMA_NODE_UNDEFINED && numa2 != UCS_NUMA_NODE_UNDEFINED) {
        /* Different NUMA nodes */
        if (numa1 != numa2) {
            *distance = (ucs_global_opts.dist.sys.bandwidth > 0) ?
                                ucs_global_opts.dist.sys :
                                gaudi_fallback_sys_distance;
            return UCS_OK;
        }
        /* Same NUMA; continue to PCIe refinement */
    }

    /* Step 2: Same NUMA or NUMA unknown or undefined */

    /* allocate three scratch buffers */
    status = ucs_string_alloc_path_buffer(&path1, "path1");
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_string_alloc_path_buffer(&path2, "path2");
    if (status != UCS_OK) {
        goto free_path1;
    }

    status = ucs_string_alloc_path_buffer(&common_path, "common_path");
    if (status != UCS_OK) {
        goto free_path2;
    }

    /* convert devices to sysfs paths */
    status = ucs_gaudi_sys_dev_to_sysfs_path(device1, path1, PATH_MAX);
    if (status != UCS_OK) {
        goto free_all;
    }

    status = ucs_gaudi_sys_dev_to_sysfs_path(device2, path2, PATH_MAX);
    if (status != UCS_OK) {
        goto free_all;
    }

    hops = ucs_gaudi_count_pcie_hops(path1, path2, common_path);

    /* Note: DO NOT treat hops==255 as system distance. In this topology,
     * Gaudi/NIC can share NUMA but different root complexes. The 255 hop
     * penalty naturally degrades the score without overriding NUMA affinity,
     * which is the correct behavior.
     */
    distance->latency = (ucs_global_opts.dist.node.bandwidth > 0 ?
                                 ucs_global_opts.dist.node.latency :
                                 gaudi_fallback_node_distance.latency) +
                        hop_latency_ns * hops;

    distance->bandwidth = (ucs_global_opts.dist.node.bandwidth > 0 ?
                                   ucs_global_opts.dist.node.bandwidth :
                                   gaudi_fallback_node_distance.bandwidth) *
                          pow(hop_bw_penalty, hops);

    status = UCS_OK;

free_all:
    ucs_free(common_path);
free_path2:
    ucs_free(path2);
free_path1:
    ucs_free(path1);
out:
    return status;
}

/* Create connection matrix */
static ucs_status_t ucs_gaudi_create_connection_matrix()
{
    ucs_sys_device_t gaudi, hnic;
    ucs_numa_node_t numa1, numa2;
    ucs_status_t status;
    ucs_gaudi_connection_t *conn;
    const char *gaudi_name;
    unsigned conn_idx;
    unsigned i, j;
    unsigned max_num_connections;
    uint16_t hnic_vendor_id;

    max_num_connections = ucs_gaudi_topo_ctx.num_gaudi_devices *
                          ucs_gaudi_topo_ctx.num_hnic_devices;
    ucs_gaudi_topo_ctx.connections = ucs_calloc(max_num_connections,
                                                sizeof(ucs_gaudi_connection_t),
                                                "gaudi_connections");
    if (!ucs_gaudi_topo_ctx.connections) {
        return UCS_ERR_NO_MEMORY;
    }

    conn_idx = 0;
    for (i = 0; i < ucs_gaudi_topo_ctx.num_gaudi_devices; i++) {
        gaudi      = ucs_gaudi_topo_ctx.gaudi_devices[i];
        gaudi_name = ucs_gaudi_topo_ctx.gaudi_devices_names[i];
        numa1      = ucs_topo_sys_device_get_numa_node(gaudi);

        for (j = 0; j < ucs_gaudi_topo_ctx.num_hnic_devices; j++) {
            hnic           = ucs_gaudi_topo_ctx.hnic_devices[j];
            hnic_vendor_id = ucs_gaudi_topo_ctx.hnic_vendor_ids[j];
            if (!ucs_gaudi_is_hnic_active(hnic, hnic_vendor_id)) {
                continue;
            }
            numa2 = ucs_topo_sys_device_get_numa_node(hnic);

            conn               = &ucs_gaudi_topo_ctx.connections[conn_idx];
            conn->gaudi_device = gaudi;
            ucs_strncpy_safe(conn->gaudi_dev_name, gaudi_name,
                             GAUDI_DEVICE_NAME_LEN);
            conn->hnic_device      = hnic;
            conn->hnic_vendor_id   = hnic_vendor_id;
            conn->common_numa_node = ((numa1 == numa2) &&
                                      (numa1 != UCS_NUMA_NODE_UNDEFINED)) ?
                                             numa1 :
                                             UCS_NUMA_NODE_UNDEFINED;

            status = ucs_gaudi_estimate_distance(gaudi, hnic, &conn->distance);
            if (status != UCS_OK) {
                ucs_debug("Failed to estimate distance between Gaudi %u and "
                          "HNIC %u",
                          gaudi, hnic);
                conn->distance = ucs_topo_default_distance;
            }
            conn_idx++;
        }
    }

    ucs_gaudi_topo_ctx.num_connections = conn_idx;
    return UCS_OK;
}

/* Fetch precomputed distance for (gaudi, hnic) from the connections list */
static int ucs_gaudi_lookup_distance(ucs_sys_device_t gaudi_device,
                                     ucs_sys_device_t hnic_device,
                                     ucs_sys_dev_distance_t *distance)
{
    const ucs_gaudi_connection_t *conn;
    unsigned i;

    /* Lookup connection matrix for distance metric */
    for (i = 0; i < ucs_gaudi_topo_ctx.num_connections; i++) {
        conn = &ucs_gaudi_topo_ctx.connections[i];
        if (conn->gaudi_device == gaudi_device &&
            conn->hnic_device == hnic_device) {
            if (distance) {
                *distance = conn->distance;
            }
            return 1;
        }
    }
    return 0;
}

/* Compare function for sorting connections by distance */
static int ucs_gaudi_compare_connections(const void *a, const void *b)
{
    const ucs_gaudi_connection_t *conn_a = (const ucs_gaudi_connection_t *)a;
    const ucs_gaudi_connection_t *conn_b = (const ucs_gaudi_connection_t *)b;
    int cmp;
    const char *hnic_a;
    const char *hnic_b;

    /* Prefer connections on same NUMA node over undefined NUMA */
    if (conn_a->common_numa_node != UCS_NUMA_NODE_UNDEFINED &&
        conn_b->common_numa_node == UCS_NUMA_NODE_UNDEFINED) {
        return -1;
    }
    if (conn_a->common_numa_node == UCS_NUMA_NODE_UNDEFINED &&
        conn_b->common_numa_node != UCS_NUMA_NODE_UNDEFINED) {
        return 1;
    }

    /* Prefer lower latency */
    cmp = COMPARE(conn_a->distance.latency, conn_b->distance.latency);
    if (cmp != 0) {
        return cmp;
    }

    /* Prefer higher bandwidth */
    cmp = COMPARE(conn_b->distance.bandwidth, conn_a->distance.bandwidth);
    if (cmp != 0) {
        return cmp;
    }

    /* Optional: tie-breaker for deterministic sort */

    /* NIC device name */
    hnic_a = ucs_topo_sys_device_get_name(conn_a->hnic_device);
    hnic_b = ucs_topo_sys_device_get_name(conn_b->hnic_device);
    cmp    = strcmp(hnic_a, hnic_b);
    if (cmp != 0) {
        return cmp;
    }

    /* Gaudi device name */
    cmp = strcmp(conn_a->gaudi_dev_name, conn_b->gaudi_dev_name);
    if (cmp != 0) {
        return cmp;
    }

    /* Numeric fallbacks (should never differ) */
    if (conn_a->hnic_device != conn_b->hnic_device) {
        return COMPARE(conn_b->hnic_device, conn_a->hnic_device);
    }
    return COMPARE(conn_a->gaudi_device, conn_b->gaudi_device);
}

static void ucs_gaudi_sys_cpuset_for_numa_node(ucs_sys_cpuset_t *cpuset,
                                               ucs_numa_node_t node)
{
    unsigned int cpu;

    CPU_ZERO(cpuset);
    for (cpu = 0; cpu < ucs_numa_num_configured_cpus(); cpu++) {
        if (ucs_numa_node_of_cpu(cpu) == node) {
            CPU_SET(cpu, cpuset);
        }
    }
}

/* Check if common_path indicates a PCIe Host Bridge (e.g., root complex) */
static int ucs_gaudi_is_host_bridge_path(const char *common_path)
{
    const char *last;
    char format_check[16];
    unsigned dom, bus;
    int n;

    if (!common_path) {
        ucs_debug("common_path is NULL");
        return 0;
    }

    last = strrchr(common_path, '/');
    last = last ? last + 1 : common_path;

    /* Ensure the segment starts with "pci" and matches "pciXXXX:YY" */
    if (strncmp(last, "pci", 3) != 0) {
        /* ucs_debug("common_path %s does not start with 'pci'", common_path); */
        return 0;
    }

    n = sscanf(last, "pci%4x:%2x%15s", &dom, &bus, format_check);
    if (n != 2) {
        /* ucs_debug("common_path %s does not match 'pciXXXX:YY' format", common_path); */
        return 0;
    }

    /* Bus 00 typically indicates a PCIe root complex (Host Bridge) */
    if (bus != 0) {
        /* ucs_debug("common_path %s has non-zero bus %02x, not a Host Bridge", common_path, bus); */
        return 0;
    }

    return 1;
}

/* Print connection matrix in a format similar to nvidia-smi topo -m */
static void ucs_gaudi_print_connection_matrix()
{
    ucs_gaudi_connection_t *conn;
    ucs_numa_node_t gaudi_numa, hnic_numa;
    ucs_sys_device_t gaudi, hnic;
    ucs_sys_cpuset_t cpuset;
    ucs_status_t status;
    const char *gaudi_name, *hnic_name;
    char *path1, *path2, *common_path;
    char *buffer;
    char numa_str[16];
    char module_id_str[16];
    char cpu_affinity[128];
    char connection_type[8];
    uint32_t module_id;
    uint16_t hnic_vendor_id;
    unsigned i, j, k, hops;
    size_t buffer_size;

    /* allocate three scratch buffers */
    status = ucs_string_alloc_path_buffer(&path1, "path1");
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_string_alloc_path_buffer(&path2, "path2");
    if (status != UCS_OK) {
        goto free_path1;
    }

    status = ucs_string_alloc_path_buffer(&common_path, "common_path");
    if (status != UCS_OK) {
        goto free_path2;
    }

    buffer_size = 256 + (ucs_gaudi_topo_ctx.num_hnic_devices * 20);
    buffer      = ucs_malloc(buffer_size, "buffer");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto free_path3;
    }

    /* Print header */
    ucs_snprintf_safe(buffer, buffer_size, "%-12s %-15s %-12s", "ModuleID",
                      "Gaudi", "NUMA ID");

    for (i = 0; i < ucs_gaudi_topo_ctx.num_hnic_devices; i++) {
        hnic           = ucs_gaudi_topo_ctx.hnic_devices[i];
        hnic_vendor_id = ucs_gaudi_topo_ctx.hnic_vendor_ids[i];
        if (!ucs_gaudi_is_hnic_active(hnic, hnic_vendor_id)) {
            continue;
        }
        hnic_name = ucs_topo_sys_device_get_name(hnic);
        ucs_snprintf_safe(buffer + strlen(buffer), buffer_size - strlen(buffer),
                          " %-15s", hnic_name);
    }
    ucs_snprintf_safe(buffer + strlen(buffer), buffer_size - strlen(buffer),
                      " %-20s", "CPU Affinity");
    ucs_info("%s", buffer);


    /* Print rows for each Gaudi device */
    for (i = 0; i < ucs_gaudi_topo_ctx.num_gaudi_devices; i++) {
        gaudi      = ucs_gaudi_topo_ctx.gaudi_devices[i];
        gaudi_name = ucs_gaudi_topo_ctx.gaudi_devices_names[i];
        gaudi_numa = ucs_topo_sys_device_get_numa_node(gaudi);

        /* Get module ID */
        status = ucs_gaudi_read_module_id(gaudi_name, &module_id);
        if (status != UCS_OK) {
            ucs_strncpy_safe(module_id_str, "N/A", sizeof(module_id_str));
        } else {
            ucs_snprintf_safe(module_id_str, sizeof(module_id_str), "%u",
                              module_id);
        }

        /* Get NUMA node for Gaudi device (NUMA column) */
        if (gaudi_numa == UCS_NUMA_NODE_UNDEFINED) {
            ucs_strncpy_safe(numa_str, "N/A", sizeof(numa_str));

            ucs_strncpy_safe(cpu_affinity, "N/A", sizeof(cpu_affinity));
        } else {
            ucs_snprintf_safe(numa_str, sizeof(numa_str), "%d", gaudi_numa);

            ucs_gaudi_sys_cpuset_for_numa_node(&cpuset, gaudi_numa);
            ucs_make_affinity_str(&cpuset, cpu_affinity, sizeof(cpu_affinity));
        }

        /* Start row with Gaudi info */
        ucs_snprintf_safe(buffer, buffer_size, "%-12s %-15s %-12s",
                          module_id_str, gaudi_name, numa_str);

        /* Fill connection types for each HNIC */
        for (j = 0; j < ucs_gaudi_topo_ctx.num_hnic_devices; j++) {
            hnic           = ucs_gaudi_topo_ctx.hnic_devices[j];
            hnic_vendor_id = ucs_gaudi_topo_ctx.hnic_vendor_ids[j];
            if (!ucs_gaudi_is_hnic_active(hnic, hnic_vendor_id)) {
                continue;
            }
            hnic_numa = ucs_topo_sys_device_get_numa_node(hnic);

            /* Find connection */
            connection_type[0] = '\0';
            for (k = 0; k < ucs_gaudi_topo_ctx.num_connections; k++) {
                conn = &ucs_gaudi_topo_ctx.connections[k];
                if (conn->gaudi_device == gaudi && conn->hnic_device == hnic) {
                    /* 1. Different NUMA - SYS */
                    if (gaudi_numa != UCS_NUMA_NODE_UNDEFINED &&
                        hnic_numa != UCS_NUMA_NODE_UNDEFINED &&
                        gaudi_numa != hnic_numa) {
                        ucs_strncpy_safe(connection_type, "SYS",
                                         sizeof(connection_type));
                        break;
                    }

                    /* 2. Same NUMA - based on PCIe hops */
                    status = ucs_gaudi_sys_dev_to_sysfs_path(gaudi, path1,
                                                             PATH_MAX);
                    if (status != UCS_OK) {
                        continue;
                    }
                    status = ucs_gaudi_sys_dev_to_sysfs_path(hnic, path2,
                                                             PATH_MAX);
                    if (status != UCS_OK) {
                        continue;
                    }
                    hops = ucs_gaudi_count_pcie_hops(path1, path2, common_path);

                    if (hops <= 1) {
                        ucs_strncpy_safe(connection_type, "PIX",
                                         sizeof(connection_type));
                    } else if (hops <= 3) {
                        ucs_strncpy_safe(connection_type, "PXB",
                                         sizeof(connection_type));
                    } else if (ucs_gaudi_is_host_bridge_path(common_path)) {
                        ucs_strncpy_safe(connection_type, "PHB",
                                         sizeof(connection_type));
                    } else if (gaudi_numa != UCS_NUMA_NODE_UNDEFINED &&
                               hnic_numa != UCS_NUMA_NODE_UNDEFINED) {
                        /* Same NUMA, no shared root complex */
                        ucs_strncpy_safe(connection_type, "NODE",
                                         sizeof(connection_type));
                    } else {
                        /* At least one NUMA unknown, or any other fallback */
                        ucs_strncpy_safe(connection_type, "SYS",
                                         sizeof(connection_type));
                    }
                    break;
                }
            }
            if (connection_type[0] == '\0') {
                ucs_strncpy_safe(connection_type, "SYS",
                                 sizeof(connection_type));
            }
            ucs_snprintf_safe(buffer + strlen(buffer),
                              buffer_size - strlen(buffer), " %-15s",
                              connection_type);
        }

        /* Append CPU affinity */
        ucs_snprintf_safe(buffer + strlen(buffer), buffer_size - strlen(buffer),
                          " %-20s", cpu_affinity);
        ucs_info("%s", buffer);
    }

    /* Print NIC Legend with NUMA nodes */
    ucs_info("\nNIC Legend:");
    for (i = 0; i < ucs_gaudi_topo_ctx.num_hnic_devices; i++) {
        hnic           = ucs_gaudi_topo_ctx.hnic_devices[i];
        hnic_vendor_id = ucs_gaudi_topo_ctx.hnic_vendor_ids[i];
        if (!ucs_gaudi_is_hnic_active(hnic, hnic_vendor_id)) {
            continue;
        }

        hnic_name = ucs_topo_sys_device_get_name(hnic);
        hnic_numa = ucs_topo_sys_device_get_numa_node(hnic);
        if (hnic_numa == UCS_NUMA_NODE_UNDEFINED) {
            ucs_snprintf_safe(numa_str, sizeof(numa_str), "N/A");
        } else {
            ucs_snprintf_safe(numa_str, sizeof(numa_str), "%d", hnic_numa);
        }
        ucs_info("  NIC%u: %s (NUMA %s)", i, hnic_name, numa_str);
    }

    /* Print Connection Legend */
    ucs_info("\nLegend:");
    ucs_info("  SYS  = Connection traversing PCIe as well as the SMP "
             "interconnect between NUMA nodes (e.g., QPI/UPI)");
    ucs_info("  NODE = Connection traversing PCIe as well as the interconnect "
             "between PCIe Host Bridges within a NUMA node");
    ucs_info("  PHB  = Connection traversing PCIe as well as a PCIe Host "
             "Bridge (typically the CPU)");
    ucs_info("  PXB  = Connection traversing multiple PCIe bridges (without "
             "traversing the PCIe Host Bridge)");
    ucs_info("  PIX  = Connection traversing at most a single PCIe bridge");

free_all:
    ucs_free(buffer);
free_path3:
    ucs_free(common_path);
free_path2:
    ucs_free(path2);
free_path1:
    ucs_free(path1);
out:
    return;
}

/* Return default UCX port for given NIC vendor ID */
static int ucs_gaudi_get_default_port(uint16_t vendor_id)
{
    switch (vendor_id) {
    case UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID: /* 0x15b3 */
        return 1; /* mlx5_X:1 */
    case UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID: /* 0x14e4 */
        return 1; /* bnxt_X:1 */
    default:
        return 1; /* fallback */
    }
}

/*
 * Get and validate the NUMA node ID for a device.
 * Returns node 0 if the node is undefined or out of range.
 */
static inline unsigned
ucs_gaudi_get_validated_numa_node(ucs_sys_device_t device,
                                  unsigned num_numa_nodes)
{
    ucs_numa_node_t node;

    node = ucs_topo_sys_device_get_numa_node(device);
    if (node == UCS_NUMA_NODE_UNDEFINED) {
        ucs_warn("Device %d has undefined NUMA node, falling back to 0",
                 device);
        return 0U;
    }
    if ((unsigned)node >= num_numa_nodes) {
        ucs_debug("Device %d NUMA node %d out of range (max %u), "
                  "falling back to 0",
                  device, node, num_numa_nodes);
        return 0U;
    }
    return (unsigned)node;
}

/* Build a balanced, topology-aware assignment.
 *
 * Pass 1: For each NUMA node that has NICs:
 *   - Balanced capacity split: floor(G_n / N_n) + remainder distributed 1:1 to first NICs
 *   - For each local Gaudi, choose among local NICs with remaining capacity:
 *     1. Minimum latency (primary)
 *     2. Maximum bandwidth (secondary)
 *     3. Least assigned Gaudis (tertiary)
 *     4. Lowest NIC index (quaternary)
 *
 * Pass 2: For any unassigned Gaudis (NUMAs w/o NICs or capacity overflow):
 *   - Search globally across all NICs
 *   - Tie-breaking:
 *     1. Minimum latency
 *     2. Maximum bandwidth
 *     3. Prefer NICs at/under capacity
 *     4. Least assigned Gaudis
 *     5. Lowest NIC index
 *
 * Note: NICs without distance data are silently skipped in both passes.
 */
static ucs_status_t ucs_gaudi_build_assignment_balanced()
{
    ucs_sys_dev_distance_t distance;
    ucs_sys_device_t gaudi_device;
    ucs_sys_device_t hnic_device;
    ucs_status_t status;
    unsigned *gaudi_per_numa    = NULL;
    unsigned *hnic_per_numa     = NULL;
    unsigned *gaudi_offset      = NULL;
    unsigned *hnic_offset       = NULL;
    unsigned *gaudi_cursor      = NULL;
    unsigned *hnic_cursor       = NULL;
    unsigned *gaudi_idx_by_numa = NULL;
    unsigned *hnic_idx_by_numa  = NULL;
    unsigned *hnic_usage        = NULL;
    unsigned *hnic_capacity     = NULL;
    unsigned assigned_count;
    unsigned best_usage;
    unsigned numa_node;
    unsigned gaudi_start;
    unsigned gaudi_end;
    unsigned hnic_start;
    unsigned hnic_end;
    unsigned num_local_gaudis;
    unsigned num_local_hnics;
    unsigned base_capacity;
    unsigned extra_gaudis;
    unsigned hidx;
    unsigned gidx;
    unsigned gi;
    unsigned i, j;
    uint16_t vendor;
    int lat_better;
    int lat_eq;
    int bw_better;
    int bw_eq;
    int use_better;
    int idx_better;
    int best_under_cap;
    int this_under_cap;
    int cap_better;
    double best_latency;
    double best_bandwidth;
    ssize_t best_hidx;

    /* 
     * ucs_gaudi_estimate_distance() uses:
     *   hop_latency_ns = 10e-9 seconds (10 nanoseconds)
     *   hop_bw_penalty = 0.95 (5% per hop)
     * 
     * Therefore, reasonable epsilons are:
     *   Latency: 0.1% of a hop = 10e-9 * 0.001 = 1e-11 seconds
     *   Bandwidth: 0.1% of typical 100GB/s = 1e11 * 0.001 = 1e8 bytes/s
     */
    const double LATENCY_EPSILON   = 1e-11; /* 10 picoseconds, 0.1% of a hop */
    const double BANDWIDTH_EPSILON = 1e8; /* 100 MB/s, 0.1% of 100GB/s */

    const unsigned num_numa_nodes    = ucs_numa_num_configured_nodes();
    const unsigned num_gaudi_devices = ucs_gaudi_topo_ctx.num_gaudi_devices;
    const unsigned num_hnic_devices  = ucs_gaudi_topo_ctx.num_hnic_devices;

    status = UCS_OK;

    /* Reset assignments */
    for (i = 0; i < num_gaudi_devices; i++) {
        ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[i] =
                UCS_SYS_DEVICE_ID_UNKNOWN;
        ucs_gaudi_topo_ctx.assigned_port_for_gaudi[i] = 1;
    }
    ucs_gaudi_topo_ctx.have_assignment = 0;

    /* Short-circuit if no devices */
    if (num_gaudi_devices == 0 || num_hnic_devices == 0) {
        ucs_debug("No devices to assign (Gaudis: %u, NICs: %u)",
                  num_gaudi_devices, num_hnic_devices);
        return status;
    }

    /* Allocate per-NUMA arrays */
    gaudi_per_numa = ucs_calloc(num_numa_nodes, sizeof(*gaudi_per_numa),
                                "gaudi_per_numa");
    hnic_per_numa  = ucs_calloc(num_numa_nodes, sizeof(*hnic_per_numa),
                                "hnic_per_numa");
    gaudi_offset   = ucs_malloc((num_numa_nodes + 1) * sizeof(*gaudi_offset),
                                "gaudi_offset");
    hnic_offset    = ucs_malloc((num_numa_nodes + 1) * sizeof(*hnic_offset),
                                "hnic_offset");
    gaudi_cursor   = ucs_malloc(num_numa_nodes * sizeof(*gaudi_cursor),
                                "gaudi_cursor");
    hnic_cursor    = ucs_malloc(num_numa_nodes * sizeof(*hnic_cursor),
                                "hnic_cursor");

    if (!gaudi_per_numa || !hnic_per_numa || !gaudi_offset || !hnic_offset ||
        !gaudi_cursor || !hnic_cursor) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    /* Count devices per NUMA node */
    for (i = 0; i < num_gaudi_devices; i++) {
        numa_node = ucs_gaudi_get_validated_numa_node(
                ucs_gaudi_topo_ctx.gaudi_devices[i], num_numa_nodes);
        gaudi_per_numa[numa_node]++;
    }
    for (i = 0; i < num_hnic_devices; i++) {
        numa_node = ucs_gaudi_get_validated_numa_node(
                ucs_gaudi_topo_ctx.hnic_devices[i], num_numa_nodes);
        hnic_per_numa[numa_node]++;
    }

    /* Build offset arrays */
    gaudi_offset[0] = 0;
    hnic_offset[0]  = 0;
    for (i = 0; i < num_numa_nodes; i++) {
        gaudi_offset[i + 1] = gaudi_offset[i] + gaudi_per_numa[i];
        hnic_offset[i + 1]  = hnic_offset[i] + hnic_per_numa[i];
    }

    /* Allocate index arrays */
    if (gaudi_offset[num_numa_nodes] > 0) {
        gaudi_idx_by_numa = ucs_malloc(gaudi_offset[num_numa_nodes] *
                                               sizeof(*gaudi_idx_by_numa),
                                       "gaudi_idx_by_numa");
        if (!gaudi_idx_by_numa) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }
    }
    if (hnic_offset[num_numa_nodes] > 0) {
        hnic_idx_by_numa = ucs_malloc(hnic_offset[num_numa_nodes] *
                                              sizeof(*hnic_idx_by_numa),
                                      "hnic_idx_by_numa");
        if (!hnic_idx_by_numa) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }
    }

    /* Initialize cursors */
    for (i = 0; i < num_numa_nodes; i++) {
        gaudi_cursor[i] = gaudi_offset[i];
        hnic_cursor[i]  = hnic_offset[i];
    }

    /* Populate NUMA-indexed arrays */
    for (i = 0; i < num_gaudi_devices; i++) {
        numa_node = ucs_gaudi_get_validated_numa_node(
                ucs_gaudi_topo_ctx.gaudi_devices[i], num_numa_nodes);
        gaudi_idx_by_numa[gaudi_cursor[numa_node]++] = i;
    }
    for (i = 0; i < num_hnic_devices; i++) {
        numa_node = ucs_gaudi_get_validated_numa_node(
                ucs_gaudi_topo_ctx.hnic_devices[i], num_numa_nodes);
        hnic_idx_by_numa[hnic_cursor[numa_node]++] = i;
    }

    /* Allocate usage and capacity tracking */
    hnic_usage    = ucs_calloc(num_hnic_devices, sizeof(*hnic_usage),
                               "hnic_usage");
    hnic_capacity = ucs_malloc(num_hnic_devices * sizeof(*hnic_capacity),
                               "hnic_capacity");
    if (!hnic_usage || !hnic_capacity) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    /* Initialize all capacities to "unlimited" for Pass 2 fallback logic */
    for (i = 0; i < num_hnic_devices; i++) {
        hnic_capacity[i] = UINT_MAX;
    }

    /* Pass 1: Local NUMA-balanced assignment */
    for (i = 0; i < num_numa_nodes; i++) {
        gaudi_start      = gaudi_offset[i];
        gaudi_end        = gaudi_offset[i + 1];
        hnic_start       = hnic_offset[i];
        hnic_end         = hnic_offset[i + 1];
        num_local_gaudis = gaudi_end - gaudi_start;
        num_local_hnics  = hnic_end - hnic_start;
        if (num_local_gaudis == 0 || num_local_hnics == 0) {
            continue;
        }

        /* Calculate balanced capacity per NIC */
        base_capacity = num_local_gaudis / num_local_hnics;
        extra_gaudis  = num_local_gaudis % num_local_hnics;
        for (j = 0; j < num_local_hnics; j++) {
            hidx                = hnic_idx_by_numa[hnic_start + j];
            hnic_capacity[hidx] = base_capacity + ((j < extra_gaudis) ? 1 : 0);
        }

        /* Assign each Gaudi to best local NIC */
        for (gi = gaudi_start; gi < gaudi_end; gi++) {
            gidx         = gaudi_idx_by_numa[gi];
            gaudi_device = ucs_gaudi_topo_ctx.gaudi_devices[gidx];

            best_hidx      = -1;
            best_latency   = DBL_MAX;
            best_bandwidth = -1.0;
            best_usage     = UINT_MAX;

            for (j = 0; j < num_local_hnics; j++) {
                hidx        = hnic_idx_by_numa[hnic_start + j];
                hnic_device = ucs_gaudi_topo_ctx.hnic_devices[hidx];

                if (!ucs_gaudi_lookup_distance(gaudi_device, hnic_device,
                                               &distance)) {
                    continue;
                }

                if (hnic_usage[hidx] >= hnic_capacity[hidx]) {
                    continue;
                }

                /* Tie-breaking: latency -> bandwidth -> usage -> index */
                lat_better = (distance.latency + LATENCY_EPSILON <
                              best_latency);
                lat_eq     = !lat_better && (fabs(distance.latency -
                                                  best_latency) <= LATENCY_EPSILON);
                bw_better  = lat_eq && (distance.bandwidth >
                                       best_bandwidth + BANDWIDTH_EPSILON);
                bw_eq      = lat_eq && !bw_better &&
                        (fabs(distance.bandwidth - best_bandwidth) <=
                         BANDWIDTH_EPSILON);
                use_better = bw_eq && (hnic_usage[hidx] < best_usage);
                idx_better = bw_eq && !use_better &&
                             ((ssize_t)hidx < best_hidx);

                if (lat_better || bw_better || use_better || idx_better) {
                    best_latency   = distance.latency;
                    best_bandwidth = distance.bandwidth;
                    best_usage     = hnic_usage[hidx];
                    best_hidx      = hidx;
                }
            }

            if (best_hidx >= 0) {
                vendor = ucs_gaudi_topo_ctx.hnic_vendor_ids[best_hidx];
                hnic_usage[best_hidx]++;
                ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[gidx] =
                        ucs_gaudi_topo_ctx.hnic_devices[best_hidx];
                ucs_gaudi_topo_ctx.assigned_port_for_gaudi[gidx] =
                        ucs_gaudi_get_default_port(vendor);
            }
        }
    }

    /* Pass 2: Global assignment with soft-cap preference */
    for (i = 0; i < num_gaudi_devices; i++) {
        if (ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[i] !=
            UCS_SYS_DEVICE_ID_UNKNOWN) {
            continue;
        }

        gaudi_device = ucs_gaudi_topo_ctx.gaudi_devices[i];

        best_hidx      = -1;
        best_latency   = DBL_MAX;
        best_bandwidth = -1.0;
        best_usage     = UINT_MAX;
        best_under_cap = 0;

        for (j = 0; j < num_hnic_devices; j++) {
            hnic_device = ucs_gaudi_topo_ctx.hnic_devices[j];

            if (!ucs_gaudi_lookup_distance(gaudi_device, hnic_device,
                                           &distance)) {
                continue;
            }

            this_under_cap = (hnic_usage[j] < hnic_capacity[j]);

            /* Tie-breaking: latency -> bandwidth -> under-cap -> usage -> index */
            lat_better = (distance.latency + LATENCY_EPSILON < best_latency);
            lat_eq     = !lat_better &&
                     (fabs(distance.latency - best_latency) <= LATENCY_EPSILON);
            bw_better = lat_eq && (distance.bandwidth >
                                   best_bandwidth + BANDWIDTH_EPSILON);
            bw_eq     = lat_eq && !bw_better &&
                    (fabs(distance.bandwidth - best_bandwidth) <=
                     BANDWIDTH_EPSILON);
            cap_better = bw_eq && (this_under_cap > best_under_cap);
            use_better = bw_eq && !cap_better && (hnic_usage[j] < best_usage);
            idx_better = bw_eq && !cap_better && !use_better &&
                         ((ssize_t)j < best_hidx);

            if (lat_better || bw_better || cap_better || use_better ||
                idx_better) {
                best_latency   = distance.latency;
                best_bandwidth = distance.bandwidth;
                best_usage     = hnic_usage[j];
                best_under_cap = this_under_cap;
                best_hidx      = j;
            }
        }

        if (best_hidx >= 0) {
            vendor = ucs_gaudi_topo_ctx.hnic_vendor_ids[best_hidx];
            hnic_usage[best_hidx]++;
            ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[i] =
                    ucs_gaudi_topo_ctx.hnic_devices[best_hidx];
            ucs_gaudi_topo_ctx.assigned_port_for_gaudi[i] =
                    ucs_gaudi_get_default_port(vendor);
        }
    }

    /* Finalize and validate assignment */
    assigned_count = 0;
    for (i = 0; i < num_gaudi_devices; i++) {
        if (ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[i] !=
            UCS_SYS_DEVICE_ID_UNKNOWN) {
            assigned_count++;
        }
    }

    ucs_gaudi_topo_ctx.have_assignment = (assigned_count == num_gaudi_devices);
    if (!ucs_gaudi_topo_ctx.have_assignment) {
        ucs_warn("Gaudi->NIC assignment incomplete: %u of %u devices assigned",
                 assigned_count, num_gaudi_devices);
    }

out:
    ucs_free(hnic_usage);
    ucs_free(hnic_capacity);
    ucs_free(gaudi_idx_by_numa);
    ucs_free(hnic_idx_by_numa);
    ucs_free(gaudi_cursor);
    ucs_free(hnic_cursor);
    ucs_free(gaudi_offset);
    ucs_free(hnic_offset);
    ucs_free(gaudi_per_numa);
    ucs_free(hnic_per_numa);

    return status;
}

/* Find best HNIC for a given Gaudi device */
ucs_status_t ucs_gaudi_find_best_connection(const char *accel_name,
                                            ucs_sys_device_t *hnic_device,
                                            int *port_num)
{
    ucs_status_t status;
    unsigned i;

    /* Validate parameters */
    if ((accel_name == NULL) || (hnic_device == NULL) || (port_num == NULL)) {
        ucs_error("Invalid NULL parameter(s) to "
                  "ucs_gaudi_find_best_connection()");
        return UCS_ERR_INVALID_PARAM;
    }

    /* Perform lazy initialization */
    status = ucs_gaudi_lazy_init();
    if (status != UCS_OK) {
        if (status == UCS_ERR_UNSUPPORTED) {
            /* Provider disabled: treat like "nothing suitable" */
            return UCS_ERR_NO_ELEM;
        }
        ucs_error("Failed to initialize Gaudi topology: %s",
                  ucs_status_string(status));
        return status;
    }

    ucs_spin_lock(&ucs_gaudi_topo_ctx.lock);

    /* Return cached balanced assignment instead of searching connections */
    for (i = 0; i < ucs_gaudi_topo_ctx.num_gaudi_devices; i++) {
        if (!strcmp(accel_name, ucs_gaudi_topo_ctx.gaudi_devices_names[i])) {
            break;
        }
    }

    if (i < ucs_gaudi_topo_ctx.num_gaudi_devices &&
        ucs_gaudi_topo_ctx.have_assignment &&
        ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[i] !=
                UCS_SYS_DEVICE_ID_UNKNOWN) {
        *hnic_device = ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi[i];
        *port_num    = ucs_gaudi_topo_ctx.assigned_port_for_gaudi[i];
        ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
        ucs_info("Selected HNIC %s:%d for Gaudi %s",
                 ucs_topo_sys_device_get_name(*hnic_device), *port_num,
                 accel_name);
        return UCS_OK;
    }

    ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
    ucs_error("No suitable HNIC found for Gaudi %s", accel_name);
    return UCS_ERR_NO_ELEM;
}

/* Get device-to-device distance for UCX topology queries */
static ucs_status_t ucs_gaudi_get_distance(ucs_sys_device_t device1,
                                           ucs_sys_device_t device2,
                                           ucs_sys_dev_distance_t *distance)
{
    ucs_gaudi_connection_t *conn;
    ucs_status_t status;
    unsigned i;
    uint16_t vendor_id1, vendor_id2;

    /* Validate parameters */
    if (distance == NULL) {
        ucs_error("distance parameter cannot be NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    /* Default distance */
    *distance = ucs_topo_default_distance;

    /* Perform lazy initialization */
    status = ucs_gaudi_lazy_init();
    if (status != UCS_OK) {
        if (status == UCS_ERR_UNSUPPORTED) {
            /* Provider disabled: fall back to generic estimation */
            goto fallback;
        }
        ucs_error("Failed to initialize Gaudi topology: %s",
                  ucs_status_string(status));
        return status;
    }

    /* If either device is unknown or they are identical, assume near topology */
    if ((device1 == UCS_SYS_DEVICE_ID_UNKNOWN) ||
        (device2 == UCS_SYS_DEVICE_ID_UNKNOWN) || (device1 == device2)) {
        goto fallback;
    }

    /* Check if either device is a Gaudi device or HNIC */
    if (ucs_gaudi_read_vendor_id(device1, &vendor_id1) != UCS_OK) {
        goto fallback;
    }
    if (ucs_gaudi_read_vendor_id(device2, &vendor_id2) != UCS_OK) {
        goto fallback;
    }

    /* If one device is Gaudi or HNIC, check connection matrix */
    if ((vendor_id1 == UCS_GAUDI_TOPO_VENDOR_ID ||
         vendor_id1 == UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID ||
         vendor_id1 == UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID) ||
        (vendor_id2 == UCS_GAUDI_TOPO_VENDOR_ID ||
         vendor_id2 == UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID ||
         vendor_id2 == UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID)) {
        ucs_spin_lock(&ucs_gaudi_topo_ctx.lock);
        for (i = 0; i < ucs_gaudi_topo_ctx.num_connections; i++) {
            conn = &ucs_gaudi_topo_ctx.connections[i];
            if ((conn->gaudi_device == device1 &&
                 conn->hnic_device == device2) ||
                (conn->gaudi_device == device2 &&
                 conn->hnic_device == device1)) {
                *distance = conn->distance;
                ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
                return UCS_OK;
            }
        }
        ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
    }

fallback:
    /* Fallback to estimate_distance for other device pairs */
    return ucs_gaudi_estimate_distance(device1, device2, distance);
}

/* Get device-to-memory distance for UCX topology queries */
static void ucs_gaudi_get_memory_distance(ucs_sys_device_t device,
                                          ucs_sys_dev_distance_t *distance)
{
    ucs_status_t status;
    ucs_sys_cpuset_t thread_cpuset;
    ucs_numa_node_t device_numa;
    unsigned cpu, num_cpus, cpuset_size;
    double total_distance;
    int full_affinity;
    uint16_t vendor_id;

    /* Validate parameters */
    if (distance == NULL) {
        ucs_error("distance parameter cannot be NULL");
        return;
    }

    *distance = ucs_topo_default_distance;

    /* If device is unknown, return default distance */
    if (device == UCS_SYS_DEVICE_ID_UNKNOWN) {
        return;
    }

    /* Check if device is Gaudi or HNIC */
    status = ucs_gaudi_read_vendor_id(device, &vendor_id);
    if (status != UCS_OK || (vendor_id != UCS_GAUDI_TOPO_VENDOR_ID &&
                             vendor_id != UCS_GAUDI_TOPO_MELLANOX_VENDOR_ID &&
                             vendor_id != UCS_GAUDI_TOPO_BROADCOM_VENDOR_ID)) {
        return;
    }

    /* Get thread CPU affinity */
    status = ucs_sys_pthread_getaffinity(&thread_cpuset);
    if (status != UCS_OK) {
        /* Assume full CPU affinity if getting affinity fails */
        full_affinity = 1;
    } else {
        full_affinity = 0;
    }

    /* Get device NUMA node */
    device_numa = ucs_topo_sys_device_get_numa_node(device);
    if (device_numa == UCS_NUMA_NODE_UNDEFINED) {
        device_numa = UCS_NUMA_NODE_DEFAULT;
    }

    /* Sum NUMA distances for CPUs in affinity set */
    num_cpus       = ucs_numa_num_configured_cpus();
    total_distance = 0;
    for (cpu = 0; cpu < num_cpus; cpu++) {
        if (!full_affinity && !CPU_ISSET(cpu, &thread_cpuset)) {
            continue;
        }
        total_distance += ucs_numa_distance(device_numa,
                                            ucs_numa_node_of_cpu(cpu));
    }

    /* Set distance: bandwidth from default, latency from average NUMA distance */
    distance->bandwidth = ucs_topo_default_distance.bandwidth;
    cpuset_size         = full_affinity ? num_cpus : CPU_COUNT(&thread_cpuset);

    /* Guard against misconfigured affinity resulting in empty cpuset.
     * This can occur if thread CPU affinity is set to an empty mask.
     */
    if (cpuset_size == 0) {
        /* Return default distance; cannot compute meaningful average */
        return;
    }

    /* Logic and baseline taken from UCX: ucs_topo_sysfs_numa_distance_to_latency
     * NUMA distance normalized to 10, latency formula assumes ~10ns per unit,
     */
    distance->latency = (total_distance / cpuset_size) * 10e-9;
}

/* Initialize spinlock exactly once */
static void ucs_gaudi_spinlock_once_init()
{
    gaudi_spinlock_init_status = ucs_spinlock_init(&ucs_gaudi_topo_ctx.lock, 0);
}

/* Initialization function */
void ucs_gaudi_topo_init()
{
    const char *disable;

    disable = getenv("UCS_GAUDI_TOPO_DISABLE");
    if (disable && strcmp(disable, "0") != 0) {
        ucs_debug("Gaudi topology provider disabled by UCS_GAUDI_TOPO_DISABLE");
        return;
    }

    /* Prevent double registration */
    if (ucs_gaudi_topo_ctx.provider_added) {
        ucs_debug("Gaudi topology provider already registered");
        return;
    }

    /* Ensure spinlock exists even if lazy init is first */
    pthread_once(&gaudi_spinlock_once_flag, ucs_gaudi_spinlock_once_init);
    if (gaudi_spinlock_init_status != UCS_OK) {
        ucs_error("Failed to initialize spinlock: %s",
                  ucs_status_string(gaudi_spinlock_init_status));
        return;
    }

    pthread_mutex_lock(&gaudi_init_mutex);
    if (!ucs_gaudi_topo_ctx.provider_added) {
        ucs_debug("Registering Gaudi topology provider");
        ucs_list_add_head(&ucs_sys_topo_providers_list,
                          &ucs_gaudi_topo_provider.list);
        ucs_gaudi_topo_ctx.provider_added = 1;
        ucs_debug("Gaudi topology provider registered");
    } else {
        ucs_debug("Gaudi topology provider already registered (raced)");
    }
    pthread_mutex_unlock(&gaudi_init_mutex);
}

static ucs_status_t ucs_gaudi_lazy_init()
{
    ucs_status_t status;
    const char *disable;

    disable = getenv("UCS_GAUDI_TOPO_DISABLE");
    if (disable && strcmp(disable, "0") != 0) {
        ucs_debug("Gaudi topology provider disabled by UCS_GAUDI_TOPO_DISABLE");
        return UCS_ERR_UNSUPPORTED;
    }

    /* Ensure spinlock exists */
    pthread_once(&gaudi_spinlock_once_flag, ucs_gaudi_spinlock_once_init);
    if (gaudi_spinlock_init_status != UCS_OK) {
        ucs_error("Failed to initialize spinlock: %s",
                  ucs_status_string(gaudi_spinlock_init_status));
        return gaudi_spinlock_init_status;
    }

    ucs_spin_lock(&ucs_gaudi_topo_ctx.lock);

    if (ucs_gaudi_topo_ctx.initialized) {
        ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
        return UCS_OK;
    }

    ucs_debug("Performing lazy initialization of Gaudi topology");

    status = ucs_gaudi_enumerate_devices();
    if (status != UCS_OK) {
        ucs_error("Failed to enumerate Gaudi devices: %s",
                  ucs_status_string(status));
        ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
        return status;
    }

    status = ucs_gaudi_create_connection_matrix();
    if (status != UCS_OK) {
        ucs_error("Failed to create connection matrix: %s",
                  ucs_status_string(status));
        goto out;
    }

    qsort(ucs_gaudi_topo_ctx.connections, ucs_gaudi_topo_ctx.num_connections,
          sizeof(ucs_gaudi_connection_t), ucs_gaudi_compare_connections);

    ucs_free(ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi);
    ucs_free(ucs_gaudi_topo_ctx.assigned_port_for_gaudi);
    ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi =
            ucs_calloc(ucs_gaudi_topo_ctx.num_gaudi_devices,
                       sizeof(*ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi),
                       "assigned_hnic_for_gaudi");
    ucs_gaudi_topo_ctx.assigned_port_for_gaudi =
            ucs_calloc(ucs_gaudi_topo_ctx.num_gaudi_devices,
                       sizeof(*ucs_gaudi_topo_ctx.assigned_port_for_gaudi),
                       "assigned_port_for_gaudi");
    if (!ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi ||
        !ucs_gaudi_topo_ctx.assigned_port_for_gaudi) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    /* Build a single, balanced assignment */
    status = ucs_gaudi_build_assignment_balanced();
    if (status != UCS_OK) {
        goto out;
    }

    if (ucs_global_opts.log_component.log_level >= UCS_LOG_LEVEL_DEBUG) {
        ucs_gaudi_print_connection_matrix();
    }

    ucs_gaudi_topo_ctx.initialized = 1;
    ucs_debug("Gaudi topology initialized");
    ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
    return UCS_OK;

out:
    /* Free allocations that happened before error */
    ucs_free(ucs_gaudi_topo_ctx.gaudi_devices);
    ucs_free(ucs_gaudi_topo_ctx.gaudi_devices_names);
    ucs_free(ucs_gaudi_topo_ctx.hnic_devices);
    ucs_free(ucs_gaudi_topo_ctx.hnic_vendor_ids);
    ucs_free(ucs_gaudi_topo_ctx.connections);
    ucs_free(ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi);
    ucs_free(ucs_gaudi_topo_ctx.assigned_port_for_gaudi);

    ucs_gaudi_topo_ctx.gaudi_devices           = NULL;
    ucs_gaudi_topo_ctx.gaudi_devices_names     = NULL;
    ucs_gaudi_topo_ctx.hnic_devices            = NULL;
    ucs_gaudi_topo_ctx.hnic_vendor_ids         = NULL;
    ucs_gaudi_topo_ctx.connections             = NULL;
    ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi = NULL;
    ucs_gaudi_topo_ctx.assigned_port_for_gaudi = NULL;

    ucs_gaudi_topo_ctx.num_gaudi_devices = 0;
    ucs_gaudi_topo_ctx.num_hnic_devices  = 0;
    ucs_gaudi_topo_ctx.num_connections   = 0;
    ucs_gaudi_topo_ctx.have_assignment   = 0;

    ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
    return status;
}

/* Cleanup function */
void ucs_gaudi_topo_cleanup()
{
    pthread_mutex_lock(&gaudi_init_mutex);

    /* If provider was never added, nothing to clean up */
    if (!ucs_gaudi_topo_ctx.provider_added) {
        pthread_mutex_unlock(&gaudi_init_mutex);
        return;
    }

    /* Provider was added, so spinlock is guaranteed initialized */
    ucs_spin_lock(&ucs_gaudi_topo_ctx.lock);

    /* Remove the Gaudi topology provider from the list */
    ucs_list_del(&ucs_gaudi_topo_provider.list);

    ucs_free(ucs_gaudi_topo_ctx.gaudi_devices);
    ucs_free(ucs_gaudi_topo_ctx.gaudi_devices_names);
    ucs_free(ucs_gaudi_topo_ctx.hnic_devices);
    ucs_free(ucs_gaudi_topo_ctx.hnic_vendor_ids);
    ucs_free(ucs_gaudi_topo_ctx.connections);
    ucs_free(ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi);
    ucs_free(ucs_gaudi_topo_ctx.assigned_port_for_gaudi);

    /* Reset pointers */
    ucs_gaudi_topo_ctx.gaudi_devices           = NULL;
    ucs_gaudi_topo_ctx.gaudi_devices_names     = NULL;
    ucs_gaudi_topo_ctx.hnic_devices            = NULL;
    ucs_gaudi_topo_ctx.hnic_vendor_ids         = NULL;
    ucs_gaudi_topo_ctx.connections             = NULL;
    ucs_gaudi_topo_ctx.assigned_hnic_for_gaudi = NULL;
    ucs_gaudi_topo_ctx.assigned_port_for_gaudi = NULL;

    /* Reset counters/flags */
    ucs_gaudi_topo_ctx.num_gaudi_devices = 0;
    ucs_gaudi_topo_ctx.num_hnic_devices  = 0;
    ucs_gaudi_topo_ctx.num_connections   = 0;
    ucs_gaudi_topo_ctx.initialized       = 0;
    ucs_gaudi_topo_ctx.provider_added    = 0;
    ucs_gaudi_topo_ctx.have_assignment   = 0;

    ucs_spin_unlock(&ucs_gaudi_topo_ctx.lock);
    pthread_mutex_unlock(&gaudi_init_mutex);
    ucs_debug("Gaudi topology cleaned up");
}
