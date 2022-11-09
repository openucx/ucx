/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/sys/topo/base/topo.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>

#define UCS_TOPO_SYSFS_PCI_PREFIX   "/sys/bus/pci/devices/"
#define UCS_TOPO_SYSFS_DEVICES_ROOT "/sys/devices"

static ucs_status_t
ucs_topo_get_sysfs_path(ucs_sys_device_t sys_dev, char *path, size_t max)
{
    const size_t prefix_length = strlen(UCS_TOPO_SYSFS_PCI_PREFIX);
    char link_path[PATH_MAX];
    unsigned num_devices;
    ucs_status_t status;

    if (max < PATH_MAX) {
        status = UCS_ERR_BUFFER_TOO_SMALL;
        goto out;
    }

    ucs_sys_topo_lock_ctx();

    num_devices = ucs_topo_num_devices_non_sync();
    if (sys_dev >= num_devices) {
        ucs_error("system device %d is invalid (max: %d)", sys_dev,
                  num_devices);
        status = UCS_ERR_INVALID_PARAM;
        goto out_unlock;
    }

    ucs_sys_topo_lock_ctx();

    ucs_strncpy_safe(link_path, UCS_TOPO_SYSFS_PCI_PREFIX, PATH_MAX);
    ucs_topo_device_bus_id_str(sys_dev, 0, link_path + prefix_length,
                               PATH_MAX - prefix_length);
    if (realpath(link_path, path) == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto out_unlock;
    }

    status = UCS_OK;

out_unlock:
    ucs_sys_topo_unlock_ctx();
out:
    return status;
}


static int ucs_topo_is_pci_root(const char *path)
{
    int count;
    sscanf(path, UCS_TOPO_SYSFS_DEVICES_ROOT "/pci%*d:%*d%n", &count);
    return count == strlen(path);
}

static int ucs_topo_is_sys_root(const char *path)
{
    return !strcmp(path, UCS_TOPO_SYSFS_DEVICES_ROOT);
}

static void ucs_topo_sys_root_distance(ucs_sys_dev_distance_t *distance)
{
    distance->latency = 500e-9;
    switch (ucs_arch_get_cpu_model()) {
    case UCS_CPU_MODEL_AMD_ROME:
    case UCS_CPU_MODEL_AMD_MILAN:
        distance->bandwidth = 5100 * UCS_MBYTE;
        break;
    default:
        distance->bandwidth = 220 * UCS_MBYTE;
        break;
    }
}

static void ucs_topo_pci_root_distance(const char *path1, const char *path2,
                                       ucs_sys_dev_distance_t *distance)
{
    size_t path_distance = ucs_path_calc_distance(path1, path2);

    ucs_trace_data("distance between '%s' and '%s' is %zu", path1, path2,
                   path_distance);
    ucs_assertv(path_distance > 0, "path1=%s path2=%s", path1, path2);

    /* TODO set latency/bandwidth by CPU model */
    distance->latency   = 300e-9;
    distance->bandwidth = ucs_min(3500.0 * UCS_MBYTE,
                                  (19200.0 * UCS_MBYTE) / path_distance);
}

static ucs_status_t
ucs_topo_get_distance_sysfs(ucs_sys_device_t device1, ucs_sys_device_t device2,
                            ucs_sys_dev_distance_t *distance)
{
    char path1[PATH_MAX], path2[PATH_MAX], common_path[PATH_MAX];
    ucs_status_t status;

    /* If one of the devices is unknown, we assume near topology */
    if ((device1 == UCS_SYS_DEVICE_ID_UNKNOWN) ||
        (device2 == UCS_SYS_DEVICE_ID_UNKNOWN) || (device1 == device2)) {
        *distance = ucs_topo_default_distance;
        return UCS_OK;
    }

    status = ucs_topo_get_sysfs_path(device1, path1, sizeof(path1));
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_get_sysfs_path(device2, path2, sizeof(path2));
    if (status != UCS_OK) {
        return status;
    }

    ucs_path_get_common_parent(path1, path2, common_path);
    if (ucs_topo_is_sys_root(common_path)) {
        ucs_topo_sys_root_distance(distance);
    } else if (ucs_topo_is_pci_root(common_path)) {
        ucs_topo_pci_root_distance(path1, path2, distance);
    } else {
        *distance = ucs_topo_default_distance;
    }

    return UCS_OK;
}

static ucs_sys_topo_method_t ucs_sys_topo_sysfs_method = {
    .name         = "sysfs",
    .get_distance = ucs_topo_get_distance_sysfs,
};

void UCS_F_CTOR ucs_topo_sysfs_init()
{
    ucs_topo_register_provider(&ucs_sys_topo_sysfs_method);
}

void UCS_F_DTOR ucs_topo_sysfs_cleanup()
{
    ucs_topo_unregister_provider(&ucs_sys_topo_sysfs_method);
}
