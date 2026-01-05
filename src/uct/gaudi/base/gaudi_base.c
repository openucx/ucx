/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gaudi_base.h"

#include <ucs/arch/atomic.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/sys/module.h>
#include <ucs/memory/numa.h>
#include <ucs/sys/topo/base/topo.h>
#include <uct/gaudi/gaudi_gdr/gaudi_gdr_md.h>

#include <inttypes.h>
#include <fcntl.h>
#include <pthread.h>
#include <hlthunk.h>
#include <synapse_api.h>


int uct_gaudi_base_get_fd(int device_id, bool *fd_created)
{
    synDeviceInfo deviceInfo;

    if (synDeviceGetInfo(-1, &deviceInfo) != synSuccess) {
        int fd = hlthunk_open_by_module_id(device_id);
        if (fd < 0) {
            ucs_info("failed to get device fd via hlthunk_open_by_module_id, "
                     "id %d",
                     device_id);
            fd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
        }

        if (fd >= 0 && fd_created != NULL) {
            *fd_created = true;
        }
        return fd;
    }

    if (fd_created != NULL) {
        *fd_created = false;
    }
    return deviceInfo.fd;
}

void uct_gaudi_base_close_fd(int fd, bool fd_created)
{
    if (fd_created && fd >= 0) {
        hlthunk_close(fd);
    }
}

void uct_gaudi_base_close_dmabuf_fd(int fd)
{
    if (fd >= 0) {
        close(fd);
    }
}

ucs_status_t uct_gaudi_base_get_sysdev(int fd, ucs_sys_device_t *sys_dev)
{
    ucs_status_t status;
    char pci_bus_id[13];
    int rc;

    rc = hlthunk_get_pci_bus_id_from_fd(fd, pci_bus_id, sizeof(pci_bus_id));
    if (rc != 0) {
        ucs_error("failed to get pci_bus_id via hlthunk_get_pci_bus_id_from_fd "
                  "(fd=%d)",
                  fd);
        return UCS_ERR_IO_ERROR;
    }

    status = ucs_topo_find_device_by_bdf_name(pci_bus_id, sys_dev);
    if (status != UCS_OK) {
        ucs_error("failed to get sys device from pci_bus_id %s", pci_bus_id);
        return status;
    }

    return UCS_OK;
}

ucs_status_t uct_gaudi_base_get_info(int fd,
                                     uint64_t *device_base_allocated_address,
                                     uint64_t *device_base_address,
                                     uint64_t *totalSize, int *dmabuf_fd)
{
    uint64_t addr, hbm_pool_start, size, offset;
    scal_handle_t scal_handle;
    scal_pool_handle_t scal_pool_handle;
    scal_memory_pool_infoV2 scal_mem_pool_info;
    int rc;

    rc = scal_get_handle_from_fd(fd, &scal_handle);
    if (rc != UCT_GAUID_SCAL_SUCCESS) {
        /*
         * If rc value equal UCT_GAUID_SCAL_SUCCESS, it mean that 
           it use synDeviceAcquireByModuleId to open Gaudi device.
         * Otherwise, the device is opened via hlthunk_open_by_module_id function.
         */
        rc = scal_init(fd, "", &scal_handle, NULL);
    }

    if (rc != UCT_GAUID_SCAL_SUCCESS) {
        ucs_error("failed to get scal handle from gaudi device (fd=%d, rc=%d)",
                  fd, rc);
        return UCS_ERR_IO_ERROR;
    }

    rc = scal_get_pool_handle_by_name(scal_handle, "global_hbm",
                                      &scal_pool_handle);
    if (rc != UCT_GAUID_SCAL_SUCCESS) {
        ucs_error("failed to get scal pool");
        return UCS_ERR_INVALID_ADDR;
    }

    rc = scal_pool_get_infoV2(scal_pool_handle, &scal_mem_pool_info);
    if (rc != UCT_GAUID_SCAL_SUCCESS) {
        ucs_error("failed to get scal pool info");
        return UCS_ERR_INVALID_ADDR;
    }

    addr           = scal_mem_pool_info.device_base_allocated_address;
    hbm_pool_start = scal_mem_pool_info.device_base_address;
    size           = scal_mem_pool_info.totalSize;
    offset         = hbm_pool_start - addr;
    *dmabuf_fd     = hlthunk_device_mapped_memory_export_dmabuf_fd(
            fd, addr, size, offset, (O_RDWR | O_CLOEXEC));
    if (*dmabuf_fd < 0) {
        ucs_error("failed to get dmabuf fd from fd %d", fd);
        return UCS_ERR_INVALID_ADDR;
    }

    *device_base_allocated_address = addr;
    *device_base_address           = hbm_pool_start;
    *totalSize                     = size;
    return UCS_OK;
}

ucs_status_t
uct_gaudi_base_query_devices(uct_md_h md,
                             uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p)
{
    uct_gaudi_md_t *gaudi_md = ucs_derived_of(md, uct_gaudi_md_t);
    ucs_sys_device_t sys_dev;
    ucs_status_t status;

    status = uct_gaudi_base_get_sysdev(gaudi_md->fd, &sys_dev);
    if (status != UCS_OK) {
        return status;
    }
    return uct_single_device_resource(md, md->component->name,
                                      UCT_DEVICE_TYPE_ACC, sys_dev,
                                      tl_devices_p, num_tl_devices_p);
}

static void
uct_gaudi_base_configure_sys_device_from_fd(int fd, int index,
                                            ucs_sys_device_t *sys_dev_p)
{
    ucs_status_t status;
    struct hlthunk_hw_ip_info hw_ip;
    const unsigned sys_device_priority = 10;
    char device_name[16];
    int rc;

    ucs_assert(fd >= 0);

    status = uct_gaudi_base_get_sysdev(fd, sys_dev_p);
    if (status != UCS_OK) {
        goto err;
    }

    memset(&hw_ip, 0, sizeof(hw_ip));
    rc = hlthunk_get_hw_ip_info(fd, &hw_ip);
    if (rc) {
        ucs_error("failed to get hw_ip info for fd %d (rc=%d)", fd, rc);
        goto err;
    }

    status = ucs_topo_sys_device_set_user_value(*sys_dev_p, hw_ip.module_id);
    if (status != UCS_OK) {
        ucs_error("failed to set user value %u for sys_dev %d", hw_ip.module_id,
                  *sys_dev_p);
        goto err;
    }

    ucs_snprintf_safe(device_name, sizeof(device_name), "GAUDI_%d", index);
    status = ucs_topo_sys_device_set_name(*sys_dev_p, device_name,
                                          sys_device_priority);
    if (status != UCS_OK) {
        ucs_warn("failed to set name for index %d: %s", index,
                 ucs_status_string(status));
    }

    status = ucs_topo_sys_device_enable_aux_path(*sys_dev_p);
    if (status != UCS_OK) {
        ucs_debug("no aux path for %s: %s", device_name,
                  ucs_status_string(status));
    }

    ucs_debug("registered %s (sys_dev %d)", device_name, *sys_dev_p);

    return;

err:
    *sys_dev_p = UCS_SYS_DEVICE_ID_UNKNOWN;
}

static int uct_gaudi_base_open_minor(int id)
{
    char buf[64];
    int fd;
    ucs_snprintf_safe(buf, sizeof(buf), HLTHUNK_DEV_NAME_CONTROL, id);
    fd = open(buf, O_RDWR | O_CLOEXEC, 0);
    return (fd >= 0) ? fd : -errno;
}

/* device discovery - enumerate all gaudi devices and register with topology */
ucs_status_t uct_gaudi_base_discover_devices(void)
{
    static pthread_mutex_t discovery_mutex = PTHREAD_MUTEX_INITIALIZER;
    static uint32_t discovery_done         = 0;

    ucs_status_t status = UCS_OK;
    ucs_sys_device_t sys_dev;

    int device_count       = 0;
    int discovered_devices = 0;
    int i, fd;

    /* check if already discovered - use atomic load for memory ordering */
    if (ucs_atomic_fadd32(&discovery_done, 0)) {
        return UCS_OK;
    }

    pthread_mutex_lock(&discovery_mutex);

    /* double-check after acquiring lock */
    if (ucs_atomic_fadd32(&discovery_done, 0)) {
        goto out;
    }

    ucs_debug("starting gaudi device discovery");

    /* we do not know what minor is in use, so try them all. */
    for (i = 0; i < HLTHUNK_MAX_MINOR; i++) {
        /* open the control device instead of the actual hardware device, */
        /* because the real device node may be busy or in use by another process. */
        fd = uct_gaudi_base_open_minor(i);
        if (fd < 0) {
            continue;
        }

        uct_gaudi_base_configure_sys_device_from_fd(fd, discovered_devices,
                                                    &sys_dev);
        close(fd);

        if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
            discovered_devices++;
        }
    }

    /* extra measure: compare with reported count */
    device_count = hlthunk_get_device_count(HLTHUNK_DEVICE_DONT_CARE);
    if (device_count >= 0 && device_count != discovered_devices) {
        ucs_warn("gaudi discovery mismatch: discovered=%d, driver=%d",
                 discovered_devices, device_count);
    }

    if (discovered_devices > 0) {
        ucs_debug("discovered %d gaudi devices", discovered_devices);
        status = UCS_OK;

        ucs_atomic_add32(&discovery_done, 1);
    } else {
        ucs_debug("no gaudi devices found");
        status = UCS_ERR_NO_DEVICE;
    }

out:
    pthread_mutex_unlock(&discovery_mutex);
    return status;
}

UCS_MODULE_INIT()
{
    return UCS_OK;
}
