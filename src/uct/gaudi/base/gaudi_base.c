/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gaudi_base.h"
#include <uct/gaudi/gaudi_gdr/gaudi_gdr_md.h>
#include <ucs/sys/module.h>
#include <ucs/memory/numa.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/topo/base/topo.h>
#include <pthread.h>

#include <inttypes.h>
#include <fcntl.h>
#include <hlthunk.h>
#include <synapse_api.h>

int uct_gaudi_base_get_fd(int device_id) {
    synDeviceInfo deviceInfo;

    if (synDeviceGetInfo(-1, &deviceInfo) != synSuccess) {
       return hlthunk_open_by_module_id(device_id);
    }

    return deviceInfo.fd;
}

ucs_status_t uct_gaudi_base_get_sysdev(int fd, ucs_sys_device_t* sys_dev) {
    ucs_status_t status;
    char pci_bus_id[13];
    int rc = hlthunk_get_pci_bus_id_from_fd(fd, pci_bus_id, sizeof(pci_bus_id));
    if (rc != 0) {
        ucs_error("Failed to get pci_bus_id via hlthunk_get_pci_bus_id_from_fd");
        return UCS_ERR_IO_ERROR;
    }

    status = ucs_topo_find_device_by_bdf_name(pci_bus_id, sys_dev);
    if (status != UCS_OK) {
       ucs_error("Failed to get sys device");
       return status;
    }

    return UCS_OK;
}

ucs_status_t uct_gaudi_base_get_info(int fd, uint64_t *device_base_allocated_address, uint64_t *device_base_address,
                                uint64_t *totalSize, int *dmabuf_fd)
{
    uint64_t addr, hbm_pool_start, size, offset;
    scal_handle_t scal_handle;
    scal_pool_handle_t      scal_pool_handle;
    scal_memory_pool_infoV2 scal_mem_pool_info;

    int rc = scal_get_handle_from_fd(fd, &scal_handle);
    if (rc != SCAL_SUCCESS) {
        /*
         * If rc value equal SCAL_SUCCESS, it mean that it use synDeviceAcquireByModuleId to open Gaudi device.
         * Otherwise, the device is opened via hlthunk_open_by_module_id function.
         */
        rc = scal_init(fd, "", &scal_handle, NULL);
    }
    if (rc != SCAL_SUCCESS) {
	ucs_error("Failed to get scal handle");
        return UCS_ERR_IO_ERROR;
    }
    rc = scal_get_pool_handle_by_name(scal_handle, "global_hbm", &scal_pool_handle);
    if (rc != SCAL_SUCCESS) {
	ucs_error("Failed to get scal pool");
        return UCS_ERR_INVALID_ADDR;
    }
    rc = scal_pool_get_infoV2(scal_pool_handle, &scal_mem_pool_info);
    if (rc != SCAL_SUCCESS) {
	ucs_error("Failed to get scal pool info");
        return UCS_ERR_INVALID_ADDR;
    }
    addr = scal_mem_pool_info.device_base_allocated_address;
    hbm_pool_start = scal_mem_pool_info.device_base_address;
    size = scal_mem_pool_info.totalSize;
    offset = hbm_pool_start - addr;
    *dmabuf_fd = hlthunk_device_mapped_memory_export_dmabuf_fd(fd, addr, size, offset, (O_RDWR | O_CLOEXEC));
    if (*dmabuf_fd < 0) {
	ucs_error("Failed to get dmabuf fd");
        return UCS_ERR_INVALID_ADDR;
    }

    *device_base_allocated_address = addr;
    *device_base_address = hbm_pool_start;
    *totalSize = size;
    return UCS_OK;
}

ucs_status_t uct_gaudi_base_query_devices(uct_md_h md,
                                         uct_tl_device_resource_t **tl_devices_p,
                                         unsigned *num_tl_devices_p)
{
    uct_gaudi_md_t *gaudi_md = ucs_derived_of(md, uct_gaudi_md_t);
    ucs_sys_device_t sys_dev;

    ucs_status_t status = uct_gaudi_base_get_sysdev(gaudi_md->fd, &sys_dev);
    if (status != UCS_OK) {
        return status;
    }
    return uct_single_device_resource(md, md->component->name,
                                      UCT_DEVICE_TYPE_ACC,
                                      sys_dev, tl_devices_p,
                                      num_tl_devices_p);
}

UCS_MODULE_INIT()
{
    return UCS_OK;
}
