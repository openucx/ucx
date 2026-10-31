/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_base.h"

#include <ucs/debug/log.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/sys/module.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/topo/base/topo.h>

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define UCT_MUSA_PCI_BUS_ID_LEN 32

#define UCT_MUSA_DMABUF_HANDLE_TYPE MU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD

static pthread_once_t uct_musa_base_once = PTHREAD_ONCE_INIT;
static ucs_status_t  uct_musa_base_init_status;
static unsigned      uct_musa_base_num_devices;

static const char *uct_musa_error_string(musaError_t error)
{
    const char *error_string = musaGetErrorString(error);
    return error_string != NULL ? error_string : "unknown MUSA error";
}

ucs_status_t uct_musa_base_status(musaError_t error, const char *operation)
{
    if (error == musaSuccess) {
        return UCS_OK;
    }

    ucs_debug("%s failed: %s", operation, uct_musa_error_string(error));
    if (error == musaErrorInvalidValue) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_ERR_IO_ERROR;
}

static void uct_musa_base_init_once(void)
{
    int count = 0;

    uct_musa_base_init_status =
            uct_musa_base_status(musaGetDeviceCount(&count),
                                 "musaGetDeviceCount");
    if (uct_musa_base_init_status != UCS_OK) {
        return;
    }

    if (count <= 0) {
        uct_musa_base_init_status = UCS_ERR_NO_DEVICE;
        return;
    }

    uct_musa_base_num_devices = (unsigned)count;
}

ucs_status_t uct_musa_base_get_device_count(unsigned *count_p)
{
    pthread_once(&uct_musa_base_once, uct_musa_base_init_once);
    if (uct_musa_base_init_status != UCS_OK) {
        return uct_musa_base_init_status;
    }

    *count_p = uct_musa_base_num_devices;
    return UCS_OK;
}

ucs_status_t
uct_musa_base_query_md_resources(uct_component_h component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    unsigned count;
    ucs_status_t status;

    status = uct_musa_base_get_device_count(&count);
    if (status != UCS_OK) {
        ucs_debug("MUSA runtime is unavailable: %s", ucs_status_string(status));
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    if (count == 0) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

static ucs_status_t
uct_musa_base_get_sys_dev(unsigned device, ucs_sys_device_t *sys_dev_p)
{
    ucs_sys_bus_id_t bus_id;
    ucs_sys_device_t sys_dev;
    char pci_bus_id[UCT_MUSA_PCI_BUS_ID_LEN];
    ucs_status_t status;
    musaError_t err;

    /* MUSA runtime exposes the PCI location only as a
     * "domain:bus:slot.function" string (no per-attribute query). */
    err = musaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), (int)device);
    if (err != musaSuccess) {
        ucs_debug("musaDeviceGetPCIBusId(device=%u) failed: %s", device,
                  musaGetErrorString(err));
        return UCS_ERR_INVALID_PARAM;
    }

    if (!uct_musa_parse_pci_bus_id(pci_bus_id, &bus_id)) {
        ucs_debug("failed to parse MUSA PCI bus id \"%s\"", pci_bus_id);
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_set_class(sys_dev, UCS_TOPO_DEVICE_CLASS_ACC);
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_topo_sys_device_enable_aux_path(sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    *sys_dev_p = sys_dev;
    return UCS_OK;
}

ucs_status_t
uct_musa_base_query_devices(uct_md_h md,
                            uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p)
{
    uct_tl_device_resource_t *devices;
    unsigned count;
    unsigned i;
    ucs_status_t status;

    status = uct_musa_base_get_device_count(&count);
    if (status != UCS_OK) {
        return status;
    }

    devices = ucs_calloc(count, sizeof(*devices), "musa tl devices");
    if (devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < count; ++i) {
        devices[i].type     = UCT_DEVICE_TYPE_ACC;
        if (uct_musa_base_get_sys_dev(i, &devices[i].sys_device) != UCS_OK) {
            devices[i].sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
        }
        ucs_snprintf_zero(devices[i].name, sizeof(devices[i].name), "musa%u",
                          i);
    }

    *tl_devices_p     = devices;
    *num_tl_devices_p = count;
    return UCS_OK;
}

ucs_status_t
uct_musa_base_get_current_device(int *device_p)
{
    return uct_musa_base_status(musaGetDevice(device_p), "musaGetDevice");
}

ucs_status_t uct_musa_base_set_device(int device)
{
    return uct_musa_base_status(musaSetDevice(device), "musaSetDevice");
}

void uct_musa_base_restore_device(int device, int valid)
{
    if (valid && (musaSetDevice(device) != musaSuccess)) {
        ucs_warn("failed to restore MUSA device %d", device);
    }
}

static ucs_status_t
uct_musa_base_query_pointer_attributes(const void *address,
                                       uct_musa_pointer_info_t *info)
{
    struct musaPointerAttributes attributes;
    musaError_t error;

    memset(&attributes, 0, sizeof(attributes));
    error = musaPointerGetAttributes(&attributes, address);
    if (error == musaErrorInvalidValue) {
        info->kind = UCT_MUSA_POINTER_HOST;
        info->device = -1;
        return UCS_OK;
    }

    if (error != musaSuccess) {
        ucs_debug("musaPointerGetAttributes(%p) failed: %s", address,
                  uct_musa_error_string(error));
        return UCS_ERR_IO_ERROR;
    }

    if (attributes.type == musaMemoryTypeDevice) {
        info->kind   = UCT_MUSA_POINTER_DEVICE;
        info->device = attributes.device;
    } else if (attributes.type == musaMemoryTypeManaged) {
        info->kind   = UCT_MUSA_POINTER_MANAGED;
        info->device = attributes.device;
    } else {
        info->kind   = UCT_MUSA_POINTER_HOST;
        info->device = -1;
    }

    return UCS_OK;
}

ucs_status_t
uct_musa_base_query_pointer(const void *address, size_t length,
                            uct_musa_pointer_info_t *info)
{
    MUdeviceptr allocation_base;
    size_t allocation_length;
    int current_device = -1;
    int have_current_device;
    ucs_status_t status;

    if ((address == NULL) || (length == 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    memset(info, 0, sizeof(*info));
    status = uct_musa_base_query_pointer_attributes(address, info);
    if (status != UCS_OK) {
        return status;
    }

    info->base_address = (void *)address;
    info->alloc_length = length;
    if ((info->kind == UCT_MUSA_POINTER_HOST) ||
        (info->kind == UCT_MUSA_POINTER_MANAGED)) {
        return UCS_OK;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    if (info->device >= 0) {
        status = uct_musa_base_set_device(info->device);
        if (status != UCS_OK) {
            return status;
        }
    }

    allocation_base   = 0;
    allocation_length = 0;
    status = uct_musa_base_status(
            muMemGetAddressRange(&allocation_base, &allocation_length,
                                 (MUdeviceptr)address),
            "muMemGetAddressRange");
    uct_musa_base_restore_device(current_device, have_current_device);
    if (status != UCS_OK) {
        return status;
    }

    info->base_address = (void *)(uintptr_t)allocation_base;
    info->alloc_length = allocation_length;
    if (!uct_musa_range_is_valid((uintptr_t)info->base_address,
                                  info->alloc_length, (uintptr_t)address,
                                  length)) {
        ucs_debug("MUSA allocation range does not contain %p len %zu",
                  address, length);
        return UCS_ERR_INVALID_ADDR;
    }

    return UCS_OK;
}

int uct_musa_base_has_dmabuf(void)
{
#ifdef HAVE_MUSA_DMABUF
    return 1;
#else
    return 0;
#endif
}

ucs_status_t
uct_musa_base_export_dmabuf(const void *address, size_t length, int *fd_p,
                            size_t *offset_p)
{
#ifdef HAVE_MUSA_DMABUF
    uct_musa_pointer_info_t info;
    int current_device = -1;
    int have_current_device;
    int fd;
    ucs_status_t status;
    MUresult result;
#endif

    *fd_p     = UCT_DMABUF_FD_INVALID;
    *offset_p = 0;

#ifndef HAVE_MUSA_DMABUF
    return UCS_ERR_UNSUPPORTED;
#else
    if (!uct_musa_base_has_dmabuf()) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_musa_base_query_pointer(address, length, &info);
    if (status != UCS_OK) {
        return status;
    }
    if (info.kind != UCT_MUSA_POINTER_DEVICE) {
        return UCS_ERR_INVALID_ADDR;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    status = uct_musa_base_set_device(info.device);
    if (status != UCS_OK) {
        return status;
    }

    fd = UCT_DMABUF_FD_INVALID;
    result = muMemGetHandleForAddressRange(
            &fd, (MUdeviceptr)info.base_address, info.alloc_length,
            UCT_MUSA_DMABUF_HANDLE_TYPE, 0);
    uct_musa_base_restore_device(current_device, have_current_device);
    if (result != MUSA_SUCCESS) {
        ucs_debug("muMemGetHandleForAddressRange failed: 0x%x", result);
        return UCS_ERR_IO_ERROR;
    }

    *fd_p = fd;
    *offset_p = UCS_PTR_BYTE_DIFF(info.base_address, address);
    return UCS_OK;
#endif
}

ucs_status_t
uct_musa_base_mem_query(const void *address, size_t length, int have_dmabuf,
                        uct_md_mem_attr_v2_t *mem_attr_p)
{
    uct_musa_pointer_info_t info;
    ucs_sys_device_t sys_dev;
    ucs_memory_type_t mem_type;
    ucs_status_t status;
    int dmabuf_fd;
    int have_dmabuf_fd;
    size_t dmabuf_offset;

    status = uct_musa_base_query_pointer(address, length, &info);
    if (status != UCS_OK) {
        return status;
    }
    if (info.kind == UCT_MUSA_POINTER_MANAGED) {
        return UCS_ERR_UNSUPPORTED;
    }
    if (info.kind != UCT_MUSA_POINTER_DEVICE) {
        return UCS_ERR_INVALID_ADDR;
    }

    mem_type = UCS_MEMORY_TYPE_MUSA;
    sys_dev  = UCS_SYS_DEVICE_ID_UNKNOWN;
    if (uct_musa_base_get_sys_dev(info.device, &sys_dev) != UCS_OK) {
        ucs_debug("failed to map MUSA device %d to a system device",
                  info.device);
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_type;
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = sys_dev;
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_BASE_ADDRESS) {
        mem_attr_p->base_address = info.base_address;
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_ALLOC_LENGTH) {
        mem_attr_p->alloc_length = info.alloc_length;
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_FLAGS) {
        mem_attr_p->mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    }

    dmabuf_fd     = UCT_DMABUF_FD_INVALID;
    dmabuf_offset = 0;
    if (have_dmabuf &&
        ((mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) ||
         (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET))) {
        status = uct_musa_base_export_dmabuf(address, length, &dmabuf_fd,
                                             &dmabuf_offset);
        if (status != UCS_OK) {
            dmabuf_fd = UCT_DMABUF_FD_INVALID;
        }
    }

    have_dmabuf_fd = dmabuf_fd != UCT_DMABUF_FD_INVALID;
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET) {
        mem_attr_p->dmabuf_offset = have_dmabuf_fd ? dmabuf_offset : 0;
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) {
        mem_attr_p->dmabuf_fd = dmabuf_fd;
    } else if (have_dmabuf_fd) {
        ucs_close_fd(&dmabuf_fd);
    }

    ucs_memtype_cache_update(info.base_address, info.alloc_length, mem_type,
                             sys_dev, UCS_MEM_FLAG_REGISTRABLE);
    return UCS_OK;
}

ucs_status_t
uct_musa_base_detect_memory_type(uct_md_h md, const void *address,
                                 size_t length,
                                 ucs_memory_type_t *mem_type_p)
{
    uct_musa_pointer_info_t info;
    ucs_status_t status;

    *mem_type_p = UCS_MEMORY_TYPE_HOST;
    status = uct_musa_base_query_pointer(address, length, &info);
    if (status != UCS_OK) {
        return status;
    }

    if (info.kind == UCT_MUSA_POINTER_MANAGED) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (info.kind == UCT_MUSA_POINTER_DEVICE) {
        *mem_type_p = UCS_MEMORY_TYPE_MUSA;
    }
    return UCS_OK;
}

ucs_status_t
uct_musa_base_query_copy_pointer(const void *address, size_t length,
                                 uct_musa_pointer_info_t *info)
{
    ucs_memory_info_t mem_info;
    ucs_status_t status;

    if ((address == NULL) || (length == 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucs_memtype_cache_lookup(address, length, &mem_info);
    if (status == UCS_ERR_NO_ELEM) {
        memset(info, 0, sizeof(*info));
        info->kind         = UCT_MUSA_POINTER_HOST;
        info->device       = -1;
        info->base_address = (void *)address;
        info->alloc_length = length;
        return UCS_OK;
    }

    if ((status == UCS_OK) && (mem_info.type == UCS_MEMORY_TYPE_MUSA)) {
        status = uct_musa_base_query_pointer_attributes(address, info);
        if (status == UCS_OK) {
            info->base_address = (void *)address;
            info->alloc_length = length;
        }
        return status;
    }

    if ((status == UCS_OK) &&
        (mem_info.type != UCS_MEMORY_TYPE_UNKNOWN)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return uct_musa_base_query_pointer(address, length, info);
}

static ucs_status_t
uct_musa_base_copy_prepare(const uct_musa_pointer_info_t *src_info,
                           const uct_musa_pointer_info_t *dst_info,
                           int *device_p, enum musaMemcpyKind *kind_p)
{
    if ((src_info->kind == UCT_MUSA_POINTER_MANAGED) ||
        (dst_info->kind == UCT_MUSA_POINTER_MANAGED)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if ((src_info->kind == UCT_MUSA_POINTER_HOST) &&
        (dst_info->kind == UCT_MUSA_POINTER_DEVICE)) {
        *kind_p = musaMemcpyHostToDevice;
    } else if ((src_info->kind == UCT_MUSA_POINTER_DEVICE) &&
               (dst_info->kind == UCT_MUSA_POINTER_HOST)) {
        *kind_p = musaMemcpyDeviceToHost;
    } else {
        *kind_p = musaMemcpyDeviceToDevice;
    }

    if ((src_info->kind == UCT_MUSA_POINTER_DEVICE) &&
        (src_info->device >= 0)) {
        *device_p = src_info->device;
    } else if ((dst_info->kind == UCT_MUSA_POINTER_DEVICE) &&
               (dst_info->device >= 0)) {
        *device_p = dst_info->device;
    }

    return (*device_p >= 0) ? UCS_OK : UCS_ERR_NO_DEVICE;
}

static ucs_status_t
uct_musa_base_copy_sync_with_info(const void *src, void *dst, size_t length,
                                  int device,
                                  const uct_musa_pointer_info_t *src_info,
                                  const uct_musa_pointer_info_t *dst_info)
{
    enum musaMemcpyKind kind;
    int current_device = -1;
    int have_current_device;
    int switched_device;
    ucs_status_t status;

    if ((src_info->kind == UCT_MUSA_POINTER_HOST) &&
        (dst_info->kind == UCT_MUSA_POINTER_HOST)) {
        memcpy(dst, src, length);
        return UCS_OK;
    }

    status = uct_musa_base_copy_prepare(src_info, dst_info, &device, &kind);
    if (status != UCS_OK) {
        return status;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    switched_device = !have_current_device || (current_device != device);
    if (switched_device) {
        status = uct_musa_base_set_device(device);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = uct_musa_base_status(musaMemcpy(dst, src, length, kind),
                                  "musaMemcpy");
    if (switched_device) {
        uct_musa_base_restore_device(current_device, have_current_device);
    }
    return status;
}

ucs_status_t uct_musa_base_copy(const void *src, void *dst, size_t length,
                                int device)
{
    uct_musa_pointer_info_t src_info;
    uct_musa_pointer_info_t dst_info;
    ucs_status_t status;

    if (length == 0) {
        return UCS_OK;
    }

    status = uct_musa_base_query_copy_pointer(src, length, &src_info);
    if (status != UCS_OK) {
        return status;
    }
    status = uct_musa_base_query_copy_pointer(dst, length, &dst_info);
    if (status != UCS_OK) {
        return status;
    }

    return uct_musa_base_copy_sync_with_info(src, dst, length, device,
                                             &src_info, &dst_info);
}

ucs_status_t
uct_musa_base_copy_async(const void *src, void *dst, size_t length, int device,
                         musaStream_t stream,
                         const uct_musa_pointer_info_t *src_info,
                         const uct_musa_pointer_info_t *dst_info)
{
    enum musaMemcpyKind kind;
    int current_device = -1;
    int have_current_device;
    int switched_device;
    ucs_status_t status;

    if (length == 0) {
        return UCS_OK;
    }

    if ((src_info->kind == UCT_MUSA_POINTER_HOST) &&
        (dst_info->kind == UCT_MUSA_POINTER_HOST)) {
        memcpy(dst, src, length);
        return UCS_OK;
    }

    status = uct_musa_base_copy_prepare(src_info, dst_info, &device, &kind);
    if (status != UCS_OK) {
        return status;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    switched_device = !have_current_device || (current_device != device);
    if (switched_device) {
        status = uct_musa_base_set_device(device);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = uct_musa_base_status(
            musaMemcpyAsync(dst, src, length, kind, stream),
            "musaMemcpyAsync");
    if (switched_device) {
        uct_musa_base_restore_device(current_device, have_current_device);
    }
    return status;
}

UCS_MODULE_INIT()
{
    return UCS_OK;
}
