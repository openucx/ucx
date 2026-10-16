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

#include <dlfcn.h>
#include <pthread.h>
#include <string.h>

#define UCT_MUSA_DMABUF_HANDLE_TYPE MU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD

typedef MUresult (*uct_musa_dmabuf_export_func_t)(void *handle,
                                                   MUdeviceptr base,
                                                   size_t length,
                                                   int handle_type,
                                                   unsigned long long flags);

static pthread_once_t uct_musa_base_once = PTHREAD_ONCE_INIT;
static ucs_status_t  uct_musa_base_status;
static unsigned      uct_musa_base_num_devices;

static pthread_once_t uct_musa_dmabuf_once = PTHREAD_ONCE_INIT;
static uct_musa_dmabuf_export_func_t uct_musa_dmabuf_export_func;

static const char *uct_musa_error_string(musaError_t error)
{
    const char *error_string = musaGetErrorString(error);
    return error_string != NULL ? error_string : "unknown MUSA error";
}

static ucs_status_t uct_musa_status(musaError_t error, const char *operation)
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

    uct_musa_base_status = uct_musa_status(musaGetDeviceCount(&count),
                                           "musaGetDeviceCount");
    if (uct_musa_base_status != UCS_OK) {
        return;
    }

    if (count <= 0) {
        uct_musa_base_status = UCS_ERR_NO_DEVICE;
        return;
    }

    uct_musa_base_num_devices = (unsigned)count;
}

ucs_status_t uct_musa_base_get_device_count(unsigned *count_p)
{
    pthread_once(&uct_musa_base_once, uct_musa_base_init_once);
    if (uct_musa_base_status != UCS_OK) {
        return uct_musa_base_status;
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

    (void)count;
    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
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

    (void)md;
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
        devices[i].sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
        ucs_snprintf_zero(devices[i].name, sizeof(devices[i].name), "musa%u",
                          i);
    }

    *tl_devices_p     = devices;
    *num_tl_devices_p = count;
    return UCS_OK;
}

static ucs_status_t
uct_musa_base_get_current_device(int *device_p)
{
    return uct_musa_status(musaGetDevice(device_p), "musaGetDevice");
}

static ucs_status_t uct_musa_base_set_device(int device)
{
    return uct_musa_status(musaSetDevice(device), "musaSetDevice");
}

static void uct_musa_base_restore_device(int device, int valid)
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
    int current_device;
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
    if (info->kind == UCT_MUSA_POINTER_HOST) {
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
    status = uct_musa_status(
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

static void uct_musa_base_dmabuf_init_once(void)
{
    uct_musa_dmabuf_export_func =
            (uct_musa_dmabuf_export_func_t)dlsym(
                    RTLD_DEFAULT, "muMemGetHandleForAddressRange");
}

int uct_musa_base_has_dmabuf(void)
{
    pthread_once(&uct_musa_dmabuf_once, uct_musa_base_dmabuf_init_once);
    return uct_musa_dmabuf_export_func != NULL;
}

ucs_status_t
uct_musa_base_export_dmabuf(const void *address, size_t length, int *fd_p,
                            size_t *offset_p)
{
    uct_musa_pointer_info_t info;
    int current_device;
    int have_current_device;
    int fd;
    ucs_status_t status;
    MUresult result;

    *fd_p     = UCT_DMABUF_FD_INVALID;
    *offset_p = 0;

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
    result = uct_musa_dmabuf_export_func(
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
}

ucs_status_t
uct_musa_base_mem_query(const void *address, size_t length, int have_dmabuf,
                        uct_md_mem_attr_v2_t *mem_attr_p)
{
    uct_musa_pointer_info_t info;
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
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_type;
    }
    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
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
                             UCS_SYS_DEVICE_ID_UNKNOWN,
                             UCS_MEM_FLAG_REGISTRABLE);
    return UCS_OK;
}

ucs_status_t
uct_musa_base_detect_memory_type(uct_md_h md, const void *address,
                                 size_t length,
                                 ucs_memory_type_t *mem_type_p)
{
    uct_musa_pointer_info_t info;
    ucs_status_t status;

    (void)md;
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

ucs_status_t uct_musa_base_copy(const void *src, void *dst, size_t length,
                                int device)
{
    uct_musa_pointer_info_t src_info;
    uct_musa_pointer_info_t dst_info;
    musaMemcpyKind kind;
    int current_device;
    int have_current_device;
    ucs_status_t status;

    if (length == 0) {
        return UCS_OK;
    }

    status = uct_musa_base_query_pointer(src, length, &src_info);
    if (status != UCS_OK) {
        return status;
    }
    status = uct_musa_base_query_pointer(dst, length, &dst_info);
    if (status != UCS_OK) {
        return status;
    }

    if ((src_info.kind == UCT_MUSA_POINTER_MANAGED) ||
        (dst_info.kind == UCT_MUSA_POINTER_MANAGED)) {
        return UCS_ERR_UNSUPPORTED;
    }
    if ((src_info.kind == UCT_MUSA_POINTER_DEVICE) &&
        (dst_info.kind == UCT_MUSA_POINTER_DEVICE) &&
        (src_info.device != dst_info.device)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if ((src_info.kind == UCT_MUSA_POINTER_HOST) &&
        (dst_info.kind == UCT_MUSA_POINTER_HOST)) {
        memcpy(dst, src, length);
        return UCS_OK;
    }

    if ((src_info.kind == UCT_MUSA_POINTER_HOST) &&
        (dst_info.kind == UCT_MUSA_POINTER_DEVICE)) {
        kind = musaMemcpyHostToDevice;
    } else if ((src_info.kind == UCT_MUSA_POINTER_DEVICE) &&
               (dst_info.kind == UCT_MUSA_POINTER_HOST)) {
        kind = musaMemcpyDeviceToHost;
    } else {
        kind = musaMemcpyDeviceToDevice;
    }

    if (device < 0) {
        device = (dst_info.kind == UCT_MUSA_POINTER_DEVICE) ?
                 dst_info.device : src_info.device;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    status = uct_musa_base_set_device(device);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_musa_status(musaMemcpy(dst, src, length, kind), "musaMemcpy");
    uct_musa_base_restore_device(current_device, have_current_device);
    return status;
}

UCS_MODULE_INIT()
{
    UCS_MODULE_FRAMEWORK_DECLARE(uct_musa);
    UCS_MODULE_FRAMEWORK_LOAD(uct_musa, 0);
    return UCS_OK;
}
