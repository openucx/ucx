/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <musa.h>
#include <musa_runtime_api.h>

#include <ucm/event/event.h>
#include <ucm/mmap/mmap.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucm/util/sys.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <pthread.h>
#include <string.h>
#include <sys/mman.h>

static musaError_t ucm_musaMalloc(void **address, size_t size);
static musaError_t ucm_musaFree(void *address);
static musaError_t ucm_musaMallocManaged(void **address, size_t size,
                                         unsigned flags);
static musaError_t ucm_musaMallocPitch(void **address, size_t *pitch,
                                       size_t width, size_t height);
static musaError_t ucm_musaMallocAsync(void **address, size_t size,
                                       musaStream_t stream);
static musaError_t
ucm_musaMallocFromPoolAsync(void **address, size_t size,
                            musaMemPool_t pool, musaStream_t stream);
static musaError_t ucm_musaMallocHost(void **address, size_t size);
static musaError_t ucm_musaFreeHost(void *address);
static musaError_t ucm_musaFreeAsync(void *address, musaStream_t stream);
static MUresult ucm_muMemAlloc(MUdeviceptr *address, size_t size);
static MUresult ucm_muMemAlloc_v2(MUdeviceptr *address, size_t size);
static MUresult ucm_muMemAllocManaged(MUdeviceptr *address, size_t size,
                                      unsigned flags);
static MUresult ucm_muMemAllocPitch(MUdeviceptr *address, size_t *pitch,
                                    size_t width, size_t height,
                                    unsigned element_size);
static MUresult ucm_muMemAllocPitch_v2(MUdeviceptr *address, size_t *pitch,
                                       size_t width, size_t height,
                                       unsigned element_size);
static MUresult ucm_muMemFree(MUdeviceptr address);
static MUresult ucm_muMemFree_v2(MUdeviceptr address);

UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMalloc, musaError_t, musaErrorUnknown,
                              void **, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaFree, musaError_t, musaErrorUnknown, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMallocManaged, musaError_t, musaErrorUnknown,
                              void **, size_t, unsigned)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMallocPitch, musaError_t, musaErrorUnknown,
                              void **, size_t *, size_t, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMallocAsync, musaError_t, musaErrorUnknown,
                              void **, size_t, musaStream_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMallocFromPoolAsync, musaError_t,
                              musaErrorUnknown, void **, size_t, musaMemPool_t,
                              musaStream_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaMallocHost, musaError_t, musaErrorUnknown,
                              void **, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaFreeHost, musaError_t, musaErrorUnknown,
                              void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(musaFreeAsync, musaError_t, musaErrorUnknown,
                              void *, musaStream_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemAlloc, MUresult, -1, MUdeviceptr *, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemAlloc_v2, MUresult, -1, MUdeviceptr *,
                              size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemAllocManaged, MUresult, -1, MUdeviceptr *,
                              size_t, unsigned)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemAllocPitch, MUresult, -1, MUdeviceptr *,
                              size_t *, size_t, size_t, unsigned)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemAllocPitch_v2, MUresult, -1,
                              MUdeviceptr *, size_t *, size_t, size_t,
                              unsigned)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemFree, MUresult, -1, MUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_FUNC(muMemFree_v2, MUresult, -1, MUdeviceptr)

static void
ucm_dispatch_musa_mem_type_event(ucm_event_type_t event_type, void *address,
                                 size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    if (address == NULL) {
        return;
    }

    event.mem_type.address   = address;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = mem_type;
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    ucm_event_dispatch(event_type, &event);
}

static size_t ucm_musa_mem_size(void *address)
{
    MUdeviceptr base;
    size_t length;

    if ((address == NULL) ||
        (muMemGetAddressRange(&base, &length, (MUdeviceptr)address) !=
         MUSA_SUCCESS)) {
        return 1;
    }

    return length;
}

static musaError_t ucm_musaMalloc(void **address, size_t size)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMalloc(address, size);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         size, UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static musaError_t ucm_musaFree(void *address)
{
    musaError_t status;
    size_t length;

    ucm_event_enter();
    length = ucm_musa_mem_size(address);
    ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_FREE, address, length,
                                     UCS_MEMORY_TYPE_MUSA);
    status = ucm_orig_musaFree(address);
    ucm_event_leave();
    return status;
}

static musaError_t
ucm_musaMallocManaged(void **address, size_t size, unsigned flags)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMallocManaged(address, size, flags);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         size, UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static musaError_t
ucm_musaMallocPitch(void **address, size_t *pitch, size_t width, size_t height)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMallocPitch(address, pitch, width, height);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         (*pitch) * height,
                                         UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static musaError_t
ucm_musaMallocAsync(void **address, size_t size, musaStream_t stream)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMallocAsync(address, size, stream);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         size, UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static musaError_t
ucm_musaMallocFromPoolAsync(void **address, size_t size, musaMemPool_t pool,
                            musaStream_t stream)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMallocFromPoolAsync(address, size, pool, stream);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         size, UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static musaError_t ucm_musaMallocHost(void **address, size_t size)
{
    musaError_t status;

    ucm_event_enter();
    status = ucm_orig_musaMallocHost(address, size);
    if ((status == musaSuccess) && (address != NULL) && (*address != NULL)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC, *address,
                                         size, UCS_MEMORY_TYPE_HOST);
    }
    ucm_event_leave();
    return status;
}

static musaError_t ucm_musaFreeHost(void *address)
{
    musaError_t status;

    ucm_event_enter();
    ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_FREE, address, 1,
                                     UCS_MEMORY_TYPE_HOST);
    status = ucm_orig_musaFreeHost(address);
    ucm_event_leave();
    return status;
}

static musaError_t
ucm_musaFreeAsync(void *address, musaStream_t stream)
{
    musaError_t status;
    size_t length;

    ucm_event_enter();
    length = ucm_musa_mem_size(address);
    ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_FREE, address, length,
                                     UCS_MEMORY_TYPE_MUSA);
    status = ucm_orig_musaFreeAsync(address, stream);
    ucm_event_leave();
    return status;
}

static MUresult ucm_muMemAlloc(MUdeviceptr *address, size_t size)
{
    MUresult status;

    ucm_event_enter();
    status = ucm_orig_muMemAlloc(address, size);
    if ((status == MUSA_SUCCESS) && (address != NULL) && (*address != 0)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC,
                                         (void *)(uintptr_t)*address, size,
                                         UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static MUresult ucm_muMemAlloc_v2(MUdeviceptr *address, size_t size)
{
    MUresult status;

    ucm_event_enter();
    status = ucm_orig_muMemAlloc_v2(address, size);
    if ((status == MUSA_SUCCESS) && (address != NULL) && (*address != 0)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC,
                                         (void *)(uintptr_t)*address, size,
                                         UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static MUresult
ucm_muMemAllocManaged(MUdeviceptr *address, size_t size, unsigned flags)
{
    MUresult status;

    ucm_event_enter();
    status = ucm_orig_muMemAllocManaged(address, size, flags);
    if ((status == MUSA_SUCCESS) && (address != NULL) && (*address != 0)) {
        ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_ALLOC,
                                         (void *)(uintptr_t)*address, size,
                                         UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static MUresult
ucm_muMemAllocPitch(MUdeviceptr *address, size_t *pitch, size_t width,
                    size_t height, unsigned element_size)
{
    MUresult status;

    ucm_event_enter();
    status = ucm_orig_muMemAllocPitch(address, pitch, width, height,
                                      element_size);
    if ((status == MUSA_SUCCESS) && (address != NULL) && (*address != 0)) {
        ucm_dispatch_musa_mem_type_event(
                UCM_EVENT_MEM_TYPE_ALLOC, (void *)(uintptr_t)*address,
                (*pitch) * height, UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static MUresult
ucm_muMemAllocPitch_v2(MUdeviceptr *address, size_t *pitch, size_t width,
                       size_t height, unsigned element_size)
{
    MUresult status;

    ucm_event_enter();
    status = ucm_orig_muMemAllocPitch_v2(address, pitch, width, height,
                                         element_size);
    if ((status == MUSA_SUCCESS) && (address != NULL) && (*address != 0)) {
        ucm_dispatch_musa_mem_type_event(
                UCM_EVENT_MEM_TYPE_ALLOC, (void *)(uintptr_t)*address,
                (*pitch) * height, UCS_MEMORY_TYPE_LAST);
    }
    ucm_event_leave();
    return status;
}

static MUresult ucm_muMemFree(MUdeviceptr address)
{
    MUresult status;

    ucm_event_enter();
    ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_FREE,
                                     (void *)(uintptr_t)address,
                                     ucm_musa_mem_size(
                                             (void *)(uintptr_t)address),
                                     UCS_MEMORY_TYPE_MUSA);
    status = ucm_orig_muMemFree(address);
    ucm_event_leave();
    return status;
}

static MUresult ucm_muMemFree_v2(MUdeviceptr address)
{
    MUresult status;

    ucm_event_enter();
    ucm_dispatch_musa_mem_type_event(UCM_EVENT_MEM_TYPE_FREE,
                                     (void *)(uintptr_t)address,
                                     ucm_musa_mem_size(
                                             (void *)(uintptr_t)address),
                                     UCS_MEMORY_TYPE_MUSA);
    status = ucm_orig_muMemFree_v2(address);
    ucm_event_leave();
    return status;
}

static ucm_reloc_patch_t ucm_musa_patches[] = {
    {UCS_PP_MAKE_STRING(musaMalloc), ucm_override_musaMalloc},
    {UCS_PP_MAKE_STRING(musaFree), ucm_override_musaFree},
    {UCS_PP_MAKE_STRING(musaMallocManaged), ucm_override_musaMallocManaged},
    {UCS_PP_MAKE_STRING(musaMallocPitch), ucm_override_musaMallocPitch},
    {UCS_PP_MAKE_STRING(musaMallocAsync), ucm_override_musaMallocAsync},
    {UCS_PP_MAKE_STRING(musaMallocFromPoolAsync),
     ucm_override_musaMallocFromPoolAsync},
    {UCS_PP_MAKE_STRING(musaMallocHost), ucm_override_musaMallocHost},
    {UCS_PP_MAKE_STRING(musaFreeHost), ucm_override_musaFreeHost},
    {UCS_PP_MAKE_STRING(musaFreeAsync), ucm_override_musaFreeAsync},
    {UCS_PP_MAKE_STRING(muMemAlloc), ucm_override_muMemAlloc},
    {UCS_PP_MAKE_STRING(muMemAlloc_v2), ucm_override_muMemAlloc_v2},
    {UCS_PP_MAKE_STRING(muMemAllocManaged), ucm_override_muMemAllocManaged},
    {UCS_PP_MAKE_STRING(muMemAllocPitch), ucm_override_muMemAllocPitch},
    {UCS_PP_MAKE_STRING(muMemAllocPitch_v2), ucm_override_muMemAllocPitch_v2},
    {UCS_PP_MAKE_STRING(muMemFree), ucm_override_muMemFree},
    {UCS_PP_MAKE_STRING(muMemFree_v2), ucm_override_muMemFree_v2},
    {NULL, NULL}
};

static ucs_status_t ucm_musamem_install(int events)
{
    static int installed;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    void *func_ptr;
    ucs_status_t status = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        return UCS_OK;
    }

    if (ucm_mmap_hook_mode() == UCM_MMAP_HOOK_NONE) {
        return UCS_ERR_UNSUPPORTED;
    }

    pthread_mutex_lock(&install_mutex);
    if (installed) {
        goto out;
    }

    for (patch = ucm_musa_patches; patch->symbol != NULL; ++patch) {
        func_ptr = ucm_reloc_get_orig(patch->symbol, patch->value);
        if (func_ptr == NULL) {
            continue;
        }

        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install MUSA hook for '%s'", patch->symbol);
            goto out;
        }
    }

    installed = 1;
    ucm_info("MUSA memory hooks are ready");

out:
    pthread_mutex_unlock(&install_mutex);
    return status;
}

static int
ucm_musamem_is_device_region(int prot, const char *path)
{
    static const char render_path[] = "/dev/dri/renderD";
    static const char mtgpu_path[]  = "/dev/mtgpu";

    if (!(prot & (PROT_READ | PROT_WRITE | PROT_EXEC))) {
        return 1;
    }

    return (path != NULL) &&
           ((!strncmp(path, render_path, sizeof(render_path) - 1)) ||
            (!strncmp(path, mtgpu_path, sizeof(mtgpu_path) - 1)));
}

static int
ucm_musamem_scan_regions_cb(void *arg, void *addr, size_t length, int prot,
                            const char *path)
{
    ucm_event_handler_t *handler = arg;
    ucm_event_t event;

    if (!ucm_musamem_is_device_region(prot, path)) {
        return 0;
    }

    event.mem_type.address   = addr;
    event.mem_type.size      = length;
    event.mem_type.mem_type  = UCS_MEMORY_TYPE_LAST;
    event.mem_type.sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    event.mem_type.mem_flags = UCS_MEM_FLAG_REGISTRABLE;

    ucm_event_enter();
    handler->cb(UCM_EVENT_MEM_TYPE_ALLOC, &event, handler->arg);
    ucm_event_leave();
    return 0;
}

static void ucm_musamem_get_existing_alloc(ucm_event_handler_t *handler)
{
    if (handler->events & UCM_EVENT_MEM_TYPE_ALLOC) {
        ucm_parse_proc_self_maps(ucm_musamem_scan_regions_cb, handler);
    }
}

static ucm_event_installer_t ucm_musa_initializer = {
    .install            = ucm_musamem_install,
    .get_existing_alloc = ucm_musamem_get_existing_alloc
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_musa_initializer.list);
}

UCS_STATIC_CLEANUP
{
    ucs_list_del(&ucm_musa_initializer.list);
}
