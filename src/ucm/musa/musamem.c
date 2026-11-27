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

static musaError_t ucm_musaMalloc(void **address, size_t size);
static musaError_t ucm_musaFree(void *address);
static musaError_t ucm_musaMallocManaged(void **address, size_t size,
                                         unsigned flags);
static musaError_t ucm_musaFreeHost(void *address);
/* The non-async and async/pool allocation families are guarded: the
 * corresponding SDK types and symbols are referenced only when configure
 * detected the family (HAVE_MUSA_MEM_ALLOC_V2 / HAVE_MUSA_MEM_POOL_ALLOC). */
#ifdef HAVE_MUSA_MEM_ALLOC_V2
static musaError_t ucm_musaMallocPitch(void **address, size_t *pitch,
                                       size_t width, size_t height);
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
static musaError_t ucm_musaMallocAsync(void **address, size_t size,
                                       musaStream_t stream);
static musaError_t
ucm_musaMallocFromPoolAsync(void **address, size_t size,
                            musaMemPool_t pool, musaStream_t stream);
static musaError_t ucm_musaFreeAsync(void *address, musaStream_t stream);
#endif
static MUresult ucm_muMemAlloc(MUdeviceptr *address, size_t size);
static MUresult ucm_muMemAllocPitch(MUdeviceptr *address, size_t *pitch,
                                    size_t width, size_t height,
                                    unsigned element_size);
static MUresult ucm_muMemFree(MUdeviceptr address);
static MUresult ucm_muMemFreeHost(void *address);
#ifdef HAVE_MUSA_MEM_ALLOC_V2
static MUresult ucm_muMemAlloc_v2(MUdeviceptr *address, size_t size);
static MUresult ucm_muMemAllocManaged(MUdeviceptr *address, size_t size,
                                      unsigned flags);
static MUresult ucm_muMemAllocPitch_v2(MUdeviceptr *address, size_t *pitch,
                                       size_t width, size_t height,
                                       unsigned element_size);
static MUresult ucm_muMemFree_v2(MUdeviceptr address);
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
static MUresult ucm_muMemAllocAsync(MUdeviceptr *address, size_t size,
                                    MUstream stream);
static MUresult
ucm_muMemAllocFromPoolAsync(MUdeviceptr *address, size_t size,
                            MUmemoryPool pool, MUstream stream);
static MUresult ucm_muMemFreeAsync(MUdeviceptr address, MUstream stream);
#endif

/* Invoke directly so SDK aliases do not expand before token concatenation. */
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaMalloc, musaError_t, musaErrorUnknown,
                                  void **, size_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaFree, musaError_t, musaErrorUnknown,
                                  void *)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaMallocManaged, musaError_t,
                                  musaErrorUnknown, void **, size_t, unsigned)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaFreeHost, musaError_t, musaErrorUnknown,
                                  void *)
#ifdef HAVE_MUSA_MEM_ALLOC_V2
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaMallocPitch, musaError_t,
                                  musaErrorUnknown, void **, size_t *, size_t,
                                  size_t)
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaMallocAsync, musaError_t,
                                  musaErrorUnknown, void **, size_t,
                                  musaStream_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaMallocFromPoolAsync, musaError_t,
                                  musaErrorUnknown, void **, size_t,
                                  musaMemPool_t, musaStream_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(musaFreeAsync, musaError_t, musaErrorUnknown,
                                  void *, musaStream_t)
#endif
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAlloc, MUresult, -1, MUdeviceptr *, size_t)
#ifdef HAVE_MUSA_MEM_ALLOC_V2
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAlloc_v2, MUresult, -1, MUdeviceptr *,
                                  size_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAllocManaged, MUresult, -1, MUdeviceptr *,
                                  size_t, unsigned)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAllocPitch_v2, MUresult, -1,
                                  MUdeviceptr *, size_t *, size_t, size_t,
                                  unsigned)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemFree_v2, MUresult, -1, MUdeviceptr)
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAllocAsync, MUresult, -1, MUdeviceptr *,
                                  size_t, MUstream)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAllocFromPoolAsync, MUresult, -1,
                                  MUdeviceptr *, size_t, MUmemoryPool, MUstream)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemFreeAsync, MUresult, -1, MUdeviceptr,
                                  MUstream)
#endif
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemAllocPitch, MUresult, -1, MUdeviceptr *,
                                  size_t *, size_t, size_t, unsigned)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemFree, MUresult, -1, MUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(muMemFreeHost, MUresult, -1, void *)

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

/* Create MUSA memory allocation replacement functions */
#define UCM_MUSA_ALLOC_FUNC(_name, _retval, _success, _size, _ptr_type, _ref, \
                            ...) \
    static _retval ucm_##_name(_ptr_type _ref ptr_arg, \
                               UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        _retval status; \
        \
        ucm_event_enter(); \
        status = ucm_orig_##_name(ptr_arg, \
                                  UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
        if ((status == (_success)) && (ptr_arg != NULL) && \
            (*ptr_arg != (_ptr_type)0)) { \
            ucm_dispatch_mem_type_alloc((void *)(uintptr_t)*ptr_arg, (_size), \
                                        UCS_MEMORY_TYPE_UNKNOWN, \
                                        UCS_MEM_FLAG_REGISTRABLE); \
        } \
        ucm_event_leave(); \
        return status; \
    }

/* Create MUSA memory release replacement functions */
#define UCM_MUSA_FREE_FUNC(_name, _retval, _ptr_type, _mem_type, _size, ...) \
    static _retval ucm_##_name(UCS_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        _retval status; \
        \
        ucm_event_enter(); \
        if (arg0 != (_ptr_type)0) { \
            ucm_dispatch_mem_type_free((void *)(uintptr_t)arg0, (_size), \
                                       _mem_type); \
        } \
        status = ucm_orig_##_name(UCS_FUNC_PASS_ARGS(__VA_ARGS__)); \
        ucm_event_leave(); \
        return status; \
    }

UCM_MUSA_ALLOC_FUNC(musaMalloc, musaError_t, musaSuccess, arg0, void*, *,
                    size_t)
UCM_MUSA_FREE_FUNC(musaFree, musaError_t, void*, UCS_MEMORY_TYPE_MUSA,
                   ucm_musa_mem_size((void *)(uintptr_t)arg0), void*)
UCM_MUSA_ALLOC_FUNC(musaMallocManaged, musaError_t, musaSuccess, arg0, void*,
                    *, size_t, unsigned)
UCM_MUSA_FREE_FUNC(musaFreeHost, musaError_t, void*, UCS_MEMORY_TYPE_HOST, 1,
                   void*)
#ifdef HAVE_MUSA_MEM_ALLOC_V2
UCM_MUSA_ALLOC_FUNC(musaMallocPitch, musaError_t, musaSuccess, (*arg0) * arg2,
                    void*, *, size_t*, size_t, size_t)
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
UCM_MUSA_ALLOC_FUNC(musaMallocAsync, musaError_t, musaSuccess, arg0, void*, *,
                    size_t, musaStream_t)
UCM_MUSA_ALLOC_FUNC(musaMallocFromPoolAsync, musaError_t, musaSuccess, arg0,
                    void*, *, size_t, musaMemPool_t, musaStream_t)
UCM_MUSA_FREE_FUNC(musaFreeAsync, musaError_t, void*, UCS_MEMORY_TYPE_MUSA,
                   ucm_musa_mem_size((void *)(uintptr_t)arg0), void*,
                   musaStream_t)
#endif

UCM_MUSA_ALLOC_FUNC(muMemAlloc, MUresult, MUSA_SUCCESS, arg0, MUdeviceptr, *,
                    size_t)
UCM_MUSA_ALLOC_FUNC(muMemAllocPitch, MUresult, MUSA_SUCCESS, (*arg0) * arg2,
                    MUdeviceptr, *, size_t*, size_t, size_t, unsigned)
UCM_MUSA_FREE_FUNC(muMemFree, MUresult, MUdeviceptr, UCS_MEMORY_TYPE_MUSA,
                   ucm_musa_mem_size((void *)(uintptr_t)arg0), MUdeviceptr)
UCM_MUSA_FREE_FUNC(muMemFreeHost, MUresult, void*, UCS_MEMORY_TYPE_HOST, 1,
                   void*)

#ifdef HAVE_MUSA_MEM_ALLOC_V2
UCM_MUSA_ALLOC_FUNC(muMemAlloc_v2, MUresult, MUSA_SUCCESS, arg0, MUdeviceptr,
                    *, size_t)
UCM_MUSA_ALLOC_FUNC(muMemAllocManaged, MUresult, MUSA_SUCCESS, arg0,
                    MUdeviceptr, *, size_t, unsigned)
UCM_MUSA_ALLOC_FUNC(muMemAllocPitch_v2, MUresult, MUSA_SUCCESS,
                    (*arg0) * arg2, MUdeviceptr, *, size_t*, size_t, size_t,
                    unsigned)
UCM_MUSA_FREE_FUNC(muMemFree_v2, MUresult, MUdeviceptr, UCS_MEMORY_TYPE_MUSA,
                   ucm_musa_mem_size((void *)(uintptr_t)arg0), MUdeviceptr)
#endif

#ifdef HAVE_MUSA_MEM_POOL_ALLOC
UCM_MUSA_ALLOC_FUNC(muMemAllocAsync, MUresult, MUSA_SUCCESS, arg0, MUdeviceptr,
                    *, size_t, MUstream)
UCM_MUSA_ALLOC_FUNC(muMemAllocFromPoolAsync, MUresult, MUSA_SUCCESS, arg0,
                    MUdeviceptr, *, size_t, MUmemoryPool, MUstream)
UCM_MUSA_FREE_FUNC(muMemFreeAsync, MUresult, MUdeviceptr, UCS_MEMORY_TYPE_MUSA,
                   ucm_musa_mem_size((void *)(uintptr_t)arg0), MUdeviceptr,
                   MUstream)
#endif

static ucm_hook_func_t ucm_musa_driver_funcs[] = {
    UCM_HOOK_FUNC_ENTRY(muMemAlloc),
    UCM_HOOK_FUNC_ENTRY(muMemAllocPitch),
    UCM_HOOK_FUNC_ENTRY(muMemFree),
    UCM_HOOK_FUNC_ENTRY(muMemFreeHost),
#ifdef HAVE_MUSA_MEM_ALLOC_V2
    UCM_HOOK_FUNC_ENTRY(muMemAlloc_v2),
    UCM_HOOK_FUNC_ENTRY(muMemAllocManaged),
    UCM_HOOK_FUNC_ENTRY(muMemAllocPitch_v2),
    UCM_HOOK_FUNC_ENTRY(muMemFree_v2),
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
    UCM_HOOK_FUNC_ENTRY(muMemAllocAsync),
    UCM_HOOK_FUNC_ENTRY(muMemAllocFromPoolAsync),
    UCM_HOOK_FUNC_ENTRY(muMemFreeAsync),
#endif
    {{NULL, NULL}, NULL}
};

static ucm_hook_func_t ucm_musa_runtime_funcs[] = {
    UCM_HOOK_FUNC_ENTRY(musaMalloc),
    UCM_HOOK_FUNC_ENTRY(musaFree),
    UCM_HOOK_FUNC_ENTRY(musaMallocManaged),
    UCM_HOOK_FUNC_ENTRY(musaFreeHost),
#ifdef HAVE_MUSA_MEM_ALLOC_V2
    UCM_HOOK_FUNC_ENTRY(musaMallocPitch),
#endif
#ifdef HAVE_MUSA_MEM_POOL_ALLOC
    UCM_HOOK_FUNC_ENTRY(musaMallocAsync),
    UCM_HOOK_FUNC_ENTRY(musaMallocFromPoolAsync),
    UCM_HOOK_FUNC_ENTRY(musaFreeAsync),
#endif
    {{NULL, NULL}, NULL}
};

static ucs_status_t ucm_musamem_install(int events)
{
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    static int installed_driver_hooks;
    static int installed_runtime_hooks;
    ucs_status_t bistro_status;
    ucs_status_t driver_reloc_status;
    ucs_status_t runtime_reloc_status;
    ucs_status_t status;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        return UCS_OK;
    }

    if (!(ucm_global_opts.musa_hook_modes & UCM_GPU_HOOK_MODES_MASK)) {
        ucm_info("MUSA memory hooks are disabled by configuration");
        return UCS_ERR_UNSUPPORTED;
    }

    pthread_mutex_lock(&install_mutex);
    bistro_status = ucm_install_hooks(ucm_musa_driver_funcs, "MUSA",
                                      UCM_MMAP_HOOK_BISTRO,
                                      ucm_global_opts.musa_hook_modes,
                                      &installed_driver_hooks);
    if (bistro_status != UCS_OK) {
        ucm_debug("failed to install MUSA bistro hooks");
    }

    driver_reloc_status = ucm_install_hooks(ucm_musa_driver_funcs, "MUSA",
                                            UCM_MMAP_HOOK_RELOC,
                                            ucm_global_opts.musa_hook_modes,
                                            &installed_driver_hooks);
    if (driver_reloc_status != UCS_OK) {
        ucm_debug("failed to install MUSA driver reloc hooks");
    }

    runtime_reloc_status = ucm_install_hooks(ucm_musa_runtime_funcs, "MUSA",
                                             UCM_MMAP_HOOK_RELOC,
                                             ucm_global_opts.musa_hook_modes,
                                             &installed_runtime_hooks);
    if (runtime_reloc_status != UCS_OK) {
        ucm_debug("failed to install MUSA runtime reloc hooks");
    }

    if ((installed_driver_hooks | installed_runtime_hooks) &
        UCM_GPU_HOOK_MODES_MASK) {
        status = UCS_OK;
        if ((bistro_status != UCS_OK) ||
            (driver_reloc_status != UCS_OK) ||
            (runtime_reloc_status != UCS_OK)) {
            ucm_warn("MUSA hooks installed partially (bistro: %d, "
                     "driver reloc: %d, runtime reloc: %d)",
                     bistro_status, driver_reloc_status,
                     runtime_reloc_status);
        } else {
            ucm_info("MUSA hooks are ready");
        }
    } else {
        status = (bistro_status != UCS_OK) ? bistro_status :
                 (driver_reloc_status != UCS_OK) ? driver_reloc_status :
                                                   runtime_reloc_status;
        ucm_diag("failed to install MUSA hooks: status %d", status);
    }

    pthread_mutex_unlock(&install_mutex);
    return status;
}

static ucm_event_installer_t ucm_musa_initializer = {
    .install      = ucm_musamem_install,
    .path_pattern = "/dev/mtgpu"
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_musa_initializer.list);
}

UCS_STATIC_CLEANUP
{
    ucs_list_del(&ucm_musa_initializer.list);
}
