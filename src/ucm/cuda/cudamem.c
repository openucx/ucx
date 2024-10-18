/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cudamem.h"

#include <ucm/event/event.h>
#include <ucm/mmap/mmap.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucm/util/sys.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <sys/mman.h>
#include <string.h>


/* Create a body of CUDA memory allocation replacement function */
#define UCM_CUDA_ALLOC_FUNC(_name, _retval, _success, _size, _ptr_type, _ref, \
                            _args_fmt, ...) \
    _retval ucm_##_name(_ptr_type _ref ptr_arg, \
                        UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        _ptr_type ptr; \
        _retval ret; \
        \
        ucm_event_enter(); \
        ret = ucm_orig_##_name(ptr_arg, UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
        if (ret == (_success)) { \
            ptr = _ref ptr_arg; \
            ucm_trace("%s(" _args_fmt ") allocated %p", __func__, \
                      UCM_FUNC_PASS_ARGS(__VA_ARGS__), (void*)ptr); \
            ucm_cuda_dispatch_mem_alloc((CUdeviceptr)ptr, (_size)); \
        } \
        ucm_event_leave(); \
        return ret; \
    }

/* Create a body of CUDA memory release replacement function */
#define UCM_CUDA_FREE_FUNC(_name, _mem_type, _retval, _ptr_arg, _size, \
                           _args_fmt, ...) \
    _retval ucm_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        _retval ret; \
        \
        ucm_event_enter(); \
        ucm_trace("%s(" _args_fmt ")", __func__, \
                  UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
        ucm_cuda_dispatch_mem_free((CUdeviceptr)(_ptr_arg), _size, _mem_type, \
                                   #_name); \
        ret = ucm_orig_##_name(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
        ucm_event_leave(); \
        return ret; \
    }

#define UCM_CUDA_FUNC_ENTRY(_func) \
    { \
        {#_func, ucm_override_##_func}, (void**)&ucm_orig_##_func \
    }

typedef struct {
    ucm_reloc_patch_t patch;
    void              **orig_func_ptr;
} ucm_cuda_func_t;


/* Driver API */
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAlloc, CUresult, -1, CUdeviceptr*,
                                  size_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAlloc_v2, CUresult, -1, CUdeviceptr*,
                                  size_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAllocManaged, CUresult, -1, CUdeviceptr*,
                                  size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAllocPitch, CUresult, -1, CUdeviceptr*,
                                  size_t*, size_t, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAllocPitch_v2, CUresult, -1,
                                  CUdeviceptr*, size_t*, size_t, size_t,
                                  unsigned int)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemMap, CUresult, -1, CUdeviceptr, size_t,
                                  size_t, CUmemGenericAllocationHandle,
                                  unsigned long long)
#if CUDA_VERSION >= 11020
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAllocAsync, CUresult, -1, CUdeviceptr*,
                                  size_t, CUstream)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemAllocFromPoolAsync, CUresult, -1,
                                  CUdeviceptr*, size_t, CUmemoryPool, CUstream)
#endif
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFree, CUresult, -1, CUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFree_v2, CUresult, -1, CUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFreeHost, CUresult, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFreeHost_v2, CUresult, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemUnmap, CUresult, -1, CUdeviceptr, size_t)
#if CUDA_VERSION >= 11020
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFreeAsync, CUresult, -1, CUdeviceptr,
                                  CUstream)
#endif

/* Runtime API */
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaFree, cudaError_t, -1, void*)
#if CUDA_VERSION >= 11020
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaFreeAsync, cudaError_t, -1, void*,
                                  cudaStream_t)
#endif
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaFreeHost, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMalloc, cudaError_t, -1, void**, size_t)
#if CUDA_VERSION >= 11020
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMallocAsync, cudaError_t, -1, void**,
                                  size_t, cudaStream_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMallocFromPoolAsync, cudaError_t, -1,
                                  void**, size_t, cudaMemPool_t, cudaStream_t)
#endif
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMallocManaged, cudaError_t, -1, void**,
                                  size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMallocPitch, cudaError_t, -1, void**,
                                  size_t*, size_t, size_t)

static void ucm_cuda_dispatch_mem_alloc(CUdeviceptr ptr, size_t length)
{
    ucm_event_t event;

    event.mem_type.address  = (void*)ptr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = UCS_MEMORY_TYPE_LAST; /* indicate unknown type
                                                       and let cuda_md detect
                                                       attributes */
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static void ucm_cuda_dispatch_mem_free(CUdeviceptr ptr, size_t length,
                                       ucs_memory_type_t mem_type,
                                       const char *func_name)
{
    ucm_event_t event;
    CUdeviceptr pbase;
    CUresult ret;

    if (ptr == 0) {
        return;
    }

    if (length == 0) {
        /* If length is unknown, try to detect it */
        ret = cuMemGetAddressRange(&pbase, &length, ptr);
        if (ret == CUDA_SUCCESS) {
            if (ptr != pbase) {
                ucm_warn("%s(%p) called with unexpected pointer (expected: %p)",
                         func_name, (void*)ptr, (void*)pbase);
            }
        } else {
            ucm_debug("cuMemGetAddressRange(devPtr=%p) failed", (void*)ptr);
            length = 1; /* set minimum length */
        }
    }

    event.mem_type.address  = (void*)ptr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

/* Driver API replacements */
UCM_CUDA_ALLOC_FUNC(cuMemAlloc, CUresult, CUDA_SUCCESS, arg0, CUdeviceptr, *,
                    "size=%zu", size_t)
UCM_CUDA_ALLOC_FUNC(cuMemAlloc_v2, CUresult, CUDA_SUCCESS, arg0, CUdeviceptr, *,
                    "size=%zu", size_t)
UCM_CUDA_ALLOC_FUNC(cuMemAllocManaged, CUresult, CUDA_SUCCESS, arg0,
                    CUdeviceptr, *, "size=%zu flags=0x%x", size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cuMemAllocPitch, CUresult, CUDA_SUCCESS,
                    ((size_t)arg1) * (arg2), CUdeviceptr, *,
                    "pitch=%p width=%zu height=%zu elem=%u", size_t*, size_t,
                    size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cuMemAllocPitch_v2, CUresult, CUDA_SUCCESS,
                    ((size_t)arg1) * (arg2), CUdeviceptr, *,
                    "pitch=%p width=%zu height=%zu elem=%u", size_t*, size_t,
                    size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cuMemMap, CUresult, CUDA_SUCCESS, arg0, CUdeviceptr, ,
                    "size=%zu offset=%zu handle=0x%llx flags=0x%llx", size_t,
                    size_t, CUmemGenericAllocationHandle, unsigned long long)
#if CUDA_VERSION >= 11020
UCM_CUDA_ALLOC_FUNC(cuMemAllocAsync, CUresult, CUDA_SUCCESS, arg0, CUdeviceptr,
                    *, "size=%zu stream=%p", size_t, CUstream)
UCM_CUDA_ALLOC_FUNC(cuMemAllocFromPoolAsync, CUresult, CUDA_SUCCESS, arg0,
                    CUdeviceptr, *, "size=%zu pool=%p stream=%p", size_t,
                    CUmemoryPool, CUstream)
#endif
UCM_CUDA_FREE_FUNC(cuMemFree, UCS_MEMORY_TYPE_CUDA, CUresult, arg0, 0,
                   "ptr=0x%llx", CUdeviceptr)
UCM_CUDA_FREE_FUNC(cuMemFree_v2, UCS_MEMORY_TYPE_CUDA, CUresult, arg0, 0,
                   "ptr=0x%llx", CUdeviceptr)
UCM_CUDA_FREE_FUNC(cuMemFreeHost, UCS_MEMORY_TYPE_HOST, CUresult, arg0, 0,
                   "ptr=%p", void*)
UCM_CUDA_FREE_FUNC(cuMemFreeHost_v2, UCS_MEMORY_TYPE_HOST, CUresult, arg0, 0,
                   "ptr=%p", void*)
UCM_CUDA_FREE_FUNC(cuMemUnmap, UCS_MEMORY_TYPE_UNKNOWN, CUresult, arg0, arg1,
                   "ptr=%llx size=%zu", CUdeviceptr, size_t)
#if CUDA_VERSION >= 11020
UCM_CUDA_FREE_FUNC(cuMemFreeAsync, UCS_MEMORY_TYPE_CUDA, CUresult, arg0, 0,
                   "ptr=0x%llx, stream=%p", CUdeviceptr, CUstream)
#endif

static ucm_cuda_func_t ucm_cuda_driver_funcs[] = {
    UCM_CUDA_FUNC_ENTRY(cuMemAlloc),
    UCM_CUDA_FUNC_ENTRY(cuMemAlloc_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocManaged),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocPitch),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocPitch_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemMap),
#if CUDA_VERSION >= 11020
    UCM_CUDA_FUNC_ENTRY(cuMemAllocAsync),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocFromPoolAsync),
#endif
    UCM_CUDA_FUNC_ENTRY(cuMemFree),
    UCM_CUDA_FUNC_ENTRY(cuMemFree_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemFreeHost),
    UCM_CUDA_FUNC_ENTRY(cuMemFreeHost_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemUnmap),
#if CUDA_VERSION >= 11020
    UCM_CUDA_FUNC_ENTRY(cuMemFreeAsync),
#endif
    {{NULL}, NULL}
};

/* Runtime API replacements */
UCM_CUDA_ALLOC_FUNC(cudaMalloc, cudaError_t, cudaSuccess, arg0, void*, *,
                    "size=%zu", size_t)
UCM_CUDA_ALLOC_FUNC(cudaMallocManaged, cudaError_t, cudaSuccess, arg0, void*, *,
                    "size=%zu flags=0x%x", size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cudaMallocPitch, cudaError_t, cudaSuccess,
                    ((size_t)arg1) * (arg2), void*, *,
                    "pitch=%p width=%zu height=%zu", size_t*, size_t, size_t)
#if CUDA_VERSION >= 11020
UCM_CUDA_ALLOC_FUNC(cudaMallocAsync, cudaError_t, cudaSuccess, arg0, void*, *,
                    "size=%zu stream=%p", size_t, cudaStream_t)
UCM_CUDA_ALLOC_FUNC(cudaMallocFromPoolAsync, cudaError_t, cudaSuccess, arg0,
                    void*, *, "size=%zu pool=%p stream=%p", size_t,
                    cudaMemPool_t, cudaStream_t)
#endif
UCM_CUDA_FREE_FUNC(cudaFree, UCS_MEMORY_TYPE_CUDA, cudaError_t, arg0, 0,
                   "devPtr=%p", void*)
UCM_CUDA_FREE_FUNC(cudaFreeHost, UCS_MEMORY_TYPE_HOST, cudaError_t, arg0, 0,
                   "ptr=%p", void*)
#if CUDA_VERSION >= 11020
UCM_CUDA_FREE_FUNC(cudaFreeAsync, UCS_MEMORY_TYPE_CUDA, cudaError_t, arg0, 0,
                   "devPtr=%p, stream=%p", void*, cudaStream_t)
#endif

static ucm_cuda_func_t ucm_cuda_runtime_funcs[] = {
    UCM_CUDA_FUNC_ENTRY(cudaFree),
#if CUDA_VERSION >= 11020
    UCM_CUDA_FUNC_ENTRY(cudaFreeAsync),
#endif
    UCM_CUDA_FUNC_ENTRY(cudaFreeHost),
    UCM_CUDA_FUNC_ENTRY(cudaMalloc),
#if CUDA_VERSION >= 11020
    UCM_CUDA_FUNC_ENTRY(cudaMallocAsync),
    UCM_CUDA_FUNC_ENTRY(cudaMallocFromPoolAsync),
#endif
    UCM_CUDA_FUNC_ENTRY(cudaMallocManaged),
    UCM_CUDA_FUNC_ENTRY(cudaMallocPitch),
    {{NULL}, NULL}
};

static ucs_status_t
ucm_cuda_install_hooks(ucm_cuda_func_t *funcs, const char *name,
                       ucm_mmap_hook_mode_t mode, int *installed_hooks_p)
{
    ucm_cuda_func_t *func;
    ucs_status_t status;
    void *func_ptr;
    int count;

    if (*installed_hooks_p & UCS_BIT(mode)) {
        return UCS_OK;
    }

    if (!(ucm_global_opts.cuda_hook_modes & UCS_BIT(mode))) {
        /* Disabled by configuration */
        ucm_debug("cuda memory hooks mode %s is disabled for %s API",
                  ucm_mmap_hook_modes[mode], name);
        return UCS_OK;
    }

    count = 0;
    for (func = funcs; func->patch.symbol != NULL; ++func) {
        func_ptr = ucm_reloc_get_orig(func->patch.symbol, func->patch.value);
        if (func_ptr == NULL) {
            continue;
        }

        if (mode == UCM_MMAP_HOOK_BISTRO) {
            status = ucm_bistro_patch(func_ptr, func->patch.value,
                                      func->patch.symbol, func->orig_func_ptr,
                                      NULL);
        } else if (mode == UCM_MMAP_HOOK_RELOC) {
            status = ucm_reloc_modify(&func->patch);
        } else {
            break;
        }

        if (status != UCS_OK) {
            ucm_diag("failed to install %s hook for '%s'",
                     ucm_mmap_hook_modes[mode], func->patch.symbol);
            return status;
        }

        ucm_debug("installed %s hook for '%s'", ucm_mmap_hook_modes[mode],
                  func->patch.symbol);
        ++count;
    }

    *installed_hooks_p |= UCS_BIT(mode);
    ucm_info("cuda memory hooks mode %s: installed %d on %s API",
             ucm_mmap_hook_modes[mode], count, name);
    return UCS_OK;
}

static ucs_status_t ucm_cudamem_install(int events)
{
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    static int driver_api_hooks          = 0;
    static int runtime_api_hooks         = 0;
    ucs_status_t status                  = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    if (ucm_global_opts.cuda_hook_modes == 0) {
        ucm_info("cuda memory hooks are disabled by configuration");
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    pthread_mutex_lock(&install_mutex);

    status = ucm_cuda_install_hooks(ucm_cuda_driver_funcs, "driver",
                                    UCM_MMAP_HOOK_BISTRO, &driver_api_hooks);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    status = ucm_cuda_install_hooks(ucm_cuda_driver_funcs, "driver",
                                    UCM_MMAP_HOOK_RELOC, &driver_api_hooks);
    if (status != UCS_OK) {
        goto out_unlock;
    }

    status = ucm_cuda_install_hooks(ucm_cuda_runtime_funcs, "runtime",
                                    UCM_MMAP_HOOK_RELOC, &runtime_api_hooks);
    if (status != UCS_OK) {
        goto out_unlock;
    }

out_unlock:
    pthread_mutex_unlock(&install_mutex);
out:
    return status;
}

static int ucm_cudamem_scan_regions_cb(void *arg, void *addr, size_t length,
                                       int prot, const char *path)
{
    static const char cuda_path_pattern[] = "/dev/nvidia";
    ucm_event_handler_t *handler          = arg;
    ucm_event_t event;

    /* we are interested in blocks which don't have any access permissions, or
     * mapped to nvidia device.
     */
    if ((prot & (PROT_READ | PROT_WRITE | PROT_EXEC)) &&
        strncmp(path, cuda_path_pattern, sizeof(cuda_path_pattern) - 1)) {
        return 0;
    }

    ucm_debug("dispatching initial memtype allocation for %p..%p %s", addr,
              UCS_PTR_BYTE_OFFSET(addr, length), path);

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = UCS_MEMORY_TYPE_LAST; /* unknown memory type */

    ucm_event_enter();
    handler->cb(UCM_EVENT_MEM_TYPE_ALLOC, &event, handler->arg);
    ucm_event_leave();

    return 0;
}

static void ucm_cudamem_get_existing_alloc(ucm_event_handler_t *handler)
{
    if (handler->events & UCM_EVENT_MEM_TYPE_ALLOC) {
        ucm_parse_proc_self_maps(ucm_cudamem_scan_regions_cb, handler);
    }
}

static ucm_event_installer_t ucm_cuda_initializer = {
    .install            = ucm_cudamem_install,
    .get_existing_alloc = ucm_cudamem_get_existing_alloc
};

UCS_STATIC_INIT
{
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_cuda_initializer.list);
}

UCS_STATIC_CLEANUP
{
    ucs_list_del(&ucm_cuda_initializer.list);
}
