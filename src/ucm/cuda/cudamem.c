/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#define UCM_CUDA_ALLOC_FUNC(_name, _mem_type, _retval, _success, _size, \
                            _ptr_type, _args_fmt, ...) \
    _retval ucm_##_name(_ptr_type *ptr_p, UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        _ptr_type ptr; \
        _retval ret; \
        \
        ucm_event_enter(); \
        ret = ucm_orig_##_name(ptr_p, UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
        if (ret == (_success)) { \
            ptr = *ptr_p; \
            ucm_trace("%s(" _args_fmt ") allocated %p", __FUNCTION__, \
                      UCM_FUNC_PASS_ARGS(__VA_ARGS__), (void*)ptr); \
            ucm_cuda_dispatch_mem_alloc((CUdeviceptr)ptr, (_size), \
                                        (_mem_type)); \
        } \
        ucm_event_leave(); \
        return ret; \
    }

/* Create a body of CUDA memory release replacement function */
#define UCM_CUDA_FREE_FUNC(_name, _retval, _ptr_type, _mem_type) \
    _retval ucm_##_name(_ptr_type ptr) \
    { \
        _retval ret; \
        \
        ucm_event_enter(); \
        ucm_trace("%s(ptr=%p)", __FUNCTION__, (void*)ptr); \
        ucm_cuda_dispatch_mem_free((CUdeviceptr)ptr, _mem_type, #_name); \
        ret = ucm_orig_##_name(ptr); \
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
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFree, CUresult, -1, CUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFree_v2, CUresult, -1, CUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFreeHost, CUresult, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cuMemFreeHost_v2, CUresult, -1, void*)

/* Runtime API */
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaFree, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaFreeHost, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMalloc, cudaError_t, -1, void**, size_t)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMallocManaged, cudaError_t, -1, void**,
                                  size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_PTR_FUNC(cudaMallocPitch, cudaError_t, -1, void**,
                                  size_t*, size_t, size_t)

static void ucm_cuda_dispatch_mem_alloc(CUdeviceptr ptr, size_t length,
                                        ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = (void*)ptr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = UCS_MEMORY_TYPE_LAST; /* indicate unknown type
                                                       and let cuda_md detect
                                                       attributes */
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static void ucm_cuda_dispatch_mem_free(CUdeviceptr ptr,
                                       ucs_memory_type_t mem_type,
                                       const char *func_name)
{
    ucm_event_t event;
    CUdeviceptr pbase;
    size_t length;
    CUresult ret;

    if (ptr == 0) {
        return;
    }

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

    event.mem_type.address  = (void*)ptr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

/* Driver API replacements */
UCM_CUDA_ALLOC_FUNC(cuMemAlloc, UCS_MEMORY_TYPE_CUDA, CUresult, CUDA_SUCCESS,
                    arg0, CUdeviceptr, "size=%zu", size_t)
UCM_CUDA_ALLOC_FUNC(cuMemAlloc_v2, UCS_MEMORY_TYPE_CUDA, CUresult, CUDA_SUCCESS,
                    arg0, CUdeviceptr, "size=%zu", size_t)
UCM_CUDA_ALLOC_FUNC(cuMemAllocManaged, UCS_MEMORY_TYPE_CUDA_MANAGED, CUresult,
                    CUDA_SUCCESS, arg0, CUdeviceptr, "size=%zu flags=0x%x",
                    size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cuMemAllocPitch, UCS_MEMORY_TYPE_CUDA, CUresult,
                    CUDA_SUCCESS, (size_t)arg1 * arg2, CUdeviceptr,
                    "pitch=%p width=%zu height=%zu elem=%u", size_t*, size_t,
                    size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cuMemAllocPitch_v2, UCS_MEMORY_TYPE_CUDA, CUresult,
                    CUDA_SUCCESS, (size_t)arg1 * arg2, CUdeviceptr,
                    "pitch=%p width=%zu height=%zu elem=%u", size_t*, size_t,
                    size_t, unsigned)
UCM_CUDA_FREE_FUNC(cuMemFree, CUresult, CUdeviceptr, UCS_MEMORY_TYPE_CUDA)
UCM_CUDA_FREE_FUNC(cuMemFree_v2, CUresult, CUdeviceptr, UCS_MEMORY_TYPE_CUDA)
UCM_CUDA_FREE_FUNC(cuMemFreeHost, CUresult, void*, UCS_MEMORY_TYPE_HOST)
UCM_CUDA_FREE_FUNC(cuMemFreeHost_v2, CUresult, void*, UCS_MEMORY_TYPE_HOST)

static ucm_cuda_func_t ucm_cuda_driver_funcs[] = {
    UCM_CUDA_FUNC_ENTRY(cuMemAlloc),
    UCM_CUDA_FUNC_ENTRY(cuMemAlloc_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocManaged),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocPitch),
    UCM_CUDA_FUNC_ENTRY(cuMemAllocPitch_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemFree),
    UCM_CUDA_FUNC_ENTRY(cuMemFree_v2),
    UCM_CUDA_FUNC_ENTRY(cuMemFreeHost),
    UCM_CUDA_FUNC_ENTRY(cuMemFreeHost_v2),
    {{NULL}, NULL}
};

/* Runtime API replacements */
UCM_CUDA_ALLOC_FUNC(cudaMalloc, UCS_MEMORY_TYPE_CUDA, cudaError_t, cudaSuccess,
                    arg0, void*, "size=%zu", size_t)
UCM_CUDA_ALLOC_FUNC(cudaMallocManaged, UCS_MEMORY_TYPE_CUDA_MANAGED,
                    cudaError_t, cudaSuccess, arg0, void*,
                    "size=%zu flags=0x%x", size_t, unsigned)
UCM_CUDA_ALLOC_FUNC(cudaMallocPitch, UCS_MEMORY_TYPE_CUDA, cudaError_t,
                    cudaSuccess, (size_t)arg1 * arg2, void*,
                    "pitch=%p width=%zu height=%zu", size_t*, size_t, size_t)
UCM_CUDA_FREE_FUNC(cudaFree, cudaError_t, void*, UCS_MEMORY_TYPE_CUDA)
UCM_CUDA_FREE_FUNC(cudaFreeHost, cudaError_t, void*, UCS_MEMORY_TYPE_HOST)

static ucm_cuda_func_t ucm_cuda_runtime_funcs[] = {
    UCM_CUDA_FUNC_ENTRY(cudaFree),
    UCM_CUDA_FUNC_ENTRY(cudaFreeHost),
    UCM_CUDA_FUNC_ENTRY(cudaMalloc),
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
    static const char *cuda_path_pattern = "/dev/nvidia";
    ucm_event_handler_t *handler         = arg;
    ucm_event_t event;

    /* we are interested in blocks which don't have any access permissions, or
     * mapped to nvidia device.
     */
    if ((prot & (PROT_READ | PROT_WRITE | PROT_EXEC)) &&
        strncmp(path, cuda_path_pattern, strlen(cuda_path_pattern))) {
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
