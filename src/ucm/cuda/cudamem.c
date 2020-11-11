/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/cuda/cudamem.h>

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucm/util/sys.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>

#include <sys/mman.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>


UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemFree, CUresult, -1, CUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemFreeHost, CUresult, -1, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemAlloc, CUresult, -1, CUdeviceptr *, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemAllocManaged, CUresult, -1, CUdeviceptr *,
                              size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemAllocPitch, CUresult, -1, CUdeviceptr *, size_t *,
                              size_t, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemHostGetDevicePointer, CUresult, -1, CUdeviceptr *,
                              void *, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemHostUnregister, CUresult, -1, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaFree, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaFreeHost, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaMalloc, cudaError_t, -1, void**, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaMallocManaged, cudaError_t, -1, void**, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaMallocPitch, cudaError_t, -1, void**, size_t *,
                              size_t, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaHostGetDevicePointer, cudaError_t, -1, void**,
                              void *, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaHostUnregister, cudaError_t, -1, void*)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_FUNC(cuMemFree,                 CUresult)
UCM_OVERRIDE_FUNC(cuMemFreeHost,             CUresult)
UCM_OVERRIDE_FUNC(cuMemAlloc,                CUresult)
UCM_OVERRIDE_FUNC(cuMemAllocManaged,         CUresult)
UCM_OVERRIDE_FUNC(cuMemAllocPitch,           CUresult)
UCM_OVERRIDE_FUNC(cuMemHostGetDevicePointer, CUresult)
UCM_OVERRIDE_FUNC(cuMemHostUnregister,       CUresult)
UCM_OVERRIDE_FUNC(cudaFree,                  cudaError_t)
UCM_OVERRIDE_FUNC(cudaFreeHost,              cudaError_t)
UCM_OVERRIDE_FUNC(cudaMalloc,                cudaError_t)
UCM_OVERRIDE_FUNC(cudaMallocManaged,         cudaError_t)
UCM_OVERRIDE_FUNC(cudaMallocPitch,           cudaError_t)
UCM_OVERRIDE_FUNC(cudaHostGetDevicePointer,  cudaError_t)
UCM_OVERRIDE_FUNC(cudaHostUnregister,        cudaError_t)
#endif


static void ucm_cuda_set_ptr_attr(CUdeviceptr dptr)
{
    unsigned int value = 1;
    CUresult ret;
    const char *cu_err_str;

    if ((void*)dptr == NULL) {
        ucm_trace("skipping cuPointerSetAttribute for null pointer");
        return;
    }

    ret = cuPointerSetAttribute(&value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    if (ret != CUDA_SUCCESS) {
        cuGetErrorString(ret, &cu_err_str);
        ucm_warn("cuPointerSetAttribute(%p) failed: %s", (void *) dptr, cu_err_str);
    }
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_alloc(void *addr, size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_free(void *addr, size_t length, ucs_memory_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

static void ucm_cudafree_dispatch_events(CUdeviceptr dptr, const char *func_name)
{
    CUresult ret;
    CUdeviceptr pbase;
    size_t psize;

    if (dptr == 0) {
        return;
    }

    ret = cuMemGetAddressRange(&pbase, &psize, dptr);
    if (ret == CUDA_SUCCESS) {
        if (dptr != pbase) {
            ucm_warn("%s(%p) called with unexpected pointer (expected: %p)",
                     func_name, (void*)dptr, (void*)pbase);
        }
    } else {
        ucm_debug("cuMemGetAddressRange(devPtr=%p) failed", (void*)dptr);
        psize = 1; /* set minimum length */
    }

    ucm_dispatch_mem_type_free((void *)dptr, psize, UCS_MEMORY_TYPE_CUDA);
}

CUresult ucm_cuMemFree(CUdeviceptr dptr)
{
    CUresult ret;

    ucm_event_enter();

    ucm_trace("ucm_cuMemFree(dptr=%p)",(void*)dptr);

    ucm_cudafree_dispatch_events(dptr, "cuMemFree");

    ret = ucm_orig_cuMemFree(dptr);

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemFreeHost(void *p)
{
    CUresult ret;

    ucm_event_enter();

    ucm_trace("ucm_cuMemFreeHost(ptr=%p)", p);

    ucm_dispatch_vm_munmap(p, 0);

    ret = ucm_orig_cuMemFreeHost(p);

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemAlloc(CUdeviceptr *dptr, size_t size)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemAlloc(dptr, size);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemAlloc(dptr=%p size:%lu)",(void *)*dptr, size);
        ucm_dispatch_mem_type_alloc((void *)*dptr, size, UCS_MEMORY_TYPE_CUDA);
        ucm_cuda_set_ptr_attr(*dptr);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemAllocManaged(CUdeviceptr *dptr, size_t size, unsigned int flags)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemAllocManaged(dptr, size, flags);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemAllocManaged(dptr=%p size:%lu, flags:%d)",
                  (void *)*dptr, size, flags);
        ucm_dispatch_mem_type_alloc((void *)*dptr, size,
                                    UCS_MEMORY_TYPE_CUDA_MANAGED);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemAllocPitch(dptr=%p size:%lu)",(void *)*dptr,
                  (WidthInBytes * Height));
        ucm_dispatch_mem_type_alloc((void *)*dptr, WidthInBytes * Height,
                                    UCS_MEMORY_TYPE_CUDA);
        ucm_cuda_set_ptr_attr(*dptr);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemHostGetDevicePointer(pdptr, p, Flags);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemHostGetDevicePointer(pdptr=%p p=%p)",(void *)*pdptr, p);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemHostUnregister(void *p)
{
    CUresult ret;

    ucm_event_enter();

    ucm_trace("ucm_cuMemHostUnregister(ptr=%p)", p);

    ret = ucm_orig_cuMemHostUnregister(p);

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaFree(void *devPtr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaFree(devPtr=%p)", devPtr);

    ucm_cudafree_dispatch_events((CUdeviceptr)devPtr, "cudaFree");

    ret = ucm_orig_cudaFree(devPtr);

    ucm_event_leave();

    return ret;
}

cudaError_t ucm_cudaFreeHost(void *ptr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaFreeHost(ptr=%p)", ptr);

    ucm_dispatch_vm_munmap(ptr, 0);

    ret = ucm_orig_cudaFreeHost(ptr);

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaMalloc(devPtr, size);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cudaMalloc(devPtr=%p size:%lu)", *devPtr, size);
        ucm_dispatch_mem_type_alloc(*devPtr, size, UCS_MEMORY_TYPE_CUDA);
        ucm_cuda_set_ptr_attr((CUdeviceptr) *devPtr);
    }

    ucm_event_leave();

    return ret;
}

cudaError_t ucm_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaMallocManaged(devPtr, size, flags);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cudaMallocManaged(devPtr=%p size:%lu flags:%d)",
                  *devPtr, size, flags);
        ucm_dispatch_mem_type_alloc(*devPtr, size, UCS_MEMORY_TYPE_CUDA_MANAGED);
    }

    ucm_event_leave();

    return ret;
}

cudaError_t ucm_cudaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaMallocPitch(devPtr, pitch, width, height);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cudaMallocPitch(devPtr=%p size:%lu)",*devPtr, (width * height));
        ucm_dispatch_mem_type_alloc(*devPtr, (width * height), UCS_MEMORY_TYPE_CUDA);
        ucm_cuda_set_ptr_attr((CUdeviceptr) *devPtr);
    }

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaHostGetDevicePointer(pDevice, pHost, flags);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cuMemHostGetDevicePointer(pDevice=%p pHost=%p)", pDevice, pHost);
    }

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaHostUnregister(void *ptr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaHostUnregister(ptr=%p)", ptr);

    ret = ucm_orig_cudaHostUnregister(ptr);

    ucm_event_leave();
    return ret;
}

static ucm_reloc_patch_t patches[] = {
    {UCS_PP_MAKE_STRING(cuMemFree),                 ucm_override_cuMemFree},
    {UCS_PP_MAKE_STRING(cuMemFreeHost),             ucm_override_cuMemFreeHost},
    {UCS_PP_MAKE_STRING(cuMemAlloc),                ucm_override_cuMemAlloc},
    {UCS_PP_MAKE_STRING(cuMemAllocManaged),         ucm_override_cuMemAllocManaged},
    {UCS_PP_MAKE_STRING(cuMemAllocPitch),           ucm_override_cuMemAllocPitch},
    {UCS_PP_MAKE_STRING(cuMemHostGetDevicePointer), ucm_override_cuMemHostGetDevicePointer},
    {UCS_PP_MAKE_STRING(cuMemHostUnregister),       ucm_override_cuMemHostUnregister},
    {UCS_PP_MAKE_STRING(cudaFree),                  ucm_override_cudaFree},
    {UCS_PP_MAKE_STRING(cudaFreeHost),              ucm_override_cudaFreeHost},
    {UCS_PP_MAKE_STRING(cudaMalloc),                ucm_override_cudaMalloc},
    {UCS_PP_MAKE_STRING(cudaMallocManaged),         ucm_override_cudaMallocManaged},
    {UCS_PP_MAKE_STRING(cudaMallocPitch),           ucm_override_cudaMallocPitch},
    {UCS_PP_MAKE_STRING(cudaHostGetDevicePointer),  ucm_override_cudaHostGetDevicePointer},
    {UCS_PP_MAKE_STRING(cudaHostUnregister),        ucm_override_cudaHostUnregister},
    {NULL,                                          NULL}
};

static ucs_status_t ucm_cudamem_install(int events)
{
    static int ucm_cudamem_installed = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status = UCS_OK;

    if (!(events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE))) {
        goto out;
    }

    if (!ucm_global_opts.enable_cuda_reloc) {
        ucm_debug("installing cudamem relocations is disabled by configuration");
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    pthread_mutex_lock(&install_mutex);

    if (ucm_cudamem_installed) {
        goto out_unlock;
    }

    for (patch = patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'", patch->symbol);
            goto out_unlock;
        }
    }

    ucm_debug("cudaFree hooks are ready");
    ucm_cudamem_installed = 1;

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
    if ((prot & (PROT_READ|PROT_WRITE|PROT_EXEC)) &&
        strncmp(path, cuda_path_pattern, strlen(cuda_path_pattern))) {
        return 0;
    }

    ucm_debug("dispatching initial memtype allocation for %p..%p %s",
              addr, UCS_PTR_BYTE_OFFSET(addr, length), path);

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

UCS_STATIC_INIT {
    ucs_list_add_tail(&ucm_event_installer_list, &ucm_cuda_initializer.list);
}

UCS_STATIC_CLEANUP {
    ucs_list_del(&ucm_cuda_initializer.list);
}
