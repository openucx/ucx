/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_mem.h"

#include "cuda_util.h"

#include <ucs/sys/ptr_arith.h>
#include <ucs/debug/log.h>

#include <limits.h>

typedef CUresult (*uct_cuda_cuCtxSetFlags_t)(unsigned);

static ucs_status_t
uct_cuda_mem_alloc_fabric(ucs_log_level_t log_level, size_t length,
                          size_t *granularity_p, uct_cuda_mem_t *mem_p)
{
#if HAVE_CUDA_FABRIC
    CUmemAllocationProp prop    = {};
    CUmemAccessDesc access_desc = {};
    size_t granularity          = *granularity_p;
    CUdevice cu_device;
    ucs_status_t status;
    CUmemGenericAllocationHandle generic_handle;
    CUdeviceptr ptr;
    uint64_t allowed_types;

    status = UCT_CUDADRV_FUNC(cuCtxGetDevice(&cu_device), log_level);
    if (status != UCS_OK) {
        return status;
    }

    prop.type                            = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes            = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.type                   = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id                     = cu_device;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    if (granularity == SIZE_MAX) {
        status = UCT_CUDADRV_FUNC(
                cuMemGetAllocationGranularity(&granularity, &prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                log_level);
        if (status != UCS_OK) {
            return status;
        }
    }

    length = ucs_align_up(length, granularity);
    status = UCT_CUDADRV_FUNC(cuMemCreate(&generic_handle, length, &prop, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err;
    }

    status = UCT_CUDADRV_FUNC(cuMemAddressReserve(&ptr, length, granularity, 0,
                                                  0),
                              log_level);
    if (status != UCS_OK) {
        goto err_mem_release;
    }

    status = UCT_CUDADRV_FUNC(cuMemMap(ptr, length, 0, generic_handle, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err_address_free;
    }

    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = cu_device;

    status = UCT_CUDADRV_FUNC(cuMemSetAccess(ptr, length, &access_desc, 1),
                              log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    }

    status = UCT_CUDADRV_FUNC(
            cuPointerGetAttribute(&allowed_types,
                                  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                                  ptr),
            log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    } else if (!(allowed_types & CU_MEM_HANDLE_TYPE_FABRIC)) {
        ucs_log(log_level,
                "allocated cuda memory at 0x%llx of size %zi does not have "
                "fabric property",
                ptr, length);
        goto err_mem_unmap;
    }

    ucs_trace("allocated cuda fabric memory at 0x%llx of size %zi", ptr,
              length);

    *granularity_p        = granularity;
    mem_p->ptr            = ptr;
    mem_p->length         = length;
    mem_p->is_vmm         = 1;
    mem_p->generic_handle = generic_handle;

    return UCS_OK;

err_mem_unmap:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(ptr, length));
err_address_free:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(ptr, length));
err_mem_release:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(generic_handle));
err:
    return UCS_ERR_NO_MEMORY;
#else
    ucs_log(log_level, "cuda fabric is not supported");
    return UCS_ERR_UNSUPPORTED;
#endif
}

static void uct_cuda_mem_free_fabric(uct_cuda_mem_t mem)
{
#if HAVE_CUDA_FABRIC
    ucs_trace("freeing cuda fabric memory at 0x%llx of size %zi", mem.ptr,
              mem.length);
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(mem.generic_handle));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(mem.ptr, mem.length));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(mem.ptr, mem.length));
#endif
}

static ucs_status_t
uct_cuda_mem_set_non_vmm(ucs_log_level_t log_level, ucs_status_t status,
                         CUdeviceptr ptr, size_t length, uct_cuda_mem_t *mem_p)
{
    if (status != UCS_OK) {
        ucs_log(log_level, "failed to allocate cuda memory of size %zu",
                length);
        return UCS_ERR_NO_MEMORY;
    }

    ucs_trace("allocated cuda memory at 0x%llx of size %zu", ptr, length);
    mem_p->ptr    = ptr;
    mem_p->length = length;
    mem_p->is_vmm = 0;
    return UCS_OK;
}

ucs_status_t
uct_cuda_mem_alloc(ucs_log_level_t log_level, ucs_memory_type_t mem_type,
                   ucs_ternary_auto_value_t enable_fabric, size_t length,
                   size_t *granularity_p, uct_cuda_mem_t *mem_p)
{
    CUdeviceptr ptr;
    ucs_status_t status;
    ucs_log_level_t alloc_fabric_log_level;

    if ((mem_type != UCS_MEMORY_TYPE_CUDA) &&
        (mem_type != UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        ucs_log(log_level,
                "unsupported memory type %s, supported types: %s, %s",
                ucs_memory_type_names[mem_type],
                ucs_memory_type_names[UCS_MEMORY_TYPE_CUDA],
                ucs_memory_type_names[UCS_MEMORY_TYPE_CUDA_MANAGED]);
        return UCS_ERR_INVALID_PARAM;
    }

    if (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED) {
        status = UCT_CUDADRV_FUNC(cuMemAllocManaged(&ptr, length,
                                                    CU_MEM_ATTACH_GLOBAL),
                                  log_level);
        return uct_cuda_mem_set_non_vmm(log_level, status, ptr, length, mem_p);
    }

    /* mem_type == UCS_MEMORY_TYPE_CUDA */
    if (enable_fabric != UCS_NO) {
        alloc_fabric_log_level = (enable_fabric == UCS_TRY) ?
                                         UCS_LOG_LEVEL_DEBUG :
                                         log_level;
        status = uct_cuda_mem_alloc_fabric(alloc_fabric_log_level, length,
                                           granularity_p, mem_p);
        if (status == UCS_OK) {
            return UCS_OK;
        }
    }

    if (enable_fabric != UCS_YES) {
        status = UCT_CUDADRV_FUNC(cuMemAlloc(&ptr, length), log_level);
        return uct_cuda_mem_set_non_vmm(log_level, status, ptr, length, mem_p);
    }

    return status;
}

void uct_cuda_mem_free(uct_cuda_mem_t mem)
{
    if (mem.is_vmm) {
        uct_cuda_mem_free_fabric(mem);
    } else {
        ucs_trace("freeing cuda memory at 0x%llx of size %zi", mem.ptr,
                  mem.length);
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemFree(mem.ptr));
    }
}

ucs_status_t uct_cuda_mem_set_ctx_sync_memops(ucs_log_level_t log_level)
{
#if HAVE_CUDA_FABRIC
    static uct_cuda_cuCtxSetFlags_t cuda_cuCtxSetFlags_func =
            (uct_cuda_cuCtxSetFlags_t)ucs_empty_function;
    CUdriverProcAddressQueryResult sym_status;
    CUresult cu_err;

    if (cuda_cuCtxSetFlags_func ==
        (uct_cuda_cuCtxSetFlags_t)ucs_empty_function) {
        cu_err = cuGetProcAddress("cuCtxSetFlags",
                                  (void**)&cuda_cuCtxSetFlags_func, 12010,
                                  CU_GET_PROC_ADDRESS_DEFAULT, &sym_status);
        if ((cu_err != CUDA_SUCCESS) ||
            (sym_status != CU_GET_PROC_ADDRESS_SUCCESS)) {
            cuda_cuCtxSetFlags_func = NULL;
        }
    }

    if (cuda_cuCtxSetFlags_func != NULL) {
        /* Synchronize future DMA operations for all memory types */
        UCT_CUDADRV_FUNC(cuda_cuCtxSetFlags_func(CU_CTX_SYNC_MEMOPS),
                         log_level);
        return UCS_OK;
    }
#endif

    return UCS_ERR_UNSUPPORTED;
}
