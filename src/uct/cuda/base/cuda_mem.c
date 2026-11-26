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

#include <ucs/debug/log.h>
#include <ucs/sys/ptr_arith.h>

static ucs_status_t uct_cuda_mem_alloc_fabric(ucs_log_level_t log_level,
                                              CUdevice cu_device, size_t length,
                                              size_t *granularity,
                                              uct_cuda_mem_t *mem)
{
#if HAVE_CUDA_FABRIC
    CUmemAllocationProp prop    = {};
    CUmemAccessDesc access_desc = {};
    ucs_status_t status;
    uint64_t allowed_types;

    prop.type                            = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes            = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.type                   = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id                     = cu_device;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    if (*granularity == SIZE_MAX) {
        status = UCT_CUDADRV_FUNC(
                cuMemGetAllocationGranularity(granularity, &prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                log_level);
        if (status != UCS_OK) {
            return status;
        }
    }

    mem->length = ucs_align_up(length, *granularity);

    status = UCT_CUDADRV_FUNC(cuMemCreate(&mem->generic_handle, mem->length,
                                          &prop, 0),
                              log_level);
    if (status != UCS_OK) {
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_CUDADRV_FUNC(cuMemAddressReserve(&mem->ptr, mem->length,
                                                  *granularity, 0, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err_mem_release;
    }

    status = UCT_CUDADRV_FUNC(cuMemMap(mem->ptr, mem->length, 0,
                                       mem->generic_handle, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err_address_free;
    }

    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = cu_device;

    status = UCT_CUDADRV_FUNC(cuMemSetAccess(mem->ptr, mem->length,
                                             &access_desc, 1),
                              log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    }

    status = UCT_CUDADRV_FUNC(
            cuPointerGetAttribute(&allowed_types,
                                  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                                  mem->ptr),
            log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    } else if (!(allowed_types & CU_MEM_HANDLE_TYPE_FABRIC)) {
        ucs_log(log_level,
                "allocated memory at %p of size %ld does not have fabric "
                "property",
                (void*)mem->ptr, mem->length);
        goto err_mem_unmap;
    }

    mem->is_vmm = 1;

    ucs_trace("allocated vmm fabric memory at %p of size %ld", (void*)mem->ptr,
              mem->length);
    return UCS_OK;

err_mem_unmap:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(mem->ptr, mem->length));
err_address_free:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(mem->ptr, mem->length));
err_mem_release:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(mem->generic_handle));
#endif
    return UCS_ERR_NO_MEMORY;
}

static void uct_cuda_mem_release_fabric(uct_cuda_mem_t *mem)
{
#if HAVE_CUDA_FABRIC
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemUnmap(mem->ptr, mem->length));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemAddressFree(mem->ptr, mem->length));
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemRelease(mem->generic_handle));
#endif
}

ucs_status_t
uct_cuda_mem_alloc(ucs_log_level_t log_level, ucs_memory_type_t mem_type,
                   ucs_ternary_auto_value_t enable_fabric, CUdevice cu_device,
                   size_t length, size_t *granularity_p, uct_cuda_mem_t *mem)
{
    CUdeviceptr ptr;
    ucs_status_t status;
    ucs_log_level_t log_level_fabric;

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
        if (status != UCS_OK) {
            return status;
        }

        goto set_non_vmm;
    }

    if (enable_fabric != UCS_NO) {
        log_level_fabric = (enable_fabric == UCS_TRY) ? UCS_LOG_LEVEL_DEBUG :
                                                        log_level;
        status = uct_cuda_mem_alloc_fabric(log_level_fabric, cu_device, length,
                                           granularity_p, mem);
        if ((status == UCS_OK) || (enable_fabric == UCS_YES)) {
            return status;
        }
    }

    status = UCT_CUDADRV_FUNC(cuMemAlloc(&ptr, length), log_level);
    if (status != UCS_OK) {
        return UCS_ERR_NO_MEMORY;
    }

set_non_vmm:
    mem->ptr    = ptr;
    mem->length = length;
    mem->is_vmm = 0;

    return UCS_OK;
}

void uct_cuda_mem_free(uct_cuda_mem_t *mem)
{
    if (mem->is_vmm) {
        uct_cuda_mem_release_fabric(mem);
    } else {
        /* TODO: The more correct behavior is to issue a warning, provided that
         *       the context still exists for which memory has been allocated */
        (void)UCT_CUDADRV_FUNC(cuMemFree(mem->ptr), UCS_LOG_LEVEL_DIAG);
    }
}
