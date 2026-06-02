/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_md.h"
#include "cuda_iface.h"
#include "cuda_util.h"

#include <ucs/sys/module.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/string.h>

#include <limits.h>
#include <stdint.h>


ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    const unsigned sys_device_priority = 10;
    ucs_sys_device_t sys_dev;
    CUdevice cuda_device;
    ucs_status_t status;
    char device_name[10];
    int i, num_gpus;

    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&num_gpus), UCS_LOG_LEVEL_DIAG);
    if ((status != UCS_OK) || (num_gpus == 0)) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    for (i = 0; i < num_gpus; ++i) {
        status = UCT_CUDADRV_FUNC(cuDeviceGet(&cuda_device, i),
                                  UCS_LOG_LEVEL_DIAG);
        if (status != UCS_OK) {
            continue;
        }

        sys_dev = uct_cuda_get_sys_dev(cuda_device);
        if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
            continue;
        }

        ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%d",
                          cuda_device);
        status = ucs_topo_sys_device_set_name(sys_dev, device_name,
                                              sys_device_priority);
        ucs_assert_always(status == UCS_OK);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

ucs_status_t
uct_cuda_base_fabric_alloc(CUdevice cuda_device, size_t *granularity_p,
                           uct_cuda_base_fabric_alloc_t *alloc_handle,
                           ucs_log_level_t log_level)
{
#if HAVE_CUDA_FABRIC
    CUmemAllocationProp prop    = {};
    CUmemAccessDesc access_desc = {};
    uint64_t allowed_types;
    ucs_status_t status;

    prop.type                            = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes            = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.type                   = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id                     = cuda_device;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    if (*granularity_p == SIZE_MAX) {
        status = UCT_CUDADRV_FUNC(cuMemGetAllocationGranularity(
                          granularity_p, &prop,
                          CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                          log_level);
        if (status != UCS_OK) {
            return status;
        }
    }

    alloc_handle->length = ucs_align_up(alloc_handle->length, *granularity_p);

    status = UCT_CUDADRV_FUNC(cuMemCreate(&alloc_handle->generic_handle,
                                          alloc_handle->length, &prop, 0),
                              log_level);
    if (status != UCS_OK) {
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_CUDADRV_FUNC(
            cuMemAddressReserve(&alloc_handle->ptr, alloc_handle->length,
                                *granularity_p, 0, 0),
            log_level);
    if (status != UCS_OK) {
        goto err_mem_release;
    }

    status = UCT_CUDADRV_FUNC(
            cuMemMap(alloc_handle->ptr, alloc_handle->length, 0,
                     alloc_handle->generic_handle, 0),
            log_level);
    if (status != UCS_OK) {
        goto err_address_free;
    }

    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = cuda_device;

    status = UCT_CUDADRV_FUNC(
            cuMemSetAccess(alloc_handle->ptr, alloc_handle->length,
                           &access_desc, 1),
            log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    }

    status = UCT_CUDADRV_FUNC(
            cuPointerGetAttribute(&allowed_types,
                                  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                                  alloc_handle->ptr),
            log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    } else if (!(allowed_types & CU_MEM_HANDLE_TYPE_FABRIC)) {
        ucs_log(log_level,
                "allocated memory at %p of size %ld does not have fabric "
                "property",
                (void*)alloc_handle->ptr, alloc_handle->length);
        status = UCS_ERR_UNSUPPORTED;
        goto err_mem_unmap;
    }

    ucs_trace("allocated vmm fabric memory at %p of size %ld",
              (void*)alloc_handle->ptr, alloc_handle->length);
    return UCS_OK;

err_mem_unmap:
    UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuMemUnmap(alloc_handle->ptr, alloc_handle->length));
err_address_free:
    UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuMemAddressFree(alloc_handle->ptr, alloc_handle->length));
err_mem_release:
    UCT_CUDADRV_FUNC_LOG_DEBUG(cuMemRelease(alloc_handle->generic_handle));
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t
uct_cuda_base_fabric_release(uct_cuda_base_fabric_alloc_t *alloc_handle,
                             ucs_log_level_t log_level)
{
#if HAVE_CUDA_FABRIC
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC(cuMemRelease(alloc_handle->generic_handle),
                              log_level);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC(cuMemUnmap(alloc_handle->ptr,
                                         alloc_handle->length),
                              log_level);
    if (status != UCS_OK) {
        return status;
    }

    return UCT_CUDADRV_FUNC(cuMemAddressFree(alloc_handle->ptr,
                                             alloc_handle->length),
                            log_level);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

int uct_cuda_base_device_supports_fabric(CUdevice cuda_device,
                                         ucs_log_level_t log_level)
{
#if HAVE_CUDA_FABRIC && \
    HAVE_DECL_CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    int supported;

    if (UCT_CUDADRV_FUNC(cuDeviceGetAttribute(
                &supported,
                CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                cuda_device), log_level) != UCS_OK) {
        return 0;
    }

    return supported;
#else
    return 0;
#endif
}

UCS_STATIC_INIT
{
    UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));
}

UCS_STATIC_CLEANUP
{
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
