/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/profile/profile.h>
#include <ucs/type/class.h>
#include <ucs/sys/ptr_arith.h>
#include <uct/cuda/base/cuda_iface.h>
#include <uct/api/v2/uct_v2.h>
#include <cuda.h>
#if CUDA_VERSION >= 11070
#include <cudaTypedefs.h>
#endif


#define UCT_CUDA_DEV_NAME_MAX_LEN 64
#define UCT_CUDA_MAX_DEVICES      32


static const char *uct_cuda_pref_loc[] = {
    [UCT_CUDA_PREF_LOC_CPU]  = "cpu",
    [UCT_CUDA_PREF_LOC_GPU]  = "gpu",
    [UCT_CUDA_PREF_LOC_LAST] = NULL,
};

static ucs_config_field_t uct_cuda_copy_md_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(uct_cuda_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"REG_WHOLE_ALLOC", "auto",
     "Allow registration of whole allocation\n"
     " auto - Let runtime decide where whole allocation registration is turned on.\n"
     "        By default this will be turned off for limited BAR GPUs (eg. T4)\n"
     " on   - Whole allocation registration is always turned on.\n"
     " off  - Whole allocation registration is always turned off.",
     ucs_offsetof(uct_cuda_copy_md_config_t, alloc_whole_reg),
     UCS_CONFIG_TYPE_ON_OFF_AUTO},

    {"MAX_REG_RATIO", "0.1",
     "If the ratio of the length of the allocation to which the user buffer belongs to"
     " to the total GPU memory capacity is below this ratio, then the whole allocation"
     " is registered. Otherwise only the user specified region is registered.",
     ucs_offsetof(uct_cuda_copy_md_config_t, max_reg_ratio), UCS_CONFIG_TYPE_DOUBLE},

    {"DMABUF", "try",
     "Enable using cross-device dmabuf file descriptor",
     ucs_offsetof(uct_cuda_copy_md_config_t, enable_dmabuf),
                  UCS_CONFIG_TYPE_TERNARY},

    {"PREF_LOC", "cpu",
     "System device designation of a CUDA managed memory buffer"
     " whose preferred location attribute is not set.\n"
     " cpu - Assume buffer is on the CPU.\n"
     " gpu - Assume buffer is on the GPU corresponding to buffer's GPU context.",
     ucs_offsetof(uct_cuda_copy_md_config_t, pref_loc),
     UCS_CONFIG_TYPE_ENUM(uct_cuda_pref_loc)},

    {"ENABLE_FABRIC", "try", "Enable fabric memory allocation",
     ucs_offsetof(uct_cuda_copy_md_config_t, enable_fabric),
     UCS_CONFIG_TYPE_TERNARY},

    {"ASYNC_MEM_TYPE", "cuda-managed",
     "Memory type which is detected for asynchronously allocated cuda memory.\n"
     "Allowed memory type is one of: cuda, cuda-managed",
     ucs_offsetof(uct_cuda_copy_md_config_t, cuda_async_mem_type),
     UCS_CONFIG_TYPE_ENUM(ucs_memory_type_names)},

    {NULL}
};

static int uct_cuda_copy_md_is_dmabuf_supported()
{
    int dmabuf_supported = 0;
    CUdevice cuda_device;

    if (UCT_CUDADRV_FUNC_LOG_DEBUG(cuDeviceGet(&cuda_device, 0)) != UCS_OK) {
        return 0;
    }

    /* Assume dmabuf support is uniform across all devices */
#if CUDA_VERSION >= 11070
    if (UCT_CUDADRV_FUNC_LOG_DEBUG(
                cuDeviceGetAttribute(&dmabuf_supported,
                                     CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                     cuda_device)) != UCS_OK) {
        return 0;
    }
#endif

    ucs_debug("dmabuf is%s supported on cuda device %d",
              dmabuf_supported ? "" : " not", cuda_device);
    return dmabuf_supported;
}

static ucs_status_t
uct_cuda_copy_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_cuda_copy_md_t *md = ucs_derived_of(uct_md, uct_cuda_copy_md_t);

    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->dmabuf_mem_types = md->config.dmabuf_supported ?
                                UCS_BIT(UCS_MEMORY_TYPE_CUDA) : 0;
    md_attr->max_alloc        = SIZE_MAX;
    return UCS_OK;
}

static ucs_status_t
uct_cuda_copy_mkey_pack(uct_md_h md, uct_mem_h memh, void *address,
                        size_t length, const uct_md_mkey_pack_params_t *params,
                        void *mkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_rkey_unpack(uct_component_t *component,
                                              const void *rkey_buffer,
                                              uct_rkey_t *rkey_p,
                                              void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_rkey_release(uct_component_t *component,
                                               uct_rkey_t rkey, void *handle)
{
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_mem_reg,
                 (md, address, length, params, memh_p),
                 uct_md_h md, void *address, size_t length,
                 const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uint64_t flags = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    ucs_log_level_t log_level;
    CUmemorytype memType;
    CUresult result;
    ucs_status_t status;

    if (!uct_cuda_base_is_context_active()) {
        ucs_debug("attempt to register memory without active context");
        return uct_md_dummy_mem_reg(md, address, length, params, memh_p);
    }

    result = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)(address));
    if ((result == CUDA_SUCCESS) && ((memType == CU_MEMORYTYPE_HOST)    ||
                                     (memType == CU_MEMORYTYPE_UNIFIED) ||
                                     (memType == CU_MEMORYTYPE_DEVICE))) {
        /* only host memory not allocated by cuda needs to be registered */
        return uct_md_dummy_mem_reg(md, address, length, params, memh_p);
    }

    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                UCS_LOG_LEVEL_ERROR;
    status    = UCT_CUDADRV_FUNC(cuMemHostRegister(address, length,
                                                   CU_MEMHOSTREGISTER_PORTABLE),
                                 log_level);
    if (status != UCS_OK) {
        return status;
    }

    *memh_p = address;
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_mem_dereg,
                 (md, params),
                 uct_md_h md, const uct_md_mem_dereg_params_t *params)
{
    void *address;
    ucs_status_t status;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    address = (void *)params->memh;
    if (address == (void*)0xdeadbeef) {
        return UCS_OK;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemHostUnregister(address));
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t
uct_cuda_copy_mem_alloc_fabric(uct_cuda_copy_md_t *md,
                               uct_cuda_copy_alloc_handle_t *alloc_handle,
                               unsigned flags)
{
#if HAVE_CUDA_FABRIC
    CUmemAllocationProp prop    = {};
    CUmemAccessDesc access_desc = {};
    ucs_log_level_t log_level   = (md->config.enable_fabric == UCS_YES) ?
                                  UCS_LOG_LEVEL_ERROR : UCS_LOG_LEVEL_DEBUG;
    ucs_status_t status;
    CUdevice cu_device;

    if (!(flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) &&
        (md->config.enable_fabric == UCS_YES)) {
        log_level = UCS_LOG_LEVEL_ERROR;
    } else {
        log_level = UCS_LOG_LEVEL_DEBUG;
    }

    status = UCT_CUDADRV_FUNC(cuCtxGetDevice(&cu_device), log_level);
    if (status != UCS_OK) {
        return status;
    }

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = cu_device;

    if (md->granularity == SIZE_MAX) {
        status = UCT_CUDADRV_FUNC(cuMemGetAllocationGranularity(
                &md->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                log_level);
        if (status != UCS_OK) {
            return status;
        }
    }

    alloc_handle->length = ucs_align_up(alloc_handle->length, md->granularity);

    status = UCT_CUDADRV_FUNC(cuMemCreate(&alloc_handle->generic_handle,
                                          alloc_handle->length, &prop, 0),
                              log_level);
    if (status != UCS_OK) {
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_CUDADRV_FUNC(cuMemAddressReserve(
                                     &alloc_handle->ptr, alloc_handle->length,
                                     md->granularity, 0, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err_mem_release;
    }

    status = UCT_CUDADRV_FUNC(cuMemMap(alloc_handle->ptr, alloc_handle->length,
                                       0, alloc_handle->generic_handle, 0),
                              log_level);
    if (status != UCS_OK) {
        goto err_address_free;
    }

    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = cu_device;

    status = UCT_CUDADRV_FUNC(cuMemSetAccess(
                     alloc_handle->ptr, alloc_handle->length, &access_desc, 1),
                     log_level);
    if (status != UCS_OK) {
        goto err_mem_unmap;
    }

    alloc_handle->is_vmm = 1;

    ucs_trace("allocated vmm fabric memory at %p of size %ld\n",
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
#endif
    return UCS_ERR_NO_MEMORY;
}

static ucs_status_t
uct_cuda_copy_mem_alloc(uct_md_h uct_md, size_t *length_p, void **address_p,
                        ucs_memory_type_t mem_type, unsigned flags,
                        const char *alloc_name, uct_mem_h *memh_p)
{
    uct_cuda_copy_md_t *md = ucs_derived_of(uct_md, uct_cuda_copy_md_t);
    ucs_status_t status;
    uct_cuda_copy_alloc_handle_t *alloc_handle;
    ucs_log_level_t log_level;

    if ((mem_type != UCS_MEMORY_TYPE_CUDA_MANAGED) &&
        (mem_type != UCS_MEMORY_TYPE_CUDA)) {
        return UCS_ERR_UNSUPPORTED;
    }

    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                UCS_LOG_LEVEL_ERROR;

    if (!uct_cuda_base_is_context_active()) {
        ucs_log(log_level,
                "attempt to allocate cuda memory without active context");
        return UCS_ERR_NO_DEVICE;
    }

    alloc_handle = ucs_malloc(sizeof(*alloc_handle),
                              "uct_cuda_copy_alloc_handle_t");
    if (NULL == alloc_handle) {
        ucs_log(log_level,
                "failed to allocate memory for uct_cuda_copy_alloc_handle_t");
        return UCS_ERR_NO_MEMORY;
    }

    alloc_handle->length = *length_p;
    alloc_handle->is_vmm = 0;

    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
        if (md->config.enable_fabric != UCS_NO) {
            status = uct_cuda_copy_mem_alloc_fabric(md, alloc_handle, flags);
            if (status == UCS_OK) {
                goto allocated;
            } else {
                /* alloc_fabric failed so revert changes to alloc handle */
                alloc_handle->length = *length_p;
                alloc_handle->is_vmm = 0;
            }
        }

        if (md->config.enable_fabric != UCS_YES) {
            status = UCT_CUDADRV_FUNC(cuMemAlloc(&alloc_handle->ptr,
                                                 alloc_handle->length),
                                      log_level);
            if (status == UCS_OK) {
                goto allocated;
            }
        }

        ucs_log(log_level, "unable to allocate cuda memory of length %ld bytes",
                alloc_handle->length);
        status = UCS_ERR_NO_MEMORY;
    } else if (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED) {
        status = UCT_CUDADRV_FUNC(
                cuMemAllocManaged(&alloc_handle->ptr, alloc_handle->length,
                                  CU_MEM_ATTACH_GLOBAL), log_level);
    } else {
        ucs_log(log_level,
                "allocation mem_types supported: cuda, cuda-managed");
        status = UCS_ERR_INVALID_PARAM;
    }

    if (status != UCS_OK) {
        ucs_free(alloc_handle);
        return status;
    }

allocated:
    *memh_p    = alloc_handle;
    *address_p = (void*)alloc_handle->ptr;
    *length_p  = alloc_handle->length;
    return UCS_OK;
}

static ucs_status_t
uct_cuda_copy_mem_release_fabric(uct_cuda_copy_alloc_handle_t *alloc_handle)
{
#if HAVE_CUDA_FABRIC
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemRelease(alloc_handle->generic_handle));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemUnmap(alloc_handle->ptr, alloc_handle->length));
    if (status != UCS_OK) {
        return status;
    }

    return UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemAddressFree(alloc_handle->ptr, alloc_handle->length));
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static int uct_cuda_copy_detect_vmm(void *address,
                                    ucs_memory_type_t *vmm_mem_type,
                                    CUdevice *cuda_device)
{
#if HAVE_CUDA_FABRIC
    ucs_status_t status      = UCS_OK;
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle alloc_handle;

    /* Check if memory is allocated using VMM API and see if host memory needs
     * to be treated as pinned device memory */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuMemRetainAllocationHandle(&alloc_handle, (void*)address));
    if (status != UCS_OK) {
        return 0;
    }

    *vmm_mem_type = UCS_MEMORY_TYPE_UNKNOWN;
    *cuda_device  = CU_DEVICE_INVALID;

    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle));
    if (status != UCS_OK) {
        goto err;
    }

    *cuda_device = (CUdevice)prop.location.id;
    if ((prop.location.type == CU_MEM_LOCATION_TYPE_HOST) ||
        (prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA) ||
        (prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT)) {
        /* TODO: Marking as CUDA to allow cuda_ipc access vmm for now */
        *vmm_mem_type = UCS_MEMORY_TYPE_CUDA;
    } else if (prop.location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        *vmm_mem_type = UCS_MEMORY_TYPE_CUDA;
    }

err:
    UCT_CUDADRV_FUNC_LOG_DEBUG(cuMemRelease(alloc_handle));
    return 1;
#else
    return 0;
#endif
}

static ucs_status_t uct_cuda_copy_mem_free(uct_md_h md, uct_mem_h memh)
{
    uct_cuda_copy_alloc_handle_t *alloc_handle = (uct_cuda_copy_alloc_handle_t*)
            memh;
    ucs_status_t status;

    if (alloc_handle->is_vmm) {
        status = uct_cuda_copy_mem_release_fabric(alloc_handle);
    } else {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemFree(alloc_handle->ptr));
    }

    ucs_free(alloc_handle);
    return status;
}


static void uct_cuda_copy_md_close(uct_md_h uct_md) {
    uct_cuda_copy_md_t *md = ucs_derived_of(uct_md, uct_cuda_copy_md_t);

    ucs_free(md);
}

static size_t uct_cuda_copy_md_get_total_device_mem(CUdevice cuda_device)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    static size_t total_bytes[UCT_CUDA_MAX_DEVICES];
    char dev_name[UCT_CUDA_DEV_NAME_MAX_LEN];

    ucs_assert(cuda_device < UCT_CUDA_MAX_DEVICES);

    pthread_mutex_lock(&lock);

    if (!total_bytes[cuda_device]) {
        if (UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceTotalMem(&total_bytes[cuda_device],
                                                      cuda_device)) != UCS_OK) {
            goto err;
        }

        if (UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetName(dev_name, sizeof(dev_name),
                                                     cuda_device)) != UCS_OK) {
            goto err;
        }

        if (!strncmp(dev_name, "T4", 2)) {
            total_bytes[cuda_device] = 1; /* should ensure that whole alloc
                                             registration is not used for t4 */
        }
    }

    pthread_mutex_unlock(&lock);
    return total_bytes[cuda_device];

err:
    pthread_mutex_unlock(&lock);
    return 1; /* return 1 byte to avoid division by zero */
}

static void
uct_cuda_copy_sync_memops(uct_cuda_copy_md_t *md, const void *address)
{
#if HAVE_CUDA_FABRIC
    ucs_status_t status;
    if (!md->sync_memops_set) {
        /* Synchronize future DMA operations for all memory types */
        status = UCT_CUDADRV_FUNC_LOG_WARN(cuCtxSetFlags(CU_CTX_SYNC_MEMOPS));
        if (status == UCS_OK) {
            md->sync_memops_set = 1;
        }
    }
#else
    unsigned value = 1;
    /* Synchronize for DMA for legacy memory types*/
    UCT_CUDADRV_FUNC_LOG_WARN(
            cuPointerSetAttribute(&value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                  (CUdeviceptr)address));
#endif
}

static ucs_status_t
uct_cuda_copy_md_query_attributes(uct_cuda_copy_md_t *md, const void *address,
                                  size_t length, ucs_memory_info_t *mem_info)
{
#define UCT_CUDA_MEM_QUERY_NUM_ATTRS 4
    CUmemorytype cuda_mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed        = 0;
    CUdevice cuda_device       = -1;
    CUcontext cuda_mem_ctx     = NULL;
    CUpointer_attribute attr_type[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    void *attr_data[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    CUdeviceptr base_address;
    size_t alloc_length;
    size_t total_bytes;
    int32_t pref_loc;
    unsigned is_vmm;
    CUresult cu_err;
    ucs_status_t status;

    is_vmm = uct_cuda_copy_detect_vmm((void*)address, &mem_info->type,
                                      &cuda_device);
    if (is_vmm) {
        if (mem_info->type == UCS_MEMORY_TYPE_UNKNOWN) {
            return UCS_ERR_INVALID_ADDR;
        }
    } else {
        attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
        attr_data[0] = &cuda_mem_type;
        attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
        attr_data[1] = &is_managed;
        attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
        attr_data[2] = &cuda_device;
        attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
        attr_data[3] = &cuda_mem_ctx;

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuPointerGetAttributes(ucs_static_array_size(attr_data),
                                       attr_type, attr_data,
                                       (CUdeviceptr)address));
        if (status != UCS_OK) {
            /* pointer not recognized */
            return status;
        }

        if (cuda_mem_type != CU_MEMORYTYPE_DEVICE) {
            /* pointer not recognized */
            return UCS_ERR_INVALID_ADDR;
        }

        if (is_managed ||
            ((cuda_mem_ctx == NULL) && md->config.cuda_async_managed)) {
            /* is_managed: cuMemGetAddress range does not support managed memory
             * so use provided address and length as base address and alloc
             * length respectively
             *
             * cuda_async_managed: currently virtual/stream-ordered CUDA
             * allocations are typed as `UCS_MEMORY_TYPE_CUDA_MANAGED`. This may
             * be changed using UCX_CUDA_COPY_ASYNC_MEM_TYPE env var.
             * Ideally checking for
             * `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE` would be better
             * here, but due to a bug in the driver `cudaMalloc` also returns false
             * in that case. Therefore, checking whether the allocation was not
             * allocated in a context should also allows us to identify
             * virtual/stream-ordered CUDA allocations. */
            mem_info->type = UCS_MEMORY_TYPE_CUDA_MANAGED;

            cu_err = cuMemRangeGetAttribute(
                    (void*)&pref_loc, sizeof(pref_loc),
                    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                    (CUdeviceptr)address, length);
            if ((cu_err != CUDA_SUCCESS) || (pref_loc == CU_DEVICE_INVALID)) {
                pref_loc = (md->config.pref_loc == UCT_CUDA_PREF_LOC_CPU) ?
                                   CU_DEVICE_CPU :
                                   cuda_device;
            }

            if (pref_loc == CU_DEVICE_CPU) {
                mem_info->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
            } else {
                uct_cuda_base_get_sys_dev(pref_loc, &mem_info->sys_dev);
                if (mem_info->sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
                    ucs_diag("cu_device %d (for address %p...%p) unrecognized",
                             pref_loc, address,
                             UCS_PTR_BYTE_OFFSET(address, length));
                }
            }

            goto out_default_range;
        } else {
            mem_info->type = UCS_MEMORY_TYPE_CUDA;
        }
    }

    if (cuda_device == CU_DEVICE_CPU) {
        mem_info->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        goto out_default_range;
    }

    uct_cuda_base_get_sys_dev(cuda_device, &mem_info->sys_dev);
    if (mem_info->sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        return UCS_ERR_NO_DEVICE;
    }

    uct_cuda_copy_sync_memops(md, address);

    /* Extending the registration range is disable by configuration */
    if (md->config.alloc_whole_reg == UCS_CONFIG_OFF) {
        goto out_default_range;
    }

    cu_err = cuMemGetAddressRange(&base_address, &alloc_length,
                                  (CUdeviceptr)address);
    if (cu_err != CUDA_SUCCESS) {
        ucs_error("cuMemGetAddressRange(%p) error: %s", address,
                  uct_cuda_base_cu_get_error_string(cu_err));
        return UCS_ERR_INVALID_ADDR;
    }

    ucs_trace("query address %p: 0x%llx..0x%llx length %zu", address,
              base_address, base_address + alloc_length, alloc_length);

    if (md->config.alloc_whole_reg == UCS_CONFIG_AUTO) {
        total_bytes = uct_cuda_copy_md_get_total_device_mem(cuda_device);
        if (alloc_length > (total_bytes * md->config.max_reg_ratio)) {
            goto out_default_range;
        }
    } else {
        ucs_assert(md->config.alloc_whole_reg == UCS_CONFIG_ON);
    }

    mem_info->base_address = (void*)base_address;
    mem_info->alloc_length = alloc_length;
    return UCS_OK;

out_default_range:
    mem_info->base_address = (void*)address;
    mem_info->alloc_length = length;
    return UCS_OK;
}

static int uct_cuda_copy_md_get_dmabuf_fd(uintptr_t address, size_t length)
{
#if CUDA_VERSION >= 11070
    PFN_cuMemGetHandleForAddressRange get_handle_func;
    CUresult cu_err;
    int fd;

    /* Get fxn ptr for cuMemGetHandleForAddressRange in case installed libcuda
     * does not have the definition for it even though 11.7 header includes the
     * declaration and avoid link error */
#if CUDA_VERSION >= 12000
    CUdriverProcAddressQueryResult proc_addr_res;
    cu_err = cuGetProcAddress("cuMemGetHandleForAddressRange",
                              (void**)&get_handle_func, 12000,
                              CU_GET_PROC_ADDRESS_DEFAULT, &proc_addr_res);
    if ((cu_err != CUDA_SUCCESS) ||
        (proc_addr_res != CU_GET_PROC_ADDRESS_SUCCESS)) {
        ucs_debug("cuMemGetHandleForAddressRange not found");
        return UCT_DMABUF_FD_INVALID;
    }
#else
    cu_err = cuGetProcAddress("cuMemGetHandleForAddressRange",
                              (void**)&get_handle_func, 11070,
                              CU_GET_PROC_ADDRESS_DEFAULT);
    if (cu_err != CUDA_SUCCESS) {
        ucs_debug("cuMemGetHandleForAddressRange not found");
        return UCT_DMABUF_FD_INVALID;
    }
#endif

    cu_err = get_handle_func((void*)&fd, address, length,
                             CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (cu_err == CUDA_SUCCESS) {
        ucs_trace("dmabuf for address 0x%lx length %zu is fd %d", address,
                  length, fd);
        return fd;
    }

    ucs_debug("cuMemGetHandleForAddressRange(address=0x%lx length=%zu "
              "DMA_BUF_FD) failed: %s",
              address, length, uct_cuda_base_cu_get_error_string(cu_err));
#endif
    return UCT_DMABUF_FD_INVALID;
}

static ucs_status_t
uct_cuda_copy_md_mem_query(uct_md_h tl_md, const void *address, size_t length,
                           uct_md_mem_attr_t *mem_attr)
{
    ucs_memory_info_t default_mem_info = {
        .type         = UCS_MEMORY_TYPE_HOST,
        .sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN,
        .base_address = (void*)address,
        .alloc_length = length
    };
    uct_cuda_copy_md_t *md = ucs_derived_of(tl_md, uct_cuda_copy_md_t);
    uintptr_t base_address, aligned_start, aligned_end;
    ucs_memory_info_t addr_mem_info;
    ucs_status_t status;

    if (!(mem_attr->field_mask &
          (UCT_MD_MEM_ATTR_FIELD_MEM_TYPE | UCT_MD_MEM_ATTR_FIELD_SYS_DEV |
           UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS |
           UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH |
           UCT_MD_MEM_ATTR_FIELD_DMABUF_FD |
           UCT_MD_MEM_ATTR_FIELD_DMABUF_OFFSET))) {
        return UCS_OK;
    }

    if (address != NULL) {
        status = uct_cuda_copy_md_query_attributes(md, address, length,
                                                   &addr_mem_info);
        if (status != UCS_OK) {
            return status;
        }

        ucs_memtype_cache_update(addr_mem_info.base_address,
                                 addr_mem_info.alloc_length, addr_mem_info.type,
                                 addr_mem_info.sys_dev);
    } else {
        addr_mem_info = default_mem_info;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr->mem_type = addr_mem_info.type;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr->sys_dev = addr_mem_info.sys_dev;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS) {
        mem_attr->base_address = addr_mem_info.base_address;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH) {
        mem_attr->alloc_length = addr_mem_info.alloc_length;
    }

    base_address  = (uintptr_t)addr_mem_info.base_address;
    aligned_start = ucs_align_down_pow2(base_address, ucs_get_page_size());

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_FD) {
        aligned_end = ucs_align_up_pow2(base_address +
                                                addr_mem_info.alloc_length,
                                        ucs_get_page_size());

        mem_attr->dmabuf_fd = uct_cuda_copy_md_get_dmabuf_fd(
                aligned_start, aligned_end - aligned_start);
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_OFFSET) {
        mem_attr->dmabuf_offset = (uintptr_t)address - aligned_start;
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_md_detect_memory_type,
                 (md, address, length, mem_type_p), uct_md_h md,
                 const void *address, size_t length,
                 ucs_memory_type_t *mem_type_p)
{
    uct_md_mem_attr_t mem_attr;
    ucs_status_t status;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;

    status = uct_cuda_copy_md_mem_query(md, address, length, &mem_attr);
    if (status != UCS_OK) {
        return status;
    }

    *mem_type_p = mem_attr.mem_type;
    return UCS_OK;
}

static uct_md_ops_t md_ops = {
    .close              = uct_cuda_copy_md_close,
    .query              = uct_cuda_copy_md_query,
    .mem_alloc          = uct_cuda_copy_mem_alloc,
    .mem_free           = uct_cuda_copy_mem_free,
    .mkey_pack          = uct_cuda_copy_mkey_pack,
    .mem_reg            = uct_cuda_copy_mem_reg,
    .mem_dereg          = uct_cuda_copy_mem_dereg,
    .mem_attach         = ucs_empty_function_return_unsupported,
    .mem_query          = uct_cuda_copy_md_mem_query,
    .detect_memory_type = uct_cuda_copy_md_detect_memory_type
};

static ucs_status_t
uct_cuda_copy_md_open(uct_component_t *component, const char *md_name,
                      const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_cuda_copy_md_config_t *config = ucs_derived_of(md_config,
                                                       uct_cuda_copy_md_config_t);
    uct_cuda_copy_md_t *md;
    int dmabuf_supported;
    ucs_status_t status;

    md = ucs_malloc(sizeof(uct_cuda_copy_md_t), "uct_cuda_copy_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_cuda_copy_md_t");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    md->super.ops               = &md_ops;
    md->super.component         = &uct_cuda_copy_component;
    md->config.alloc_whole_reg  = config->alloc_whole_reg;
    md->config.max_reg_ratio    = config->max_reg_ratio;
    md->config.pref_loc         = config->pref_loc;
    md->config.enable_fabric    = config->enable_fabric;
    md->config.dmabuf_supported = 0;
    md->sync_memops_set         = 0;
    md->granularity             = SIZE_MAX;

    if ((config->cuda_async_mem_type != UCS_MEMORY_TYPE_CUDA) &&
        (config->cuda_async_mem_type != UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        ucs_warn("wrong memory type for async memory allocations: \"%s\";"
                " cuda-managed will be used instead",
                ucs_memory_type_names[config->cuda_async_mem_type]);
    }

    md->config.cuda_async_managed =
                          (config->cuda_async_mem_type != UCS_MEMORY_TYPE_CUDA);

    dmabuf_supported = uct_cuda_copy_md_is_dmabuf_supported();
    if ((config->enable_dmabuf == UCS_YES) && !dmabuf_supported) {
        ucs_error("dmabuf support requested but not found");
        status = UCS_ERR_UNSUPPORTED;
        goto err_free_md;
    }

    if (config->enable_dmabuf != UCS_NO) {
        md->config.dmabuf_supported = dmabuf_supported;
    }

    *md_p = (uct_md_h)md;

    return UCS_OK;

err_free_md:
    ucs_free(md);
err:
    return status;
}

uct_component_t uct_cuda_copy_component = {
    .query_md_resources = uct_cuda_base_query_md_resources,
    .md_open            = uct_cuda_copy_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_cuda_copy_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_cuda_copy_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "cuda_cpy",
    .md_config          = {
        .name           = "Cuda-copy memory domain",
        .prefix         = "CUDA_COPY_",
        .table          = uct_cuda_copy_md_config_table,
        .size           = sizeof(uct_cuda_copy_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_cuda_copy_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_cuda_copy_component);
