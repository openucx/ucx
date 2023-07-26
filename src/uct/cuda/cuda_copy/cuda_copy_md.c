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
#include <ucs/sys/math.h>
#include <uct/cuda/base/cuda_iface.h>
#include <uct/api/v2/uct_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#if CUDA_VERSION >= 11070
#include <cudaTypedefs.h>
#endif


#define UCT_CUDA_DEV_NAME_MAX_LEN 64
#define UCT_CUDA_MAX_DEVICES      32


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
uct_cuda_copy_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    md_attr->flags                  = UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC;
    md_attr->reg_mem_types          = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                      UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                      UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->reg_nonblock_mem_types = 0;
    md_attr->cache_mem_types        = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                      UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->alloc_mem_types        = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                      UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->access_mem_types       = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                      UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->detect_mem_types       = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                      UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->dmabuf_mem_types       = 0;
    if (uct_cuda_copy_md_is_dmabuf_supported()) {
        md_attr->dmabuf_mem_types |= UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    }
    md_attr->max_alloc        = SIZE_MAX;
    md_attr->max_reg          = ULONG_MAX;
    md_attr->rkey_packed_size = 0;
    md_attr->reg_cost         = UCS_LINEAR_FUNC_ZERO;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t
uct_cuda_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                        const uct_md_mkey_pack_params_t *params,
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

    result = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)(address));
    if ((result == CUDA_SUCCESS) && ((memType == CU_MEMORYTYPE_HOST)    ||
                                     (memType == CU_MEMORYTYPE_UNIFIED) ||
                                     (memType == CU_MEMORYTYPE_DEVICE))) {
        /* only host memory not allocated by cuda needs to be registered */
        /* using deadbeef as VA to avoid gtest error */
        UCS_STATIC_ASSERT((uint64_t)0xdeadbeef != (uint64_t)UCT_MEM_HANDLE_NULL);
        *memh_p = (void *)0xdeadbeef;
        return UCS_OK;
    }

    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                UCS_LOG_LEVEL_ERROR;
    status    = UCT_CUDA_FUNC(cudaHostRegister(address, length,
                                               cudaHostRegisterPortable),
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

    status = UCT_CUDA_FUNC_LOG_ERR(cudaHostUnregister(address));
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_alloc(uct_md_h md, size_t *length_p,
                                            void **address_p,
                                            ucs_memory_type_t mem_type,
                                            unsigned flags,
                                            const char *alloc_name,
                                            uct_mem_h *memh_p)
{
    ucs_status_t status;

    if ((mem_type != UCS_MEMORY_TYPE_CUDA_MANAGED) &&
        (mem_type != UCS_MEMORY_TYPE_CUDA)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!uct_cuda_base_is_context_active()) {
        ucs_error("attempt to allocate cuda memory without active context");
        return UCS_ERR_NO_DEVICE;
    }

    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemAlloc((CUdeviceptr*)address_p,
                                                     *length_p));
    } else {
        status =
            UCT_CUDADRV_FUNC_LOG_ERR(cuMemAllocManaged((CUdeviceptr*)address_p,
                                                       *length_p,
                                                       CU_MEM_ATTACH_GLOBAL));
    }

    if (status != UCS_OK) {
        return status;
    }

    *memh_p = *address_p;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_free(uct_md_h md, uct_mem_h memh)
{
    return UCT_CUDADRV_FUNC_LOG_ERR(cuMemFree((CUdeviceptr)memh));
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

static ucs_status_t
uct_cuda_copy_md_query_attributes(uct_cuda_copy_md_t *md, const void *address,
                                  size_t length, ucs_memory_info_t *mem_info)
{
#define UCT_CUDA_MEM_QUERY_NUM_ATTRS 4
    CUmemorytype cuda_mem_mype = (CUmemorytype)0;
    uint32_t is_managed        = 0;
    CUdevice cuda_device       = -1;
    CUcontext cuda_mem_ctx     = NULL;
    CUpointer_attribute attr_type[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    void *attr_data[UCT_CUDA_MEM_QUERY_NUM_ATTRS];
    const char *cu_err_str;
    CUdeviceptr base_address;
    size_t alloc_length;
    ucs_status_t status;
    size_t total_bytes;
    CUresult cu_err;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &cuda_mem_mype;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &cuda_device;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &cuda_mem_ctx;

    cu_err = cuPointerGetAttributes(ucs_static_array_size(attr_data), attr_type,
                                    attr_data, (CUdeviceptr)address);
    if ((cu_err != CUDA_SUCCESS) || (cuda_mem_mype != CU_MEMORYTYPE_DEVICE)) {
        /* pointer not recognized */
        return UCS_ERR_INVALID_ADDR;
    }

    status = uct_cuda_base_get_sys_dev(cuda_device, &mem_info->sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    if (is_managed || (cuda_mem_ctx == NULL)) {
        /* is_managed: cuMemGetAddress range does not support managed memory so
         * use provided address and length as base address and alloc length
         * respectively.
         *
         * cuda_mem_ctx == NULL: currently virtual/stream-ordered CUDA
         * allocations are typed as `UCS_MEMORY_TYPE_CUDA_MANAGED`. This may
         * change in the future. Ideally checking for
         * `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE` would be better
         * here, but due to a bug in the driver `cudaMalloc` also returns false
         * in that case. Therefore, checking whether the allocation was not
         * allocated in a context should also allows us to identify
         * virtual/stream-ordered CUDA allocations. */
        mem_info->type = UCS_MEMORY_TYPE_CUDA_MANAGED;
        goto out_default_range;
    }

    mem_info->type = UCS_MEMORY_TYPE_CUDA;

    /* Extending the registration range is disable by configuration */
    if (md->config.alloc_whole_reg == UCS_CONFIG_OFF) {
        goto out_default_range;
    }

    cu_err = cuMemGetAddressRange(&base_address, &alloc_length,
                                  (CUdeviceptr)address);
    if (cu_err != CUDA_SUCCESS) {
        cuGetErrorString(cu_err, &cu_err_str);
        ucs_error("cuMemGetAddressRange(%p) error: %s", address, cu_err_str);
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
    const char *cu_err_str;
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

    cuGetErrorString(cu_err, &cu_err_str);
    ucs_debug("cuMemGetHandleForAddressRange(address=0x%lx length=%zu "
              "DMA_BUF_FD) failed: %s",
              address, length, cu_err_str);
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
    unsigned value         = 1;
    uintptr_t base_address, aligned_start, aligned_end;
    ucs_memory_info_t addr_mem_info;
    const char *cu_err_str;
    ucs_status_t status;
    CUresult cu_err;

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

        /* Synchronize for DMA */
        cu_err = cuPointerSetAttribute(&value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       (CUdeviceptr)address);
        if (cu_err != CUDA_SUCCESS) {
            cuGetErrorString(cu_err, &cu_err_str);
            ucs_warn("cuPointerSetAttribute(%p, SYNC_MEMOPS) error: %s",
                     address, cu_err_str);
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
    .close                  = uct_cuda_copy_md_close,
    .query                  = uct_cuda_copy_md_query,
    .mem_alloc              = uct_cuda_copy_mem_alloc,
    .mem_free               = uct_cuda_copy_mem_free,
    .mkey_pack              = uct_cuda_copy_mkey_pack,
    .mem_reg                = uct_cuda_copy_mem_reg,
    .mem_dereg              = uct_cuda_copy_mem_dereg,
    .mem_attach             = ucs_empty_function_return_unsupported,
    .mem_query              = uct_cuda_copy_md_mem_query,
    .is_sockaddr_accessible = ucs_empty_function_return_zero_int,
    .detect_memory_type     = uct_cuda_copy_md_detect_memory_type
};

static ucs_status_t
uct_cuda_copy_md_open(uct_component_t *component, const char *md_name,
                      const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_cuda_copy_md_config_t *config = ucs_derived_of(md_config,
                                                       uct_cuda_copy_md_config_t);
    uct_cuda_copy_md_t *md;

    md = ucs_malloc(sizeof(uct_cuda_copy_md_t), "uct_cuda_copy_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_cuda_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops              = &md_ops;
    md->super.component        = &uct_cuda_copy_component;
    md->config.alloc_whole_reg = config->alloc_whole_reg;
    md->config.max_reg_ratio   = config->max_reg_ratio;
    *md_p                      = (uct_md_h)md;

    return UCS_OK;
}

uct_component_t uct_cuda_copy_component = {
    .query_md_resources = uct_cuda_base_query_md_resources,
    .md_open            = uct_cuda_copy_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_cuda_copy_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_cuda_copy_rkey_release,
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
