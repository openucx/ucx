/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
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
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>
#include <uct/cuda/base/cuda_iface.h>
#include <cuda_runtime.h>
#include <cuda.h>


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

static ucs_status_t uct_cuda_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->cap.alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->cap.detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->cap.max_alloc        = SIZE_MAX;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->rkey_packed_size     = 0;
    md_attr->reg_cost             = ucs_linear_func_make(0, 0);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t
uct_cuda_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                        const uct_md_mkey_pack_params_t *params,
                        void *rkey_buffer)
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
                 (md, address, length, flags, memh_p),
                 uct_md_h md, void *address, size_t length,
                 unsigned flags, uct_mem_h *memh_p)
{
    ucs_log_level_t log_level;
    CUmemorytype memType;
    CUresult result;
    ucs_status_t status;

    if (address == NULL) {
        *memh_p = address;
        return UCS_OK;
    }

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

static uct_md_ops_t md_ops = {
    .close                  = uct_cuda_copy_md_close,
    .query                  = uct_cuda_copy_md_query,
    .mem_alloc              = uct_cuda_copy_mem_alloc,
    .mem_free               = uct_cuda_copy_mem_free,
    .mkey_pack              = uct_cuda_copy_mkey_pack,
    .mem_reg                = uct_cuda_copy_mem_reg,
    .mem_dereg              = uct_cuda_copy_mem_dereg,
    .mem_query              = uct_cuda_base_mem_query,
    .is_sockaddr_accessible = ucs_empty_function_return_zero_int,
    .detect_memory_type     = uct_cuda_base_detect_memory_type
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
