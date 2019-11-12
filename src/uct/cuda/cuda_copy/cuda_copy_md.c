/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cuda_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <uct/cuda/base/cuda_iface.h>
#include <cuda_runtime.h>
#include <cuda.h>


static ucs_config_field_t uct_cuda_copy_md_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(uct_cuda_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_cuda_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags            = UCT_MD_FLAG_REG;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.access_mem_type  = UCS_MEMORY_TYPE_CUDA;
    md_attr->cap.detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->rkey_packed_size     = 0;
    md_attr->reg_cost.overhead    = 0;
    md_attr->reg_cost.growth      = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
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

static ucs_status_t uct_cuda_copy_mem_reg(uct_md_h md, void *address, size_t length,
                                          unsigned flags, uct_mem_h *memh_p)
{
    CUmemorytype memType;
    CUresult result;
    ucs_status_t status;

    if (address == NULL) {
        *memh_p = address;
        return UCS_OK;
    }

    result = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)(address));
    if ((result == CUDA_SUCCESS) && (memType == CU_MEMORYTYPE_HOST)) {
        /* memory is allocated with cudaMallocHost which is already registered */
        *memh_p = NULL;
        return UCS_OK;
    }

    status = UCT_CUDA_FUNC(cudaHostRegister(address, length,
                                            cudaHostRegisterPortable));
    if (status != UCS_OK) {
        return status;
    }

    *memh_p = address;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    void *address = (void *)memh;
    ucs_status_t status;

    if (address == NULL) {
        return UCS_OK;
    }

    status = UCT_CUDA_FUNC(cudaHostUnregister(address));
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static void uct_cuda_copy_md_close(uct_md_h uct_md) {
    uct_cuda_copy_md_t *md = ucs_derived_of(uct_md, uct_cuda_copy_md_t);

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close               = uct_cuda_copy_md_close,
    .query               = uct_cuda_copy_md_query,
    .mkey_pack           = uct_cuda_copy_mkey_pack,
    .mem_reg             = uct_cuda_copy_mem_reg,
    .mem_dereg           = uct_cuda_copy_mem_dereg,
    .detect_memory_type  = uct_cuda_base_detect_memory_type,
};

static ucs_status_t
uct_cuda_copy_md_open(uct_component_t *component, const char *md_name,
                      const uct_md_config_t *config, uct_md_h *md_p)
{
    uct_cuda_copy_md_t *md;

    md = ucs_malloc(sizeof(uct_cuda_copy_md_t), "uct_cuda_copy_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_cuda_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_cuda_copy_component;
    *md_p               = (uct_md_h)md;
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
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_cuda_copy_component),
    .flags              = 0
};
UCT_COMPONENT_REGISTER(&uct_cuda_copy_component);

