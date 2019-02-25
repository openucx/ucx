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
#include <cuda_runtime.h>
#include <cuda.h>


static ucs_config_field_t uct_cuda_copy_md_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(uct_cuda_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_cuda_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_CUDA;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->rkey_packed_size  = 0;
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                            void *rkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_rkey_unpack(uct_md_component_t *mdc,
                                              const void *rkey_buffer, uct_rkey_t *rkey_p,
                                              void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                               void *handle)
{
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_reg(uct_md_h md, void *address, size_t length,
                                          unsigned flags, uct_mem_h *memh_p)
{
    cudaError_t cuerr = cudaSuccess;

    if(address == NULL) {
        *memh_p = address;
        return UCS_OK;
    }

    cuerr = cudaHostRegister(address, length, cudaHostRegisterPortable);
    if (cuerr != cudaSuccess) {
        return UCS_ERR_IO_ERROR;
    }

    *memh_p = address;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    void *address = (void *)memh;
    cudaError_t cuerr;

    if (address == NULL) {
        return UCS_OK;
    }
    cuerr = cudaHostUnregister(address);
    if (cuerr != cudaSuccess) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                     unsigned *num_resources_p)
{
    int num_gpus;
    cudaError_t cudaErr;

    cudaErr = cudaGetDeviceCount(&num_gpus);
    if ((cudaErr!= cudaSuccess) || (num_gpus == 0)) {
        ucs_debug("Not found cuda devices");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    return uct_single_md_resource(&uct_cuda_copy_md_component, resources_p, num_resources_p);
}

static void uct_cuda_copy_md_close(uct_md_h uct_md) {
    uct_cuda_copy_md_t *md = ucs_derived_of(uct_md, uct_cuda_copy_md_t);

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close              = uct_cuda_copy_md_close,
    .query              = uct_cuda_copy_md_query,
    .mkey_pack          = uct_cuda_copy_mkey_pack,
    .mem_reg            = uct_cuda_copy_mem_reg,
    .mem_dereg          = uct_cuda_copy_mem_dereg,
    .is_mem_type_owned  = uct_cuda_is_mem_type_owned,
};

static ucs_status_t uct_cuda_copy_md_open(const char *md_name, const uct_md_config_t *md_config,
                                          uct_md_h *md_p)
{
    uct_cuda_copy_md_t *md;

    md = ucs_malloc(sizeof(uct_cuda_copy_md_t), "uct_cuda_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_cuda_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_cuda_copy_md_component;

    *md_p = (uct_md_h) md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_cuda_copy_md_component, UCT_CUDA_COPY_MD_NAME,
                        uct_cuda_copy_query_md_resources, uct_cuda_copy_md_open, NULL,
                        uct_cuda_copy_rkey_unpack, uct_cuda_copy_rkey_release, "CUDA_COPY_",
                        uct_cuda_copy_md_config_table, uct_cuda_copy_md_config_t,
                        ucs_empty_function_return_unsupported);
