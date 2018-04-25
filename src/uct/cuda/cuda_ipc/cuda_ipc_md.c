/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include "cuda_ipc_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static ucs_config_field_t uct_cuda_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},
    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* Using registration cost for size 0 */
     ucs_offsetof(uct_cuda_ipc_md_config_t, uc_reg_cost.overhead), UCS_CONFIG_TYPE_TIME},
    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate",
     ucs_offsetof(uct_cuda_ipc_md_config_t, uc_reg_cost.growth), UCS_CONFIG_TYPE_TIME},
    {NULL}
};

static ucs_status_t uct_cuda_ipc_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_CUDA);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_CUDA;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = UCT_CUDA_IPC_MAX_ALLOC_SZ; /*FIX THIS*/
    md_attr->rkey_packed_size  = sizeof(uct_cuda_ipc_key_t);;
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_cuda_ipc_key_t *packed   = (uct_cuda_ipc_key_t *) rkey_buffer;
    uct_cuda_ipc_mem_t *mem_hndl = (uct_cuda_ipc_mem_t *) memh;

    packed->ph         = mem_hndl->ph;
    packed->d_rem_ptr  = mem_hndl->d_ptr;
    packed->d_rem_bptr = mem_hndl->d_bptr;
    packed->b_rem_len  = mem_hndl->b_len;
    packed->dev_num    = mem_hndl->dev_num;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_rkey_unpack(uct_md_component_t *mdc,
                                             const void *rkey_buffer, uct_rkey_t *rkey_p,
                                             void **handle_p)
{
    uct_cuda_ipc_key_t *packed = (uct_cuda_ipc_key_t *) rkey_buffer;
    uct_cuda_ipc_key_t *key;
    ucs_status_t status;
    CUdevice cu_device;

    status = UCT_CUDADRV_FUNC(cuCtxGetDevice(&cu_device));
    if (UCS_OK != status) {
        return status;
    }

    key = ucs_malloc(sizeof(uct_cuda_ipc_key_t), "uct_cuda_ipc_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }
    *key = *packed;
    *handle_p = NULL;
    *rkey_p   = (uintptr_t) key;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                              void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_mem_reg_internal(uct_md_h uct_md, void *address, size_t length,
                              unsigned flags, uct_cuda_ipc_mem_t *mem_hndl)
{
    CUdevice cu_device;
    ucs_status_t status;

    if (!length) {
        return UCS_OK;
    }
    status = UCT_CUDADRV_FUNC(cuIpcGetMemHandle(&(mem_hndl->ph), (CUdeviceptr) address));
    if (UCS_OK != status) {
        return status;
    }

    /* TODO: There are limitations when process has >1 contexts */

    status = UCT_CUDADRV_FUNC(cuCtxGetDevice(&cu_device));
    if (UCS_OK != status) {
        return status;
    }

    UCT_CUDADRV_FUNC(cuMemGetAddressRange(&(mem_hndl->d_bptr), &(mem_hndl->b_len),
                                          (CUdeviceptr) address));
    mem_hndl->d_ptr    = (CUdeviceptr) address;
    mem_hndl->reg_size = length;
    mem_hndl->dev_num  = (int) cu_device;
    ucs_trace("registered memory:%p..%p length:%lu d_ptr:%p dev_num:%d",
              address, address + length, length, address, (int) cu_device);
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_cuda_ipc_mem_t *mem_hndl = NULL;
    ucs_status_t status;

    mem_hndl = ucs_malloc(sizeof(uct_cuda_ipc_mem_t), "cuda_ipc handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for cuda_ipc_mem_t");
        return UCS_ERR_NO_MEMORY;
    }
    status = uct_cuda_ipc_mem_reg_internal(md, address, length, 0, mem_hndl);
    if (status != UCS_OK) {
        ucs_free(mem_hndl);
        return status;
    }
    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    ucs_free(memh);
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                    unsigned *num_resources_p)
{
    ucs_status_t status;
    int num_gpus;

    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&num_gpus));
    if ((UCS_OK != status) || (num_gpus == 0)) {
        ucs_debug("not found cuda devices");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }
    return uct_single_md_resource(&uct_cuda_ipc_md_component, resources_p, num_resources_p);
}

static ucs_status_t uct_cuda_ipc_md_open(const char *md_name, const uct_md_config_t *md_config,
                                         uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close             = (void*)ucs_empty_function,
        .query             = uct_cuda_ipc_md_query,
        .mkey_pack         = uct_cuda_ipc_mkey_pack,
        .mem_reg           = uct_cuda_ipc_mem_reg,
        .mem_dereg         = uct_cuda_ipc_mem_dereg,
        .is_mem_type_owned = uct_cuda_is_mem_type_owned,
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_cuda_ipc_md_component
    };
    *md_p = &md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_cuda_ipc_md_component, UCT_CUDA_IPC_MD_NAME,
                        uct_cuda_ipc_query_md_resources, uct_cuda_ipc_md_open, NULL,
                        uct_cuda_ipc_rkey_unpack, uct_cuda_ipc_rkey_release, "CUDA_IPC_",
                        uct_cuda_ipc_md_config_table, uct_cuda_ipc_md_config_t);
