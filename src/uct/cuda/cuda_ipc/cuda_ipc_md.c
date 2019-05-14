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
#include <sys/types.h>
#include <unistd.h>


static ucs_config_field_t uct_cuda_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_cuda_ipc_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_CUDA);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_CUDA;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = UCT_CUDA_IPC_MAX_ALLOC_SZ;
    md_attr->rkey_packed_size  = sizeof(uct_cuda_ipc_key_t);
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_cuda_ipc_key_t *packed   = (uct_cuda_ipc_key_t *) rkey_buffer;
    uct_cuda_ipc_key_t *mem_hndl = (uct_cuda_ipc_key_t *) memh;

    *packed = *mem_hndl;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_rkey_unpack(uct_md_component_t *mdc,
                                             const void *rkey_buffer, uct_rkey_t *rkey_p,
                                             void **handle_p)
{
    uct_cuda_ipc_key_t *packed = (uct_cuda_ipc_key_t *) rkey_buffer;
    uct_cuda_ipc_md_t *md = mdc->priv; /* cuda_ipc_mdc->priv points to uct_md */
    uct_cuda_ipc_key_t *key;
    ucs_status_t status;
    CUdevice cu_device;
    CUdevice peer_cu_device;
    int peer_accessble;
    int i;

    UCT_CUDA_IPC_GET_DEVICE(cu_device);

    peer_cu_device = -1;
    for (i = 0; i < md->uuid_map_len; i++) {
        
        if (!memcmp((void *) &(md->uuid_map[i]), (void *) &packed->uuid, sizeof(packed->uuid))) {
            peer_cu_device = i;
        }
    }
    
    status = UCT_CUDADRV_FUNC(cuDeviceCanAccessPeer(&peer_accessble,
                                                    cu_device, peer_cu_device));
    if ((status != UCS_OK) || (peer_accessble == 0)) {
        return UCS_ERR_UNREACHABLE;
    }

    key = ucs_malloc(sizeof(uct_cuda_ipc_key_t), "uct_cuda_ipc_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    *key      = *packed;
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
uct_cuda_ipc_mem_reg_internal(uct_md_h uct_md, void *addr, size_t length,
                              unsigned flags, uct_cuda_ipc_key_t *key)
{
    CUdevice cu_device;
    ucs_status_t status;

    if (!length) {
        return UCS_OK;
    }

    status = UCT_CUDADRV_FUNC(cuIpcGetMemHandle(&(key->ph), (CUdeviceptr) addr));
    if (UCS_OK != status) {
        return status;
    }

    UCT_CUDA_IPC_GET_DEVICE(cu_device);

    UCT_CUDADRV_FUNC(cuMemGetAddressRange(&(key->d_bptr),
                                          &(key->b_len),
                                          (CUdeviceptr) addr));
    key->dev_num  = (int) cu_device;

    status = UCT_CUDADRV_FUNC(cuDeviceGetUuid(&(key->uuid), key->dev_num));
    if (UCS_OK != status) {
        return status;
    }
    
    ucs_trace("registered memory:%p..%p length:%lu dev_num:%d",
              addr, addr + length, length, (int) cu_device);
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_cuda_ipc_key_t *key;

    key = ucs_malloc(sizeof(uct_cuda_ipc_key_t), "uct_cuda_ipc_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    if (UCS_OK != uct_cuda_ipc_mem_reg_internal(md, address, length, 0, key)) {
        ucs_free(key);
        return UCS_ERR_IO_ERROR;
    }
    *memh_p = key;

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
    int num_gpus;
    cudaError_t cudaErr;

    cudaErr = cudaGetDeviceCount(&num_gpus);
    if ((cudaErr!= cudaSuccess) || (num_gpus == 0)) {
        ucs_debug("Not found cuda devices");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    return uct_single_md_resource(&uct_cuda_ipc_md_component, resources_p, num_resources_p);
}

static void uct_cuda_ipc_md_close(uct_md_h uct_md)
{
    uct_cuda_ipc_md_t *md = ucs_derived_of(uct_md, uct_cuda_ipc_md_t);
    if (NULL != md->uuid_map) {
        ucs_free(md->uuid_map);
    }
    ucs_free(md);
}

static ucs_status_t uct_cuda_ipc_md_open(const char *md_name, const uct_md_config_t *md_config,
                                         uct_md_h *md_p)
{
    uct_cuda_ipc_md_t *cuda_ipc_md;
    ucs_status_t status;
    int device_count;
    int i;
    CUuuid own_uuid;
    CUdevice own_cu_device;

    cuda_ipc_md = ucs_malloc(sizeof(*cuda_ipc_md), "uct_cuda_ipc_md_t");
    if (cuda_ipc_md == NULL) {
        ucs_error("Failed to allocate memory for uct_cuda_ipc_md_t");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }
    
    cuda_ipc_md->uuid_map = NULL;
    cuda_ipc_md->uuid_map_len = -1;
        
    status = UCT_CUDADRV_FUNC(cuDeviceGetCount(&device_count));
    if (status != UCS_OK) {
	return UCS_ERR_IO_ERROR;
    }
        
    cuda_ipc_md->uuid_map = ucs_malloc(sizeof(CUuuid) * device_count, "uct_cuda_ipc_uuid_map");
    if (NULL == cuda_ipc_md->uuid_map) {
	ucs_error("failed to allocate memory for uct_cuda_ipc_uuid_map");
	return UCS_ERR_NO_MEMORY;
    }
        
    for (i = 0; i < device_count; i++) {

	status = UCT_CUDADRV_FUNC(cuDeviceGet(&own_cu_device, i));
	if (status != UCS_OK) {
	    return UCS_ERR_IO_ERROR;
	}
        
	status = UCT_CUDADRV_FUNC(cuDeviceGetUuid(&own_uuid, own_cu_device));
	if (status != UCS_OK) {
	    return UCS_ERR_IO_ERROR;
	}
            
	cuda_ipc_md->uuid_map[i] = own_uuid;
    }
        
    cuda_ipc_md->uuid_map_len = device_count;
    
    static uct_md_ops_t md_ops = {
        .close        = uct_cuda_ipc_md_close,
        .query        = uct_cuda_ipc_md_query,
        .mkey_pack    = uct_cuda_ipc_mkey_pack,
        .mem_reg      = uct_cuda_ipc_mem_reg,
        .mem_dereg    = uct_cuda_ipc_mem_dereg,
        .is_mem_type_owned = uct_cuda_is_mem_type_owned,
    };

    cuda_ipc_md->super.ops = &md_ops;
    cuda_ipc_md->super.component = &uct_cuda_ipc_md_component;
    cuda_ipc_md->super.component->priv = (void *) cuda_ipc_md;

    *md_p = &cuda_ipc_md->super;
    return UCS_OK;
    
err:
    return status;
}

UCT_MD_COMPONENT_DEFINE(uct_cuda_ipc_md_component, UCT_CUDA_IPC_MD_NAME,
                        uct_cuda_ipc_query_md_resources, uct_cuda_ipc_md_open, NULL,
                        uct_cuda_ipc_rkey_unpack, uct_cuda_ipc_rkey_release, "CUDA_IPC_",
                        uct_cuda_ipc_md_config_table, uct_cuda_ipc_md_config_t);
