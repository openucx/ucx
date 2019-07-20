/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_ipc_md.h"

#include <uct/rocm/base/rocm_base.h>

static ucs_config_field_t uct_rocm_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rocm_ipc_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_rocm_ipc_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                    unsigned *num_resources_p)
{
    if (uct_rocm_base_init() != HSA_STATUS_SUCCESS) {
        ucs_debug("Could not initialize ROCm support");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    return uct_single_md_resource(&uct_rocm_ipc_md_component, resources_p,
				  num_resources_p);
}

static ucs_status_t uct_rocm_ipc_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->rkey_packed_size     = sizeof(uct_rocm_ipc_key_t);
    md_attr->cap.flags            = UCT_MD_FLAG_REG |
                                    UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cap.access_mem_type  = UCS_MEMORY_TYPE_ROCM;
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;

    /* TODO: get accurate number */
    md_attr->reg_cost.overhead    = 9e-9;
    md_attr->reg_cost.growth      = 0;

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_mkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_rocm_ipc_key_t *packed   = (uct_rocm_ipc_key_t *) rkey_buffer;
    uct_rocm_ipc_key_t *key = (uct_rocm_ipc_key_t *) memh;

    *packed = *key;

    return UCS_OK;
}

static hsa_status_t uct_rocm_ipc_pack_key(void *address, size_t length,
                                          uct_rocm_ipc_key_t *key)
{
    hsa_status_t status;
    hsa_agent_t agent;
    void *base_ptr;
    size_t size;

    status = uct_rocm_base_get_ptr_info(address, length, &base_ptr, &size, &agent);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("pack none ROCM ptr %p/%lx", address, length);
        return status;
    }

    status = hsa_amd_ipc_memory_create(base_ptr, size, &key->ipc);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failed to create ipc for %p/%lx", address, length);
        return status;
    }

    key->address = (uintptr_t)base_ptr;
    key->length = size;
    key->dev_num = uct_rocm_base_get_dev_num(agent);

    return HSA_STATUS_SUCCESS;
}

static ucs_status_t uct_rocm_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_rocm_ipc_key_t *key;
    hsa_status_t status;

    key = ucs_malloc(sizeof(*key), "uct_rocm_ipc_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_rocm_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_rocm_ipc_pack_key(address, length, key);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_free(key);
        return UCS_ERR_INVALID_ADDR;
    }

    *memh_p = key;

    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_rocm_ipc_key_t *key = (uct_rocm_ipc_key_t *)memh;

    ucs_free(key);
    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_md_open(const char *md_name,
                                         const uct_md_config_t *uct_md_config,
                                         uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = (void*)ucs_empty_function,
        .query              = uct_rocm_ipc_md_query,
        .mkey_pack          = uct_rocm_ipc_mkey_pack,
        .mem_reg            = uct_rocm_ipc_mem_reg,
        .mem_dereg          = uct_rocm_ipc_mem_dereg,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    };
    static uct_md_t md = {
        .ops       = &md_ops,
        .component = &uct_rocm_ipc_md_component,
    };

    *md_p = &md;
    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_rkey_unpack(uct_md_component_t *mdc,
                                             const void *rkey_buffer, uct_rkey_t *rkey_p,
                                             void **handle_p)
{
    uct_rocm_ipc_key_t *packed = (uct_rocm_ipc_key_t *)rkey_buffer;
    uct_rocm_ipc_key_t *key;

    key = ucs_malloc(sizeof(uct_rocm_ipc_key_t), "uct_rocm_ipc_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_rocm_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    *key      = *packed;
    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;

    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_rkey_release(uct_md_component_t *mdc,
                                              uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_rocm_ipc_md_component,
                        UCT_ROCM_IPC_MD_NAME,
                        uct_rocm_ipc_query_md_resources,
                        uct_rocm_ipc_md_open, 0,
                        uct_rocm_ipc_rkey_unpack,
                        uct_rocm_ipc_rkey_release,
                        "ROCM_IPC_MD_",
                        uct_rocm_ipc_md_config_table,
                        uct_rocm_ipc_md_config_t,
                        ucs_empty_function_return_unsupported);
