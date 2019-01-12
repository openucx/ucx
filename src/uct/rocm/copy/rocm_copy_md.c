/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_copy_md.h"

#include <uct/rocm/base/rocm_base.h>

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>

#include <hsa_ext_amd.h>

static ucs_config_field_t uct_rocm_copy_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rocm_copy_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_rocm_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_ROCM;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->rkey_packed_size  = 0;
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                            void *rkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_rkey_unpack(uct_md_component_t *mdc,
                                              const void *rkey_buffer, uct_rkey_t *rkey_p,
                                              void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                               void *handle)
{
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_reg(uct_md_h md, void *address, size_t length,
                                          unsigned flags, uct_mem_h *memh_p)
{
    hsa_status_t status;
    void *lock_addr;

    if(address == NULL) {
        *memh_p = address;
        return UCS_OK;
    }

    status = hsa_amd_memory_lock(address, length, NULL, 0, &lock_addr);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    *memh_p = address;
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    void *address = (void *)memh;
    hsa_status_t status;

    if (address == NULL) {
        return UCS_OK;
    }

    status = hsa_amd_memory_unlock(address);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                     unsigned *num_resources_p)
{
    if (uct_rocm_base_init() != HSA_STATUS_SUCCESS) {
        ucs_error("Could not initialize ROCm support");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    return uct_single_md_resource(&uct_rocm_copy_md_component, resources_p,
                                  num_resources_p);
}

static void uct_rocm_copy_md_close(uct_md_h uct_md) {
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close              = uct_rocm_copy_md_close,
    .query              = uct_rocm_copy_md_query,
    .mkey_pack          = uct_rocm_copy_mkey_pack,
    .mem_reg            = uct_rocm_copy_mem_reg,
    .mem_dereg          = uct_rocm_copy_mem_dereg,
    .is_mem_type_owned  = uct_rocm_base_is_mem_type_owned,
};

static ucs_status_t uct_rocm_copy_md_open(const char *md_name, const uct_md_config_t *md_config,
                                          uct_md_h *md_p)
{
    uct_rocm_copy_md_t *md;

    md = ucs_malloc(sizeof(uct_rocm_copy_md_t), "uct_rocm_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_rocm_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_rocm_copy_md_component;

    *md_p = (uct_md_h) md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_rocm_copy_md_component, UCT_ROCM_COPY_MD_NAME,
                        uct_rocm_copy_query_md_resources, uct_rocm_copy_md_open, NULL,
                        uct_rocm_copy_rkey_unpack, uct_rocm_copy_rkey_release, "ROCM_COPY_",
                        uct_rocm_copy_md_config_table, uct_rocm_copy_md_config_t);
