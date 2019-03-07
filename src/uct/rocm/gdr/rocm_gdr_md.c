/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_gdr_md.h"

#include <uct/rocm/base/rocm_base.h>

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>

#include <hsa_ext_amd.h>

static ucs_config_field_t uct_rocm_gdr_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rocm_gdr_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_rocm_gdr_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_ROCM);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_ROCM;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->rkey_packed_size  = sizeof(uct_rocm_gdr_key_t);
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_rocm_gdr_mkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_rocm_gdr_key_t *packed      = (uct_rocm_gdr_key_t *)rkey_buffer;
    //uct_rocm_gdr_mem_t *mem_hndl    = (uct_rocm_gdr_mem_t *)memh;
    packed->dummy = 0;
    return UCS_OK;
}

static ucs_status_t uct_rocm_gdr_rkey_unpack(uct_md_component_t *mdc,
                                             const void *rkey_buffer, uct_rkey_t *rkey_p,
                                             void **handle_p)
{
    //uct_rocm_gdr_key_t *packed = (uct_rocm_gdr_key_t *)rkey_buffer;
    uct_rocm_gdr_key_t *key;

    key = ucs_malloc(sizeof(uct_rocm_gdr_key_t), "uct_rocm_gdr_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_rocm_gdr_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    key->dummy = 0;

    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;

    return UCS_OK;
}

static ucs_status_t uct_rocm_gdr_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                              void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_rocm_gdr_mem_reg(uct_md_h md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_rocm_gdr_mem_t *mem_hndl = NULL;

    mem_hndl = ucs_malloc(sizeof(uct_rocm_gdr_mem_t), "rocm_gdr handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for rocm_gdr_mem_t");
        return UCS_ERR_NO_MEMORY;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_rocm_gdr_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_rocm_gdr_mem_t *mem_hndl = memh;

    ucs_free(mem_hndl);
    return UCS_OK;
}

static ucs_status_t uct_rocm_gdr_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                    unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_rocm_gdr_md_component, resources_p,
                                  num_resources_p);
}

static void uct_rocm_gdr_md_close(uct_md_h uct_md) {
    uct_rocm_gdr_md_t *md = ucs_derived_of(uct_md, uct_rocm_gdr_md_t);

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close              = uct_rocm_gdr_md_close,
    .query              = uct_rocm_gdr_md_query,
    .mkey_pack          = uct_rocm_gdr_mkey_pack,
    .mem_reg            = uct_rocm_gdr_mem_reg,
    .mem_dereg          = uct_rocm_gdr_mem_dereg,
    .is_mem_type_owned  = uct_rocm_base_is_mem_type_owned,
};

static ucs_status_t uct_rocm_gdr_md_open(const char *md_name, const uct_md_config_t *md_config,
                                         uct_md_h *md_p)
{
    uct_rocm_gdr_md_t *md;

    md = ucs_malloc(sizeof(uct_rocm_gdr_md_t), "uct_rocm_gdr_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_rocm_gdr_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_rocm_gdr_md_component;

    *md_p = (uct_md_h) md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_rocm_gdr_md_component, UCT_ROCM_GDR_MD_NAME,
                        uct_rocm_gdr_query_md_resources, uct_rocm_gdr_md_open, NULL,
                        uct_rocm_gdr_rkey_unpack, uct_rocm_gdr_rkey_release, "ROCM_GDR_",
                        uct_rocm_gdr_md_config_table, uct_rocm_gdr_md_config_t);
