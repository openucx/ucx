/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

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
    md_attr->cap.flags            = UCT_MD_FLAG_REG;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cap.detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->rkey_packed_size     = 0;
    md_attr->reg_cost             = ucs_linear_func_make(0, 0);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                            void *rkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_rkey_unpack(uct_component_t *component,
                                              const void *rkey_buffer,
                                              uct_rkey_t *rkey_p, void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_rkey_release(uct_component_t *component,
                                               uct_rkey_t rkey, void *handle)
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

static void uct_rocm_copy_md_close(uct_md_h uct_md) {
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close               = uct_rocm_copy_md_close,
    .query               = uct_rocm_copy_md_query,
    .mkey_pack           = uct_rocm_copy_mkey_pack,
    .mem_reg             = uct_rocm_copy_mem_reg,
    .mem_dereg           = uct_rocm_copy_mem_dereg,
    .mem_query           = uct_rocm_base_mem_query,
    .detect_memory_type  = uct_rocm_base_detect_memory_type
};

static ucs_status_t
uct_rocm_copy_md_open(uct_component_h component, const char *md_name,
                      const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_rocm_copy_md_t *md;

    md = ucs_malloc(sizeof(uct_rocm_copy_md_t), "uct_rocm_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_rocm_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_rocm_copy_component;

    *md_p = (uct_md_h) md;
    return UCS_OK;
}

uct_component_t uct_rocm_copy_component = {
    .query_md_resources = uct_rocm_base_query_md_resources,
    .md_open            = uct_rocm_copy_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_rocm_copy_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_rocm_copy_rkey_release,
    .name               = "rocm_cpy",
    .md_config          = {
        .name           = "ROCm-copy memory domain",
        .prefix         = "ROCM_COPY_",
        .table          = uct_rocm_copy_md_config_table,
        .size           = sizeof(uct_rocm_copy_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_rocm_copy_component),
    .flags              = 0
};
UCT_COMPONENT_REGISTER(&uct_rocm_copy_component);

