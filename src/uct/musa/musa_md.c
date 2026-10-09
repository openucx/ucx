/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_md.h"
#include "musa_base.h"

#include <ucs/debug/log.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/type/class.h>

static const char *uct_musa_reg_mode_names[] = {
    [UCT_MUSA_REG_MODE_AUTO]    = "auto",
    [UCT_MUSA_REG_MODE_DMABUF]  = "dmabuf",
    [UCT_MUSA_REG_MODE_PEERMEM] = "peermem",
    [UCT_MUSA_REG_MODE_STAGING] = "staging",
    [UCT_MUSA_REG_MODE_LAST]    = NULL
};

static ucs_config_field_t uct_musa_md_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_musa_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"REGISTRATION_MODE", "auto",
     "MUSA registration path: auto, dmabuf, peermem, or staging",
     ucs_offsetof(uct_musa_md_config_t, mode),
     UCS_CONFIG_TYPE_ENUM(uct_musa_reg_mode_names)},

    {NULL}
};

static ucs_status_t uct_musa_md_query(uct_md_h uct_md,
                                      uct_md_attr_v2_t *md_attr)
{
    uct_musa_md_t *md = ucs_derived_of(uct_md, uct_musa_md_t);

    uct_md_base_md_query(md_attr);
    md_attr->flags             = UCT_MD_FLAG_REG;
    md_attr->reg_mem_types     = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->cache_mem_types   = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->access_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                 UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->detect_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->required_mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    md_attr->dmabuf_mem_types  = md->have_dmabuf ?
                                 UCS_BIT(UCS_MEMORY_TYPE_MUSA) : 0;
    md_attr->max_alloc         = SIZE_MAX;
    return UCS_OK;
}

static ucs_status_t
uct_musa_md_mem_reg(uct_md_h md, void *address, size_t length,
                    const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_musa_pointer_info_t info;
    uct_musa_mem_t *memh;
    ucs_memory_type_t mem_type;
    ucs_status_t status;

    (void)md;
    mem_type = UCT_MD_MEM_REG_FIELD_VALUE(params, mem_type, FIELD_MEM_TYPE,
                                          UCS_MEMORY_TYPE_HOST);
    if (mem_type != UCS_MEMORY_TYPE_MUSA) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_musa_base_query_pointer(address, length, &info);
    if (status != UCS_OK) {
        return status;
    }
    if (info.kind != UCT_MUSA_POINTER_DEVICE) {
        return UCS_ERR_INVALID_ADDR;
    }

    memh = ucs_malloc(sizeof(*memh), "musa memory handle");
    if (memh == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memh->address       = address;
    memh->base_address  = info.base_address;
    memh->length        = length;
    memh->alloc_length  = info.alloc_length;
    memh->device        = info.device;
    *memh_p              = memh;
    return UCS_OK;
}

static ucs_status_t
uct_musa_md_mem_dereg(uct_md_h md, const uct_md_mem_dereg_params_t *params)
{
    uct_musa_mem_t *memh;

    (void)md;
    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    memh = (uct_musa_mem_t *)params->memh;
    ucs_free(memh);
    return UCS_OK;
}

static ucs_status_t
uct_musa_md_mkey_pack(uct_md_h md, uct_mem_h memh, void *address,
                      size_t length,
                      const uct_md_mkey_pack_params_t *params,
                      void *buffer)
{
    (void)md;
    (void)memh;
    (void)address;
    (void)length;
    (void)params;
    (void)buffer;
    return UCS_OK;
}

ucs_status_t uct_musa_md_mem_query(uct_md_h md, const void *address,
                                    size_t length,
                                    uct_md_mem_attr_v2_t *mem_attr)
{
    uct_musa_md_t *musa_md = ucs_derived_of(md, uct_musa_md_t);

    return uct_musa_base_mem_query(address, length, musa_md->have_dmabuf,
                                   mem_attr);
}

static void uct_musa_md_close(uct_md_h uct_md)
{
    uct_musa_md_t *md = ucs_derived_of(uct_md, uct_musa_md_t);
    ucs_free(md);
}

static uct_md_ops_t uct_musa_md_ops = {
    .close              = uct_musa_md_close,
    .query              = uct_musa_md_query,
    .mem_alloc          = (uct_md_mem_alloc_func_t)
                          ucs_empty_function_return_unsupported,
    .mem_free           = (uct_md_mem_free_func_t)
                          ucs_empty_function_return_unsupported,
    .mem_advise         = (uct_md_mem_advise_func_t)
                          ucs_empty_function_return_unsupported,
    .mem_reg            = uct_musa_md_mem_reg,
    .mem_dereg          = uct_musa_md_mem_dereg,
    .mem_query          = uct_musa_md_mem_query,
    .mkey_pack          = uct_musa_md_mkey_pack,
    .mem_attach         = (uct_md_mem_attach_func_t)
                          ucs_empty_function_return_unsupported,
    .detect_memory_type = uct_musa_base_detect_memory_type,
    .mem_elem_pack      = (uct_md_mem_elem_pack_func_t)
                          ucs_empty_function_return_unsupported,
    .mem_elem_release   = (uct_md_mem_elem_release_func_t)
                          ucs_empty_function
};

static ucs_status_t
uct_musa_md_open(uct_component_h component, const char *md_name,
                 const uct_md_config_t *config, uct_md_h *md_p)
{
    const uct_musa_md_config_t *md_config =
            ucs_derived_of(config, uct_musa_md_config_t);
    uct_musa_md_t *md;
    unsigned device_count;
    ucs_status_t status;

    (void)md_name;

    status = uct_musa_base_get_device_count(&device_count);
    if (status != UCS_OK) {
        return status;
    }

    md = ucs_calloc(1, sizeof(*md), "musa md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &uct_musa_md_ops;
    md->super.component = component;
    md->mode            = md_config->mode;
    md->have_dmabuf     = uct_musa_base_has_dmabuf() &&
                          ((md->mode == UCT_MUSA_REG_MODE_AUTO) ||
                           (md->mode == UCT_MUSA_REG_MODE_DMABUF));

    if ((md->mode == UCT_MUSA_REG_MODE_DMABUF) && !md->have_dmabuf) {
        ucs_error("MUSA DMA-BUF registration was requested but the runtime "
                  "does not export muMemGetHandleForAddressRange");
        ucs_free(md);
        return UCS_ERR_UNSUPPORTED;
    }

    (void)device_count;
    *md_p = &md->super;
    return UCS_OK;
}

uct_component_t uct_musa_component = {
    .query_md_resources = uct_musa_base_query_md_resources,
    .md_open            = uct_musa_md_open,
    .cm_open            = (uct_component_cm_open_func_t)
                          ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_md_stub_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)
                          ucs_empty_function_return_unsupported,
    .rkey_release       = (uct_component_rkey_release_func_t)
                          ucs_empty_function_return_success,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "musa_cpy",
    .md_config          = {
        .name   = "MUSA memory domain",
        .prefix = "MUSA_",
        .table  = uct_musa_md_config_table,
        .size   = sizeof(uct_musa_md_config_t)
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_musa_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)
                          ucs_empty_function
};

UCT_COMPONENT_REGISTER(&uct_musa_component);
