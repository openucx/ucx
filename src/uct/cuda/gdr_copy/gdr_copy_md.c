/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "gdr_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>
#include <ucm/api/ucm.h>
#include <uct/api/v2/uct_v2.h>
#include <uct/cuda/base/cuda_md.h>

static ucs_config_field_t uct_gdr_copy_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_gdr_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* TODO take default from device */
     ucs_offsetof(uct_gdr_copy_md_config_t, uc_reg_cost.m), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
     ucs_offsetof(uct_gdr_copy_md_config_t, uc_reg_cost.c), UCS_CONFIG_TYPE_TIME},

    {NULL}
};

static ucs_status_t
uct_gdr_copy_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_RKEY;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->rkey_packed_size = sizeof(uct_gdr_copy_key_t);
    md_attr->reg_alignment    = GPU_PAGE_SIZE;
    return UCS_OK;
}

static ucs_status_t
uct_gdr_copy_mkey_pack(uct_md_h md, uct_mem_h memh, void *address,
                       size_t length, const uct_md_mkey_pack_params_t *params,
                       void *mkey_buffer)
{
    uct_gdr_copy_key_t *packed   = mkey_buffer;
    uct_gdr_copy_mem_t *mem_hndl = memh;

    packed->vaddr   = mem_hndl->info.va;
    packed->bar_ptr = mem_hndl->bar_ptr;
    packed->mh      = mem_hndl->mh;

    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_rkey_unpack(uct_component_t *component,
                                             const void *rkey_buffer,
                                             uct_rkey_t *rkey_p, void **handle_p)
{
    uct_gdr_copy_key_t *packed = (uct_gdr_copy_key_t *)rkey_buffer;
    uct_gdr_copy_key_t *key;

    key = ucs_malloc(sizeof(uct_gdr_copy_key_t), "uct_gdr_copy_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_gdr_copy_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    key->vaddr      = packed->vaddr;
    key->bar_ptr    = packed->bar_ptr;
    key->mh         = packed->mh;

    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;

    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_rkey_release(uct_component_t *component,
                                              uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_gdr_copy_mem_reg_internal,
                 (uct_md, address, length, flags, mem_hndl),
                 uct_md_h uct_md, void *address, size_t length,
                 unsigned flags, uct_gdr_copy_mem_t *mem_hndl)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    unsigned long d_ptr   = ((unsigned long)(char*)address);
    ucs_log_level_t log_level;
    int ret;

    ucs_assert((address != NULL) && (length != 0));

    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                UCS_LOG_LEVEL_ERROR;

    ret = gdr_pin_buffer(md->gdrcpy_ctx, d_ptr, length, 0, 0, &mem_hndl->mh);
    if (ret) {
        ucs_log(log_level, "gdr_pin_buffer failed. length :%lu ret:%d",
                length, ret);
        goto err;
    }

    ret = gdr_map(md->gdrcpy_ctx, mem_hndl->mh, &mem_hndl->bar_ptr, length);
    if (ret) {
        ucs_log(log_level, "gdr_map failed. length :%lu ret:%d", length, ret);
        goto unpin_buffer;
    }

    mem_hndl->reg_size = length;

    ret = gdr_get_info(md->gdrcpy_ctx, mem_hndl->mh, &mem_hndl->info);
    if (ret) {
        ucs_error("gdr_get_info failed. ret:%d", ret);
        goto unmap_buffer;
    }

    ucs_trace("registered memory:%p..%p length:%lu info.va:0x%"PRIx64" bar_ptr:%p",
              address, UCS_PTR_BYTE_OFFSET(address, length), length,
              mem_hndl->info.va, mem_hndl->bar_ptr);

    return UCS_OK;

unmap_buffer:
    ret = gdr_unmap(md->gdrcpy_ctx, mem_hndl->mh, mem_hndl->bar_ptr, mem_hndl->reg_size);
    if (ret) {
        ucs_warn("gdr_unmap failed. unpin_size:%lu ret:%d", mem_hndl->reg_size, ret);
    }
unpin_buffer:
    ret = gdr_unpin_buffer(md->gdrcpy_ctx, mem_hndl->mh);
    if (ret) {
        ucs_warn("gdr_unpin_buffer failed. ret;%d", ret);
    }
err:
    return UCS_ERR_IO_ERROR;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_gdr_copy_mem_dereg_internal,
                 (uct_md, mem_hndl),
                 uct_md_h uct_md, uct_gdr_copy_mem_t *mem_hndl)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    int ret;

    ret = gdr_unmap(md->gdrcpy_ctx, mem_hndl->mh, mem_hndl->bar_ptr, mem_hndl->reg_size);
    if (ret) {
        ucs_error("gdr_unmap failed. unpin_size:%lu ret:%d", mem_hndl->reg_size, ret);
        return UCS_ERR_IO_ERROR;
    }

    ret = gdr_unpin_buffer(md->gdrcpy_ctx, mem_hndl->mh);
    if (ret) {
        ucs_error("gdr_unpin_buffer failed. ret:%d", ret);
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("deregistered memory. info.va:0x%"PRIx64" bar_ptr:%p",
              mem_hndl->info.va, mem_hndl->bar_ptr);
    return UCS_OK;
}

static ucs_status_t
uct_gdr_copy_mem_reg(uct_md_h uct_md, void *address, size_t length,
                     const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_gdr_copy_mem_t *mem_hndl = NULL;
    ucs_status_t status;

    mem_hndl = ucs_malloc(sizeof(uct_gdr_copy_mem_t), "gdr_copy handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for gdr_copy_mem_t");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_ptr_check_align(address, length, GPU_PAGE_SIZE);
    status = uct_gdr_copy_mem_reg_internal(uct_md, address, length, 0,
                                           mem_hndl);
    if (status != UCS_OK) {
        ucs_free(mem_hndl);
        return status;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t
uct_gdr_copy_mem_dereg(uct_md_h uct_md,
                       const uct_md_mem_dereg_params_t *params)
{
    uct_gdr_copy_mem_t *mem_hndl;
    ucs_status_t status;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    mem_hndl = params->memh;
    status   = uct_gdr_copy_mem_dereg_internal(uct_md, mem_hndl);
    if (status != UCS_OK) {
        ucs_warn("failed to deregister memory handle");
    }

    ucs_free(mem_hndl);
    return status;
}

static ucs_status_t
uct_gdr_copy_query_md_resources(uct_component_t *component,
                                uct_md_resource_desc_t **resources_p,
                                unsigned *num_resources_p)
{
    gdr_t ctx;

    ctx = gdr_open();
    if (ctx == NULL) {
        ucs_debug("could not open gdr copy. disabling gdr copy resource");
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }
    gdr_close(ctx);

    return uct_cuda_base_query_md_resources(component, resources_p,
                                            num_resources_p);
}

static void uct_gdr_copy_md_close(uct_md_h uct_md)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    int ret;

    ret = gdr_close(md->gdrcpy_ctx);
    if (ret) {
        ucs_warn("failed to close gdrcopy. ret:%d", ret);
    }

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close              = uct_gdr_copy_md_close,
    .query              = uct_gdr_copy_md_query,
    .mkey_pack          = uct_gdr_copy_mkey_pack,
    .mem_reg            = uct_gdr_copy_mem_reg,
    .mem_dereg          = uct_gdr_copy_mem_dereg,
    .mem_attach         = ucs_empty_function_return_unsupported,
    .detect_memory_type = ucs_empty_function_return_unsupported
};

static ucs_status_t
uct_gdr_copy_md_open(uct_component_t *component, const char *md_name,
                     const uct_md_config_t *config, uct_md_h *md_p)
{
    const uct_gdr_copy_md_config_t *md_config =
                    ucs_derived_of(config, uct_gdr_copy_md_config_t);
    ucs_status_t status;
    uct_gdr_copy_md_t *md;

    md = ucs_malloc(sizeof(uct_gdr_copy_md_t), "uct_gdr_copy_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_gdr_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_gdr_copy_component;
    md->reg_cost        = md_config->uc_reg_cost;

    md->gdrcpy_ctx = gdr_open();
    if (md->gdrcpy_ctx == NULL) {
        ucs_error("failed to open gdr copy");
        status = UCS_ERR_IO_ERROR;
        goto err_free_md;
    }

    *md_p = (uct_md_h) md;
    return UCS_OK;

err_free_md:
    ucs_free(md);
    return status;
}

uct_component_t uct_gdr_copy_component = {
    .query_md_resources = uct_gdr_copy_query_md_resources,
    .md_open            = uct_gdr_copy_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_gdr_copy_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_gdr_copy_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "gdr_copy",
    .md_config          = {
        .name           = "GDR-copy memory domain",
        .prefix         = "GDR_COPY_",
        .table          = uct_gdr_copy_md_config_table,
        .size           = sizeof(uct_gdr_copy_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_gdr_copy_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_gdr_copy_component);

