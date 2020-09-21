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
#include <ucs/sys/math.h>
#include <ucs/debug/memtrack.h>
#include <ucm/api/ucm.h>
#include <ucs/type/class.h>

#include <hsa_ext_amd.h>

static ucs_config_field_t uct_rocm_copy_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rocm_copy_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"RCACHE", "try", "Enable using memory registration cache",
     ucs_offsetof(uct_rocm_copy_md_config_t, enable_rcache),
     UCS_CONFIG_TYPE_TERNARY},

    {NULL}
};

static ucs_status_t uct_rocm_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                    UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cap.detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->rkey_packed_size     = sizeof(uct_rocm_copy_key_t);
    md_attr->reg_cost             = ucs_linear_func_make(0, 0);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                            void *rkey_buffer)
{
    uct_rocm_copy_key_t *packed   = (uct_rocm_copy_key_t *)rkey_buffer;
    uct_rocm_copy_mem_t *mem_hndl = (uct_rocm_copy_mem_t *)memh;

    packed->vaddr   = (uint64_t) mem_hndl->vaddr;
    packed->dev_ptr = mem_hndl->dev_ptr;

    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_rkey_unpack(uct_component_t *component,
                                              const void *rkey_buffer,
                                              uct_rkey_t *rkey_p, void **handle_p)
{
    uct_rocm_copy_key_t *packed = (uct_rocm_copy_key_t *)rkey_buffer;
    uct_rocm_copy_key_t *key;

    key = ucs_malloc(sizeof(uct_rocm_copy_key_t), "uct_rocm_copy_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_rocm_copy_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    key->vaddr   = packed->vaddr;
    key->dev_ptr = packed->dev_ptr;

    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;

    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_rkey_release(uct_component_t *component,
                                               uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_reg_internal(
        uct_md_h uct_md, void *address, size_t length,
        unsigned flags, uct_rocm_copy_mem_t *mem_hndl)
{
    void *dev_addr = NULL;
    hsa_status_t status;

    if(address == NULL) {
        memset(mem_hndl, 0, sizeof(*mem_hndl));
        return UCS_OK;
    }

    status = hsa_amd_memory_lock(address, length, NULL, 0, &dev_addr);
    if ((status != HSA_STATUS_SUCCESS) || (dev_addr == NULL)) {
        return UCS_ERR_IO_ERROR;
    }

    mem_hndl->vaddr    = address;
    mem_hndl->dev_ptr  = dev_addr;
    mem_hndl->reg_size = length;

    ucs_trace("Registered addr %p len %zu dev addr %p", address, length, dev_addr);
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_reg(uct_md_h md, void *address, size_t length,
                                          unsigned flags, uct_mem_h *memh_p)
{
    uct_rocm_copy_mem_t *mem_hndl = NULL;
    void *start, *end;
    size_t len, page_size;
    ucs_status_t status;

    mem_hndl = ucs_malloc(sizeof(uct_rocm_copy_mem_t), "rocm_copy handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for rocm_copy_mem_t");
        return UCS_ERR_NO_MEMORY;
    }

    page_size = ucs_get_page_size();
    start     = ucs_align_down_pow2_ptr(address, page_size);
    end       = ucs_align_up_pow2_ptr(UCS_PTR_BYTE_OFFSET(address, length), page_size);
    len       = UCS_PTR_BYTE_DIFF(start, end);
    ucs_assert_always(start <= end);

    status = uct_rocm_copy_mem_reg_internal(md, address, len, 0, mem_hndl);
    if (status != UCS_OK) {
        ucs_free(mem_hndl);
        return status;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_rocm_copy_mem_t *mem_hndl = (uct_rocm_copy_mem_t *)memh;
    void *address = mem_hndl->vaddr;
    hsa_status_t status;

    if (address == NULL) {
        return UCS_OK;
    }

    status = hsa_amd_memory_unlock(address);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("Deregistered addr %p len %zu", address, mem_hndl->reg_size);
    return UCS_OK;
}

static void uct_rocm_copy_md_close(uct_md_h uct_md) {
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);

    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }

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

static inline uct_rocm_copy_rcache_region_t*
uct_rocm_copy_rache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_rocm_copy_rcache_region_t, memh);
}

static ucs_status_t
uct_rocm_copy_mem_rcache_reg(uct_md_h uct_md, void *address, size_t length,
                             unsigned flags, uct_mem_h *memh_p)
{
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;
    uct_rocm_copy_mem_t *memh;

    status = ucs_rcache_get(md->rcache, (void *)address, length, PROT_READ|PROT_WRITE,
                            &flags, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    memh    = &ucs_derived_of(rregion, uct_rocm_copy_rcache_region_t)->memh;
    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);
    uct_rocm_copy_rcache_region_t *region = uct_rocm_copy_rache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t md_rcache_ops = {
    .close              = uct_rocm_copy_md_close,
    .query              = uct_rocm_copy_md_query,
    .mkey_pack          = uct_rocm_copy_mkey_pack,
    .mem_reg            = uct_rocm_copy_mem_rcache_reg,
    .mem_dereg          = uct_rocm_copy_mem_rcache_dereg,
    .detect_memory_type = uct_rocm_base_detect_memory_type,
};

static ucs_status_t
uct_rocm_copy_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                void *arg, ucs_rcache_region_t *rregion,
                                uint16_t rcache_mem_reg_flags)
{
    uct_rocm_copy_md_t *md = context;
    int *flags = arg;
    uct_rocm_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_rocm_copy_rcache_region_t);
    return uct_rocm_copy_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                          region->super.super.end -
                                          region->super.super.start,
                                          *flags, &region->memh);
}

static void uct_rocm_copy_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                              ucs_rcache_region_t *rregion)
{
    uct_rocm_copy_md_t *md = context;
    uct_rocm_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_rocm_copy_rcache_region_t);
    (void)uct_rocm_copy_mem_dereg(&md->super, &region->memh);
}

static void uct_rocm_copy_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                                ucs_rcache_region_t *rregion, char *buf,
                                                size_t max)
{
    uct_rocm_copy_rcache_region_t *region = ucs_derived_of(rregion,
                                                           uct_rocm_copy_rcache_region_t);
    uct_rocm_copy_mem_t *memh = &region->memh;

    snprintf(buf, max, "dev ptr:%p", memh->dev_ptr);
}

static ucs_rcache_ops_t uct_rocm_copy_rcache_ops = {
    .mem_reg     = uct_rocm_copy_rcache_mem_reg_cb,
    .mem_dereg   = uct_rocm_copy_rcache_mem_dereg_cb,
    .dump_region = uct_rocm_copy_rcache_dump_region_cb
};

static ucs_status_t
uct_rocm_copy_md_open(uct_component_h component, const char *md_name,
                      const uct_md_config_t *config, uct_md_h *md_p)
{
    const uct_rocm_copy_md_config_t *md_config =
                    ucs_derived_of(config, uct_rocm_copy_md_config_t);
    ucs_status_t status;
    uct_rocm_copy_md_t *md;
    ucs_rcache_params_t rcache_params;

    md = ucs_malloc(sizeof(uct_rocm_copy_md_t), "uct_rocm_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_rocm_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_rocm_copy_component;
    md->rcache          = NULL;
    md->reg_cost        = ucs_linear_func_make(0, 0);

    if (md_config->enable_rcache != UCS_NO) {
        rcache_params.region_struct_size = sizeof(uct_rocm_copy_rcache_region_t);
        rcache_params.alignment          = ucs_get_page_size();
        rcache_params.max_alignment      = ucs_get_page_size();
        rcache_params.ucm_events         = UCM_EVENT_MEM_TYPE_FREE;
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = md;
        rcache_params.ops                = &uct_rocm_copy_rcache_ops;
        rcache_params.flags              = 0;
        status = ucs_rcache_create(&rcache_params, "rocm_copy", NULL, &md->rcache);
        if (status == UCS_OK) {
            md->super.ops = &md_rcache_ops;
            md->reg_cost  = ucs_linear_func_make(0, 0);
        } else {
            ucs_assert(md->rcache == NULL);
            if (md_config->enable_rcache == UCS_YES) {
                status = UCS_ERR_IO_ERROR;
                goto err;
            } else {
                ucs_debug("could not create registration cache for: %s",
                          ucs_status_string(status));
            }
        }
    }

    *md_p = (uct_md_h) md;
    status = UCS_OK;
out:
    return status;
err:
    ucs_free(md);
    goto out;
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

