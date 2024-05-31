/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
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
#include <ucs/sys/ptr_arith.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack_int.h>
#include <ucm/api/ucm.h>
#include <ucs/type/class.h>
#include <uct/api/v2/uct_v2.h>
#include <hsa_ext_amd.h>

static ucs_config_field_t uct_rocm_copy_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rocm_copy_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"RCACHE", "try", "Enable using memory registration cache",
     ucs_offsetof(uct_rocm_copy_md_config_t, enable_rcache),
     UCS_CONFIG_TYPE_TERNARY},

    {"", "", NULL,
     ucs_offsetof(uct_rocm_copy_md_config_t, rcache),
     UCS_CONFIG_TYPE_TABLE(ucs_config_rcache_table)},

    {"DMABUF", "no",
     "Enable using cross-device dmabuf file descriptor",
     ucs_offsetof(uct_rocm_copy_md_config_t, enable_dmabuf),
     UCS_CONFIG_TYPE_TERNARY},

    {NULL}
};

static ucs_status_t
uct_rocm_copy_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);

    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_RKEY |
                                UCT_MD_FLAG_ALLOC;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    if (md->have_dmabuf) {
        md_attr->dmabuf_mem_types |= UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    }
    md_attr->max_alloc        = SIZE_MAX;
    md_attr->rkey_packed_size = sizeof(uct_rocm_copy_key_t);

    return UCS_OK;
}

static ucs_status_t
uct_rocm_copy_mkey_pack(uct_md_h uct_md, uct_mem_h memh, void *address,
                        size_t length, const uct_md_mkey_pack_params_t *params,
                        void *mkey_buffer)
{
    uct_rocm_copy_key_t *packed   = mkey_buffer;
    uct_rocm_copy_mem_t *mem_hndl = memh;

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

static void uct_rocm_copy_pg_align_addr(void **address, size_t *length)
{
    void *start, *end;
    size_t page_size;

    page_size = ucs_get_page_size();
    start     = ucs_align_down_pow2_ptr(*address, page_size);
    end       = ucs_align_up_pow2_ptr(UCS_PTR_BYTE_OFFSET(*address, *length), page_size);
    ucs_assert_always(start <= end);

    *address  = start;
    *length   = UCS_PTR_BYTE_DIFF(start, end);
}

static ucs_status_t uct_rocm_copy_mem_reg_internal(
        uct_md_h uct_md, void *address, size_t length,
        int pg_align_addr, uct_rocm_copy_mem_t *mem_hndl)
{
    void *dev_addr = NULL;
    hsa_status_t status;
    hsa_amd_pointer_type_t mem_type;

    ucs_assert((address != NULL) && (length != 0));

    status = uct_rocm_base_get_ptr_info(address, length, NULL, NULL, &mem_type,
                                        NULL, NULL);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("failed to detect memory type for addr %p len %zu", address, length);
        return UCS_ERR_IO_ERROR;
    }

    if (mem_type == HSA_EXT_POINTER_TYPE_HSA) {
        /* This covers device memory and memory allocated
           with hipHostMalloc */
        dev_addr = address;
    } else {
        if (pg_align_addr) {
            uct_rocm_copy_pg_align_addr(&address, &length);
        }

        status = hsa_amd_memory_lock(address, length, NULL, 0, &dev_addr);
        if ((status != HSA_STATUS_SUCCESS) || (dev_addr == NULL)) {
            return UCS_ERR_IO_ERROR;
        }
    }

    mem_hndl->vaddr    = address;
    mem_hndl->dev_ptr  = dev_addr;
    mem_hndl->reg_size = length;

    ucs_trace("Registered addr %p len %zu dev addr %p", address, length, dev_addr);
    return UCS_OK;
}

static ucs_status_t
uct_rocm_copy_mem_reg(uct_md_h md, void *address, size_t length,
                      const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_rocm_copy_mem_t *mem_hndl = NULL;
    ucs_status_t status;

    mem_hndl = ucs_malloc(sizeof(uct_rocm_copy_mem_t), "rocm_copy handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for rocm_copy_mem_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_rocm_copy_mem_reg_internal(md, address, length, 1, mem_hndl);
    if (status != UCS_OK) {
        ucs_free(mem_hndl);
        return status;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t
uct_rocm_copy_mem_dereg_internal(uct_md_h md,
                                 uct_rocm_copy_mem_t *mem_hndl)
{
    void *address = mem_hndl->vaddr;
    void *dev_ptr = mem_hndl->dev_ptr;
    hsa_status_t status;

    /* address == dev_ptr implies address was not host memory */
    if ((address == NULL) || (address == dev_ptr)) {
        return UCS_OK;
    }

    status = hsa_amd_memory_unlock(address);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("Deregistered addr %p len %zu", address, mem_hndl->reg_size);
    return UCS_OK;
}

static ucs_status_t
uct_rocm_copy_mem_dereg(uct_md_h md,
                        const uct_md_mem_dereg_params_t *params)
{
    ucs_status_t status;
    uct_rocm_copy_mem_t *mem_hndl;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    mem_hndl = (uct_rocm_copy_mem_t *)params->memh;
    status   = uct_rocm_copy_mem_dereg_internal(md, mem_hndl);
    ucs_free(mem_hndl);

    return status;
}

static void uct_rocm_copy_md_close(uct_md_h uct_md) {
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);

    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }

    ucs_free(md);
}

static ucs_status_t
uct_rocm_copy_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                        ucs_memory_type_t mem_type, unsigned flags,
                        const char *alloc_name, uct_mem_h *memh_p)
{
    ucs_status_t status;
    hsa_status_t hsa_status;
    hsa_amd_memory_pool_t pool;

    if (mem_type != UCS_MEMORY_TYPE_ROCM) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_rocm_base_get_last_device_pool(&pool);
    if (status != UCS_OK) {
        return status;
    }

    hsa_status = hsa_amd_memory_pool_allocate(pool, *length_p, 0, address_p);
    if (hsa_status != HSA_STATUS_SUCCESS) {
        ucs_debug("could not allocate HSA memory: 0x%x", hsa_status);
        return UCS_ERR_UNSUPPORTED;
    }

    *memh_p = *address_p;
    return UCS_OK;
}

static ucs_status_t uct_rocm_copy_mem_free(uct_md_h md, uct_mem_h memh)
{
    hsa_status_t hsa_status;

    hsa_status = hsa_amd_memory_pool_free((void*)memh);
    if ((hsa_status != HSA_STATUS_SUCCESS) &&
        (hsa_status != HSA_STATUS_INFO_BREAK)) {
        ucs_debug("could not free HSA memory 0x%x", hsa_status);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static uct_md_ops_t md_ops = {
    .close              = uct_rocm_copy_md_close,
    .query              = uct_rocm_copy_md_query,
    .mkey_pack          = uct_rocm_copy_mkey_pack,
    .mem_alloc          = uct_rocm_copy_mem_alloc,
    .mem_free           = uct_rocm_copy_mem_free,
    .mem_reg            = uct_rocm_copy_mem_reg,
    .mem_dereg          = uct_rocm_copy_mem_dereg,
    .mem_attach         = ucs_empty_function_return_unsupported,
    .mem_query          = uct_rocm_base_mem_query,
    .detect_memory_type = uct_rocm_base_detect_memory_type,
};

static inline uct_rocm_copy_rcache_region_t*
uct_rocm_copy_rache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_rocm_copy_rcache_region_t, memh);
}

static ucs_status_t
uct_rocm_copy_mem_rcache_reg(uct_md_h uct_md, void *address, size_t length,
                             const uct_md_mem_reg_params_t *params,
                             uct_mem_h *memh_p)
{
    uint64_t flags         = UCT_MD_MEM_REG_FIELD_VALUE(params, flags,
                                                        FIELD_FLAGS, 0);
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;
    uct_rocm_copy_mem_t *memh;

    status = ucs_rcache_get(md->rcache, (void *)address, length,
                            ucs_get_page_size(), PROT_READ | PROT_WRITE, &flags,
                            &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    memh    = &ucs_derived_of(rregion, uct_rocm_copy_rcache_region_t)->memh;
    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t
uct_rocm_copy_mem_rcache_dereg(uct_md_h uct_md,
                               const uct_md_mem_dereg_params_t *params)
{
    uct_rocm_copy_md_t *md = ucs_derived_of(uct_md, uct_rocm_copy_md_t);
    uct_rocm_copy_rcache_region_t *region;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    region = uct_rocm_copy_rache_region_from_memh(params->memh);
    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t md_rcache_ops = {
    .close              = uct_rocm_copy_md_close,
    .query              = uct_rocm_copy_md_query,
    .mem_alloc          = uct_rocm_copy_mem_alloc,
    .mem_free           = uct_rocm_copy_mem_free,
    .mkey_pack          = uct_rocm_copy_mkey_pack,
    .mem_reg            = uct_rocm_copy_mem_rcache_reg,
    .mem_dereg          = uct_rocm_copy_mem_rcache_dereg,
    .mem_attach         = ucs_empty_function_return_unsupported,
    .mem_query          = uct_rocm_base_mem_query,
    .detect_memory_type = uct_rocm_base_detect_memory_type,
};

static ucs_status_t
uct_rocm_copy_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                void *arg, ucs_rcache_region_t *rregion,
                                uint16_t rcache_mem_reg_flags)
{
    uct_rocm_copy_md_t *md = context;
    uct_rocm_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_rocm_copy_rcache_region_t);
    return uct_rocm_copy_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                          region->super.super.end -
                                          region->super.super.start,
                                          0, &region->memh);
}

static void uct_rocm_copy_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                              ucs_rcache_region_t *rregion)
{
    uct_rocm_copy_md_t *md = context;
    uct_rocm_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_rocm_copy_rcache_region_t);
    (void)uct_rocm_copy_mem_dereg_internal(&md->super, &region->memh);
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
    .merge       = (void*)ucs_empty_function,
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
    int have_dmabuf;

    md = ucs_malloc(sizeof(uct_rocm_copy_md_t), "uct_rocm_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_rocm_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_rocm_copy_component;
    md->rcache          = NULL;
    md->reg_cost        = UCS_LINEAR_FUNC_ZERO;
    md->have_dmabuf     = 0;

    have_dmabuf = uct_rocm_base_is_dmabuf_supported();
    if ((md_config->enable_dmabuf == UCS_YES) && !have_dmabuf) {
        ucs_error("ROCm dmabuf support requested but not found");
        return UCS_ERR_UNSUPPORTED;
    }

    if (md_config->enable_dmabuf != UCS_NO) {
        md->have_dmabuf = have_dmabuf;
    }

    if (md_config->enable_rcache != UCS_NO) {
        ucs_rcache_set_params(&rcache_params, &md_config->rcache);
        rcache_params.region_struct_size = sizeof(uct_rocm_copy_rcache_region_t);
        rcache_params.ucm_events         = UCM_EVENT_MEM_TYPE_FREE;
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = md;
        rcache_params.ops                = &uct_rocm_copy_rcache_ops;
        rcache_params.flags              = UCS_RCACHE_FLAG_PURGE_ON_FORK;

        status = ucs_rcache_create(&rcache_params, "rocm_copy", NULL, &md->rcache);
        if (status == UCS_OK) {
            md->super.ops = &md_rcache_ops;
            md->reg_cost  = UCS_LINEAR_FUNC_ZERO;
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
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "rocm_cpy",
    .md_config          = {
        .name           = "ROCm-copy memory domain",
        .prefix         = "ROCM_COPY_",
        .table          = uct_rocm_copy_md_config_table,
        .size           = sizeof(uct_rocm_copy_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_rocm_copy_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_rocm_copy_component);

