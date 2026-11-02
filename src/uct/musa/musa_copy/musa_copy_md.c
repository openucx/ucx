/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_copy_md.h"

#include <uct/musa/base/musa_base.h>

#include <ucs/debug/log.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/type/class.h>

/* Handle returned for MUSA-allocated memory: no registration is needed for
 * the copy transport, but a valid handle keeps this md in the packed rkey,
 * so the peer can select the musa_copy lane (same as cuda_copy). */
static struct {} uct_musa_dummy_memh;

typedef struct {
    uct_md_t super;
    int      have_dmabuf;
} uct_musa_md_t;

typedef struct {
    void *address;
    int  device;
} uct_musa_alloc_handle_t;

static ucs_status_t uct_musa_md_query(uct_md_h uct_md,
                                      uct_md_attr_v2_t *md_attr)
{
    uct_musa_md_t *md = ucs_derived_of(uct_md, uct_musa_md_t);

    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
    md_attr->dmabuf_mem_types = md->have_dmabuf ?
                                UCS_BIT(UCS_MEMORY_TYPE_MUSA) : 0;
    md_attr->max_alloc        = SIZE_MAX;
    return UCS_OK;
}

static ucs_status_t
uct_musa_md_mem_alloc(uct_md_h uct_md, size_t *length_p, void **address_p,
                      ucs_memory_type_t mem_type, ucs_sys_device_t sys_dev,
                      unsigned flags, const char *alloc_name,
                      uct_mem_h *memh_p)
{
    uct_musa_alloc_handle_t *alloc_handle;
    ucs_status_t status;
    musaError_t error;
    int current_device = -1;
    int have_current_device;
    int device = -1;

    if (mem_type != UCS_MEMORY_TYPE_MUSA) {
        return UCS_ERR_UNSUPPORTED;
    }

    alloc_handle = ucs_malloc(sizeof(*alloc_handle), alloc_name);
    if (alloc_handle == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        status = uct_musa_base_get_device_by_sys_dev(sys_dev, &device);
        if (status != UCS_OK) {
            goto err_free_handle;
        }
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    if (device >= 0) {
        status = uct_musa_base_set_device(device);
        if (status != UCS_OK) {
            goto err_free_handle;
        }
    } else if (have_current_device) {
        device = current_device;
    }

    error = musaMalloc(address_p, *length_p);
    status = uct_musa_base_status(error, "musaMalloc");
    uct_musa_base_restore_device(current_device, have_current_device);
    if (status != UCS_OK) {
        goto err_free_handle;
    }

    alloc_handle->address = *address_p;
    alloc_handle->device  = device;
    *memh_p               = alloc_handle;
    return UCS_OK;

err_free_handle:
    ucs_free(alloc_handle);
    return status;
}

static ucs_status_t
uct_musa_md_mem_free(uct_md_h uct_md, uct_mem_h memh)
{
    uct_musa_alloc_handle_t *alloc_handle = memh;
    ucs_status_t status;
    musaError_t error;
    int current_device = -1;
    int have_current_device;

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    if (alloc_handle->device >= 0) {
        status = uct_musa_base_set_device(alloc_handle->device);
        if (status != UCS_OK) {
            ucs_free(alloc_handle);
            return status;
        }
    }

    error = musaFree(alloc_handle->address);
    status = uct_musa_base_status(error, "musaFree");
    uct_musa_base_restore_device(current_device, have_current_device);
    ucs_free(alloc_handle);
    return status;
}

static ucs_status_t
uct_musa_md_mem_query(uct_md_h md, const void *address, size_t length,
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

static ucs_status_t uct_musa_md_ensure_context(void)
{
    MUcontext context = NULL;
    MUresult result;
    ucs_status_t status;
    int device;

    result = muCtxGetCurrent(&context);
    if ((result == MUSA_SUCCESS) && (context != NULL)) {
        return UCS_OK;
    }

    if ((result != MUSA_SUCCESS) && (result != MUSA_ERROR_NOT_INITIALIZED)) {
        ucs_debug("muCtxGetCurrent failed: 0x%x", result);
        return UCS_ERR_IO_ERROR;
    }

    /* Driver registration needs a current context; preserve an existing
     * application context and initialize the runtime only if none exists. */
    status = uct_musa_base_get_current_device(&device);
    if (status != UCS_OK) {
        return status;
    }

    return uct_musa_base_set_device(device);
}

static ucs_status_t
uct_musa_md_mem_reg(uct_md_h uct_md, void *address, size_t length,
                    const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uint64_t flags = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    uct_musa_pointer_info_t info;
    ucs_log_level_t log_level;
    ucs_status_t status;
    MUresult result;

    status = uct_musa_md_ensure_context();
    if (status != UCS_OK) {
        return status;
    }

    status = uct_musa_base_query_pointer(address, length, &info);
    if (status != UCS_OK) {
        return status;
    }

    if (info.kind == UCT_MUSA_POINTER_MANAGED) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (info.kind == UCT_MUSA_POINTER_DEVICE) {
        /* MUSA allocations do not need registration for the copy transport */
        *memh_p = &uct_musa_dummy_memh;
        return UCS_OK;
    }

    /* Driver registration leaves the application's runtime last-error intact.
     * Already pinned memory uses a dummy handle so dereg cannot unpin it. */
    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                UCS_LOG_LEVEL_ERROR;
    result    = muMemHostRegister(address, length, MU_MEMHOSTREGISTER_PORTABLE);
    if (result == MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
        *memh_p = &uct_musa_dummy_memh;
        return UCS_OK;
    }

    if (result != MUSA_SUCCESS) {
        ucs_log(log_level, "muMemHostRegister address %p length %zu "
                "failed: 0x%x",
                address, length, result);
        return (result == MUSA_ERROR_INVALID_VALUE) ? UCS_ERR_INVALID_PARAM :
                                                     UCS_ERR_IO_ERROR;
    }

    *memh_p = address;
    return UCS_OK;
}

static ucs_status_t
uct_musa_md_mem_dereg(uct_md_h uct_md, const uct_md_mem_dereg_params_t *params)
{
    MUresult result;
    ucs_status_t status;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    /* Only memory pinned by mem_reg (memh == base address) needs to be
     * unpinned; dummy handles have nothing to deregister. */
    if (params->memh != &uct_musa_dummy_memh) {
        status = uct_musa_md_ensure_context();
        if (status != UCS_OK) {
            return status;
        }

        result = muMemHostUnregister(params->memh);
        if (result != MUSA_SUCCESS) {
            ucs_debug("muMemHostUnregister(%p) failed: 0x%x", params->memh,
                      result);
            return UCS_ERR_IO_ERROR;
        }
    }

    return UCS_OK;
}

static uct_md_ops_t uct_musa_md_ops = {
    .close              = uct_musa_md_close,
    .query              = uct_musa_md_query,
    .mem_alloc          = uct_musa_md_mem_alloc,
    .mem_free           = uct_musa_md_mem_free,
    .mem_advise         = (uct_md_mem_advise_func_t)
                          ucs_empty_function_return_unsupported,
    .mem_reg            = uct_musa_md_mem_reg,
    .mem_dereg          = uct_musa_md_mem_dereg,
    .mem_query          = uct_musa_md_mem_query,
    .mkey_pack          = (uct_md_mkey_pack_func_t)
                          ucs_empty_function_return_success,
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
    uct_musa_md_t *md;
    unsigned device_count;
    ucs_status_t status;

    status = uct_musa_base_get_device_count(&device_count);
    if (status != UCS_OK) {
        return status;
    }
    if (device_count == 0) {
        return UCS_ERR_NO_DEVICE;
    }

    md = ucs_calloc(1, sizeof(*md), "musa md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &uct_musa_md_ops;
    md->super.component = component;
    md->have_dmabuf     = uct_musa_base_has_dmabuf();

    *md_p = &md->super;

    /* Keep the global cache lazy, but install its handler before MUSA events
     * (same rationale as cuda_copy): musaMalloc calls issued after ucp_init
     * must be tracked, otherwise the memtype cache never learns about them. */
    ucs_memtype_cache_global_create();

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
        .prefix = "MUSA_COPY_",
        .table  = uct_md_config_table,
        .size   = sizeof(uct_md_config_t)
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_musa_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)
                          ucs_empty_function
};

UCT_COMPONENT_REGISTER(&uct_musa_component);
