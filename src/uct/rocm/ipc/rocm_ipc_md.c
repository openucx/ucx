/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_md.h"
#include "rocm_ipc_cache.h"

#include <uct/rocm/base/rocm_base.h>
#include <uct/api/v2/uct_v2.h>
#include <uct/api/device/uct_device_types.h>

#include <unistd.h>


static ucs_config_field_t uct_rocm_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_rocm_ipc_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_rocm_ipc_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->rkey_packed_size = sizeof(uct_rocm_ipc_key_t);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_RKEY |
                                UCT_MD_FLAG_INVALIDATE |
                                UCT_MD_FLAG_INVALIDATE_RMA |
                                UCT_MD_FLAG_INVALIDATE_AMO |
                                UCT_MD_FLAG_IPC_MEMTYPE_COPY;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ROCM);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ROCM);

    /* TODO: get accurate number */
    md_attr->reg_cost         = ucs_linear_func_make(9e-9, 0);
    return UCS_OK;
}

static ucs_status_t
uct_rocm_ipc_mkey_pack(uct_md_h uct_md, uct_mem_h memh, void *address,
                       size_t length, const uct_md_mkey_pack_params_t *params,
                       void *mkey_buffer)
{
    uct_rocm_ipc_key_t *packed = mkey_buffer;
    uct_rocm_ipc_key_t *key    = memh;

    *packed = *key;

    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_pack_key(void *address, size_t length,
                                          uct_rocm_ipc_key_t *key)
{
    void *base_ptr = NULL;
    size_t size    = 0;
    hsa_status_t hsa_status;
    ucs_status_t status;
    hsa_agent_t agent;
    hsa_amd_pointer_type_t mem_type;

    hsa_status = uct_rocm_base_get_ptr_info(address, length, &base_ptr, &size,
                                            &mem_type, &agent, NULL);
    if ((hsa_status != HSA_STATUS_SUCCESS) ||
        (mem_type == HSA_EXT_POINTER_TYPE_UNKNOWN)) {
        ucs_error("failed to get base ptr for %p/%lx, ROCm returned %p/%lx",
                   address, length, base_ptr, size);
        return UCS_ERR_INVALID_ADDR;
    }

    hsa_status = hsa_amd_ipc_memory_create(base_ptr, size, &key->ipc);
    if (hsa_status != HSA_STATUS_SUCCESS) {
        ucs_error("Failed to create ipc for %p/%lx", address, length);
        return UCS_ERR_INVALID_ADDR;
    }

    key->address = (uintptr_t)base_ptr;
    key->length  = size;
    key->dev_num = uct_rocm_base_get_dev_num(agent);
    key->pid     = getpid();
    key->pid_ns  = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);

    /* One IPC signal per allocation. Every importer of this key watches it. */
    memset(&key->ipc_signal, 0, sizeof(key->ipc_signal));
    status = uct_rocm_ipc_export_add(key->address, key->length,
                                     &key->ipc_signal);
    if (status != UCS_OK) {
        ucs_warn("failed to export rocm ipc invalidation signal for %p: %s",
                 address, ucs_status_string(status));
    }

    return UCS_OK;
}

static ucs_status_t
uct_rocm_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                     const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_rocm_ipc_key_t *key;
    ucs_status_t status;

    key = ucs_malloc(sizeof(*key), "uct_rocm_ipc_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_rocm_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_rocm_ipc_pack_key(address, length, key);
    if (status != UCS_OK) {
        ucs_free(key);
        return status;
    }

    *memh_p = key;

    return UCS_OK;
}

static ucs_status_t
uct_rocm_ipc_mem_dereg(uct_md_h md,
                       const uct_md_mem_dereg_params_t *params)
{
    uct_rocm_ipc_key_t *mem_hndl;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    /* The IPC signal is owned by the export table and stays until free. */
    mem_hndl = params->memh;
    ucs_free(mem_hndl);
    return UCS_OK;
}

static ucs_status_t
uct_rocm_ipc_md_mem_elem_pack(uct_md_h md_h, uct_mem_h memh, uct_rkey_t rkey,
                              uct_device_mem_elem_t *mem_elem_p,
                              void **release_handle_p)
{
    uct_rocm_ipc_unpacked_rkey_t *unpacked =
            ucs_derived_of((uct_rocm_ipc_key_t*)rkey,
                           uct_rocm_ipc_unpacked_rkey_t);
    uct_rocm_ipc_device_mem_element_t *rocm_ipc_mem_element =
            (uct_rocm_ipc_device_mem_element_t*)mem_elem_p;
    uct_rocm_ipc_cache_region_t *region;
    void *mapped_addr;
    ucs_status_t status;

    /* GPU kernels use the mapping directly, so it must outlive the element */
    status = uct_rocm_ipc_cache_pin_memhandle(&unpacked->super, &mapped_addr,
                                              &region);
    if (status != UCS_OK) {
        ucs_error("Failed to map IPC handle: %s", ucs_status_string(status));
        return status;
    }

    /* Calculate offset: base_address - mapped_address */
    rocm_ipc_mem_element->mapped_offset =
            UCS_PTR_BYTE_DIFF(unpacked->super.address, mapped_addr);

    ucs_trace("rocm_ipc mem_elem_pack: key_addr=%p mapped=%p offset=%ld",
              (void*)unpacked->super.address, mapped_addr,
              rocm_ipc_mem_element->mapped_offset);

    *release_handle_p = region;
    return UCS_OK;
}

static void uct_rocm_ipc_md_mem_elem_release(uct_md_h md, void *release_handle)
{
    uct_rocm_ipc_cache_unpin(release_handle);
}

static ucs_status_t
uct_rocm_ipc_md_open(uct_component_h component, const char *md_name,
                     const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = (uct_md_close_func_t)ucs_empty_function,
        .query              = uct_rocm_ipc_md_query,
        .mem_alloc          = (uct_md_mem_alloc_func_t)
                ucs_empty_function_return_unsupported,
        .mem_free           = (uct_md_mem_free_func_t)
                ucs_empty_function_return_unsupported,
        .mem_advise         = (uct_md_mem_advise_func_t)
                ucs_empty_function_return_unsupported,
        .mem_reg            = uct_rocm_ipc_mem_reg,
        .mem_dereg          = uct_rocm_ipc_mem_dereg,
        .mem_query          = (uct_md_mem_query_func_t)
                ucs_empty_function_return_unsupported,
        .mkey_pack          = uct_rocm_ipc_mkey_pack,
        .mem_elem_pack      = uct_rocm_ipc_md_mem_elem_pack,
        .mem_attach         = (uct_md_mem_attach_func_t)
                ucs_empty_function_return_unsupported,
        .detect_memory_type = (uct_md_detect_memory_type_func_t)
                ucs_empty_function_return_unsupported,
        .mem_elem_release   = uct_rocm_ipc_md_mem_elem_release
    };
    static uct_md_t md         = {
        .ops       = &md_ops,
        .component = &uct_rocm_ipc_component.super,
    };

    uct_rocm_ipc_export_init();
    *md_p = &md;
    return UCS_OK;
}

static ucs_status_t
uct_rocm_ipc_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                         const uct_rkey_unpack_params_t *params,
                         uct_rkey_t *rkey_p, void **handle_p)
{
    uct_rocm_ipc_key_t *packed = (uct_rocm_ipc_key_t *)rkey_buffer;
    uct_rocm_ipc_unpacked_rkey_t *unpacked;

    unpacked = ucs_malloc(sizeof(*unpacked), "uct_rocm_ipc_unpacked_rkey_t");
    if (NULL == unpacked) {
        ucs_error("Failed to allocate memory for uct_rocm_ipc_unpacked_rkey_t");
        return UCS_ERR_NO_MEMORY;
    }

    unpacked->super       = *packed;
    unpacked->region      = NULL;
    unpacked->mapped_addr = NULL;
    *handle_p             = NULL;
    *rkey_p               = (uintptr_t)unpacked;

    return UCS_OK;
}

static ucs_status_t uct_rocm_ipc_rkey_release(uct_component_t *component,
                                              uct_rkey_t rkey, void *handle)
{
    uct_rocm_ipc_unpacked_rkey_t *unpacked =
            ucs_derived_of((uct_rocm_ipc_key_t*)rkey,
                           uct_rocm_ipc_unpacked_rkey_t);

    ucs_assert(NULL == handle);
    if (unpacked->region != NULL) {
        uct_rocm_ipc_cache_unpin(unpacked->region);
    }
    ucs_free(unpacked);
    return UCS_OK;
}

ucs_status_t uct_rocm_ipc_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                                   void *handle, uint64_t raddr, void **laddr_p)
{
    uct_rocm_ipc_unpacked_rkey_t *unpacked =
            ucs_derived_of((uct_rocm_ipc_key_t*)rkey,
                           uct_rocm_ipc_unpacked_rkey_t);
    ptrdiff_t offset;
    ucs_status_t status;

    /* The returned address is valid until the rkey is released */
    if (unpacked->region == NULL) {
        status = uct_rocm_ipc_cache_pin_memhandle(&unpacked->super,
                                                  &unpacked->mapped_addr,
                                                  &unpacked->region);
        if (status != UCS_OK) {
            ucs_error("Failed to map IPC handle: %s",
                      ucs_status_string(status));
            return status;
        }
    }

    /* Calculate offset from base address */
    offset   = UCS_PTR_BYTE_DIFF(unpacked->super.address, raddr);
    *laddr_p = UCS_PTR_BYTE_OFFSET(unpacked->mapped_addr, offset);

    ucs_trace("rocm_ipc rkey_ptr: raddr=%p mapped=%p offset=%ld laddr=%p",
              (void*)raddr, unpacked->mapped_addr, offset, *laddr_p);

    return UCS_OK;
}

uct_rocm_ipc_component_t uct_rocm_ipc_component = {
    .super = {.query_md_resources = uct_rocm_base_query_md_resources,
              .md_open            = uct_rocm_ipc_md_open,
              .cm_open            = (uct_component_cm_open_func_t)
                                 ucs_empty_function_return_unsupported,
              .rkey_unpack        = uct_rocm_ipc_rkey_unpack,
              .rkey_ptr           = uct_rocm_ipc_rkey_ptr,
              .rkey_release       = uct_rocm_ipc_rkey_release,
              .rkey_compare       = uct_base_rkey_compare,
              .name               = "rocm_ipc",
              .md_config =
                      {
                              .name   = "ROCm-IPC memory domain",
                              .prefix = "ROCM_IPC_MD_",
                              .table  = uct_rocm_ipc_md_config_table,
                              .size   = sizeof(uct_rocm_ipc_md_config_t),
                      },
              .cm_config   = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
              .tl_list     = UCT_COMPONENT_TL_LIST_INITIALIZER(
                      &uct_rocm_ipc_component.super),
              .flags       = 0,
              .md_vfs_init = (uct_component_md_vfs_init_func_t)
                      ucs_empty_function}
};
UCT_COMPONENT_REGISTER(&uct_rocm_ipc_component.super);
