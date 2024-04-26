/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_copy_md.h"

#include <uct/ze/base/ze_base.h>

#include <string.h>
#include <limits.h>
#include <ucm/api/ucm.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/class.h>
#include <ucs/memory/memtype_cache.h>


static ucs_config_field_t uct_ze_copy_md_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_ze_copy_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"DEVICE_ORDINAL", "0",
     "Ordinal of the GPU device to allocate memory from.",
     ucs_offsetof(uct_ze_copy_md_config_t, device_ordinal),
     UCS_CONFIG_TYPE_INT},

    {NULL}
};

static ucs_status_t uct_ze_copy_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_ALLOC;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->alloc_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_MANAGED);
    md_attr->dmabuf_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                                UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE);
    md_attr->max_alloc        = SIZE_MAX;
    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_mem_alloc(uct_md_h tl_md, size_t *length_p, void **address_p,
                      ucs_memory_type_t mem_type, unsigned flags,
                      const char *alloc_name, uct_mem_h *memh_p)
{
    uct_ze_copy_md_t *md = ucs_derived_of(tl_md, uct_ze_copy_md_t);
    ze_host_mem_alloc_desc_t host_desc  = {};
    ze_device_mem_alloc_desc_t dev_desc = {};
    size_t alignment                    = ucs_get_page_size();
    ucs_status_t status;

    switch (mem_type) {
    case UCS_MEMORY_TYPE_ZE_HOST:
        status = UCT_ZE_FUNC_LOG_ERR(zeMemAllocHost(md->ze_context, &host_desc,
                                                    *length_p, alignment,
                                                    address_p));
        break;
    case UCS_MEMORY_TYPE_ZE_DEVICE:
        status = UCT_ZE_FUNC_LOG_ERR(zeMemAllocDevice(md->ze_context, &dev_desc,
                                                      *length_p, alignment,
                                                      md->ze_device,
                                                      address_p));
        break;
    case UCS_MEMORY_TYPE_ZE_MANAGED:
        status = UCT_ZE_FUNC_LOG_ERR(zeMemAllocShared(md->ze_context, &dev_desc,
                                                      &host_desc, *length_p,
                                                      alignment, md->ze_device,
                                                      address_p));
        break;
    default:
        ucs_debug("unsupported mem_type: %d", mem_type);
        status = UCS_ERR_UNSUPPORTED;
        break;
    }

    if (status == UCS_OK) {
        *memh_p = *address_p;
    }

    return status;
}

static ucs_status_t uct_ze_copy_mem_free(uct_md_h tl_md, uct_mem_h memh)
{
    uct_ze_copy_md_t *md = ucs_derived_of(tl_md, uct_ze_copy_md_t);

    return UCT_ZE_FUNC_LOG_ERR(zeMemFree(md->ze_context, (void*)memh));
}

static ucs_status_t uct_ze_copy_rkey_unpack(uct_component_t *component,
                                            const void *rkey_buffer,
                                            uct_rkey_t *rkey_p, void **handle_p)
{
    *handle_p = NULL;
    *rkey_p   = 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_mem_reg(uct_md_h md, void *address, size_t length,
                    const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    /* memory registration it not needed for ZE */
    *memh_p = (uct_mem_h)0xdeadbeef;
    return UCS_OK;
}

static void uct_ze_copy_md_close(uct_md_h uct_md)
{
    uct_ze_copy_md_t *md = ucs_derived_of(uct_md, uct_ze_copy_md_t);

    zeContextDestroy(md->ze_context);
    ucs_free(md);
}

static ucs_status_t
uct_ze_copy_md_query_attributes(uct_md_h md, const void *addr, size_t length,
                                ucs_memory_info_t *mem_info, int *dmabuf_fd)
{
    uct_ze_copy_md_t *ze_md = ucs_derived_of(md, uct_ze_copy_md_t);
    ze_external_memory_export_fd_t export_fd = {
        .stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
        .flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF
    };
    ze_memory_allocation_properties_t props  = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
        .pNext = dmabuf_fd ? &export_fd : NULL
    };
    ze_result_t ret;
    void *base_address;
    size_t alloc_length;

    ret = zeMemGetAllocProperties(ze_md->ze_context, addr, &props, NULL);
    if ((ret != ZE_RESULT_SUCCESS) || (props.type == ZE_MEMORY_TYPE_UNKNOWN)) {
        return UCS_ERR_INVALID_ADDR;
    }

    ret = zeMemGetAddressRange(ze_md->ze_context, addr, &base_address,
                               &alloc_length);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (props.type == ZE_MEMORY_TYPE_HOST) {
        mem_info->type = UCS_MEMORY_TYPE_ZE_HOST;
    } else if (props.type == ZE_MEMORY_TYPE_DEVICE) {
        mem_info->type = UCS_MEMORY_TYPE_ZE_DEVICE;
    } else {
        mem_info->type = UCS_MEMORY_TYPE_ZE_MANAGED;
    }
    mem_info->base_address = base_address;
    mem_info->alloc_length = alloc_length;
    mem_info->sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;
    if (dmabuf_fd) {
        *dmabuf_fd = export_fd.fd;
    }
    return UCS_OK;
}

static ucs_status_t uct_ze_copy_md_mem_query(uct_md_h md, const void *addr,
                                             const size_t length,
                                             uct_md_mem_attr_t *mem_attr_p)
{
    int dmabuf_fd = UCT_DMABUF_FD_INVALID;
    ucs_status_t status;
    ucs_memory_info_t mem_info;

    status = uct_ze_copy_md_query_attributes(md, addr, length, &mem_info,
                                             &dmabuf_fd);
    if (status != UCS_OK) {
        return status;
    }

    ucs_memtype_cache_update(mem_info.base_address, mem_info.alloc_length,
                             mem_info.type, mem_info.sys_dev);

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_info.type;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = mem_info.sys_dev;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS) {
        mem_attr_p->base_address = mem_info.base_address;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH) {
        mem_attr_p->alloc_length = mem_info.alloc_length;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_FD) {
        mem_attr_p->dmabuf_fd = dup(dmabuf_fd);
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_OFFSET) {
        mem_attr_p->dmabuf_offset = UCS_PTR_BYTE_DIFF(mem_info.base_address,
                                                      addr);
    }
    return UCS_OK;
}

static ucs_status_t
uct_ze_copy_md_detect_memory_type(uct_md_h md, const void *addr, size_t length,
                                  ucs_memory_type_t *mem_type_p)
{
    uct_md_mem_attr_t mem_attr;
    ucs_status_t status;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;
    status = uct_ze_copy_md_mem_query(md, addr, length, &mem_attr);
    if (status != UCS_OK) {
        return status;
    }

    *mem_type_p = mem_attr.mem_type;
    return UCS_OK;
}

static uct_md_ops_t md_ops = {
    .close              = uct_ze_copy_md_close,
    .query              = uct_ze_copy_md_query,
    .mem_alloc          = uct_ze_copy_mem_alloc,
    .mem_free           = uct_ze_copy_mem_free,
    .mkey_pack          = ucs_empty_function_return_success,
    .mem_reg            = uct_ze_copy_mem_reg,
    .mem_dereg          = ucs_empty_function_return_success,
    .mem_attach         = ucs_empty_function_return_unsupported,
    .mem_query          = uct_ze_copy_md_mem_query,
    .detect_memory_type = uct_ze_copy_md_detect_memory_type,
};

static ucs_status_t
uct_ze_copy_md_open(uct_component_h component, const char *md_name,
                    const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_ze_copy_md_config_t *config = ucs_derived_of(md_config,
                                                     uct_ze_copy_md_config_t);
    uct_ze_copy_md_t *md;
    ze_driver_handle_t ze_driver;
    ze_context_desc_t context_desc = {};
    ze_result_t ret;

    ze_driver = uct_ze_base_get_driver();
    if (ze_driver == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    md = ucs_malloc(sizeof(uct_ze_copy_md_t), "uct_ze_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_ze_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->ze_device = uct_ze_base_get_device(config->device_ordinal);
    if (md->ze_device == NULL) {
        ucs_error("Failed to get device at ordial %d", config->device_ordinal);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    ret = zeContextCreate(ze_driver, &context_desc, &md->ze_context);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("zeContextCreate failed with error %x", ret);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_ze_copy_component;

    *md_p = (uct_md_h)md;
    return UCS_OK;
}

uct_component_t uct_ze_copy_component = {
    .query_md_resources = uct_ze_base_query_md_resources,
    .md_open            = uct_ze_copy_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ze_copy_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = "ze_cpy",
    .md_config = {
        .name       = "ze-copy memory domain",
        .prefix     = "ZE_COPY_",
        .table      = uct_ze_copy_md_config_table,
        .size       = sizeof(uct_ze_copy_md_config_t),
    },
    .cm_config      = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list        = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ze_copy_component),
    .flags          = 0,
    .md_vfs_init    = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_ze_copy_component);
