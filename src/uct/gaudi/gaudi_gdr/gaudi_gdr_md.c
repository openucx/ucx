/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gaudi_gdr_md.h"

#include <ucs/memory/memtype_cache.h>
#include <uct/gaudi/base/gaudi_base.h>
#include <ucs/sys/module.h>

#include <inttypes.h>
#include <fcntl.h>
#include <pthread.h>
#include <hlthunk.h>


static ucs_config_field_t uct_gaudi_md_config_table[] =
        {{"", "", NULL, ucs_offsetof(uct_gaudi_md_config_t, super),
          UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

         {"DEVICE_ID", "0", "Index of the HPU devices to query memory from.",
          ucs_offsetof(uct_gaudi_md_config_t, device_id), UCS_CONFIG_TYPE_INT},

         {NULL}};

static ucs_status_t uct_gaudi_md_query(uct_md_h md, uct_md_attr_v2_t *attr)
{
    uct_md_base_md_query(attr);
    attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_GAUDI);
    attr->dmabuf_mem_types = UCS_BIT(UCS_MEMORY_TYPE_GAUDI);
    return UCS_OK;
}

static void uct_gaudi_md_close(uct_md_h uct_md)
{
    uct_gaudi_md_t *md = ucs_derived_of(uct_md, uct_gaudi_md_t);
    uct_gaudi_base_close_dmabuf_fd(md->dmabuf_fd);
    uct_gaudi_base_close_fd(md->fd, md->fd_created);
    ucs_free(md);
}

static ucs_status_t
uct_gaudi_md_query_attributes(uct_md_h md, const void *addr, size_t length,
                              ucs_memory_info_t *mem_info, int *dmabuf_fd)
{
    uct_gaudi_md_t *gaudi_md = ucs_derived_of(md, uct_gaudi_md_t);

    void *begin = (void*)gaudi_md->device_base_address;
    void *end   = (uint8_t*)begin + gaudi_md->totalSize;

    if ((addr < begin) || (addr >= end)) {
        mem_info->type = UCS_MEMORY_TYPE_LAST;
        return UCS_ERR_OUT_OF_RANGE;
    }

    *dmabuf_fd             = gaudi_md->dmabuf_fd;
    mem_info->type         = UCS_MEMORY_TYPE_GAUDI;
    mem_info->base_address = (void*)gaudi_md->device_base_address;
    mem_info->alloc_length = (size_t)gaudi_md->totalSize;
    mem_info->sys_dev      = gaudi_md->sys_dev;
    return UCS_OK;
}

static ucs_status_t uct_gaudi_md_mem_query(uct_md_h md, const void *addr,
                                           const size_t length,
                                           uct_md_mem_attr_t *mem_attr_p)
{
    int dmabuf_fd = UCT_DMABUF_FD_INVALID;
    ucs_status_t status;
    ucs_memory_info_t mem_info;

    status = uct_gaudi_md_query_attributes(md, addr, length, &mem_info,
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
        int dup_fd = dup(dmabuf_fd);
        if (dup_fd < 0) {
            return UCS_ERR_IO_ERROR;
        }
        mem_attr_p->dmabuf_fd = dup_fd;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_OFFSET) {
        mem_attr_p->dmabuf_offset = UCS_PTR_BYTE_DIFF(mem_info.base_address,
                                                      addr);
    }
    return UCS_OK;
}

static ucs_status_t
uct_gaudi_md_detect_memory_type(uct_md_h md, const void *addr, size_t length,
                                ucs_memory_type_t *mem_type_p)
{
    uct_md_mem_attr_t mem_attr;
    ucs_status_t status;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;
    status              = uct_gaudi_md_mem_query(md, addr, length, &mem_attr);
    if (status != UCS_OK) {
        return status;
    }

    *mem_type_p = mem_attr.mem_type;
    return UCS_OK;
}

static uct_md_ops_t md_ops = {
    .close     = uct_gaudi_md_close,
    .query     = uct_gaudi_md_query,
    .mem_alloc = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
    .mem_free  = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
    .mem_advise = (uct_md_mem_advise_func_t)
            ucs_empty_function_return_unsupported,
    .mem_reg    = (uct_md_mem_reg_func_t)ucs_empty_function_return_unsupported,
    .mem_dereg = (uct_md_mem_dereg_func_t)ucs_empty_function_return_unsupported,
    .mem_query = uct_gaudi_md_mem_query,
    .mkey_pack = (uct_md_mkey_pack_func_t)ucs_empty_function_return_unsupported,
    .mem_attach         = (uct_md_mem_attach_func_t)
            ucs_empty_function_return_unsupported,
    .detect_memory_type = uct_gaudi_md_detect_memory_type,
};

static ucs_status_t
uct_gaudi_md_open(uct_component_h component, const char *md_name,
                  const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_gaudi_md_config_t *config = ucs_derived_of(md_config,
                                                   uct_gaudi_md_config_t);
    uct_gaudi_md_t *md;
    ucs_status_t status;
    bool fd_created = false;
    int fd;

    md = ucs_malloc(sizeof(uct_gaudi_md_t), "uct_gaudi_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_gaudi_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    fd = uct_gaudi_base_get_fd(config->device_id, &fd_created);
    if (fd < 0) {
        ucs_error("failed to get device fd");
        status = UCS_ERR_NO_DEVICE;
        goto err_free_md;
    }

    status = uct_gaudi_base_get_info(fd, &md->device_base_allocated_address,
                                     &md->device_base_address, &md->totalSize,
                                     &md->dmabuf_fd);

    if (status != UCS_OK) {
        ucs_error("failed to get dmabuf information");
        goto err_close_fd;
    }

    status = uct_gaudi_base_get_sysdev(fd, &md->sys_dev);
    if (status != UCS_OK) {
        ucs_error("failed to get sys dev");
        goto err_close_dmabuf;
    }

    md->fd              = fd;
    md->fd_created      = fd_created;
    md->super.ops       = &md_ops;
    md->super.component = &uct_gaudi_gdr_component;

    *md_p = (uct_md_h)md;
    return UCS_OK;

err_close_dmabuf:
    uct_gaudi_base_close_dmabuf_fd(md->dmabuf_fd);
err_close_fd:
    uct_gaudi_base_close_fd(fd, fd_created);
err_free_md:
    ucs_free(md);
    return status;
}

ucs_status_t uct_gaudi_query_md_resources(uct_component_h component,
                                          uct_md_resource_desc_t **resources_p,
                                          unsigned *num_resources_p)
{
    ucs_status_t status;

    status = uct_gaudi_base_discover_devices();
    if (status != UCS_OK) {
        ucs_debug("gaudi device discovery failed, no devices available");
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

uct_component_t uct_gaudi_gdr_component = {
    .query_md_resources = uct_gaudi_query_md_resources,
    .md_open            = uct_gaudi_md_open,
    .cm_open            = (uct_component_cm_open_func_t)
            ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_md_stub_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)
            ucs_empty_function_return_unsupported,
    .rkey_release       = (uct_component_rkey_release_func_t)
            ucs_empty_function_return_success,
    .name               = "gaudi_gdr",
    .md_config =
            {
                    .name   = "gaudi-gdr memory domain",
                    .prefix = "GAUDI_GDR_",
                    .table  = uct_gaudi_md_config_table,
                    .size   = sizeof(uct_gaudi_md_config_t),
            },
    .cm_config   = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list     = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_gaudi_gdr_component),
    .flags       = 0,
    .md_vfs_init = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_gaudi_gdr_component);
