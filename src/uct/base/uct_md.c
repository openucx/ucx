/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017. ALL RIGHTS RESERVED.
* Copyright (c) Triad National Security, LLC. 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_md.h"
#include "uct_iface.h"

#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/memory/rcache.h>
#include <ucs/type/class.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <ucs/time/time.h>
#include <ucs/arch/cpu.h>
#include <ucs/vfs/base/vfs_obj.h>


#define UCT_MD_ATTR_V2_FIELD_COPY(_md_attr_dst, _md_attr_src, _field_name, \
                                  _field_flag) \
    { \
        if ((_md_attr_dst)->field_mask & (_field_flag)) { \
            ucs_assert((_md_attr_src)->field_mask &(_field_flag)); \
            memcpy(&((_md_attr_dst)->_field_name), \
                   &((_md_attr_src)->_field_name), \
                   sizeof((_md_attr_src)->_field_name)); \
        } \
    }


ucs_config_field_t uct_md_config_table[] = {

  {NULL}
};


const char *uct_device_type_names[] = {
    [UCT_DEVICE_TYPE_NET]  = "network",
    [UCT_DEVICE_TYPE_SHM]  = "intra-node",
    [UCT_DEVICE_TYPE_ACC]  = "accelerator",
    [UCT_DEVICE_TYPE_SELF] = "loopback",
};

ucs_status_t uct_md_open(uct_component_h component, const char *md_name,
                         const uct_md_config_t *config, uct_md_h *md_p)
{
    ucs_status_t status;
    uct_md_h md;

    status = component->md_open(component, md_name, config, &md);
    if (status != UCS_OK) {
        return status;
    }

    uct_md_vfs_init(component, md, md_name);
    *md_p = md;

    ucs_assert_always(md->component == component);
    return UCS_OK;
}

void uct_md_close(uct_md_h md)
{
    ucs_vfs_obj_remove(md);
    md->ops->close(md);
}

ucs_status_t uct_md_query_tl_resources(uct_md_h md,
                                       uct_tl_resource_desc_t **resources_p,
                                       unsigned *num_resources_p)
{
    uct_component_t *component = md->component;
    uct_tl_resource_desc_t *resources, *tmp;
    uct_tl_device_resource_t *tl_devices;
    unsigned i, num_resources, num_tl_devices;
    ucs_status_t status;
    uct_tl_t *tl;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(tl, &component->tl_list, list) {
        status = tl->query_devices(md, &tl_devices, &num_tl_devices);
        if (status != UCS_OK) {
            ucs_debug("failed to query %s resources: %s", tl->name,
                      ucs_status_string(status));
            continue;
        }

        if (num_tl_devices == 0) {
            ucs_free(tl_devices);
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_tl_devices),
                          "md_resources");
        if (tmp == NULL) {
            ucs_free(tl_devices);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        /* add tl devices to overall list of resources */
        for (i = 0; i < num_tl_devices; ++i) {
            ucs_strncpy_zero(tmp[num_resources + i].tl_name, tl->name,
                             sizeof(tmp[num_resources + i].tl_name));
            ucs_strncpy_zero(tmp[num_resources + i].dev_name, tl_devices[i].name,
                             sizeof(tmp[num_resources + i].dev_name));
            tmp[num_resources + i].dev_type   = tl_devices[i].type;
            tmp[num_resources + i].sys_device = tl_devices[i].sys_device;
        }

        resources      = tmp;
        num_resources += num_tl_devices;
        ucs_free(tl_devices);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err:
    ucs_free(resources);
    return status;
}

void uct_release_tl_resource_list(uct_tl_resource_desc_t *resources)
{
    ucs_free(resources);
}

ucs_status_t
uct_md_query_single_md_resource(uct_component_t *component,
                                uct_md_resource_desc_t **resources_p,
                                unsigned *num_resources_p)
{
    uct_md_resource_desc_t *resource;

    resource = ucs_malloc(sizeof(*resource), "md resource");
    if (resource == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->md_name, UCT_MD_NAME_MAX, "%s",
                      component->name);

    *resources_p     = resource;
    *num_resources_p = 1;
    return UCS_OK;
}

ucs_status_t
uct_md_query_empty_md_resource(uct_md_resource_desc_t **resources_p,
                               unsigned *num_resources_p)
{
    *resources_p     = NULL;
    *num_resources_p = 0;
    return UCS_OK;
}

ucs_status_t uct_md_stub_rkey_unpack(uct_component_t *component,
                                     const void *rkey_buffer, uct_rkey_t *rkey_p,
                                     void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static uct_tl_t *uct_find_tl(uct_component_h component, const char *tl_name)
{
    uct_tl_t *tl;

    ucs_list_for_each(tl, &component->tl_list, list) {
        if ((tl_name != NULL) && !strcmp(tl_name, tl->name)) {
            return tl;
        }
    }
    return NULL;
}

ucs_status_t uct_md_iface_config_read(uct_md_h md, const char *tl_name,
                                      const char *env_prefix, const char *filename,
                                      uct_iface_config_t **config_p)
{
    uct_config_bundle_t *bundle = NULL;
    ucs_status_t status;
    uct_tl_t *tl;

    tl = uct_find_tl(md->component, tl_name);
    if (tl == NULL) {
        if (tl_name == NULL) {
            ucs_error("There is no sockaddr transport registered on the md");
        } else {
            ucs_error("Transport '%s' does not exist", tl_name);
        }
        status = UCS_ERR_NO_DEVICE; /* Non-existing transport */
        return status;
    }

    status = uct_config_read(&bundle, &tl->config, env_prefix);
    if (status != UCS_OK) {
        ucs_error("Failed to read iface config");
        return status;
    }

    *config_p = (uct_iface_config_t*) bundle->data;
    /* coverity[leaked_storage] */
    return UCS_OK;
}

ucs_status_t uct_iface_open(uct_md_h md, uct_worker_h worker,
                            const uct_iface_params_t *params,
                            const uct_iface_config_t *config,
                            uct_iface_h *iface_p)
{
    ucs_status_t status;
    uct_tl_t *tl;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");

    if (params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE) {
        tl = uct_find_tl(md->component, params->mode.device.tl_name);
    } else if ((params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_CLIENT) ||
               (params->open_mode & UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER)) {
        tl = uct_find_tl(md->component, NULL);
    } else {
        ucs_error("Invalid open mode %"PRIu64, params->open_mode);
        return UCS_ERR_INVALID_PARAM;
    }

    if (tl == NULL) {
        /* Non-existing transport */
        return UCS_ERR_NO_DEVICE;
    }

    status = tl->iface_open(md, worker, params, config, iface_p);
    if (status != UCS_OK) {
        return status;
    }

    ucs_vfs_obj_add_dir(worker, *iface_p, "iface/%p", *iface_p);
    ucs_vfs_obj_add_sym_link(*iface_p, md, "memory_domain");
    ucs_vfs_obj_set_dirty(*iface_p, uct_iface_vfs_refresh);

    return UCS_OK;
}

ucs_status_t uct_md_config_read(uct_component_h component,
                                const char *env_prefix, const char *filename,
                                uct_md_config_t **config_p)
{
    uct_config_bundle_t *bundle = NULL;
    ucs_status_t status;

    status = uct_config_read(&bundle, &component->md_config, env_prefix);
    if (status != UCS_OK) {
        ucs_error("Failed to read MD config");
        return status;
    }

    *config_p = (uct_md_config_t*) bundle->data;
    /* coverity[leaked_storage] */
    return UCS_OK;
}

void uct_config_release(void *config)
{
    uct_config_bundle_t *bundle = (uct_config_bundle_t *)config - 1;

    ucs_config_parser_release_opts(config, bundle->table);
    ucs_free((void*)(bundle->table_prefix));
    ucs_free(bundle);
}

ucs_status_t uct_config_get(void *config, const char *name, char *value,
                            size_t max)
{
    uct_config_bundle_t *bundle = (uct_config_bundle_t *)config - 1;
    return ucs_config_parser_get_value(bundle->data, bundle->table, name, value,
                                       max);
}

ucs_status_t uct_config_modify(void *config, const char *name, const char *value)
{
    uct_config_bundle_t *bundle = (uct_config_bundle_t *)config - 1;
    return ucs_config_parser_set_value(bundle->data, bundle->table,
                                       bundle->table_prefix, name, value);
}

static ucs_status_t
uct_md_mkey_pack_params_check(uct_md_h md, uct_mem_h memh, void *mkey_buffer)
{
    if (ENABLE_PARAMS_CHECK) {
        return ((md != NULL) && (memh != NULL) && (mkey_buffer != NULL)) ?
               UCS_OK : UCS_ERR_INVALID_PARAM;
    } else {
        return UCS_OK;
    }
}

ucs_status_t uct_md_mkey_pack_v2(uct_md_h md, uct_mem_h memh,
                                 void *address, size_t length,
                                 const uct_md_mkey_pack_params_t *params,
                                 void *mkey_buffer)
{
    ucs_status_t status;

    status = uct_md_mkey_pack_params_check(md, memh, mkey_buffer);
    if (status != UCS_OK) {
        return status;
    }

    return md->ops->mkey_pack(md, memh, address, length, params, mkey_buffer);
}

ucs_status_t uct_md_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_md_mkey_pack_params_t params = {
        .field_mask = 0
    };

    return uct_md_mkey_pack_v2(md, memh, NULL, SIZE_MAX, &params, rkey_buffer);
}

ucs_status_t uct_md_mem_attach(uct_md_h md, const void *mkey_buffer,
                               uct_md_mem_attach_params_t *params,
                               uct_mem_h *memh_p)
{
    return md->ops->mem_attach(md, mkey_buffer, params, memh_p);
}

ucs_status_t uct_rkey_unpack(uct_component_h component, const void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob)
{
    return component->rkey_unpack(component, rkey_buffer, &rkey_ob->rkey,
                                  &rkey_ob->handle);
}

ucs_status_t uct_rkey_ptr(uct_component_h component, uct_rkey_bundle_t *rkey_ob,
                          uint64_t remote_addr, void **local_addr_p)
{
    return component->rkey_ptr(component, rkey_ob->rkey, rkey_ob->handle,
                               remote_addr, local_addr_p);
}

ucs_status_t uct_rkey_release(uct_component_h component,
                              const uct_rkey_bundle_t *rkey_ob)
{
    return component->rkey_release(component, rkey_ob->rkey, rkey_ob->handle);
}

ucs_status_t uct_base_rkey_compare(uct_component_t *component, uct_rkey_t rkey1,
                                   uct_rkey_t rkey2,
                                   const uct_rkey_compare_params_t *params,
                                   int *result)
{
    if ((params->field_mask != 0) || (result == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    *result = (rkey1 > rkey2) ? 1 : (rkey1 < rkey2) ? -1 : 0;
    return UCS_OK;
}

ucs_status_t
uct_rkey_compare(uct_component_h component, uct_rkey_t rkey1, uct_rkey_t rkey2,
                 const uct_rkey_compare_params_t *params, int *result)
{
    return component->rkey_compare(component, rkey1, rkey2, params, result);
}

static void uct_md_attr_from_v2(uct_md_attr_t *dst, const uct_md_attr_v2_t *src)
{
    dst->cap.max_alloc        = src->max_alloc;
    dst->cap.max_reg          = src->max_reg;
    dst->cap.flags            = src->flags;
    dst->cap.reg_mem_types    = src->reg_mem_types;
    dst->cap.detect_mem_types = src->detect_mem_types;
    dst->cap.alloc_mem_types  = src->alloc_mem_types;
    dst->cap.access_mem_types = src->access_mem_types;
    dst->reg_cost             = src->reg_cost;
    dst->rkey_packed_size     = src->rkey_packed_size;

    memcpy(&dst->local_cpus, &src->local_cpus, sizeof(src->local_cpus));
    memcpy(&dst->component_name, &src->component_name,
           sizeof(src->component_name));
}

static void
uct_md_attr_v2_copy(uct_md_attr_v2_t *dst, const uct_md_attr_v2_t *src)
{
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, max_alloc, UCT_MD_ATTR_FIELD_MAX_ALLOC);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, max_reg, UCT_MD_ATTR_FIELD_MAX_REG);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, flags, UCT_MD_ATTR_FIELD_FLAGS);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, reg_mem_types,
                              UCT_MD_ATTR_FIELD_REG_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, reg_nonblock_mem_types,
                              UCT_MD_ATTR_FIELD_REG_NONBLOCK_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, cache_mem_types,
                              UCT_MD_ATTR_FIELD_CACHE_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, detect_mem_types,
                              UCT_MD_ATTR_FIELD_DETECT_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, alloc_mem_types,
                              UCT_MD_ATTR_FIELD_ALLOC_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, access_mem_types,
                              UCT_MD_ATTR_FIELD_ACCESS_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, gva_mem_types,
                              UCT_MD_ATTR_FIELD_GVA_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, dmabuf_mem_types,
                              UCT_MD_ATTR_FIELD_DMABUF_MEM_TYPES);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, reg_cost, UCT_MD_ATTR_FIELD_REG_COST);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, rkey_packed_size,
                              UCT_MD_ATTR_FIELD_RKEY_PACKED_SIZE);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, local_cpus,
                              UCT_MD_ATTR_FIELD_LOCAL_CPUS);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, component_name,
                              UCT_MD_ATTR_FIELD_COMPONENT_NAME);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, exported_mkey_packed_size,
                              UCT_MD_ATTR_FIELD_EXPORTED_MKEY_PACKED_SIZE);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, global_id,
                              UCT_MD_ATTR_FIELD_GLOBAL_ID);
    UCT_MD_ATTR_V2_FIELD_COPY(dst, src, reg_alignment,
                              UCT_MD_ATTR_FIELD_REG_ALIGNMENT);
}

static ucs_status_t uct_md_attr_v2_init(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    ucs_status_t status;

    memset(md_attr, 0, sizeof(*md_attr));

    md_attr->field_mask = UINT64_MAX;

    status = md->ops->query(md, md_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* Component name + data */
    memcpy(md_attr->component_name, md->component->name,
           sizeof(md_attr->component_name));

    return UCS_OK;
}

ucs_status_t uct_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    ucs_status_t status;
    uct_md_attr_v2_t md_attr_v2;

    status = uct_md_attr_v2_init(md, &md_attr_v2);
    if (status != UCS_OK) {
        return status;
    }

    uct_md_attr_from_v2(md_attr, &md_attr_v2);

    return UCS_OK;
}

ucs_status_t uct_md_query_v2(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    ucs_status_t status;
    uct_md_attr_v2_t md_attr_v2;

    status = uct_md_attr_v2_init(md, &md_attr_v2);
    if (status != UCS_OK) {
        return status;
    }

    /* Populate fields based on field mask set by user in md_attr  */
    uct_md_attr_v2_copy(md_attr, &md_attr_v2);

    return UCS_OK;
}

void uct_md_base_md_query(uct_md_attr_v2_t *md_attr)
{
    md_attr->reg_mem_types             = 0;
    md_attr->reg_nonblock_mem_types    = 0;
    md_attr->cache_mem_types           = 0;
    md_attr->detect_mem_types          = 0;
    md_attr->alloc_mem_types           = 0;
    md_attr->access_mem_types          = 0;
    md_attr->dmabuf_mem_types          = 0;
    md_attr->gva_mem_types             = 0;
    md_attr->max_alloc                 = 0;
    md_attr->max_reg                   = ULONG_MAX;
    md_attr->reg_cost                  = UCS_LINEAR_FUNC_ZERO;
    md_attr->rkey_packed_size          = 0;
    md_attr->exported_mkey_packed_size = 0;
    md_attr->reg_alignment             = 1;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
}

ucs_status_t uct_mem_alloc_check_params(size_t length,
                                        const uct_alloc_method_t *methods,
                                        unsigned num_methods,
                                        const uct_mem_alloc_params_t *params)
{
    if (params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_FLAGS) {
        if (params->flags & UCT_MD_MEM_FLAG_FIXED) {
            if (!(params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS)) {
                ucs_debug("UCT_MD_MEM_FLAG_FIXED requires setting of"
                          " UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS field");
                return UCS_ERR_INVALID_PARAM;
            }

            if ((params->address == NULL) ||
                ((uintptr_t)params->address % ucs_get_page_size())) {
                ucs_debug("UCT_MD_MEM_FLAG_FIXED requires valid page size aligned address");
                return UCS_ERR_INVALID_PARAM;
            }
        }
    }

    if (length == 0) {
        ucs_debug("the length value for allocating memory is set to zero: %s",
                  ucs_status_string(UCS_ERR_INVALID_PARAM));
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t uct_md_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              ucs_memory_type_t mem_type, unsigned flags,
                              const char *alloc_name, uct_mem_h *memh_p)
{
    return md->ops->mem_alloc(md, length_p, address_p, mem_type, flags,
                              alloc_name, memh_p);
}

ucs_status_t uct_md_mem_free(uct_md_h md, uct_mem_h memh)
{
    return md->ops->mem_free(md, memh);
}

ucs_status_t
uct_md_mem_advise(uct_md_h md, uct_mem_h memh, void *addr, size_t length,
                  uct_mem_advice_t advice)
{
    if ((length == 0) || (addr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return md->ops->mem_advise(md, memh, addr, length, advice);
}

ucs_status_t uct_md_mem_reg(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p)
{
    uct_md_mem_reg_params_t params = {
        .field_mask = UCT_MD_MEM_REG_FIELD_FLAGS,
        .flags      = flags
    };

    return uct_md_mem_reg_v2(md, address, length, &params, memh_p);
}

ucs_status_t uct_md_mem_reg_v2(uct_md_h md, void *address, size_t length,
                               const uct_md_mem_reg_params_t *params,
                               uct_mem_h *memh_p)
{
    uint64_t flags = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);

    if (length == 0) {
        uct_md_log_mem_reg_error(flags,
                                 "uct_md_mem_reg(address=%p length=%zu): "
                                 "invalid parameters", address, length);
        return UCS_ERR_INVALID_PARAM;
    }

    return md->ops->mem_reg(md, address, length, params, memh_p);
}

ucs_status_t uct_md_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_md_mem_dereg_params_t params = {
        .field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH,
        .memh       = memh
    };

    return md->ops->mem_dereg(md, &params);
}

ucs_status_t uct_md_mem_dereg_v2(uct_md_h md,
                                 const uct_md_mem_dereg_params_t *params)
{
    return md->ops->mem_dereg(md, params);
}

ucs_status_t uct_md_mem_query(uct_md_h md, const void *address, size_t length,
                              uct_md_mem_attr_t *mem_attr)
{
    return md->ops->mem_query(md, address, length, mem_attr);
}

int uct_md_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                  uct_sockaddr_accessibility_t mode)
{
    return 0; /* Retained for API backward compatibility */
}

ucs_status_t uct_md_detect_memory_type(uct_md_h md, const void *addr, size_t length,
                                       ucs_memory_type_t *mem_type_p)
{
    return md->ops->detect_memory_type(md, addr, length, mem_type_p);
}

ucs_status_t uct_md_dummy_mem_reg(uct_md_h md, void *address, size_t length,
                                  const uct_md_mem_reg_params_t *params,
                                  uct_mem_h *memh_p)
{
    /* We have to emulate memory registration. Return dummy pointer */
    *memh_p = (void*)0xdeadbeef;
    return UCS_OK;
}

ucs_status_t uct_md_dummy_mem_dereg(uct_md_h uct_md,
                                    const uct_md_mem_dereg_params_t *params)
{
    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    ucs_assert(params->memh == (void*)0xdeadbeef);

    return UCS_OK;
}
