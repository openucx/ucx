/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_md.h"
#include "uct_iface.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/memory/rcache.h>
#include <ucs/type/class.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <ucs/time/time.h>
#include <ucs/arch/cpu.h>
#include <ucs/vfs/base/vfs_obj.h>


ucs_config_field_t uct_md_config_table[] = {

  {NULL}
};

ucs_config_field_t uct_md_config_rcache_table[] = {
    {"RCACHE_MEM_PRIO", "1000", "Registration cache memory event priority",
     ucs_offsetof(uct_md_rcache_config_t, event_prio), UCS_CONFIG_TYPE_UINT},

    {"RCACHE_OVERHEAD", "auto", "Registration cache lookup overhead",
     ucs_offsetof(uct_md_rcache_config_t, overhead), UCS_CONFIG_TYPE_TIME_UNITS},

    {"RCACHE_ADDR_ALIGN", UCS_PP_MAKE_STRING(UCS_SYS_CACHE_LINE_SIZE),
     "Registration cache address alignment, must be power of 2\n"
     "between " UCS_PP_MAKE_STRING(UCS_PGT_ADDR_ALIGN) "and system page size",
     ucs_offsetof(uct_md_rcache_config_t, alignment), UCS_CONFIG_TYPE_UINT},

    {"RCACHE_MAX_REGIONS", "inf",
     "Maximal number of regions in the registration cache",
     ucs_offsetof(uct_md_rcache_config_t, max_regions),
     UCS_CONFIG_TYPE_ULUNITS},

    {"RCACHE_MAX_SIZE", "inf",
     "Maximal total size of registration cache regions",
     ucs_offsetof(uct_md_rcache_config_t, max_size), UCS_CONFIG_TYPE_MEMUNITS},

    {"RCACHE_MAX_UNRELEASED", "512M",
     "Maximal size of total memory regions in invalidate queue and garbage,\n"
     "after which a cleanup is triggered.",
     ucs_offsetof(uct_md_rcache_config_t, max_unreleased),
     UCS_CONFIG_TYPE_MEMUNITS},

    {"RCACHE_PURGE_ON_FORK", "y",
     "Purge registration cache upon fork",
     ucs_offsetof(uct_md_rcache_config_t, purge_on_fork), UCS_CONFIG_TYPE_BOOL},

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

    status = uct_config_read(&bundle, tl->config.table, tl->config.size,
                             env_prefix, tl->config.prefix);
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

    status = uct_config_read(&bundle, component->md_config.table,
                             component->md_config.size, env_prefix,
                             component->md_config.prefix);
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
    return ucs_config_parser_set_value(bundle->data, bundle->table, name, value);
}

static ucs_status_t
uct_md_mkey_pack_params_check(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    if (ENABLE_PARAMS_CHECK) {
        return ((md != NULL) && (memh != NULL) && (rkey_buffer != NULL)) ?
               UCS_OK : UCS_ERR_INVALID_PARAM;
    } else {
        return UCS_OK;
    }
}

ucs_status_t uct_md_mkey_pack_v2(uct_md_h md, uct_mem_h memh,
                                 const uct_md_mkey_pack_params_t *params,
                                 void *rkey_buffer)
{
    ucs_status_t status;

    status = uct_md_mkey_pack_params_check(md, memh, rkey_buffer);
    if (status != UCS_OK) {
        return status;
    }

    return md->ops->mkey_pack(md, memh, params, rkey_buffer);
}

ucs_status_t uct_md_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_md_mkey_pack_params_t params = {
        .field_mask = 0
    };

    return uct_md_mkey_pack_v2(md, memh, &params, rkey_buffer);
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

ucs_status_t uct_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    ucs_status_t status;

    status = md->ops->query(md, md_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* Component name + data */
    memcpy(md_attr->component_name, md->component->name, UCT_COMPONENT_NAME_MAX);

    return UCS_OK;
}

static ucs_status_t uct_mem_check_flags(unsigned flags)
{
    if (!(flags & UCT_MD_MEM_ACCESS_ALL)) {
        return UCS_ERR_INVALID_PARAM;
    }
    return UCS_OK;
}

ucs_status_t uct_mem_alloc_check_params(size_t length,
                                        const uct_alloc_method_t *methods,
                                        unsigned num_methods,
                                        const uct_mem_alloc_params_t *params)
{
    ucs_status_t status;

    if (params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_FLAGS) {
        status = uct_mem_check_flags(params->flags);
        if (status != UCS_OK) {
            return status;
        }

        /* assuming flags are valid */
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
                  unsigned advice)
{
    if ((length == 0) || (addr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return md->ops->mem_advise(md, memh, addr, length, advice);
}

ucs_status_t uct_md_mem_reg(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p)
{
    ucs_status_t status;

    if ((length == 0) || (address == NULL)) {
        uct_md_log_mem_reg_error(flags,
                                 "uct_md_mem_reg(address=%p length=%zu): "
                                 "invalid parameters", address, length);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_mem_check_flags(flags);
    if (status != UCS_OK) {
        uct_md_log_mem_reg_error(flags,
                                 "uct_md_mem_reg(flags=0x%x): invalid flags",
                                 flags);
        return status;
    }

    return md->ops->mem_reg(md, address, length, flags, memh_p);
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
    return md->ops->is_sockaddr_accessible(md, sockaddr, mode);
}

ucs_status_t uct_md_detect_memory_type(uct_md_h md, const void *addr, size_t length,
                                       ucs_memory_type_t *mem_type_p)
{
    return md->ops->detect_memory_type(md, addr, length, mem_type_p);
}

void uct_md_set_rcache_params(ucs_rcache_params_t *rcache_params,
                              const uct_md_rcache_config_t *rcache_config)
{
    rcache_params->alignment          = rcache_config->alignment;
    rcache_params->ucm_event_priority = rcache_config->event_prio;
    rcache_params->max_regions        = rcache_config->max_regions;
    rcache_params->max_size           = rcache_config->max_size;
    rcache_params->max_unreleased     = rcache_config->max_unreleased;
    rcache_params->flags              = !rcache_config->purge_on_fork ? 0 :
                                        UCS_RCACHE_FLAG_PURGE_ON_FORK;
}

double uct_md_rcache_overhead(const uct_md_rcache_config_t *rcache_config)
{
    if (rcache_config->overhead == UCS_TIME_AUTO) {
        if (ucs_arch_get_cpu_vendor() == UCS_CPU_VENDOR_FUJITSU_ARM) {
            return 360e-9;
        } else {
            return 180e-9;
        }
    } else {
        return ucs_time_to_sec(rcache_config->overhead);
    }
}
