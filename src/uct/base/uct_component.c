/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_component.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <uct/tcp/tcp.h>
#include <uct/sm/self/self.h>
#include <uct/sm/mm/base/mm_iface.h>
#include <limits.h>
#include <string.h>


UCS_LIST_HEAD(uct_components_list);

UCT_TL_DECL(self)
UCT_TL_DECL(tcp)
UCT_TL_DECL(posix)
UCT_TL_DECL(sysv)

void UCS_F_CTOR uct_init()
{
    uct_self_init();
    uct_tcp_init();
    uct_sysv_init();
    uct_posix_init();
}

void UCS_F_DTOR uct_cleanup()
{
    uct_posix_cleanup();
    uct_sysv_cleanup();
    uct_tcp_cleanup();
    uct_self_cleanup();
}

ucs_status_t uct_query_components(uct_component_h **components_p,
                                  unsigned *num_components_p)
{
    UCS_MODULE_FRAMEWORK_DECLARE(uct);
    uct_component_h *components;
    uct_component_t *component;
    size_t num_components;

    UCS_MODULE_FRAMEWORK_LOAD(uct, 0);
    num_components = ucs_list_length(&uct_components_list);
    components = ucs_malloc(num_components * sizeof(*components),
                            "uct_components");
    if (components == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_assert(num_components < UINT_MAX);
    *num_components_p = num_components;
    *components_p     = components;

    ucs_list_for_each(component, &uct_components_list, list) {
        *(components++) = component;
        ucs_vfs_obj_add_dir(NULL, component, "uct/component/%s",
                            component->name);
    }

    return UCS_OK;
}

void uct_release_component_list(uct_component_h *components)
{
    ucs_free(components);
}

ucs_status_t uct_component_query(uct_component_h component,
                                 uct_component_attr_t *component_attr)
{
    uct_md_resource_desc_t *resources = NULL;
    unsigned num_resources            = 0;
    ucs_status_t status;

    if (component_attr->field_mask & (UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT|
                                      UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES)) {
        status = component->query_md_resources(component, &resources,
                                               &num_resources);
        if (status != UCS_OK) {
            return status;
        }

        ucs_assertv((num_resources == 0) || (resources != NULL),
                    "component=%s", component->name);
    }

    if (component_attr->field_mask & UCT_COMPONENT_ATTR_FIELD_NAME) {
        ucs_snprintf_zero(component_attr->name, sizeof(component_attr->name),
                          "%s", component->name);
    }

    if (component_attr->field_mask & UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT) {
        component_attr->md_resource_count = num_resources;

    }

    if ((resources != NULL) &&
        (component_attr->field_mask & UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES))
    {
        memcpy(component_attr->md_resources, resources,
               sizeof(uct_md_resource_desc_t) * num_resources);
    }

    if (component_attr->field_mask & UCT_COMPONENT_ATTR_FIELD_FLAGS) {
        component_attr->flags = component->flags;
    }

    ucs_free(resources);
    return UCS_OK;
}

ucs_status_t uct_config_read(uct_config_bundle_t **bundle,
                             ucs_config_field_t *config_table,
                             size_t config_size, const char *env_prefix,
                             const char *cfg_prefix)
{
    char full_prefix[128] = UCS_DEFAULT_ENV_PREFIX;
    uct_config_bundle_t *config_bundle;
    ucs_status_t status;

    if (config_table == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    config_bundle = ucs_calloc(1, sizeof(*config_bundle) + config_size, "uct_config");
    if (config_bundle == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* TODO use env_prefix */
    if ((env_prefix != NULL) && (strlen(env_prefix) > 0)) {
        ucs_snprintf_zero(full_prefix, sizeof(full_prefix), "%s_%s",
                          env_prefix, UCS_DEFAULT_ENV_PREFIX);
    }

    status = ucs_config_parser_fill_opts(config_bundle->data, config_table,
                                         full_prefix, cfg_prefix, 0);
    if (status != UCS_OK) {
        goto err_free_bundle;
    }

    config_bundle->table = config_table;
    config_bundle->table_prefix = ucs_strdup(cfg_prefix, "uct_config");
    if (config_bundle->table_prefix == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_bundle;
    }

    *bundle = config_bundle;
    return UCS_OK;

err_free_bundle:
    ucs_free(config_bundle);
err:
    return status;
}

void uct_component_register(uct_component_t *component)
{
    ucs_list_add_tail(&uct_components_list, &component->list);
    ucs_list_add_tail(&ucs_config_global_list, &component->md_config.list);
    ucs_list_add_tail(&ucs_config_global_list, &component->cm_config.list);
}

void uct_component_unregister(uct_component_t *component)
{
    /* TODO: add ucs_list_del(uct_components_list) */
    ucs_list_del(&component->md_config.list);
    ucs_list_del(&component->cm_config.list);
}
