/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "uct_pd.h"
#include "uct_iface.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <malloc.h>

UCS_LIST_HEAD(uct_pd_components_list);

ucs_config_field_t uct_pd_config_table[] = {

  {NULL}
};

/**
 * Keeps information about allocated configuration structure, to be used when
 * releasing the options.
 */
typedef struct uct_config_bundle {
    ucs_config_field_t *table;
    const char         *table_prefix;
    char               data[];
} uct_config_bundle_t;


ucs_status_t uct_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_pd_resource_desc_t *resources, *pd_resources, *tmp;
    unsigned i, num_resources, num_pd_resources;
    uct_pd_component_t *pdc;
    ucs_status_t status;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        status = pdc->query_resources(&pd_resources, &num_pd_resources);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s* resources: %s", pdc->name,
                      ucs_status_string(status));
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_pd_resources),
                          "pd_resources");
        if (tmp == NULL) {
            ucs_free(pd_resources);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        for (i = 0; i < num_pd_resources; ++i) {
            ucs_assertv_always(!strncmp(pdc->name, pd_resources[i].pd_name,
                                       strlen(pdc->name)),
                               "PD name must begin with PD component name");
        }
        resources = tmp;
        memcpy(resources + num_resources, pd_resources,
               sizeof(*pd_resources) * num_pd_resources);
        num_resources += num_pd_resources;
        ucs_free(pd_resources);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err:
    ucs_free(resources);
    return status;
}

void uct_release_pd_resource_list(uct_pd_resource_desc_t *resources)
{
    ucs_free(resources);
}

ucs_status_t uct_pd_open(const char *pd_name, const uct_pd_config_t *config,
                         uct_pd_h *pd_p)
{
    uct_pd_component_t *pdc;
    ucs_status_t status;
    uct_pd_h pd;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        if (!strncmp(pd_name, pdc->name, strlen(pdc->name))) {
            status = pdc->pd_open(pd_name, config, &pd);
            if (status != UCS_OK) {
                return status;
            }

            ucs_assert_always(pd->component == pdc);
            *pd_p = pd;
            return UCS_OK;
        }
    }

    ucs_error("PD '%s' does not exist", pd_name);
    return UCS_ERR_NO_DEVICE;
}

void uct_pd_close(uct_pd_h pd)
{
    pd->ops->close(pd);
}

ucs_status_t uct_pd_query_tl_resources(uct_pd_h pd,
                                       uct_tl_resource_desc_t **resources_p,
                                       unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources, *tl_resources, *tmp;
    unsigned i, num_resources, num_tl_resources;
    uct_pd_component_t *pdc = pd->component;
    uct_pd_registered_tl_t *tlr;
    uct_tl_component_t *tlc;
    ucs_status_t status;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(tlr, &pdc->tl_list, list) {
        tlc = tlr->tl;

        status = tlc->query_resources(pd, &tl_resources, &num_tl_resources);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s resources: %s", tlc->name,
                      ucs_status_string(status));
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_tl_resources),
                          "pd_resources");
        if (tmp == NULL) {
            ucs_free(tl_resources);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        for (i = 0; i < num_tl_resources; ++i) {
            ucs_assert_always(!strcmp(tlc->name, tl_resources[i].tl_name));
        }
        resources = tmp;
        memcpy(resources + num_resources, tl_resources,
               sizeof(*tl_resources) * num_tl_resources);
        num_resources += num_tl_resources;
        ucs_free(tl_resources);
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

ucs_status_t uct_single_pd_resource(uct_pd_component_t *pdc,
                                    uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_pd_resource_desc_t *resource;

    resource = ucs_malloc(sizeof(*resource), "pd resource");
    if (resource == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->pd_name, UCT_PD_NAME_MAX, "%s", pdc->name);

    *resources_p     = resource;
    *num_resources_p = 1;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    self->async       = async;
    self->thread_mode = thread_mode;
    ucs_callbackq_init(&self->progress_q, 64);
    ucs_list_head_init(&self->tl_data);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_worker_t)
{
    ucs_callbackq_cleanup(&self->progress_q);
}

void uct_worker_progress(uct_worker_h worker)
{
    ucs_callbackq_dispatch(&worker->progress_q);
}


void uct_worker_progress_register(uct_worker_h worker,
                                  ucs_callback_t func, void *arg)
{
    ucs_callbackq_add(&worker->progress_q, func, arg);
}

void uct_worker_progress_unregister(uct_worker_h worker,
                                    ucs_callback_t func, void *arg)
{
    ucs_callbackq_remove(&worker->progress_q, func, arg);
}

UCS_CLASS_DEFINE(uct_worker_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_worker_create, uct_worker_t, uct_worker_t,
                                ucs_async_context_t*, ucs_thread_mode_t)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(uct_worker_destroy, uct_worker_t, uct_worker_t)


static ucs_status_t uct_config_read(uct_config_bundle_t **bundle,
                                    ucs_config_field_t *config_table,
                                    size_t config_size, const char *env_prefix,
                                    const char *cfg_prefix)
{
    uct_config_bundle_t *config_bundle;
    ucs_status_t status;

    config_bundle = ucs_calloc(1, sizeof(*config_bundle) + config_size, "uct_config");
    if (config_bundle == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* TODO use env_prefix */
    status = ucs_config_parser_fill_opts(config_bundle->data, config_table,
                                         env_prefix, cfg_prefix, 0);
    if (status != UCS_OK) {
        goto err_free_bundle;
    }

    config_bundle->table = config_table;
    config_bundle->table_prefix = strdup(cfg_prefix);
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

static uct_tl_component_t *uct_find_tl_on_pd(uct_pd_component_t *pdc,
                                             const char *tl_name)
{
    uct_pd_registered_tl_t *tlr;

    ucs_list_for_each(tlr, &pdc->tl_list, list) {
        if (!strcmp(tl_name, tlr->tl->name)) {
            return tlr->tl;
        }
    }
    return NULL;
}

static uct_tl_component_t *uct_find_tl(const char *tl_name)
{
    uct_pd_component_t *pdc;
    uct_tl_component_t *tlc;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        tlc = uct_find_tl_on_pd(pdc, tl_name);
        if (tlc != NULL) {
            return tlc;
        }
    }
    return NULL;
}

ucs_status_t uct_iface_config_read(const char *tl_name, const char *env_prefix,
                                   const char *filename,
                                   uct_iface_config_t **config_p)
{
    uct_config_bundle_t *bundle = NULL;
    uct_tl_component_t *tlc;
    ucs_status_t status;

    tlc = uct_find_tl(tl_name);
    if (tlc == NULL) {
        ucs_error("Transport '%s' does not exist", tl_name);
        status = UCS_ERR_NO_DEVICE; /* Non-existing transport */
        return status;
    }

    status = uct_config_read(&bundle, tlc->iface_config_table,
                             tlc->iface_config_size, env_prefix, tlc->cfg_prefix);
    if (status != UCS_OK) {
        ucs_error("Failed to read iface config");
        return status;
    }

    *config_p = (uct_iface_config_t*) bundle->data;
    return UCS_OK;
}

ucs_status_t uct_iface_open(uct_pd_h pd, uct_worker_h worker, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            const uct_iface_config_t *config, uct_iface_h *iface_p)
{
    uct_tl_component_t *tlc;

    tlc = uct_find_tl_on_pd(pd->component, tl_name);
    if (tlc == NULL) {
        /* Non-existing transport */
        return UCS_ERR_NO_DEVICE;
    }

    return tlc->iface_open(pd, worker, dev_name, rx_headroom, config, iface_p);
}

static uct_pd_component_t *uct_find_pdc(const char *name)
{
    uct_pd_component_t *pdc;

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        if (!strncmp(name, pdc->name, strlen(pdc->name))) {
            return pdc;
        }
    }
    return NULL;
}

ucs_status_t uct_pd_config_read(const char *name, const char *env_prefix,
                                const char *filename,
                                uct_pd_config_t **config_p)
{
    uct_config_bundle_t *bundle = NULL;
    uct_pd_component_t *pdc;
    ucs_status_t status;

    /* find the matching pdc. the search can be by pd_name or by pdc_name.
     * (depending on the caller) */
    pdc = uct_find_pdc(name);
    if (pdc == NULL) {
        ucs_error("PD component does not exist for '%s'", name);
        status = UCS_ERR_INVALID_PARAM; /* Non-existing PDC */
        return status;
    }

    status = uct_config_read(&bundle, pdc->pd_config_table,
                             pdc->pd_config_size, env_prefix, pdc->cfg_prefix);
    if (status != UCS_OK) {
        ucs_error("Failed to read PD config");
        return status;
    }

    *config_p = (uct_pd_config_t*) bundle->data;
    return UCS_OK;
}

void uct_config_release(void *config)
{
    uct_config_bundle_t *bundle = (uct_config_bundle_t *)config - 1;

    ucs_config_parser_release_opts(config, bundle->table);
    ucs_free((void*)(bundle->table_prefix));
    ucs_free(bundle);
}

void uct_config_print(const void *config, FILE *stream, const char *title,
                      ucs_config_print_flags_t print_flags)
{
    uct_config_bundle_t *bundle = (uct_config_bundle_t *)config - 1;
    ucs_config_parser_print_opts(stream, title, bundle->data, bundle->table,
                                 bundle->table_prefix, print_flags);
}

ucs_status_t uct_config_modify(void *config, const char *name, const char *value)
{
    uct_config_bundle_t *bundle = (uct_config_bundle_t *)config - 1;
    return ucs_config_parser_set_value(bundle->data, bundle->table, name, value);
}

void uct_pd_component_config_print(ucs_config_print_flags_t print_flags)
{
    uct_pd_component_t *pdc;
    uct_pd_config_t *pd_config;
    char cfg_title[UCT_TL_NAME_MAX + 128];
    ucs_status_t status;

    /* go over the list of pd components and print the config table per each */
    ucs_list_for_each(pdc, &uct_pd_components_list, list)
    {
        snprintf(cfg_title, sizeof(cfg_title), "%s PD component configuration",
                 pdc->name);
        status = uct_pd_config_read(pdc->name, NULL, NULL, &pd_config);
        if (status != UCS_OK) {
            ucs_error("Failed to read pd_config for PD component %s", pdc->name);
            continue;
        }
        uct_config_print(pd_config, stdout, cfg_title, print_flags);
        uct_config_release(pd_config);
    }
}

ucs_status_t uct_pd_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer)
{
    memcpy(rkey_buffer, pd->component->name, UCT_PD_COMPONENT_NAME_MAX);
    return pd->ops->mkey_pack(pd, memh, rkey_buffer + UCT_PD_COMPONENT_NAME_MAX);
}

ucs_status_t uct_rkey_unpack(const void *rkey_buffer, uct_rkey_bundle_t *rkey_ob)
{
    uct_pd_component_t *pdc;
    ucs_status_t status;
    char pdc_name[UCT_PD_COMPONENT_NAME_MAX + 1];

    ucs_list_for_each(pdc, &uct_pd_components_list, list) {
        if (!strncmp(rkey_buffer, pdc->name, UCT_PD_COMPONENT_NAME_MAX)) {
            status = pdc->rkey_unpack(pdc, rkey_buffer + UCT_PD_COMPONENT_NAME_MAX,
                                      &rkey_ob->rkey, &rkey_ob->handle);
            if (status == UCS_OK) {
                rkey_ob->type = pdc;
            }

            return status;
        }
    }

    ucs_snprintf_zero(pdc_name, sizeof(pdc_name), "%s", (const char*)rkey_buffer);
    ucs_debug("No matching PD component found for '%s'", pdc_name);
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_rkey_release(const uct_rkey_bundle_t *rkey_ob)
{
    uct_pd_component_t *pdc = rkey_ob->type;
    return pdc->rkey_release(pdc, rkey_ob->rkey, rkey_ob->handle);
}

ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    ucs_status_t status;

    status = pd->ops->query(pd, pd_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* PD component name + data */
    memcpy(pd_attr->component_name, pd->component->name, UCT_PD_COMPONENT_NAME_MAX);
    pd_attr->rkey_packed_size += UCT_PD_COMPONENT_NAME_MAX;

    return UCS_OK;
}

ucs_status_t uct_pd_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                              const char *alloc_name, uct_mem_h *memh_p)
{
    return pd->ops->mem_alloc(pd, length_p, address_p, memh_p UCS_MEMTRACK_VAL);
}

ucs_status_t uct_pd_mem_free(uct_pd_h pd, uct_mem_h memh)
{
    return pd->ops->mem_free(pd, memh);
}

ucs_status_t uct_pd_mem_reg(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p)
{
    if ((length == 0) || (address == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return pd->ops->mem_reg(pd, address, length, memh_p);
}

ucs_status_t uct_pd_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    return pd->ops->mem_dereg(pd, memh);
}
