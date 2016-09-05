/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "uct_md.h"
#include "uct_iface.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <malloc.h>

UCS_LIST_HEAD(uct_md_components_list);

ucs_config_field_t uct_md_config_table[] = {

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


ucs_status_t uct_query_md_resources(uct_md_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_md_resource_desc_t *resources, *md_resources, *tmp;
    unsigned i, num_resources, num_md_resources;
    uct_md_component_t *mdc;
    ucs_status_t status;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(mdc, &uct_md_components_list, list) {
        status = mdc->query_resources(&md_resources, &num_md_resources);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s* resources: %s", mdc->name,
                      ucs_status_string(status));
            continue;
        }

        if (num_md_resources == 0) {
            ucs_free(md_resources);
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_md_resources),
                          "md_resources");
        if (tmp == NULL) {
            ucs_free(md_resources);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        for (i = 0; i < num_md_resources; ++i) {
            ucs_assertv_always(!strncmp(mdc->name, md_resources[i].md_name,
                                       strlen(mdc->name)),
                               "MD name must begin with MD component name");
        }
        resources = tmp;
        memcpy(resources + num_resources, md_resources,
               sizeof(*md_resources) * num_md_resources);
        num_resources += num_md_resources;
        ucs_free(md_resources);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err:
    ucs_free(resources);
    return status;
}

void uct_release_md_resource_list(uct_md_resource_desc_t *resources)
{
    ucs_free(resources);
}

ucs_status_t uct_md_open(const char *md_name, const uct_md_config_t *config,
                         uct_md_h *md_p)
{
    uct_md_component_t *mdc;
    ucs_status_t status;
    uct_md_h md;

    ucs_list_for_each(mdc, &uct_md_components_list, list) {
        if (!strncmp(md_name, mdc->name, strlen(mdc->name))) {
            status = mdc->md_open(md_name, config, &md);
            if (status != UCS_OK) {
                return status;
            }

            ucs_assert_always(md->component == mdc);
            *md_p = md;
            return UCS_OK;
        }
    }

    ucs_error("MD '%s' does not exist", md_name);
    return UCS_ERR_NO_DEVICE;
}

void uct_md_close(uct_md_h md)
{
    md->ops->close(md);
}

ucs_status_t uct_md_query_tl_resources(uct_md_h md,
                                       uct_tl_resource_desc_t **resources_p,
                                       unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources, *tl_resources, *tmp;
    unsigned i, num_resources, num_tl_resources;
    uct_md_component_t *mdc = md->component;
    uct_md_registered_tl_t *tlr;
    uct_tl_component_t *tlc;
    ucs_status_t status;

    resources     = NULL;
    num_resources = 0;

    ucs_list_for_each(tlr, &mdc->tl_list, list) {
        tlc = tlr->tl;

        status = tlc->query_resources(md, &tl_resources, &num_tl_resources);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s resources: %s", tlc->name,
                      ucs_status_string(status));
            continue;
        }

        if (num_tl_resources == 0) {
            ucs_free(tl_resources);
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (num_resources + num_tl_resources),
                          "md_resources");
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

ucs_status_t uct_single_md_resource(uct_md_component_t *mdc,
                                    uct_md_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_md_resource_desc_t *resource;

    resource = ucs_malloc(sizeof(*resource), "md resource");
    if (resource == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->md_name, UCT_MD_NAME_MAX, "%s", mdc->name);

    *resources_p     = resource;
    *num_resources_p = 1;
    return UCS_OK;
}

ucs_status_t uct_md_stub_rkey_unpack(uct_md_component_t *mdc,
                                     const void *rkey_buffer, uct_rkey_t *rkey_p,
                                     void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    self->async       = async;
    self->thread_mode = thread_mode;
    ucs_callbackq_init(&self->progress_q, 64, async);
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

void uct_worker_slow_progress_register(uct_worker_h worker,
                                       ucs_callbackq_slow_elem_t *elem)
{
    ucs_callbackq_add_slow_path(&worker->progress_q, elem);
}

void uct_worker_slow_progress_unregister(uct_worker_h worker,
                                         ucs_callbackq_slow_elem_t *elem)
{
    ucs_callbackq_remove_slow_path(&worker->progress_q, elem);
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

static uct_tl_component_t *uct_find_tl_on_md(uct_md_component_t *mdc,
                                             const char *tl_name)
{
    uct_md_registered_tl_t *tlr;

    ucs_list_for_each(tlr, &mdc->tl_list, list) {
        if (!strcmp(tl_name, tlr->tl->name)) {
            return tlr->tl;
        }
    }
    return NULL;
}

static uct_tl_component_t *uct_find_tl(const char *tl_name)
{
    uct_md_component_t *mdc;
    uct_tl_component_t *tlc;

    ucs_list_for_each(mdc, &uct_md_components_list, list) {
        tlc = uct_find_tl_on_md(mdc, tl_name);
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

ucs_status_t uct_iface_open(uct_md_h md, uct_worker_h worker, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            const uct_iface_config_t *config, uct_iface_h *iface_p)
{
    uct_tl_component_t *tlc;

    tlc = uct_find_tl_on_md(md->component, tl_name);
    if (tlc == NULL) {
        /* Non-existing transport */
        return UCS_ERR_NO_DEVICE;
    }

    return tlc->iface_open(md, worker, dev_name, rx_headroom, config, iface_p);
}

static uct_md_component_t *uct_find_mdc(const char *name)
{
    uct_md_component_t *mdc;

    ucs_list_for_each(mdc, &uct_md_components_list, list) {
        if (!strncmp(name, mdc->name, strlen(mdc->name))) {
            return mdc;
        }
    }
    return NULL;
}

ucs_status_t uct_md_config_read(const char *name, const char *env_prefix,
                                const char *filename,
                                uct_md_config_t **config_p)
{
    uct_config_bundle_t *bundle = NULL;
    uct_md_component_t *mdc;
    ucs_status_t status;

    /* find the matching mdc. the search can be by md_name or by mdc_name.
     * (depending on the caller) */
    mdc = uct_find_mdc(name);
    if (mdc == NULL) {
        ucs_error("MD component does not exist for '%s'", name);
        status = UCS_ERR_INVALID_PARAM; /* Non-existing MDC */
        return status;
    }

    status = uct_config_read(&bundle, mdc->md_config_table,
                             mdc->md_config_size, env_prefix, mdc->cfg_prefix);
    if (status != UCS_OK) {
        ucs_error("Failed to read MD config");
        return status;
    }

    *config_p = (uct_md_config_t*) bundle->data;
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

void uct_md_component_config_print(ucs_config_print_flags_t print_flags)
{
    uct_md_component_t *mdc;
    uct_md_config_t *md_config;
    char cfg_title[UCT_TL_NAME_MAX + 128];
    ucs_status_t status;

    /* go over the list of md components and print the config table per each */
    ucs_list_for_each(mdc, &uct_md_components_list, list)
    {
        snprintf(cfg_title, sizeof(cfg_title), "%s MD component configuration",
                 mdc->name);
        status = uct_md_config_read(mdc->name, NULL, NULL, &md_config);
        if (status != UCS_OK) {
            ucs_error("Failed to read md_config for MD component %s", mdc->name);
            continue;
        }
        uct_config_print(md_config, stdout, cfg_title, print_flags);
        uct_config_release(md_config);
    }
}

ucs_status_t uct_md_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    memcpy(rkey_buffer, md->component->name, UCT_MD_COMPONENT_NAME_MAX);
    return md->ops->mkey_pack(md, memh, rkey_buffer + UCT_MD_COMPONENT_NAME_MAX);
}

ucs_status_t uct_rkey_unpack(const void *rkey_buffer, uct_rkey_bundle_t *rkey_ob)
{
    uct_md_component_t *mdc;
    ucs_status_t status;
    char mdc_name[UCT_MD_COMPONENT_NAME_MAX + 1];

    ucs_list_for_each(mdc, &uct_md_components_list, list) {
        if (!strncmp(rkey_buffer, mdc->name, UCT_MD_COMPONENT_NAME_MAX)) {
            status = mdc->rkey_unpack(mdc, rkey_buffer + UCT_MD_COMPONENT_NAME_MAX,
                                      &rkey_ob->rkey, &rkey_ob->handle);
            if (status == UCS_OK) {
                rkey_ob->type = mdc;
            }

            return status;
        }
    }

    ucs_snprintf_zero(mdc_name, sizeof(mdc_name), "%s", (const char*)rkey_buffer);
    ucs_debug("No matching MD component found for '%s'", mdc_name);
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_rkey_release(const uct_rkey_bundle_t *rkey_ob)
{
    uct_md_component_t *mdc = rkey_ob->type;
    return mdc->rkey_release(mdc, rkey_ob->rkey, rkey_ob->handle);
}

ucs_status_t uct_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    ucs_status_t status;

    status = md->ops->query(md, md_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* MD component name + data */
    memcpy(md_attr->component_name, md->component->name, UCT_MD_COMPONENT_NAME_MAX);
    md_attr->rkey_packed_size += UCT_MD_COMPONENT_NAME_MAX;

    return UCS_OK;
}

ucs_status_t uct_md_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              const char *alloc_name, uct_mem_h *memh_p)
{
    return md->ops->mem_alloc(md, length_p, address_p, memh_p UCS_MEMTRACK_VAL);
}

ucs_status_t uct_md_mem_free(uct_md_h md, uct_mem_h memh)
{
    return md->ops->mem_free(md, memh);
}

ucs_status_t uct_md_mem_reg(uct_md_h md, void *address, size_t length,
                            uct_mem_h *memh_p)
{
    if ((length == 0) || (address == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return md->ops->mem_reg(md, address, length, memh_p);
}

ucs_status_t uct_md_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    return md->ops->mem_dereg(md, memh);
}
