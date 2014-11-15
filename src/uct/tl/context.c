/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "context.h"

#include <uct/api/uct.h>
#include <ucs/debug/memtrack.h>

UCS_COMPONENT_LIST_DEFINE(uct_context_t);

ucs_status_t uct_init(uct_context_h *context_p)
{
    ucs_status_t status;
    uct_context_t *context;

    context = ucs_malloc(ucs_components_total_size(uct_context_t), "uct context");
    if (context == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    context->num_tls  = 0;
    context->tls      = NULL;
    ucs_notifier_chain_init(&context->progress_chain);

    status = ucs_components_init_all(uct_context_t, context);
    if (status != UCS_OK) {
        goto err_free;
    }

    *context_p = context;
    return UCS_OK;

err_free:
    ucs_free(context);
err:
    return status;
}

void uct_cleanup(uct_context_h context)
{
    ucs_free(context->tls);
    ucs_components_cleanup_all(uct_context_t, context);
    ucs_free(context);
}

void uct_progress(uct_context_h context)
{
    ucs_notifier_chain_call(&context->progress_chain);
}

ucs_status_t uct_register_tl(uct_context_h context, const char *tl_name,
                             uct_tl_ops_t *tl_ops)
{
    uct_context_tl_info_t *tls;

    tls = ucs_realloc(context->tls, (context->num_tls + 1) * sizeof(*tls));
    if (tls == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    context->tls = tls;
    context->tls[context->num_tls].name = tl_name;
    context->tls[context->num_tls].ops  = tl_ops;
    ++context->num_tls;
    return UCS_OK;
}

ucs_status_t uct_query_resources(uct_context_h context,
                                 uct_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    uct_resource_desc_t *resources, *tl_resources, *p;
    unsigned num_resources, num_tl_resources, i;
    uct_context_tl_info_t *tl;
    ucs_status_t status;

    resources = NULL;
    num_resources = 0;
    for (tl = context->tls; tl < context->tls + context->num_tls; ++tl) {

        /* Get TL resources */
        status = tl->ops->query_resources(context,&tl_resources,
                                          &num_tl_resources);
        if (status != UCS_OK) {
            continue; /* Skip transport */
        }

        /* Enlarge the array */
        p = ucs_realloc(resources, (num_resources + num_tl_resources) * sizeof(*resources));
        if (p == NULL) {
            goto err_free;
        }
        resources = p;

        /* Append TL resources to the array. Set TL name. */
        for (i = 0; i < num_tl_resources; ++i) {
            resources[num_resources] = tl_resources[i];
            ucs_snprintf_zero(resources[num_resources].tl_name,
                              sizeof(resources[num_resources].tl_name),
                              "%s", tl->name);
            ++num_resources;
        }

        /* Release array returned from TL */
        ucs_free(tl_resources);
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    return UCS_OK;

err_free:
    ucs_free(resources);
    return UCS_ERR_NO_MEMORY;
}

void uct_release_resource_list(uct_resource_desc_t *resources)
{
    ucs_free(resources);
}

ucs_status_t uct_iface_open(uct_context_h context, const char *tl_name,
                            const char *dev_name, uct_iface_h *iface_p)
{
    uct_context_tl_info_t *tl;

    for (tl = context->tls; tl < context->tls + context->num_tls; ++tl) {
        if (!strcmp(tl_name, tl->name)) {
            return tl->ops->iface_open(context, dev_name, iface_p);
        }
    }

    /* Non-existing transport */
    return UCS_ERR_NO_DEVICE;
}

ucs_status_t uct_rkey_unpack(uct_context_h context, void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob)
{
    uct_context_tl_info_t *tl;
    ucs_status_t status;

    for (tl = context->tls; tl < context->tls + context->num_tls; ++tl) {
        status = tl->ops->rkey_unpack(context, rkey_buffer, rkey_ob);
        if (status != UCS_ERR_UNSUPPORTED) {
            return status;
        }
    }

    return UCS_ERR_INVALID_PARAM;
}

void uct_rkey_release(uct_context_h context, uct_rkey_bundle_t *rkey_ob)
{
    uct_rkey_release_func_t release = rkey_ob->type;
    release(context, rkey_ob->rkey);
}

