/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <src/ucp/api/ucp.h>
#include <ucs/type/component.h>

UCS_COMPONENT_LIST_DEFINE(ucp_context_t);
UCS_COMPONENT_LIST_DEFINE(ucp_iface_t);
UCS_COMPONENT_LIST_DEFINE(ucp_ep_t);

ucs_status_t ucp_init(ucp_context_h *context_p)
{
    ucs_status_t status;
    ucp_context_t *context;
    uct_resource_desc_t *resources;
    unsigned num_resources;

    /* allocate a ucp context */
    context = ucs_malloc(ucs_components_total_size(ucp_context_t),"ucp context");
    if (context == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* initialize uct */
    status = uct_init(&context->uct_context);
    if (status != UCS_OK) {
        ucs_error("Failed to initialize UCT: %s", ucs_status_string(status));
        goto err_free_ctx;
    }

    status = uct_query_resources(context->uct_context, &resources, &num_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s\n",ucs_status_string(status));
        goto err_free_uct;
    }

    context->resources = resources;
    *context_p = context;

    return UCS_OK;

err_free_uct:
    uct_cleanup(context->uct_context);
err_free_ctx:
    ucs_free(context);
err:
    return status;
}

void ucp_cleanup(ucp_context_h context_p)
{
    uct_release_resource_list(context_p->resources);
    uct_cleanup(context_p->uct_context);
    ucs_free(context_p);
}

ucs_status_t ucp_iface_create(ucp_context_h ucp_context, ucp_iface_h *ucp_iface_p)
{
    ucp_iface_t *ucp_iface;
    uct_iface_config_t *iface_config;
    ucs_status_t status;

    ucp_iface = ucs_malloc(ucs_components_total_size(ucp_iface_t), "ucp iface");

    status = uct_iface_config_read(ucp_context->uct_context,
                                   ucp_context->resources->tl_name, NULL, NULL,
                                   &iface_config);
    if (status != UCS_OK) {
        ucs_error("Failed to read UCT config: %s", ucs_status_string(status));
        goto err;
    }

    // need to open all the resources, one iface per resource?
    status = uct_iface_open(ucp_context->uct_context,
                            ucp_context->resources->tl_name,
                            ucp_context->resources->dev_name, 0, iface_config,
                            &ucp_iface->uct_iface);
    if (status != UCS_OK) {
        goto err_release_cfg;
    }

    uct_iface_config_release(iface_config);

    ucp_iface->context = ucp_context;
    *ucp_iface_p = ucp_iface;


    return UCS_OK;

err_release_cfg:
    uct_iface_config_release(iface_config);
err:
    return status;

}

void ucp_iface_close(ucp_iface_h ucp_iface)
{
    uct_iface_close(ucp_iface->uct_iface);
    ucs_free(ucp_iface);
}

ucs_status_t ucp_ep_create(ucp_iface_h ucp_iface, ucp_ep_h *ucp_ep_p)
{
    ucp_ep_t *ucp_ep;
    ucs_status_t status;

    ucp_ep = ucs_malloc(ucs_components_total_size(ucp_ep_t), "ucp ep");
    if (ucp_ep == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = uct_ep_create(ucp_iface->uct_iface, &ucp_ep->uct_ep);
    if (UCS_OK != status) {
        goto err_free_ep;
    }

    *ucp_ep_p = ucp_ep;
    return UCS_OK;

err_free_ep:
    ucs_free(ucp_ep);
err:
    return status;
}

void ucp_ep_destroy(ucp_ep_h ucp_ep)
{
    uct_ep_destroy(ucp_ep->uct_ep);
    ucs_free(ucp_ep);
}
