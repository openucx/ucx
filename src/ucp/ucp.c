/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <ucp/api/ucp.h>
#include <ucs/type/component.h>

#define UCP_CONFIG_ENV_PREFIX "UCP_"

static UCS_CONFIG_DEFINE_ARRAY(port_spec,
                               sizeof(ucp_ib_port_spec_t),
                               UCS_CONFIG_TYPE_STRING);

static const char *device_policy_names[] = {
    [UCP_DEVICE_POLICY_TRY]   = "try",
    [UCP_DEVICE_POLICY_FORCE] = "force",
    [UCP_DEVICE_POLICY_LAST]   = NULL
};

ucs_config_field_t ucp_iface_config_table[] = {
  {"DEVICES", "*",
   "Specifies which Infiniband ports to use.",
   ucs_offsetof(ucp_iface_config_t, ports), UCS_CONFIG_TYPE_ARRAY(port_spec)},

  {"DEVICE_POLICY", "try",
   "The ports selection policy."
   "try     - try to using the available devices from the devices list.\n"
   "force   - force the usage of all the devices from the devices list.\n",
   ucs_offsetof(ucp_iface_config_t, device_policy), UCS_CONFIG_TYPE_ENUM(device_policy_names)},

  {NULL}
};

static int ucp_device_match(uct_resource_desc_t *resources,
                            ucp_iface_config_t *ucp_config,
                            uct_resource_desc_t *final_resources,
                            unsigned num_resources) {

    int config_idx, resource_idx, final_idx, is_force, found_match = 0;
    uct_resource_desc_t *resource_device = NULL;

    /* if we got here then num_resources > 0.
     * if the user's list is *, use all the available resources */
    if (0 == ucp_config->ports.count
        || !strcmp(ucp_config->ports.spec[0].device_name_port_num, "*")) {
        for (resource_idx = 0; resource_idx < num_resources; resource_idx++) {
            final_resources[resource_idx] = resources[resource_idx];
        }
        return 1;
    }

    is_force = (ucp_config->device_policy == UCP_DEVICE_POLICY_FORCE);

    /* go over the device list from the user and check (against the available resources)
     * which can be satisfied */
    final_idx = 0;
    for (config_idx = 0; config_idx < ucp_config->ports.count; config_idx++) {
        if (is_force) {
            found_match = 0;
        }
        for (resource_idx = 0; resource_idx < num_resources; resource_idx++) {

            resource_device = &resources[resource_idx];
            if (!strcmp(ucp_config->ports.spec[config_idx].device_name_port_num, resource_device->dev_name)) {
                /* there is a match */
                final_resources[final_idx] = *resource_device;
                final_idx++;
                found_match = 1;
                ucs_debug("port %s can be used",resource_device->dev_name);
                break;
            }
        }
        if ((!found_match) && (is_force)) {
            ucs_error("Device '%s' is not available",
                      ucp_config->ports.spec[config_idx].device_name_port_num);
            break;
        }
    }

    if (!found_match) {
        ucs_error("One or more of the ports from the UCP_DEVICES list, is not available");
    }

    return found_match;

}

ucs_status_t ucp_init(ucp_context_h *context_p)
{
    ucs_status_t status;
    ucp_context_t *context;
    uct_resource_desc_t *resources, *final_resources;
    unsigned num_resources;
    ucp_iface_config_t ucp_config;
    int match;

    /* allocate a ucp context */
    context = ucs_malloc(sizeof(*context), "ucp context");
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

    /* check what are the available uct resources */
    status = uct_query_resources(context->uct_context, &resources, &num_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s",ucs_status_string(status));
        goto err_free_uct;
    }

    if (0 == num_resources) {
        ucs_error("There are no available resources on the host");
        goto err_free_resources;
    }

    /* fill ucp configure options */
    status = ucs_config_parser_fill_opts(&ucp_config, ucp_iface_config_table, UCP_CONFIG_ENV_PREFIX, NULL, 0);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    final_resources = ucs_calloc(num_resources, sizeof(uct_resource_desc_t), "final resources list");
    if (final_resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_resources;
    }

    /* test the user's ucp_configuration against the uct resource */
    match = ucp_device_match(resources, &ucp_config, final_resources, num_resources);
    if (0 == match) {
        /* the requested device list cannot be satisfied */
        status = UCS_ERR_NO_ELEM;
        goto err_free_final_resources;
    }

    /* release the original list of resources */
    uct_release_resource_list(resources);

    context->resources = final_resources;
    *context_p = context;
    return UCS_OK;

err_free_final_resources:
    ucs_free(final_resources);
err_free_resources:
    uct_release_resource_list(resources);
err_free_uct:
    uct_cleanup(context->uct_context);
err_free_ctx:
    ucs_free(context);
err:
    return status;
}

void ucp_cleanup(ucp_context_h context)
{
    uct_release_resource_list(context->resources);
    uct_cleanup(context->uct_context);
    ucs_free(context);
}

ucs_status_t ucp_iface_create(ucp_context_h ucp_context, const char *env_prefix,
                              ucp_iface_h *ucp_iface_p)
{
    ucp_iface_t *ucp_iface;
    uct_iface_config_t *iface_config;
    ucs_status_t status;

    ucp_iface = ucs_malloc(sizeof(*ucp_iface), "ucp iface");
    if (ucp_iface == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* fill uct configure options */
    status = uct_iface_config_read(ucp_context->uct_context,
                                   ucp_context->resources[0].tl_name, env_prefix, NULL,
                                   &iface_config);
    if (status != UCS_OK) {
        ucs_error("Failed to read UCT config: %s", ucs_status_string(status));
        goto err_free_iface;
    }

    /* TODO open the matched resources. for now we open just the 1st */
    status = uct_iface_open(ucp_context->uct_context,
                            ucp_context->resources[0].tl_name,
                            ucp_context->resources[0].dev_name, 0, iface_config,
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
err_free_iface:
    ucs_free(ucp_iface);
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

    ucp_ep = ucs_malloc(sizeof(*ucp_ep), "ucp ep");
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
