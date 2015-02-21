/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <ucp/api/ucp.h>
#include <ucs/type/component.h>

#define UCP_CONFIG_ENV_PREFIX "UCX_"

static UCS_CONFIG_DEFINE_ARRAY(device_spec,
                               sizeof(char*),
                               UCS_CONFIG_TYPE_STRING);

static UCS_CONFIG_DEFINE_ARRAY(tl_names,
                               sizeof(char*),
                               UCS_CONFIG_TYPE_STRING);

ucs_config_field_t ucp_iface_config_table[] = {
  {"DEVICES", "all",
   "Specifies which device to use.\n"
   "all - use all the available devices\n"
   "or a comma separated list of devices\n",
   ucs_offsetof(ucp_iface_config_t, devices), UCS_CONFIG_TYPE_ARRAY(device_spec)},

  {"FORCE_ALL_DEVICES", "no",
   "The devices selection policy.\n"
   "yes - force the usage of all the devices from the devices list.\n"
   "no  - try using the available devices from the devices list.\n",
   ucs_offsetof(ucp_iface_config_t, device_policy_force), UCS_CONFIG_TYPE_BOOL},

   {"TLS", "all",
   "Comma-separated list of transports to use. The order is not significant.\n"
   "all - use all the available TLs\n",
   ucs_offsetof(ucp_iface_config_t, tls), UCS_CONFIG_TYPE_ARRAY(tl_names)},

  {NULL}
};

/**
 * @ingroup CONTEXT
 * @brief create a list of resources to use. (compare the user's device list against the available resources)
 *
 * @param [in]  resources            Available resources on the host. (filled by the uct layer)
 * @param [in]  ucp_config           User's ucp configuration parameters.
 * @param [in]  num_resources        Number of available resources on the host.
 * @param [out] final_resources      Filled with the final list of resources to use.
 * @param [out] final_num_resources  Filled with the number of resources to use.
 *
 * @return Error code.
 */
static ucs_status_t ucp_device_match(uct_resource_desc_t *resources,
                                     ucp_iface_config_t *ucp_config,
                                     uct_resource_desc_t *final_resources,
                                     unsigned num_resources,
                                     unsigned *final_num_resources)
{
    int config_idx, resource_idx, final_idx, is_force, found_match = 0;
    uct_resource_desc_t *resource_device = NULL;

    /* if we got here then num_resources > 0.
     * if the user's device list is empty, there is no match */
    if (0 == ucp_config->devices.count) {
        ucs_error("The device list is empty. Please specify the devices you would like to use "
                  "or omit the UCX_DEVICES so that the default will be used.");
        return UCS_ERR_NO_ELEM;
    }

    /* if the user's list is 'all', use all the available resources */
    if (!strcmp(ucp_config->devices.device_name[0], "all")) {
        memcpy(final_resources, resources, num_resources*sizeof(uct_resource_desc_t));
        *final_num_resources = num_resources;
        return UCS_OK;
    }

    is_force = ucp_config->device_policy_force;

    /* go over the device list from the user and check (against the available resources)
     * which can be satisfied */
    final_idx = 0;
    for (config_idx = 0; config_idx < ucp_config->devices.count; config_idx++) {
        if (is_force) {
            found_match = 0;
        }
        for (resource_idx = 0; resource_idx < num_resources; resource_idx++) {

            resource_device = &resources[resource_idx];
            if (!strcmp(ucp_config->devices.device_name[config_idx], resource_device->dev_name)) {
                /* there is a match */
                final_resources[final_idx] = *resource_device;
                final_idx++;
                found_match = 1;
                ucs_debug("device %s can be used. tl name: %s",resource_device->dev_name, resource_device->tl_name);
            }
        }
        if ((!found_match) && (is_force)) {
            ucs_error("Device '%s' is not available",
                      ucp_config->devices.device_name[config_idx]);
            break;
        }
    }

    *final_num_resources = final_idx;

    if (!found_match) {
        ucs_error("One or more of the devices from the UCX_DEVICES list, is not available");
        return UCS_ERR_NO_ELEM;
    }

    return UCS_OK;
}

/**
 * @ingroup CONTEXT
 * @brief create the final list of resources to use. (compare the user's tl list against the available resources)
 *
 * @param [in]  resources            Available resources on the host. (filtered to match the user's device list)
 * @param [in]  ucp_config           User's ucp configuration parameters.
 * @param [in]  num_resources        Number of available resources in the given resources list.
 * @param [out] final_resources      Filled with the final list of resources to use.
 * @param [out] final_num_resources  Filled with the number of resources to use.
 *
 * @return Error code.
 */
static ucs_status_t ucp_tl_match(uct_resource_desc_t *resources,
                                 ucp_iface_config_t *ucp_config,
                                 uct_resource_desc_t *final_resources,
                                 unsigned num_resources,
                                 unsigned *final_num_resources)
{
    int resource_idx, final_idx, config_idx;
    uct_resource_desc_t *resource_device = NULL;

    /* if we got here then num_resources > 0.
     * if the user's tls list is empty, there is no match */
    if (0 == ucp_config->tls.count) {
        ucs_error("The TLs list is empty. Please specify the transport layers you would like to use "
                  "or omit the UCX_TLS so that the default will be used.");
        return UCS_ERR_NO_ELEM;
    }

    /* if the user's list is 'all', use all the available tls */
    if (!strcmp(ucp_config->tls.tl_name[0], "all")) {
        memcpy(final_resources, resources, num_resources * sizeof(uct_resource_desc_t));
        *final_num_resources = num_resources;
        return UCS_OK;
    }

    final_idx = 0;
    /* go over the tls list from the user and compare it against the available resources */
    for (config_idx = 0; config_idx < ucp_config->tls.count; config_idx++) {
        for (resource_idx = 0; resource_idx < num_resources; resource_idx++) {

            resource_device = &resources[resource_idx];
            if (!strcmp(ucp_config->tls.tl_name[config_idx], resource_device->tl_name)) {
                /* there is a match */
                final_resources[final_idx] = *resource_device;
                final_idx++;
                ucs_debug("TL %s, Device: %s can be used",
                          resource_device->tl_name, resource_device->dev_name);
            }
        }
    }

    if (0 == final_idx) {
        ucs_error("None of the TLs in the UCX_TLS list can be used.");
        return UCS_ERR_NO_ELEM;
    }

    *final_num_resources = final_idx;
    return UCS_OK;
}

ucs_status_t ucp_init(ucp_context_h *context_p)
{
    ucs_status_t status;
    ucp_context_t *context;
    uct_resource_desc_t *resources, *tmp_resources;
    unsigned num_resources, final_num_resources;
    ucp_iface_config_t ucp_config;

    /* allocate a ucp context */
    context = ucs_malloc(sizeof(*context), "ucp context");
    if (NULL == context) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* initialize uct */
    status = uct_init(&context->uct_context);
    if (UCS_OK != status) {
        ucs_error("Failed to initialize UCT: %s", ucs_status_string(status));
        goto err_free_ctx;
    }

    /* check what are the available uct resources */
    status = uct_query_resources(context->uct_context, &resources, &num_resources);
    if (UCS_OK != status) {
        ucs_error("Failed to query resources: %s",ucs_status_string(status));
        goto err_free_uct;
    }

    if (0 == num_resources) {
        ucs_error("There are no available resources on the host");
        goto err_free_resources;
    }

    /* fill ucp configure options */
    status = ucs_config_parser_fill_opts(&ucp_config, ucp_iface_config_table, UCP_CONFIG_ENV_PREFIX, NULL, 0);
    if (UCS_OK != status) {
        goto err_free_resources;
    }

    tmp_resources = ucs_calloc(num_resources, sizeof(uct_resource_desc_t), "temporary resources list");
    if (NULL == tmp_resources) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_ucp_config;
    }

    /* test the user's ucp_configuration (devices list) against the uct resources */
    status = ucp_device_match(resources, &ucp_config, tmp_resources, num_resources, &final_num_resources);
    if (UCS_OK != status) {
        /* the requested device list cannot be satisfied */
        goto err_free_tmp_resources;
    }

    context->resources = ucs_calloc(num_resources, sizeof(uct_resource_desc_t), "final resources list");
    if (NULL == context->resources) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_tmp_resources;
    }

    /* test the user's ucp_configuration (tls list) against the uct resources */
    status = ucp_tl_match(tmp_resources, &ucp_config, context->resources, final_num_resources, &final_num_resources);
    if (UCS_OK != status) {
        /* the requested tl list cannot be satisfied */
        goto err_free_final_resources;
    }

    /* release the original and the tmp lists of resources */
    uct_release_resource_list(tmp_resources);
    uct_release_resource_list(resources);

    ucs_config_parser_release_opts(&ucp_config, ucp_iface_config_table);

    context->num_resources = final_num_resources;
    *context_p = context;
    return UCS_OK;

err_free_final_resources:
    ucs_free(context->resources);
err_free_tmp_resources:
    ucs_free(tmp_resources);
err_free_ucp_config:
    ucs_config_parser_release_opts(&ucp_config, ucp_iface_config_table);
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
    if (NULL == ucp_iface) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* fill uct configure options */
    status = uct_iface_config_read(ucp_context->uct_context,
                                   ucp_context->resources[0].tl_name, env_prefix, NULL,
                                   &iface_config);
    if (UCS_OK != status) {
        ucs_error("Failed to read UCT config: %s", ucs_status_string(status));
        goto err_free_iface;
    }

    /* TODO open the matched resources. for now we open just the 1st */
    status = uct_iface_open(ucp_context->uct_context,
                            ucp_context->resources[0].tl_name,
                            ucp_context->resources[0].dev_name, 0, iface_config,
                            &ucp_iface->uct_iface);
    if (UCS_OK != status) {
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
    if (NULL == ucp_ep) {
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
