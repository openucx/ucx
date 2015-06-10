/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"

#include <ucs/config/parser.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <string.h>


static UCS_CONFIG_DEFINE_ARRAY(device_names,
                               sizeof(char*),
                               UCS_CONFIG_TYPE_STRING);
static UCS_CONFIG_DEFINE_ARRAY(tl_names,
                               sizeof(char*),
                               UCS_CONFIG_TYPE_STRING);

static ucs_config_field_t ucp_config_table[] = {
  {"DEVICES", "all",
   "Specifies which device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices), UCS_CONFIG_TYPE_ARRAY(device_names)},

  {"FORCE_ALL_DEVICES", "no",
   "Device selection policy.\n"
   "yes - force using of all the devices from the device list.\n"
   "no  - try using the available devices from the device list.",
   ucs_offsetof(ucp_config_t, device_policy_force), UCS_CONFIG_TYPE_BOOL},

  {"TLS", "all",
   "Comma-separated list of transports to use. The order is not meaningful.\n"
   "all - use all the available transports.",
   ucs_offsetof(ucp_config_t, tls), UCS_CONFIG_TYPE_ARRAY(tl_names)},

  {NULL}
};


ucs_status_t ucp_config_read(const char *env_prefix, const char *filename,
                             ucp_config_t **config_p)
{
    ucp_config_t *config;
    ucs_status_t status;

    config = ucs_malloc(sizeof(*config), "ucp config");
    if (config == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = ucs_config_parser_fill_opts(config, ucp_config_table,
                                         UCP_CONFIG_ENV_PREFIX, NULL, 0);
    if (status != UCS_OK) {
        goto err_free;
    }

    *config_p = config;
    return UCS_OK;

err_free:
    ucs_free(config);
err:
    return status;
}

void ucp_config_release(ucp_config_t *config)
{
    ucs_config_parser_release_opts(config, ucp_config_table);
    ucs_free(config);
}

static int ucp_is_resource_enabled(uct_tl_resource_desc_t *resource,
                                   const ucp_config_t *config,
                                   uint64_t *devices_mask_p)
{
    int device_enabled, tl_enabled;
    unsigned config_idx;

    ucs_assert(config->devices.count > 0);
    if (!strcmp(config->devices.names[0], "all")) {
        /* if the user's list is 'all', use all the available resources */
        device_enabled  = 1;
        *devices_mask_p = 1;
    } else {
        /* go over the device list from the user and check (against the available resources)
         * which can be satisfied */
        device_enabled  = 0;
        *devices_mask_p = 0;
        ucs_assert_always(config->devices.count <= 64); /* Using uint64_t bitmap */
        for (config_idx = 0; config_idx < config->devices.count; ++config_idx) {
            if (!strcmp(config->devices.names[config_idx], resource->dev_name)) {
                device_enabled  = 1;
               *devices_mask_p |= UCS_MASK(config_idx);
            }
        }
    }

    ucs_assert(config->tls.count > 0);
    if (!strcmp(config->tls.names[0], "all")) {
        /* if the user's list is 'all', use all the available tls */
        tl_enabled = 1;
    } else {
        /* go over the tls list from the user and compare it against the available resources */
        tl_enabled = 0;
        for (config_idx = 0; config_idx < config->tls.count; ++config_idx) {
            if (!strcmp(config->tls.names[config_idx], resource->tl_name)) {
                tl_enabled = 1;
                break;
            }
        }
    }

    ucs_trace(UCT_TL_RESOURCE_DESC_FMT " is %sabled",
              UCT_TL_RESOURCE_DESC_ARG(resource),
              (device_enabled && tl_enabled) ? "en" : "dis");
    return device_enabled && tl_enabled;
}

static ucs_status_t ucp_add_tl_resources(ucp_context_h context,
                                         ucp_rsc_index_t pd_index,
                                         const ucp_config_t *config)
{
    uint64_t used_devices_mask, mask, config_devices_mask;
    uct_tl_resource_desc_t *tl_resources;
    ucp_tl_resource_desc_t *tmp;
    unsigned num_resources;
    ucs_status_t status;
    ucp_rsc_index_t i;

    /* check what are the available uct resources */
    status = uct_pd_query_tl_resources(context->pds[pd_index], &tl_resources,
                                       &num_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s", ucs_status_string(status));
        goto err;
    }

    tmp = ucs_realloc(context->tl_rscs,
                      sizeof(*context->tl_rscs) * (context->num_tls + num_resources),
                      "ucp resources");
    if (tmp == NULL) {
        ucs_error("Failed to allocate resources");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_resources;
    }

    /* mask of all devices from configuration which were used */
    used_devices_mask = 0;

    /* copy only the resources enabled by user configuration */
    context->tl_rscs = tmp;
    for (i = 0; i < num_resources; ++i) {
        if (ucp_is_resource_enabled(&tl_resources[i], config, &mask)) {
            context->tl_rscs[context->num_tls].tl_rsc   = tl_resources[i];
            context->tl_rscs[context->num_tls].pd_index = pd_index;
            ++context->num_tls;
            used_devices_mask |= mask;
        }
    }

    /* if all devices should be used, check that */
    config_devices_mask = UCS_MASK_SAFE(config->devices.count);
    if (config->device_policy_force && (used_devices_mask != config_devices_mask)) {
        i = ucs_ffs64(used_devices_mask ^ config_devices_mask);
        ucs_error("device %s is not available", config->devices.names[i]);
        status = UCS_ERR_NO_DEVICE;
        goto err_free_resources;
    }

    uct_release_tl_resource_list(tl_resources);
    return UCS_OK;

err_free_resources:
    uct_release_tl_resource_list(tl_resources);
err:
    return status;
}

static void ucp_free_pds(ucp_context_h context)
{
    ucp_rsc_index_t i;

    for (i = 0; i < context->num_pds; ++i) {
        if (context->pds[i] != NULL) {
            uct_pd_close(context->pds[i]);
        }
    }
    ucs_free(context->pds);
}

static ucs_status_t ucp_fill_resources(ucp_context_h context, const ucp_config_t *config)
{
    unsigned num_resources;
    ucp_rsc_index_t i;
    ucs_status_t status;

    /* if we got here then num_resources > 0.
     * if the user's device list is empty, there is no match */
    if (0 == config->devices.count) {
        ucs_error("The device list is empty. Please specify the devices you would like to use "
                  "or omit the UCX_DEVICES so that the default will be used.");
        status = UCS_ERR_NO_ELEM;
        goto err;
    }

    /* if we got here then num_resources > 0.
     * if the user's tls list is empty, there is no match */
    if (0 == config->tls.count) {
        ucs_error("The TLs list is empty. Please specify the transports you would like to use "
                  "or omit the UCX_TLS so that the default will be used.");
        status = UCS_ERR_NO_ELEM;
        goto err;
    }

    /* List protection domain resources */
    status = uct_query_pd_resources(&context->pd_rscs, &num_resources);
    if (status != UCS_OK) {
        goto err;
    }

    if (num_resources >= UINT8_MAX) {
        ucs_error("Only up to %d resources are supported", UINT8_MAX);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err_release_pd_resources;
    }

    /* Allocate array of protection domains */
    context->num_pds = num_resources;
    context->pds = ucs_calloc(context->num_pds, sizeof(*context->pds), "ucp_pds");
    if (context->pds == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_release_pd_resources;
    }

    /* Open all protection domains
     * TODO add configuration to select which protection domains to use
     */
    for (i = 0; i < context->num_pds; ++i) {
        status = uct_pd_open(context->pd_rscs[i].pd_name, &context->pds[i]);
        if (status != UCS_OK) {
            goto err_free_pds;
        }
    }

    context->tl_rscs = NULL;
    context->num_tls = 0;

    /* Add communication resources of each PD */
    for (i = 0; i < context->num_pds; ++i) {
        status = ucp_add_tl_resources(context, i, config);
        if (status != UCS_OK) {
            goto err_free_context_resources;
        }
    }

    if (0 == context->num_tls) {
        ucs_error("There are no available resources matching the configured criteria");
        status = UCS_ERR_NO_DEVICE;
        goto err_free_context_resources;
    }

    if (context->num_tls >= UCP_MAX_TLS) {
        ucs_error("Exceeded resources limit (%u requested, up to %d are supported)",
                  context->num_tls, UCP_MAX_TLS);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err_free_context_resources;
    }

    return UCS_OK;

err_free_context_resources:
    ucs_free(context->tl_rscs);
err_free_pds:
    ucp_free_pds(context);
err_release_pd_resources:
    uct_release_pd_resource_list(context->pd_rscs);
err:
    return status;
}

ucs_status_t ucp_init(const ucp_config_t *config, size_t request_headroom,
                      ucp_context_h *context_p)
{
    ucp_context_t *context;
    ucs_status_t status;

    /* allocate a ucp context */
    context = ucs_malloc(sizeof(*context), "ucp context");
    if (context == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* fill resources we should use */
    status = ucp_fill_resources(context, config);
    if (status != UCS_OK) {
        goto err_free_ctx;
    }

    /* initialize tag matching */
    status = ucp_tag_init(context);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    *context_p = context;
    return UCS_OK;

err_free_resources:
    ucs_free(context->tl_rscs);
err_free_ctx:
    ucs_free(context);
err:
    return status;
}

void ucp_cleanup(ucp_context_h context)
{
    ucp_tag_cleanup(context);
    ucs_free(context->tl_rscs);
    ucp_free_pds(context);
    uct_release_pd_resource_list(context->pd_rscs);
    ucs_free(context);
}
