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


#define UCP_CONFIG_ENV_PREFIX "UCX_"

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

static int ucp_is_resource_enabled(uct_resource_desc_t *resource,
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
        ucs_assert(config->devices.count <= 64);
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

    ucs_trace("%s/%s is %sabled", resource->tl_name, resource->dev_name,
              (device_enabled && tl_enabled) ? "en" : "dis");
    return device_enabled && tl_enabled;
}

static ucs_status_t ucp_fill_resources(ucp_context_h context, const ucp_config_t *config)
{
    uct_resource_desc_t *resources;
    unsigned i, num_resources;
    ucs_status_t status;
    uint64_t used_devices_mask, mask, config_devices_mask;

    /* if we got here then num_resources > 0.
     * if the user's device list is empty, there is no match */
    if (0 == config->devices.count) {
        ucs_error("The device list is empty. Please specify the devices you would like to use "
                  "or omit the UCX_DEVICES so that the default will be used.");
        return UCS_ERR_NO_ELEM;
    }

    /* if we got here then num_resources > 0.
     * if the user's tls list is empty, there is no match */
    if (0 == config->tls.count) {
        ucs_error("The TLs list is empty. Please specify the transports you would like to use "
                  "or omit the UCX_TLS so that the default will be used.");
        return UCS_ERR_NO_ELEM;
    }

    /* check what are the available uct resources */
    status = uct_query_resources(context->uct, &resources, &num_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s",ucs_status_string(status));
        goto err;
    }

    if (0 == num_resources) {
        ucs_error("There are no available resources on the host");
        ucs_assert(resources == NULL);
        status = UCS_ERR_NO_DEVICE;
        goto err;
    }

    context->resources = ucs_calloc(num_resources, sizeof(uct_resource_desc_t),
                                    "ucp resources list");
    if (context->resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_resources;
    }

    /* mask of all devices from configuration which were used */
    used_devices_mask = 0;

    /* copy only the resources enabled by user configuration */
    context->num_resources = 0;
    for (i = 0; i < num_resources; ++i) {
        if (ucp_is_resource_enabled(&resources[i], config, &mask)) {
            context->resources[context->num_resources] = resources[i];
            ++context->num_resources;
            used_devices_mask |= mask;
        }
    }

    /* if all devices should be used, check that */
    config_devices_mask = UCS_MASK_SAFE(config->devices.count);
    if (config->device_policy_force && (used_devices_mask != config_devices_mask)) {
        i = ucs_ffs64(used_devices_mask ^ config_devices_mask);
        ucs_error("device %s is not available", config->devices.names[i]);
        status = UCS_ERR_NO_DEVICE;
        goto err_free_context_resources;
    }

    if (0 == context->num_resources) {
        ucs_error("There are no available resources matching the configured criteria");
        status = UCS_ERR_NO_DEVICE;
        goto err_free_context_resources;
    }

    uct_release_resource_list(resources);
    return UCS_OK;

err_free_context_resources:
    ucs_free(context->resources);
err_free_resources:
    uct_release_resource_list(resources);
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

    /* initialize uct */
    status = uct_init(&context->uct);
    if (status != UCS_OK) {
        ucs_error("Failed to initialize UCT: %s", ucs_status_string(status));
        goto err_free_ctx;
    }

    /* fill resources we should use */
    status = ucp_fill_resources(context, config);
    if (status != UCS_OK) {
        goto err_cleanup_uct;
    }

    /* initialize tag matching */
    status = ucp_tag_init(context);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    *context_p = context;
    return UCS_OK;

err_free_resources:
    ucs_free(context->resources);
err_cleanup_uct:
    uct_cleanup(context->uct);
err_free_ctx:
    ucs_free(context);
err:
    return status;
}

void ucp_cleanup(ucp_context_h context)
{
    ucp_tag_cleanup(context);
    ucs_free(context->resources);
    uct_cleanup(context->uct);
    ucs_free(context);
}

static void ucp_worker_close_ifaces(ucp_worker_h worker)
{
    unsigned i;

    for (i = 0; i < worker->num_ifaces; ++i) {
        uct_iface_close(worker->ifaces[i]);
    }
}

static ucs_status_t ucp_worker_add_iface(ucp_worker_h worker,
                                         uct_resource_desc_t *resource)
{
    ucp_context_h context = worker->context;
    uct_iface_config_t *iface_config;
    uct_iface_attr_t iface_attr;
    ucs_status_t status;
    uct_iface_h iface;

    /* Read configuration */
    status = uct_iface_config_read(context->uct, resource->tl_name,
                                   UCP_CONFIG_ENV_PREFIX, NULL,
                                   &iface_config);
    if (status != UCS_OK) {
        goto out;
    }

    /* Open UCT interface */
    status = uct_iface_open(worker->uct, resource->tl_name, resource->dev_name,
                            sizeof(ucp_recv_desc_t), iface_config, &iface);
    uct_iface_config_release(iface_config);

    if (status != UCS_OK) {
        goto out;
    }

    status = uct_iface_query(iface, &iface_attr);
    if (status != UCS_OK) {
        goto out;
    }


    if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT)) {
        status = UCS_OK;
        goto out_close_iface;
    }

    /* Set active message handlers for tag matching */
    status = ucp_tag_set_am_handlers(context, iface);
    if (status != UCS_OK) {
        goto out_close_iface;
    }

    ucs_debug("created interface[%d] using %s/%s on worker %p",
              worker->num_ifaces, resource->tl_name, resource->dev_name, worker);

    worker->ifaces[worker->num_ifaces] = iface;
    ++worker->num_ifaces;
    return UCS_OK;

out_close_iface:
    uct_iface_close(iface);
out:
    return status;
}

ucs_status_t ucp_worker_create(ucp_context_h context, ucs_thread_mode_t thread_mode,
                               ucp_worker_h *worker_p)
{
    uct_resource_desc_t *resource;
    ucp_worker_h worker;
    ucs_status_t status;

    worker = ucs_malloc(sizeof(*worker) + sizeof(worker->ifaces[0]) * context->num_resources,
                        "ucp worker");
    if (worker == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    worker->context    = context;
    worker->num_ifaces = 0;
    ucs_queue_head_init(&worker->completed);

    /* Create the underlying UCT worker */
    status = uct_worker_create(context->uct, thread_mode, &worker->uct);
    if (status != UCS_OK) {
        goto err_free;
    }

    /* Open all resources as interfaces on this worker */
    for (resource = context->resources;
         resource < context->resources + context->num_resources; ++resource)
    {
        status = ucp_worker_add_iface(worker, resource);
        if (status != UCS_OK) {
            goto err_close_ifaces;
        }
    }

    if (worker->num_ifaces == 0) {
        ucs_error("No valid transport interfaces");
        status = UCS_ERR_NO_DEVICE;
        goto err_close_ifaces;
    }

    *worker_p = worker;
    return UCS_OK;

err_close_ifaces:
    ucp_worker_close_ifaces(worker);
    uct_worker_destroy(worker->uct);
err_free:
    ucs_free(worker);
err:
    return status;
}

void ucp_worker_destroy(ucp_worker_h worker)
{
    ucp_worker_close_ifaces(worker);
    uct_worker_destroy(worker->uct);
    ucs_free(worker);
}

void ucp_worker_progress(ucp_worker_h worker)
{
    uct_worker_progress(worker->uct);
}
