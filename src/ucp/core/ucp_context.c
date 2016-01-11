/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_context.h"


#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/arch/bitops.h>
#include <string.h>


ucp_am_handler_t ucp_am_handlers[UCP_AM_ID_LAST] = {{0, NULL, NULL}};


static ucs_config_field_t ucp_config_table[] = {
  {"DEVICES", "all",
   "Specifies which device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"TLS", "all",
   "Comma-separated list of transports to use. The order is not meaningful.\n"
   "In addition it's possible to use a combination of the following aliases:\n"
   " - all    : use all the available transports."
   " - sm/shm : all shared memory transports.\n"
   " - ugni   : ugni_rdma and ugni_udt.\n"
   " - rc     : rc and cm."
   " - rc-x   : rc with accelerated verbs and cm."
   " - ud-x   : ud with accelerated verbs.\n",
   ucs_offsetof(ucp_config_t, tls), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"FORCE_ALL_DEVICES", "no",
   "Device selection policy.\n"
   "yes - force using of all the devices from the device list.\n"
   "no  - try using the available devices from the device list.",
   ucs_offsetof(ucp_config_t, force_all_devices), UCS_CONFIG_TYPE_BOOL},

  {"ALLOC_PRIO", "pd:sysv,pd:posix,huge,pd:*,mmap,heap",
   "Priority of memory allocation methods. Each item in the list can be either\n"
   "an allocation method (huge, mmap, libc) or pd:<NAME> which means to use the\n"
   "specified protection domain for allocation. NAME can be either a PD component\n"
   "name, or a wildcard - '*' - which expands to all PD components.",
   ucs_offsetof(ucp_config_t, alloc_prio), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"BCOPY_THRESH", "0",
   "Threshold for switching from short to bcopy protocol",
   ucs_offsetof(ucp_config_t, bcopy_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_THRESH", "1gb",
   "Threshold for switching from eager to rendezvous protocol",
   ucs_offsetof(ucp_config_t, rndv_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"LOG_DATA", "0",
   "Size of packet data that is dumped to the log system in debug mode (0 - nothing).",
   ucs_offsetof(ucp_config_t, log_data_size), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};

static ucp_tl_alias_t ucp_tl_aliases[] = {
  { "sm",    { "mm", "knem", "sysv", "posix", "cma", "xpmem", NULL } },
  { "shm",   { "mm", "knem", "sysv", "posix", "cma", "xpmem", NULL } },
  { "rc",    { "rc", "cm", NULL } },
  { "rc_x",  { "rc_mlx5", "cm", NULL } },
  { "ud_x",  { "ud_mlx5", NULL } },
  { "ugni",  { "ugni_udt", "ugni_rdma", NULL } },
  { NULL }
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

    status = ucs_config_parser_fill_opts(config, ucp_config_table, env_prefix,
                                         NULL, 0);
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

void ucp_config_print(const ucp_config_t *config, FILE *stream,
                      const char *title, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, ucp_config_table, NULL,
                                 print_flags);
}

static int ucp_str_array_search(const char **array, unsigned length,
                                const char *str)
{
    unsigned i;

    for (i = 0; i < length; ++i) {
        if (!strcmp(array[i], str)) {
            return i;
        }
    }
    return -1;
}

static unsigned ucp_tl_alias_count(ucp_tl_alias_t *alias)
{
    unsigned count;
    for (count = 0; alias->tls[count] != NULL; ++count);
    return count;
}

static int ucp_config_is_tl_enabled(const ucp_config_t *config, const char *tl_name)
{
    const char **names = (const char**)config->tls.names;
    return (ucp_str_array_search(names, config->tls.count, tl_name) >= 0) ||
           (ucp_str_array_search(names, config->tls.count, "all"  ) >= 0);
}

static int ucp_is_resource_enabled(uct_tl_resource_desc_t *resource,
                                   const ucp_config_t *config,
                                   uint64_t *devices_mask_p)
{
    int device_enabled, tl_enabled;
    ucp_tl_alias_t *alias;
    int config_idx;

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
        config_idx = ucp_str_array_search((const char**)config->devices.names,
                                          config->devices.count,
                                          resource->dev_name);
        if (config_idx >= 0) {
            device_enabled  = 1;
            *devices_mask_p |= UCS_MASK(config_idx);
        }
    }

    /* Disable the posix mmap and xpmem 'devices'. ONLY for now - use sysv for mm .
     * This will be removed after multi-rail is supported */
    if (!strcmp(resource->dev_name,"posix") || !strcmp(resource->dev_name, "xpmem")) {
        device_enabled  = 0;
    }

    ucs_assert(config->tls.count > 0);
    if (ucp_config_is_tl_enabled(config, resource->tl_name)) {
        tl_enabled = 1;
    } else {
        tl_enabled = 0;

        /* check aliases */
        for (alias = ucp_tl_aliases; alias->alias != NULL; ++alias) {

            /* If an alias is enabled, and the transport is part of this alias,
             * enable the transport.
             */
            if (ucp_config_is_tl_enabled(config, alias->alias) &&
                (ucp_str_array_search(alias->tls, ucp_tl_alias_count(alias),
                                     resource->tl_name) >= 0))
            {
                tl_enabled = 1;
                ucs_trace("enabling tl '%s' for alias '%s'", resource->tl_name,
                          alias->alias);
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
                                         uct_pd_h pd, ucp_rsc_index_t pd_index,
                                         const ucp_config_t *config,
                                         unsigned *num_resources_p)
{
    uint64_t used_devices_mask, mask, config_devices_mask;
    uct_tl_resource_desc_t *tl_resources;
    ucp_tl_resource_desc_t *tmp;
    unsigned num_resources;
    ucs_status_t status;
    ucp_rsc_index_t i;

    *num_resources_p = 0;

    /* check what are the available uct resources */
    status = uct_pd_query_tl_resources(pd, &tl_resources, &num_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s", ucs_status_string(status));
        goto err;
    }

    if (num_resources == 0) {
        ucs_debug("No tl resources found for pd %s", context->pd_rscs[pd_index].pd_name);
        goto out_free_resources;
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
            ++(*num_resources_p);
        }
    }

    /* if all devices should be used, check that */
    config_devices_mask = UCS_MASK_SAFE(config->devices.count);
    if (config->force_all_devices && (used_devices_mask != config_devices_mask)) {
        i = ucs_ffs64(used_devices_mask ^ config_devices_mask);
        ucs_error("device %s is not available", config->devices.names[i]);
        status = UCS_ERR_NO_DEVICE;
        goto err_free_resources;
    }

out_free_resources:
    uct_release_tl_resource_list(tl_resources);
    return UCS_OK;

err_free_resources:
    uct_release_tl_resource_list(tl_resources);
err:
    return status;
}

static void ucp_free_resources(ucp_context_t *context)
{
    ucp_rsc_index_t i;

    ucs_free(context->tl_rscs);
    for (i = 0; i < context->num_pds; ++i) {
        if (context->pds[i] != NULL) {
            uct_pd_close(context->pds[i]);
        }
    }
    ucs_free(context->pd_attrs);
    ucs_free(context->pds);
    ucs_free(context->pd_rscs);
}

static ucs_status_t ucp_fill_resources(ucp_context_h context,
                                       const ucp_config_t *config)
{
    unsigned num_tl_resources;
    unsigned num_pd_resources;
    uct_pd_resource_desc_t *pd_rscs;
    ucs_status_t status;
    ucp_rsc_index_t i;
    unsigned pd_index;
    uct_pd_h pd;
    uct_pd_config_t *pd_config;

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
    status = uct_query_pd_resources(&pd_rscs, &num_pd_resources);
    if (status != UCS_OK) {
        goto err;
    }

    /* Error check: Make sure there is at least one PD */
    if (num_pd_resources == 0) {
        ucs_error("No pd resources found");
        status = UCS_ERR_NO_DEVICE;
        goto err_release_pd_resources;
    }

    if (num_pd_resources >= UCP_MAX_PDS) {
        ucs_error("Only up to %ld PDs are supported", UCP_MAX_PDS);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err_release_pd_resources;
    }

    context->num_pds  = 0;
    context->pd_rscs  = NULL;
    context->pds      = NULL;
    context->pd_attrs = NULL;
    context->num_tls  = 0;
    context->tl_rscs  = NULL;

    /* Allocate array of PD resources we would actually use */
    context->pd_rscs = ucs_calloc(num_pd_resources, sizeof(*context->pd_rscs),
                                  "ucp_pd_resources");
    if (context->pd_rscs == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context_resources;
    }

    /* Allocate array of protection domains */
    context->pds = ucs_calloc(num_pd_resources, sizeof(*context->pds), "ucp_pds");
    if (context->pds == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context_resources;
    }

    /* Allocate array of protection domains attributes */
    context->pd_attrs = ucs_calloc(num_pd_resources, sizeof(*context->pd_attrs),
                                   "ucp_pd_attrs");
    if (context->pd_attrs == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context_resources;
    }

    /* Open all protection domains, keep only those which have at least one TL
     * resources selected on them.
     */
    pd_index = 0;
    for (i = 0; i < num_pd_resources; ++i) {
        status = uct_pd_config_read(pd_rscs[i].pd_name, NULL, NULL, &pd_config);
        if (status != UCS_OK) {
            goto err_free_context_resources;
        }

        status = uct_pd_open(pd_rscs[i].pd_name, pd_config, &pd);
        uct_config_release(pd_config);
        if (status != UCS_OK) {
            goto err_free_context_resources;
        }

        context->pd_rscs[pd_index] = pd_rscs[i];
        context->pds[pd_index]     = pd;

        /* Save PD attributes */
        status = uct_pd_query(pd, &context->pd_attrs[pd_index]);
        if (status != UCS_OK) {
            goto err_free_context_resources;
        }

        /* Add communication resources of each PD */
        status = ucp_add_tl_resources(context, pd, pd_index, config,
                                      &num_tl_resources);
        if (status != UCS_OK) {
            goto err_free_context_resources;
        }

        /* If the PD does not have transport resources, don't use it */
        if (num_tl_resources > 0) {
            ++pd_index;
            ++context->num_pds;
        } else {
            ucs_debug("closing pd %s because it has no selected transport resources",
                      pd_rscs[i].pd_name);
            uct_pd_close(pd);
        }
    }

    /* Error check: Make sure there is at least one transport */
    if (0 == context->num_tls) {
        ucs_error("There are no available resources matching the configured criteria");
        status = UCS_ERR_NO_DEVICE;
        goto err_free_context_resources;
    }

    /* Error check: Make sure there are no too many transports */
    if (context->num_tls >= UCP_MAX_RESOURCES) {
        ucs_error("Exceeded resources limit (%u requested, up to %d are supported)",
                  context->num_tls, UCP_MAX_RESOURCES);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err_free_context_resources;
    }

    uct_release_pd_resource_list(pd_rscs);
    return UCS_OK;

err_free_context_resources:
    ucp_free_resources(context);
err_release_pd_resources:
    uct_release_pd_resource_list(pd_rscs);
err:
    return status;
}

static ucs_status_t ucp_fill_config(ucp_context_h context,
                                    const ucp_params_t *params,
                                    const ucp_config_t *config)
{
    unsigned i, num_alloc_methods, method;
    const char *method_name;
    ucs_status_t status;

    if (0 == params->features) {
        ucs_error("empty features set passed to ucp context create");
        return UCS_ERR_INVALID_PARAM;
    }

    context->config.features        = params->features;
    context->config.request.size    = params->request_size;
    context->config.request.init    = params->request_init;
    context->config.request.cleanup = params->request_cleanup;
    context->config.bcopy_thresh    = config->bcopy_thresh;
    context->config.rndv_thresh     = config->rndv_thresh;
    context->config.log_data_size   = config->log_data_size;

    /* Get allocation alignment from configuration, make sure it's valid */
    if (config->alloc_prio.count == 0) {
        ucs_error("No allocation methods specified - aborting");
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    num_alloc_methods = config->alloc_prio.count;
    context->config.num_alloc_methods = num_alloc_methods;

    /* Allocate an array to hold the allocation methods configuration */
    context->config.alloc_methods = ucs_calloc(num_alloc_methods,
                                               sizeof(*context->config.alloc_methods),
                                               "ucp_alloc_methods");
    if (context->config.alloc_methods == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* Parse the allocation methods specified in the configuration */
    for (i = 0; i < num_alloc_methods; ++i) {
        method_name = config->alloc_prio.methods[i];
        if (!strncasecmp(method_name, "pd:", 3)) {
            /* If the method name begins with 'pd:', treat it as protection domain
             * component name.
             */
            context->config.alloc_methods[i].method = UCT_ALLOC_METHOD_PD;
            strncpy(context->config.alloc_methods[i].pdc_name,
                    method_name + 3, UCT_PD_COMPONENT_NAME_MAX);
            ucs_debug("allocation method[%d] is pd '%s'", i, method_name + 3);
        } else {
            /* Otherwise, this is specific allocation method name.
             */
            context->config.alloc_methods[i].method = UCT_ALLOC_METHOD_LAST;
            for (method = 0; method < UCT_ALLOC_METHOD_LAST; ++method) {
                if ((method != UCT_ALLOC_METHOD_PD) &&
                    !strcmp(method_name, uct_alloc_method_names[method]))
                {
                    /* Found the allocation method in the internal name list */
                    context->config.alloc_methods[i].method = method;
                    strcpy(context->config.alloc_methods[i].pdc_name, "");
                    ucs_debug("allocation method[%d] is '%s'", i, method_name);
                    break;
                }
            }
            if (context->config.alloc_methods[i].method == UCT_ALLOC_METHOD_LAST) {
                ucs_error("Invalid allocation method: %s", method_name);
                status = UCS_ERR_INVALID_PARAM;
                goto err_free;
            }
        }
    }

    return UCS_OK;

err_free:
    ucs_free(context->config.alloc_methods);
err:
    return status;
}

static void ucp_free_config(ucp_context_h context)
{
    ucs_free(context->config.alloc_methods);
}

ucs_status_t ucp_init_version(unsigned api_major_version, unsigned api_minor_version,
                              const ucp_params_t *params, const ucp_config_t *config,
                              ucp_context_h *context_p)
{
    unsigned major_version, minor_version, release_number;
    ucp_context_t *context;
    ucs_status_t status;

    ucp_get_version(&major_version, &minor_version, &release_number);

    if ((api_major_version != major_version) || (api_minor_version != minor_version)) {
        ucs_error("UCP version is incompatible, required: %d.%d, actual: %d.%d (release %d)",
                  api_major_version, api_minor_version,
                  major_version, minor_version, release_number);
        status = UCS_ERR_NOT_IMPLEMENTED;
        goto err;
    }

    /* allocate a ucp context */
    context = ucs_calloc(1, sizeof(*context), "ucp context");
    if (context == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = ucp_fill_config(context, params, config);
    if (status != UCS_OK) {
        goto err_free_ctx;
    }

    /* fill resources we should use */
    status = ucp_fill_resources(context, config);
    if (status != UCS_OK) {
        goto err_free_config;
    }

    /* initialize tag matching */
    ucs_queue_head_init(&context->tag.expected);
    ucs_queue_head_init(&context->tag.unexpected);

    *context_p = context;
    return UCS_OK;

err_free_config:
    ucp_free_config(context);
err_free_ctx:
    ucs_free(context);
err:
    return status;
}

void ucp_cleanup(ucp_context_h context)
{
    ucp_free_resources(context);
    ucp_free_config(context);
    ucs_free(context);
}

void ucp_dump_payload(ucp_context_h context, char *buffer, size_t max,
                      const void *data, size_t length)
{
    char *p, *endp;
    size_t offset;

    if (context->config.log_data_size == 0) {
        return;
    }

    p    = buffer;
    endp = buffer + max;

    strncat(p, " : ", endp - p);
    p = p + strlen(p);

    offset = 0;
    while ((offset < length) && (offset < context->config.log_data_size) && (p < endp)) {
        snprintf(p, endp - p, "%02x", ((const uint8_t*)data)[offset]);
        p += strlen(p);
        ++offset;
    }
}
