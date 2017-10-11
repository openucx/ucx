/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_context.h"
#include "ucp_request.h"
#include <ucp/proto/proto.h>

#include <ucs/config/parser.h>
#include <ucs/algorithm/crc.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/arch/bitops.h>
#include <string.h>


ucp_am_handler_t ucp_am_handlers[UCP_AM_ID_LAST] = {{0, NULL, NULL}};

static const char *ucp_atomic_modes[] = {
    [UCP_ATOMIC_MODE_CPU]    = "cpu",
    [UCP_ATOMIC_MODE_DEVICE] = "device",
    [UCP_ATOMIC_MODE_GUESS]  = "guess",
    [UCP_ATOMIC_MODE_LAST]   = NULL,
};

static const char * ucp_device_type_names[] = {
    [UCT_DEVICE_TYPE_NET]  = "network",
    [UCT_DEVICE_TYPE_SHM]  = "intra-node",
    [UCT_DEVICE_TYPE_ACC]  = "accelerator",
    [UCT_DEVICE_TYPE_SELF] = "loopback",
};

static ucs_config_field_t ucp_config_table[] = {
  {"NET_DEVICES", "all",
   "Specifies which network device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_NET]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SHM_DEVICES", "all",
   "Specifies which intra-node device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_SHM]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"ACC_DEVICES", "all",
   "Specifies which accelerator device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_ACC]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SELF_DEVICES", "all",
    "Specifies which loop-back device(s) to use. The order is not meaningful.\n"
    "\"all\" would use all available devices.",
    ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_SELF]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"TLS", "all",
   "Comma-separated list of transports to use. The order is not meaningful.\n"
   "In addition it's possible to use a combination of the following aliases:\n"
   " - all    : use all the available transports.\n"
   " - sm/shm : all shared memory transports.\n"
   " - mm     : shared memory transports - only memory mappers.\n"
   " - ugni   : ugni_rdma and ugni_udt.\n"
   " - ib     : all infiniband transports.\n"
   " - rc     : rc and ud.\n"
   " - rc_x   : rc with accelerated verbs and ud.\n"
   " - ud_x   : ud with accelerated verbs.\n"
   " - dc_x   : dc with accelerated verbs.\n"
   " Using a \\ prefix before a transport name treats it as an explicit transport name\n"
   " and disables aliasing.\n",
   ucs_offsetof(ucp_config_t, tls), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"ALLOC_PRIO", "md:sysv,md:posix,huge,thp,md:*,mmap,heap",
   "Priority of memory allocation methods. Each item in the list can be either\n"
   "an allocation method (huge, thp, mmap, libc) or md:<NAME> which means to use the\n"
   "specified memory domain for allocation. NAME can be either a MD component\n"
   "name, or a wildcard - '*' - which expands to all MD components.",
   ucs_offsetof(ucp_config_t, alloc_prio), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"BCOPY_THRESH", "0",
   "Threshold for switching from short to bcopy protocol",
   ucs_offsetof(ucp_config_t, ctx.bcopy_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_THRESH", "auto",
   "Threshold for switching from eager to rendezvous protocol",
   ucs_offsetof(ucp_config_t, ctx.rndv_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_THRESH_FALLBACK", "inf",
   "Message size to start using the rendezvous protocol in case the calculated threshold "
   "is zero or negative",
   ucs_offsetof(ucp_config_t, ctx.rndv_thresh_fallback), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_PERF_DIFF", "1",
   "The percentage allowed for performance difference between rendezvous and "
   "the eager_zcopy protocol",
   ucs_offsetof(ucp_config_t, ctx.rndv_perf_diff), UCS_CONFIG_TYPE_DOUBLE},

  {"ZCOPY_THRESH", "auto",
   "Threshold for switching from buffer copy to zero copy protocol",
   ucs_offsetof(ucp_config_t, ctx.zcopy_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"BCOPY_BW", "5800mb",
   "Estimation of buffer copy bandwidth",
   ucs_offsetof(ucp_config_t, ctx.bcopy_bw), UCS_CONFIG_TYPE_MEMUNITS},

  {"ATOMIC_MODE", "guess",
   "Atomic operations synchronization mode.\n"
   " cpu    - atomic operations are consistent with respect to the CPU.\n"
   " device - atomic operations are performed on one of the transport devices,\n"
   "          and there is guarantee of consistency with respect to the CPU."
   " guess  - atomic operations mode is configured based on underlying\n"
   "          transport capabilities. If one of active transports supports\n"
   "          the DEVICE atomic mode, the DEVICE mode is selected.\n"
   "          Otherwise the CPU mode is selected.",
   ucs_offsetof(ucp_config_t, ctx.atomic_mode), UCS_CONFIG_TYPE_ENUM(ucp_atomic_modes)},

  {"MAX_WORKER_NAME", UCS_PP_MAKE_STRING(UCP_WORKER_NAME_MAX),
   "Maximal length of worker name. Affects the size of worker address in debug builds.",
   ucs_offsetof(ucp_config_t, ctx.max_worker_name), UCS_CONFIG_TYPE_UINT},

  {"USE_MT_MUTEX", "n", "Use mutex for multithreading support in UCP.\n"
   "n      - Not use mutex for multithreading support in UCP (use spinlock by default).\n"
   "y      - Use mutex for multithreading support in UCP.\n",
   ucs_offsetof(ucp_config_t, ctx.use_mt_mutex), UCS_CONFIG_TYPE_BOOL},

  {"ADAPTIVE_PROGRESS", "y",
   "Enable apaptive progress mechanism, which turns on polling only on active\n"
   "transport interfaces.",
   ucs_offsetof(ucp_config_t, ctx.adaptive_progress), UCS_CONFIG_TYPE_BOOL},

  {"SEG_SIZE", "8192",
   "Size of a segment in the worker preregistered memory pool.",
   ucs_offsetof(ucp_config_t, ctx.seg_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_OFFLOAD", "y", "Enable tag matching offload",
   ucs_offsetof(ucp_config_t, ctx.tm_offload), UCS_CONFIG_TYPE_BOOL},

  {"TM_THRESH", "1024", /* TODO: calculate automaticlly */
   "Threshold for using tag matching offload capabilities.\n"
   "Smaller buffers will not be posted to the transport.",
   ucs_offsetof(ucp_config_t, ctx.tm_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_MAX_BCOPY", "1024", /* TODO: calculate automaticlly */
   "Maximal size for posting \"bounce buffer\" (UCX interal preregistered memory) for\n"
   "tag offload receives. When message arrives, it is copied into the user buffer (similar\n"
   "to eager protocol). The size values has to be equal or less than segment size.\n"
   "Also the value has to be bigger than UCX_TM_THRESH to take an effect." ,
   ucs_offsetof(ucp_config_t, ctx.tm_max_bcopy), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};

static ucp_tl_alias_t ucp_tl_aliases[] = {
  { "sm",    { "mm", "knem", "sysv", "posix", "cma", "xpmem", NULL } },
  { "shm",   { "mm", "knem", "sysv", "posix", "cma", "xpmem", NULL } },
  { "ib",    { "rc", "ud", "rc_mlx5", "ud_mlx5", NULL } },
  { "rc",    { "rc", "ud", NULL } },
  { "rc_x",  { "rc_mlx5", "ud_mlx5", NULL } },
  { "ud_x",  { "ud_mlx5", NULL } },
  { "dc_x",  { "dc_mlx5", NULL } },
  { "ugni",  { "ugni_smsg", "ugni_udt", "ugni_rdma", NULL } },
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

ucs_status_t ucp_config_modify(ucp_config_t *config, const char *name,
                               const char *value)
{
    return ucs_config_parser_set_value(config, ucp_config_table, name, value);
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

static int ucp_config_is_tl_enabled(const ucp_config_t *config, const char *tl_name,
                                    int is_alias)
{
    const char **names = (const char**)config->tls.names;
    char buf[UCT_TL_NAME_MAX + 1];

    snprintf(buf, sizeof(buf), "\\%s", tl_name);
    return (!is_alias && ucp_str_array_search(names, config->tls.count, buf) >= 0) ||
           ((ucp_str_array_search(names, config->tls.count, tl_name) >= 0)) ||
           (ucp_str_array_search(names, config->tls.count, "all"  ) >= 0);
}

static int ucp_is_resource_in_device_list(uct_tl_resource_desc_t *resource,
                                          const ucs_config_names_array_t *devices,
                                          uint64_t *masks, int index)
{
    int device_enabled, config_idx;

    if (devices[index].count == 0) {
        return 0;
    }

    if (!strcmp(devices[index].names[0], "all")) {
        /* if the user's list is 'all', use all the available resources */
        device_enabled  = 1;
        masks[index] = -1;      /* using all available devices. can satisfy 'all' */
    } else {
        /* go over the device list from the user and check (against the available resources)
         * which can be satisfied */
        device_enabled  = 0;
        ucs_assert_always(devices[index].count <= 64); /* Using uint64_t bitmap */
        config_idx = ucp_str_array_search((const char**)devices[index].names,
                                          devices[index].count,
                                          resource->dev_name);
        if (config_idx >= 0) {
            device_enabled  = 1;
            masks[index] |= UCS_BIT(config_idx);
        }
    }

    return device_enabled;
}

static int ucp_is_resource_enabled(uct_tl_resource_desc_t *resource,
                                   const ucp_config_t *config,
                                   uint64_t *masks)
{
    int device_enabled, tl_enabled;
    ucp_tl_alias_t *alias;

    /* Find the enabled devices */
    device_enabled = ucp_is_resource_in_device_list(resource, config->devices,
                                                    masks, resource->dev_type);

    /* Find the enabled UCTs */
    ucs_assert(config->tls.count > 0);
    if (ucp_config_is_tl_enabled(config, resource->tl_name, 0)) {
        tl_enabled = 1;
    } else {
        tl_enabled = 0;

        /* check aliases */
        for (alias = ucp_tl_aliases; alias->alias != NULL; ++alias) {

            /* If an alias is enabled, and the transport is part of this alias,
             * enable the transport.
             */
            if (ucp_config_is_tl_enabled(config, alias->alias, 1) &&
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

static ucs_status_t ucp_add_tl_resources(ucp_context_h context, ucp_tl_md_t *md,
                                         ucp_rsc_index_t md_index,
                                         const ucp_config_t *config,
                                         unsigned *num_resources_p,
                                         uint64_t *masks)
{
    uct_tl_resource_desc_t *tl_resources;
    ucp_tl_resource_desc_t *tmp;
    unsigned num_tl_resources;
    ucs_status_t status;
    ucp_rsc_index_t i;

    *num_resources_p = 0;

    /* check what are the available uct resources */
    status = uct_md_query_tl_resources(md->md, &tl_resources, &num_tl_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s", ucs_status_string(status));
        goto err;
    }

    if (num_tl_resources == 0) {
        ucs_debug("No tl resources found for md %s", md->rsc.md_name);
        goto out_free_resources;
    }

    tmp = ucs_realloc(context->tl_rscs,
                      sizeof(*context->tl_rscs) * (context->num_tls + num_tl_resources),
                      "ucp resources");
    if (tmp == NULL) {
        ucs_error("Failed to allocate resources");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_resources;
    }

    /* print configuration */
    for (i = 0; i < config->tls.count; ++i) {
        ucs_trace("allowed transport %d : '%s'", i, config->tls.names[i]);
    }

    /* copy only the resources enabled by user configuration */
    context->tl_rscs = tmp;
    for (i = 0; i < num_tl_resources; ++i) {
        if (ucp_is_resource_enabled(&tl_resources[i], config, masks)) {
            context->tl_rscs[context->num_tls].tl_rsc   = tl_resources[i];
            context->tl_rscs[context->num_tls].md_index = md_index;
            context->tl_rscs[context->num_tls].tl_name_csum =
                            ucs_crc16_string(tl_resources[i].tl_name);
            ++context->num_tls;
            ++(*num_resources_p);
        }
    }

out_free_resources:
    uct_release_tl_resource_list(tl_resources);
    return UCS_OK;

err_free_resources:
    uct_release_tl_resource_list(tl_resources);
err:
    return status;
}

static void ucp_report_unavailable_devices(const ucs_config_names_array_t *devices,
                                           uint64_t *masks)
{
    int dev_type_idx, i;

    /* Go over the devices lists and check which devices were marked as unavailable */
    for (dev_type_idx = 0; dev_type_idx < UCT_DEVICE_TYPE_LAST; dev_type_idx++) {
        if (masks[dev_type_idx] == 0) {
            ucs_info("No %s devices are available",
                     ucp_device_type_names[dev_type_idx]);
        } else {
            for (i = 0; i < devices[dev_type_idx].count; i++) {
                if (!(masks[dev_type_idx] & UCS_BIT(i))) {
                    ucs_info("Device %s is not available",
                             devices[dev_type_idx].names[i]);
                }
            }
        }
    }
}

const char * ucp_find_tl_name_by_csum(ucp_context_t *context, uint16_t tl_name_csum)
{
    ucp_tl_resource_desc_t *rsc;

    for (rsc = context->tl_rscs; rsc < context->tl_rscs + context->num_tls; ++rsc) {
         if (rsc->tl_name_csum == tl_name_csum) {
             return rsc->tl_rsc.tl_name;
         }
    }
    return NULL;
}

static ucs_status_t ucp_check_tl_names(ucp_context_t *context)
{
    ucp_tl_resource_desc_t *rsc;
    const char *tl_name;

    /* Make there we don't have two different transports with same checksum. */
    for (rsc = context->tl_rscs; rsc < context->tl_rscs + context->num_tls; ++rsc) {
        tl_name = ucp_find_tl_name_by_csum(context, rsc->tl_name_csum);
        if ((tl_name != NULL) && strcmp(rsc->tl_rsc.tl_name, tl_name)) {
            ucs_error("Transports '%s' and '%s' have same checksum (0x%x), "
                            "please rename one of them to avoid collision",
                            rsc->tl_rsc.tl_name, tl_name, rsc->tl_name_csum);
            return UCS_ERR_ALREADY_EXISTS;
        }
    }
    return UCS_OK;
}

static void ucp_free_resources(ucp_context_t *context)
{
    ucp_rsc_index_t i;

    ucs_free(context->tl_rscs);
    for (i = 0; i < context->num_mds; ++i) {
        uct_md_close(context->tl_mds[i].md);
    }
    ucs_free(context->tl_mds);
}

static ucs_status_t ucp_check_resource_config(const ucp_config_t *config)
{
    /* if we got here then num_resources > 0.
      * if the user's device list is empty, there is no match */
     if ((0 == config->devices[UCT_DEVICE_TYPE_NET].count) &&
         (0 == config->devices[UCT_DEVICE_TYPE_SHM].count) &&
         (0 == config->devices[UCT_DEVICE_TYPE_ACC].count) &&
         (0 == config->devices[UCT_DEVICE_TYPE_SELF].count)) {
         ucs_error("The device lists are empty. Please specify the devices you would like to use "
                   "or omit the UCX_*_DEVICES so that the default will be used.");
         return UCS_ERR_NO_ELEM;
     }

     /* if we got here then num_resources > 0.
      * if the user's tls list is empty, there is no match */
     if (0 == config->tls.count) {
         ucs_error("The TLs list is empty. Please specify the transports you would like to use "
                   "or omit the UCX_TLS so that the default will be used.");
         return UCS_ERR_NO_ELEM;
     }

     return UCS_OK;
}

static ucs_status_t ucp_fill_tl_md(const uct_md_resource_desc_t *md_rsc,
                                   ucp_tl_md_t *tl_md)
{

    uct_md_config_t *md_config;
    ucs_status_t status;

    /* Save MD resource */
    tl_md->rsc = *md_rsc;

    /* Read MD configuration */
    status = uct_md_config_read(md_rsc->md_name, NULL, NULL, &md_config);
    if (status != UCS_OK) {
        return status;
    }

    /* Open MD */
    status = uct_md_open(md_rsc->md_name, md_config, &tl_md->md);
    uct_config_release(md_config);
    if (status != UCS_OK) {
        return status;
    }

    VALGRIND_MAKE_MEM_UNDEFINED(&tl_md->attr, sizeof(tl_md->attr));
    /* Save MD attributes */
    status = uct_md_query(tl_md->md, &tl_md->attr);
    if (status != UCS_OK) {
        uct_md_close(tl_md->md);
        return status;
    }

    return UCS_OK;
}

static void ucp_resource_config_array_str(const ucs_config_names_array_t *array,
                                          const char *title, char *buf, size_t max)
{
    char *p, *endp;
    unsigned i;

    if (ucp_str_array_search((const char**)array->names, array->count, "all") >= 0) {
        strncpy(buf, "", max);
        return;
    }

    p    = buf;
    endp = buf + max;

    if (strlen(title)) {
        snprintf(p, endp - p, "%s:", title);
        p += strlen(p);
    }

    for (i = 0; i < array->count; ++i) {
        snprintf(p, endp - p, "%s%c", array->names[i],
                  (i == array->count - 1) ? ' ' : ',');
        p += strlen(p);
    }
}

static void ucp_resource_config_str(const ucp_config_t *config, char *buf,
                                    size_t max)
{
    int dev_type_idx;
    char *p, *endp, *devs_p;

    p    = buf;
    endp = buf + max;

    ucp_resource_config_array_str(&config->tls, "", p, endp - p);

    if (strlen(p)) {
        p += strlen(p);
        snprintf(p, endp - p, "on ");
        p += strlen(p);
    }

    devs_p = p;
    for (dev_type_idx = 0; dev_type_idx < UCT_DEVICE_TYPE_LAST; ++dev_type_idx) {
        ucp_resource_config_array_str(&config->devices[dev_type_idx],
                                      ucp_device_type_names[dev_type_idx], p,
                                      endp - p);
        p += strlen(p);
    }

    if (devs_p == p) {
        snprintf(p, endp - p, "all devices");
    }
}

static ucs_status_t ucp_check_resources(ucp_context_h context,
                                        const ucp_config_t *config)
{
    char info_str[128];

    /* Error check: Make sure there is at least one transport */
    if (0 == context->num_tls) {
        ucp_resource_config_str(config, info_str, sizeof(info_str));
        ucs_error("No matching transports/devices, using %s", info_str);
        return UCS_ERR_NO_DEVICE;
    }

    /* Error check: Make sure there are not too many transports */
    if (context->num_tls >= UCP_MAX_RESOURCES) {
        ucs_error("Exceeded transports/devices limit (%u requested, up to %d are supported)",
                  context->num_tls, UCP_MAX_RESOURCES);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    return ucp_check_tl_names(context);
}

static ucs_status_t ucp_fill_resources(ucp_context_h context,
                                       const ucp_config_t *config)
{
    unsigned num_tl_resources;
    unsigned num_md_resources;
    uct_md_resource_desc_t *md_rscs;
    ucs_status_t status;
    ucp_rsc_index_t i;
    unsigned md_index;
    uint64_t masks[UCT_DEVICE_TYPE_LAST] = {0};

    context->tl_mds      = NULL;
    context->num_mds     = 0;
    context->tl_rscs     = NULL;
    context->num_tls     = 0;

    status = ucp_check_resource_config(config);
    if (status != UCS_OK) {
        goto err;
    }

    /* List memory domain resources */
    status = uct_query_md_resources(&md_rscs, &num_md_resources);
    if (status != UCS_OK) {
        goto err;
    }

    /* Error check: Make sure there is at least one MD */
    if (num_md_resources == 0) {
        ucs_error("No memory domain resources found");
        status = UCS_ERR_NO_DEVICE;
        goto err_release_md_resources;
    }

    /* Allocate actual array of MDs */
    context->tl_mds = ucs_malloc(num_md_resources * sizeof(*context->tl_mds),
                                 "ucp_tl_mds");
    if (context->tl_mds == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_release_md_resources;
    }

    /* Open all memory domains */
    md_index = 0;
    for (i = 0; i < num_md_resources; ++i) {
        status = ucp_fill_tl_md(&md_rscs[i], &context->tl_mds[md_index]);
        if (status != UCS_OK) {
            goto err_free_context_resources;
        }

        /* Add communication resources of each MD */
        status = ucp_add_tl_resources(context, &context->tl_mds[md_index],
                                      md_index, config, &num_tl_resources, masks);
        if (status != UCS_OK) {
            uct_md_close(context->tl_mds[md_index].md);
            goto err_free_context_resources;
        }

        /* If the MD does not have transport resources, and if md does not support
         * client-server connection establishment via sockaddr, don't use it */
        if ((num_tl_resources > 0) ||
            (context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_SOCKADDR)) {
            ++md_index;
            ++context->num_mds;
        } else {
            ucs_debug("closing md %s because it has no selected transport resources",
                      md_rscs[i].md_name);
            uct_md_close(context->tl_mds[md_index].md);
        }
    }

    /* Validate context resources */
    status = ucp_check_resources(context, config);
    if (status != UCS_OK) {
        goto err_free_context_resources;
    }

    uct_release_md_resource_list(md_rscs);

    /* Notify the user if there are devices from the command line that are not available */
    ucp_report_unavailable_devices(config->devices, masks);

    return UCS_OK;

err_free_context_resources:
    ucp_free_resources(context);
err_release_md_resources:
    uct_release_md_resource_list(md_rscs);
err:
    return status;
}

static void ucp_apply_params(ucp_context_h context, const ucp_params_t *params,
                             ucp_mt_type_t mt_type)
{
    if (params->field_mask & UCP_PARAM_FIELD_FEATURES) {
        context->config.features = params->features;
    } else {
        context->config.features = 0;
    }
    if (!context->config.features) {
        ucs_warn("empty features set passed to ucp context create");
    }

    if (params->field_mask & UCP_PARAM_FIELD_TAG_SENDER_MASK) {
        context->config.tag_sender_mask = params->tag_sender_mask;
    } else {
        context->config.tag_sender_mask = 0;
    }

    if (params->field_mask & UCP_PARAM_FIELD_REQUEST_SIZE) {
        context->config.request.size = params->request_size;
    } else {
        context->config.request.size = 0;
    }

    if (params->field_mask & UCP_PARAM_FIELD_REQUEST_INIT) {
        context->config.request.init = params->request_init;
    } else {
        context->config.request.init = NULL;
    }

    if (params->field_mask & UCP_PARAM_FIELD_REQUEST_CLEANUP) {
        context->config.request.cleanup = params->request_cleanup;
    } else {
        context->config.request.cleanup = NULL;
    }

    if (params->field_mask & UCP_PARAM_FIELD_ESTIMATED_NUM_EPS) {
        context->config.est_num_eps = params->estimated_num_eps;
    } else {
        context->config.est_num_eps = 1;
    }

    if ((params->field_mask & UCP_PARAM_FIELD_MT_WORKERS_SHARED) &&
        params->mt_workers_shared) {
        context->mt_lock.mt_type = mt_type;
    } else {
        context->mt_lock.mt_type = UCP_MT_TYPE_NONE;
    }
}

static ucs_status_t ucp_fill_config(ucp_context_h context,
                                    const ucp_params_t *params,
                                    const ucp_config_t *config)
{
    unsigned i, num_alloc_methods, method;
    const char *method_name;
    ucs_status_t status;

    ucp_apply_params(context, params,
                     config->ctx.use_mt_mutex ? UCP_MT_TYPE_MUTEX
                                              : UCP_MT_TYPE_SPINLOCK);
    context->config.ext = config->ctx;

    /* always init MT lock in context even though it is disabled by user,
     * because we need to use context lock to protect ucp_mm_ and ucp_rkey_
     * routines */
    UCP_THREAD_LOCK_INIT(&context->mt_lock);

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
        if (!strncasecmp(method_name, "md:", 3)) {
            /* If the method name begins with 'md:', treat it as memory domain
             * component name.
             */
            context->config.alloc_methods[i].method = UCT_ALLOC_METHOD_MD;
            ucs_strncpy_zero(context->config.alloc_methods[i].mdc_name,
                             method_name + 3, UCT_MD_COMPONENT_NAME_MAX);
            ucs_debug("allocation method[%d] is md '%s'", i, method_name + 3);
        } else {
            /* Otherwise, this is specific allocation method name.
             */
            context->config.alloc_methods[i].method = UCT_ALLOC_METHOD_LAST;
            for (method = 0; method < UCT_ALLOC_METHOD_LAST; ++method) {
                if ((method != UCT_ALLOC_METHOD_MD) &&
                    !strcmp(method_name, uct_alloc_method_names[method]))
                {
                    /* Found the allocation method in the internal name list */
                    context->config.alloc_methods[i].method = method;
                    strcpy(context->config.alloc_methods[i].mdc_name, "");
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

    /* Need to check MAX_BCOPY value if it is enabled only */
    if (context->config.ext.tm_max_bcopy > context->config.ext.tm_thresh) {
        if (context->config.ext.tm_max_bcopy < sizeof(ucp_request_hdr_t)) {
            /* In case of expected SW RNDV message, the header (ucp_request_hdr_t) is
             * scattered to UCP user buffer. Make sure that bounce buffer is used for
             * messages which can not fit SW RNDV hdr. */
            context->config.ext.tm_max_bcopy = sizeof(ucp_request_hdr_t);
            ucs_info("UCX_TM_MAX_BCOPY value: %zu, adjusted to: %zu",
                     context->config.ext.tm_max_bcopy, sizeof(ucp_request_hdr_t));
        }

        if (context->config.ext.tm_max_bcopy > context->config.ext.seg_size) {
            context->config.ext.tm_max_bcopy = context->config.ext.seg_size;
            ucs_info("Wrong UCX_TM_MAX_BCOPY value: %zu, adjusted to: %zu",
                     context->config.ext.tm_max_bcopy,
                     context->config.ext.seg_size);
        }
    }

    return UCS_OK;

err_free:
    ucs_free(context->config.alloc_methods);
err:
    UCP_THREAD_LOCK_FINALIZE(&context->mt_lock);
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
    status = ucp_tag_match_init(&context->tm);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    ucs_debug("created ucp context %p [%d mds %d tls] features 0x%lx", context,
              context->num_mds, context->num_tls, context->config.features);

    *context_p = context;
    return UCS_OK;

err_free_resources:
    ucp_free_resources(context);
err_free_config:
    ucp_free_config(context);
err_free_ctx:
    ucs_free(context);
err:
    return status;
}

void ucp_cleanup(ucp_context_h context)
{
    ucp_tag_match_cleanup(&context->tm);
    ucp_free_resources(context);
    ucp_free_config(context);
    UCP_THREAD_LOCK_FINALIZE(&context->mt_lock);
    ucs_free(context);
}

void ucp_dump_payload(ucp_context_h context, char *buffer, size_t max,
                      const void *data, size_t length)
{
    size_t data_size = ucs_global_opts.log_data_size;
    char *p, *endp;
    size_t offset;

    if (data_size == 0) {
        return;
    }

    p    = buffer;
    endp = buffer + max;

    strncat(p, " : ", endp - p);
    p = p + strlen(p);

    offset = 0;
    while ((offset < length) && (offset < data_size) && (p < endp)) {
        snprintf(p, endp - p, "%02x", ((const uint8_t*)data)[offset]);
        p += strlen(p);
        ++offset;
    }
}

uint64_t ucp_context_uct_atomic_iface_flags(ucp_context_h context)
{
    return ((context->config.features & UCP_FEATURE_AMO32) ?
            UCP_UCT_IFACE_ATOMIC32_FLAGS : 0) |
           ((context->config.features & UCP_FEATURE_AMO64) ?
            UCP_UCT_IFACE_ATOMIC64_FLAGS : 0);
}

ucs_status_t ucp_context_query(ucp_context_h context, ucp_context_attr_t *attr)
{
    if (attr->field_mask & UCP_ATTR_FIELD_REQUEST_SIZE) {
        attr->request_size = sizeof(ucp_request_t);
    }
    if (attr->field_mask & UCP_ATTR_FIELD_THREAD_MODE) {
        if (UCP_THREAD_IS_REQUIRED(&context->mt_lock)) {
            attr->thread_mode = UCS_THREAD_MODE_MULTI;
        } else {
            attr->thread_mode = UCS_THREAD_MODE_SINGLE;
        }
    }

    return UCS_OK;
}

void ucp_context_print_info(ucp_context_h context, FILE *stream)
{
    ucp_rsc_index_t md_index, rsc_index;

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP context\n");
    fprintf(stream, "#\n");

    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        fprintf(stream, "#                   md[%d]:  %s\n",
                md_index, context->tl_mds[md_index].rsc.md_name);
    }

    fprintf(stream, "#\n");

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        fprintf(stream, "#    resource[%2d] / md[%d]:  "UCT_TL_RESOURCE_DESC_FMT"\n",
                rsc_index, context->tl_rscs[rsc_index].md_index,
                UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc)
                );
    }

    fprintf(stream, "#\n");
}

void ucp_context_tag_offload_enable(ucp_context_h context)
{
    ucp_worker_iface_t *offload_iface;

    /* Enable offload, if only one tag offload capable interface is present
     * (multiple offload ifaces are not supported yet). */
    if (context->config.ext.tm_offload &&
        (ucs_queue_length(&context->tm.offload.ifaces) == 1)) {
        context->tm.offload.thresh       = context->config.ext.tm_thresh;
        context->tm.offload.zcopy_thresh = context->config.ext.tm_max_bcopy;

        offload_iface = ucs_queue_head_elem_non_empty(&context->tm.offload.ifaces,
                                                      ucp_worker_iface_t, queue);
        ucp_worker_iface_activate(offload_iface, 0);

        ucs_debug("Enable TM offload: thresh %zu, zcopy_thresh %zu",
                  context->tm.offload.thresh, context->tm.offload.zcopy_thresh);
    } else {
        context->tm.offload.thresh = SIZE_MAX;
        ucs_debug("Disable TM offload, multiple offload ifaces are not supported");
    }
}

