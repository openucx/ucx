/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_context.h"
#include "ucp_request.h"

#include <ucs/config/parser.h>
#include <ucs/algorithm/crc.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/string_set.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <string.h>


#define UCP_RSC_CONFIG_ALL    "all"

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

static const char * ucp_rndv_modes[] = {
    [UCP_RNDV_MODE_GET_ZCOPY] = "get_zcopy",
    [UCP_RNDV_MODE_PUT_ZCOPY] = "put_zcopy",
    [UCP_RNDV_MODE_AUTO]      = "auto",
    [UCP_RNDV_MODE_LAST]      = NULL,
};

const char* ucp_operation_names[] = {
    [UCP_OP_ID_TAG_SEND]      = "tag_send",
    [UCP_OP_ID_TAG_SEND_SYNC] = "tag_send_sync",
    [UCP_OP_ID_PUT]           = "put",
    [UCP_OP_ID_GET]           = "get",
    [UCP_OP_ID_LAST]          = NULL
};

static ucs_config_field_t ucp_config_table[] = {
  {"NET_DEVICES", UCP_RSC_CONFIG_ALL,
   "Specifies which network device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_NET]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SHM_DEVICES", UCP_RSC_CONFIG_ALL,
   "Specifies which intra-node device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_SHM]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"ACC_DEVICES", UCP_RSC_CONFIG_ALL,
   "Specifies which accelerator device(s) to use. The order is not meaningful.\n"
   "\"all\" would use all available devices.",
   ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_ACC]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SELF_DEVICES", UCP_RSC_CONFIG_ALL,
    "Specifies which loop-back device(s) to use. The order is not meaningful.\n"
    "\"all\" would use all available devices.",
    ucs_offsetof(ucp_config_t, devices[UCT_DEVICE_TYPE_SELF]), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"TLS", UCP_RSC_CONFIG_ALL,
   "Comma-separated list of transports to use. The order is not meaningful.\n"
   " - all     : use all the available transports.\n"
   " - sm/shm  : all shared memory transports (mm, cma, knem).\n"
   " - mm      : shared memory transports - only memory mappers.\n"
   " - ugni    : ugni_smsg and ugni_rdma (uses ugni_udt for bootstrap).\n"
   " - ib      : all infiniband transports (rc/rc_mlx5, ud/ud_mlx5, dc_mlx5).\n"
   " - rc_v    : rc verbs (uses ud for bootstrap).\n"
   " - rc_x    : rc with accelerated verbs (uses ud_mlx5 for bootstrap).\n"
   " - rc      : rc_v and rc_x (preferably if available).\n"
   " - ud_v    : ud verbs.\n"
   " - ud_x    : ud with accelerated verbs.\n"
   " - ud      : ud_v and ud_x (preferably if available).\n"
   " - dc/dc_x : dc with accelerated verbs.\n"
   " - tcp     : sockets over TCP/IP.\n"
   " - cuda    : CUDA (NVIDIA GPU) memory support.\n"
   " - rocm    : ROCm (AMD GPU) memory support.\n"
   " Using a \\ prefix before a transport name treats it as an explicit transport name\n"
   " and disables aliasing.\n",
   ucs_offsetof(ucp_config_t, tls), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"ALLOC_PRIO", "md:sysv,md:posix,huge,thp,md:*,mmap,heap",
   "Priority of memory allocation methods. Each item in the list can be either\n"
   "an allocation method (huge, thp, mmap, libc) or md:<NAME> which means to use the\n"
   "specified memory domain for allocation. NAME can be either a UCT component\n"
   "name, or a wildcard - '*' - which is equivalent to all UCT components.",
   ucs_offsetof(ucp_config_t, alloc_prio), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SOCKADDR_TLS_PRIORITY", "rdmacm,tcp,sockcm",
   "Priority of sockaddr transports for client/server connection establishment.\n"
   "The '*' wildcard expands to all the available sockaddr transports.",
   ucs_offsetof(ucp_config_t, sockaddr_cm_tls), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SOCKADDR_AUX_TLS", "ud",
   "Transports to use for exchanging additional address information while\n"
   "establishing client/server connection. ",
   ucs_offsetof(ucp_config_t, sockaddr_aux_tls), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"SELECT_DISTANCE_MD", "cuda_cpy",
   "MD whose distance is queried when evaluating transport selection score",
   ucs_offsetof(ucp_config_t, selection_cmp), UCS_CONFIG_TYPE_STRING},

  {"WARN_INVALID_CONFIG", "y",
   "Issue a warning in case of invalid device and/or transport configuration.",
   ucs_offsetof(ucp_config_t, warn_invalid_config), UCS_CONFIG_TYPE_BOOL},

  {"BCOPY_THRESH", "0",
   "Threshold for switching from short to bcopy protocol",
   ucs_offsetof(ucp_config_t, ctx.bcopy_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_THRESH", UCS_VALUE_AUTO_STR,
   "Threshold for switching from eager to rendezvous protocol",
   ucs_offsetof(ucp_config_t, ctx.rndv_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_SEND_NBR_THRESH", "256k",
   "Threshold for switching from eager to rendezvous protocol in ucp_tag_send_nbr().\n"
   "Relevant only if UCX_RNDV_THRESH is set to \"auto\".",
   ucs_offsetof(ucp_config_t, ctx.rndv_send_nbr_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_THRESH_FALLBACK", "inf",
   "Message size to start using the rendezvous protocol in case the calculated threshold "
   "is zero or negative",
   ucs_offsetof(ucp_config_t, ctx.rndv_thresh_fallback), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_PERF_DIFF", "1",
   "The percentage allowed for performance difference between rendezvous and "
   "the eager_zcopy protocol",
   ucs_offsetof(ucp_config_t, ctx.rndv_perf_diff), UCS_CONFIG_TYPE_DOUBLE},

  {"MULTI_LANE_MAX_RATIO", "10",
   "Maximal allowed ratio between slowest and fastest lane in a multi-lane "
   "protocol. Lanes slower than the specified ratio will not be used.",
   ucs_offsetof(ucp_config_t, ctx.multi_lane_max_ratio), UCS_CONFIG_TYPE_DOUBLE},

  {"MAX_EAGER_LANES", NULL, "",
   ucs_offsetof(ucp_config_t, ctx.max_eager_lanes), UCS_CONFIG_TYPE_UINT},

  {"MAX_EAGER_RAILS", "1",
   "Maximal number of devices on which an eager operation may be executed in parallel",
   ucs_offsetof(ucp_config_t, ctx.max_eager_lanes), UCS_CONFIG_TYPE_UINT},

  {"MAX_RNDV_LANES", NULL,"",
   ucs_offsetof(ucp_config_t, ctx.max_rndv_lanes), UCS_CONFIG_TYPE_UINT},

  {"MAX_RNDV_RAILS", "2",
   "Maximal number of devices on which a rendezvous operation may be executed in parallel",
   ucs_offsetof(ucp_config_t, ctx.max_rndv_lanes), UCS_CONFIG_TYPE_UINT},

  {"RNDV_SCHEME", "auto",
   "Communication scheme in RNDV protocol.\n"
   " get_zcopy - use get_zcopy scheme in RNDV protocol.\n"
   " put_zcopy - use put_zcopy scheme in RNDV protocol.\n"
   " auto      - runtime automatically chooses optimal scheme to use.\n",
   ucs_offsetof(ucp_config_t, ctx.rndv_mode), UCS_CONFIG_TYPE_ENUM(ucp_rndv_modes)},

  {"RKEY_PTR_SEG_SIZE", "512k",
   "Segment size that is used to perform data transfer when doing RKEY PTR progress",
   ucs_offsetof(ucp_config_t, ctx.rkey_ptr_seg_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"ZCOPY_THRESH", "auto",
   "Threshold for switching from buffer copy to zero copy protocol",
   ucs_offsetof(ucp_config_t, ctx.zcopy_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"BCOPY_BW", "auto",
   "Estimation of buffer copy bandwidth",
   ucs_offsetof(ucp_config_t, ctx.bcopy_bw), UCS_CONFIG_TYPE_BW},

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

  {"ADDRESS_DEBUG_INFO",
#if ENABLE_DEBUG_DATA
   "y",
#else
   "n",
#endif
   "Add debugging information to worker address.",
   ucs_offsetof(ucp_config_t, ctx.address_debug_info), UCS_CONFIG_TYPE_BOOL},

  {"MAX_WORKER_NAME", UCS_PP_MAKE_STRING(UCP_WORKER_NAME_MAX),
   "Maximal length of worker name. Sent to remote peer as part of worker address\n"
   "if UCX_ADDRESS_DEBUG_INFO is set to 'yes'",
   ucs_offsetof(ucp_config_t, ctx.max_worker_name), UCS_CONFIG_TYPE_UINT},

  {"USE_MT_MUTEX", "n", "Use mutex for multithreading support in UCP.\n"
   "n      - Not use mutex for multithreading support in UCP (use spinlock by default).\n"
   "y      - Use mutex for multithreading support in UCP.\n",
   ucs_offsetof(ucp_config_t, ctx.use_mt_mutex), UCS_CONFIG_TYPE_BOOL},

  {"ADAPTIVE_PROGRESS", "y",
   "Enable adaptive progress mechanism, which turns on polling only on active\n"
   "transport interfaces.",
   ucs_offsetof(ucp_config_t, ctx.adaptive_progress), UCS_CONFIG_TYPE_BOOL},

  {"SEG_SIZE", "8192",
   "Size of a segment in the worker preregistered memory pool.",
   ucs_offsetof(ucp_config_t, ctx.seg_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_THRESH", "1024", /* TODO: calculate automatically */
   "Threshold for using tag matching offload capabilities.\n"
   "Smaller buffers will not be posted to the transport.",
   ucs_offsetof(ucp_config_t, ctx.tm_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_MAX_BB_SIZE", "1024", /* TODO: calculate automatically */
   "Maximal size for posting \"bounce buffer\" (UCX internal preregistered memory) for\n"
   "tag offload receives. When message arrives, it is copied into the user buffer (similar\n"
   "to eager protocol). The size values has to be equal or less than segment size.\n"
   "Also the value has to be bigger than UCX_TM_THRESH to take an effect." ,
   ucs_offsetof(ucp_config_t, ctx.tm_max_bb_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_FORCE_THRESH", "8192", /* TODO: calculate automatically */
   "Threshold for forcing tag matching offload mode. Every tag receive operation\n"
   "with buffer bigger than this threshold would force offloading of all uncompleted\n"
   "non-offloaded receive operations to the transport (e. g. operations with\n"
   "buffers below the UCX_TM_THRESH value). Offloading may be unsuccessful in certain\n"
   "cases (non-contig buffer, or sender wildcard).",
   ucs_offsetof(ucp_config_t, ctx.tm_force_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_SW_RNDV", "n",
   "Use software rendezvous protocol with tag offload. If enabled, tag offload\n"
   "mode will be used for messages sent with eager protocol only.",
   ucs_offsetof(ucp_config_t, ctx.tm_sw_rndv), UCS_CONFIG_TYPE_BOOL},

  {"NUM_EPS", "auto",
   "An optimization hint of how many endpoints would be created on this context.\n"
   "Does not affect semantics, but only transport selection criteria and the\n"
   "resulting performance.\n"
   " If set to a value different from \"auto\" it will override the value passed\n"
   "to ucp_init()",
   ucs_offsetof(ucp_config_t, ctx.estimated_num_eps), UCS_CONFIG_TYPE_ULUNITS},

  {"NUM_PPN", "auto",
   "An optimization hint for the number of processes expected to be launched\n"
   "on a single node. Does not affect semantics, only transport selection criteria\n"
   "and the resulting performance.\n",
   ucs_offsetof(ucp_config_t, ctx.estimated_num_ppn), UCS_CONFIG_TYPE_ULUNITS},

  {"RNDV_FRAG_SIZE", "512k",
   "RNDV fragment size \n",
   ucs_offsetof(ucp_config_t, ctx.rndv_frag_size), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_PIPELINE_SEND_THRESH", "inf",
   "RNDV size threshold to enable sender side pipeline for mem type\n",
   ucs_offsetof(ucp_config_t, ctx.rndv_pipeline_send_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"MEMTYPE_CACHE", "y",
   "Enable memory type (cuda/rocm) cache \n",
   ucs_offsetof(ucp_config_t, ctx.enable_memtype_cache), UCS_CONFIG_TYPE_BOOL},

  {"FLUSH_WORKER_EPS", "y",
   "Enable flushing the worker by flushing its endpoints. Allows completing\n"
   "the flush operation in a bounded time even if there are new requests on\n"
   "another thread, or incoming active messages, but consumes more resources.",
   ucs_offsetof(ucp_config_t, ctx.flush_worker_eps), UCS_CONFIG_TYPE_BOOL},

  {"UNIFIED_MODE", "n",
   "Enable various optimizations intended for homogeneous environment.\n"
   "Enabling this mode implies that the local transport resources/devices\n"
   "of all entities which connect to each other are the same.",
   ucs_offsetof(ucp_config_t, ctx.unified_mode), UCS_CONFIG_TYPE_BOOL},

  {"SOCKADDR_CM_ENABLE", "y",
   "Enable alternative wireup protocol for sockaddr connected endpoints.\n"
   "Enabling this mode changes underlying UCT mechanism for connection\n"
   "establishment and enables synchronized close protocol which does not\n"
   "require out of band synchronization before destroying UCP resources.",
   ucs_offsetof(ucp_config_t, ctx.sockaddr_cm_enable), UCS_CONFIG_TYPE_TERNARY},

  {"CM_USE_ALL_DEVICES", "y",
   "When creating client/server endpoints, use all available devices.\n"
   "If disabled, use only the one device on which the connection\n"
   "establishment is done\n",
   ucs_offsetof(ucp_config_t, ctx.cm_use_all_devices), UCS_CONFIG_TYPE_BOOL},

  {"LISTENER_BACKLOG", "auto",
   "'auto' means that each transport would use its maximal allowed value.\n"
   "If a value larger than what a transport supports is set, the backlog value\n"
   "would be cut to that maximal value.",
   ucs_offsetof(ucp_config_t, ctx.listener_backlog), UCS_CONFIG_TYPE_ULUNITS},

  {"PROTO_ENABLE", "n",
   "Experimental: enable new protocol selection logic",
   ucs_offsetof(ucp_config_t, ctx.proto_enable), UCS_CONFIG_TYPE_BOOL},

  /* TODO: set for keepalive more reasonable values */
  {"KEEPALIVE_INTERVAL", "60s",
   "Time interval between keepalive rounds (0 - disabled).",
   ucs_offsetof(ucp_config_t, ctx.keepalive_interval), UCS_CONFIG_TYPE_TIME},

  {"KEEPALIVE_NUM_EPS", "128",
   "Maximal number of endpoints to check on every keepalive round\n"
   "(inf - check all endpoints on every round, must be greater than 0)",
   ucs_offsetof(ucp_config_t, ctx.keepalive_num_eps), UCS_CONFIG_TYPE_UINT},

  {"PROTO_INDIRECT_ID", "auto",
   "Enable indirect IDs to object pointers (endpoint, request) in wire protocols.\n"
   "A value of 'auto' means to enable only if error handling is enabled on the\n"
   "endpoint.",
   ucs_offsetof(ucp_config_t, ctx.proto_indirect_id), UCS_CONFIG_TYPE_ON_OFF_AUTO},

   {NULL}
};
UCS_CONFIG_REGISTER_TABLE(ucp_config_table, "UCP context", NULL, ucp_config_t,
                          &ucs_config_global_list)


static ucp_tl_alias_t ucp_tl_aliases[] = {
  { "mm",    { "posix", "sysv", "xpmem", NULL } }, /* for backward compatibility */
  { "sm",    { "posix", "sysv", "xpmem", "knem", "cma", "rdmacm", "sockcm", NULL } },
  { "shm",   { "posix", "sysv", "xpmem", "knem", "cma", "rdmacm", "sockcm", NULL } },
  { "ib",    { "rc_verbs", "ud_verbs", "rc_mlx5", "ud_mlx5", "dc_mlx5", "rdmacm", NULL } },
  { "ud_v",  { "ud_verbs", "rdmacm", NULL } },
  { "ud_x",  { "ud_mlx5", "rdmacm", NULL } },
  { "ud",    { "ud_mlx5", "ud_verbs", "rdmacm", NULL } },
  { "rc_v",  { "rc_verbs", "ud_verbs:aux", "rdmacm", NULL } },
  { "rc_x",  { "rc_mlx5", "ud_mlx5:aux", "rdmacm", NULL } },
  { "rc",    { "rc_mlx5", "ud_mlx5:aux", "rc_verbs", "ud_verbs:aux", "rdmacm", NULL } },
  { "dc",    { "dc_mlx5", "rdmacm", NULL } },
  { "dc_x",  { "dc_mlx5", "rdmacm", NULL } },
  { "ugni",  { "ugni_smsg", "ugni_udt:aux", "ugni_rdma", NULL } },
  { "cuda",  { "cuda_copy", "cuda_ipc", "gdr_copy", NULL } },
  { "rocm",  { "rocm_copy", "rocm_ipc", "rocm_gdr", NULL } },
  { NULL }
};


const char *ucp_feature_str[] = {
    [ucs_ilog2(UCP_FEATURE_TAG)]    = "UCP_FEATURE_TAG",
    [ucs_ilog2(UCP_FEATURE_RMA)]    = "UCP_FEATURE_RMA",
    [ucs_ilog2(UCP_FEATURE_AMO32)]  = "UCP_FEATURE_AMO32",
    [ucs_ilog2(UCP_FEATURE_AMO64)]  = "UCP_FEATURE_AMO64",
    [ucs_ilog2(UCP_FEATURE_WAKEUP)] = "UCP_FEATURE_WAKEUP",
    [ucs_ilog2(UCP_FEATURE_STREAM)] = "UCP_FEATURE_STREAM",
    [ucs_ilog2(UCP_FEATURE_AM)]     = "UCP_FEATURE_AM",
    NULL
};


ucs_status_t ucp_config_read(const char *env_prefix, const char *filename,
                             ucp_config_t **config_p)
{
    unsigned full_prefix_len = sizeof(UCS_DEFAULT_ENV_PREFIX) + 1;
    unsigned env_prefix_len  = 0;
    ucp_config_t *config;
    ucs_status_t status;

    config = ucs_malloc(sizeof(*config), "ucp config");
    if (config == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    if (env_prefix != NULL) {
        env_prefix_len   = strlen(env_prefix);
        full_prefix_len += env_prefix_len;
    }

    config->env_prefix = ucs_malloc(full_prefix_len, "ucp config");
    if (config->env_prefix == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_config;
    }

    if (env_prefix_len != 0) {
        ucs_snprintf_zero(config->env_prefix, full_prefix_len, "%s_%s",
                          env_prefix, UCS_DEFAULT_ENV_PREFIX);
    } else {
        ucs_snprintf_zero(config->env_prefix, full_prefix_len, "%s",
                          UCS_DEFAULT_ENV_PREFIX);
    }

    status = ucs_config_parser_fill_opts(config, ucp_config_table,
                                         config->env_prefix, NULL, 0);
    if (status != UCS_OK) {
        goto err_free_prefix;
    }

    *config_p = config;
    return UCS_OK;

err_free_prefix:
    ucs_free(config->env_prefix);
err_free_config:
    ucs_free(config);
err:
    return status;
}

void ucp_config_release(ucp_config_t *config)
{
    ucs_config_parser_release_opts(config, ucp_config_table);
    ucs_free(config->env_prefix);
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
    ucs_config_parser_print_opts(stream, title, config, ucp_config_table,
                                 NULL, UCS_DEFAULT_ENV_PREFIX, print_flags);
}

/* Search str in the array. If str_suffix is specified, search for
 * 'str:str_suffix' string.
 * @return bitmap of indexes in which the string appears in the array.
 */
static uint64_t ucp_str_array_search(const char **array, unsigned array_len,
                                     const char *str, const char *str_suffix)
{
    const size_t len = strlen(str);
    uint64_t result;
    const char *p;
    int i;

    result = 0;
    for (i = 0; i < array_len; ++i) {
        if (str_suffix == NULL) {
            if (!strcmp(array[i], str)) {
                result |= UCS_BIT(i);
            }
        } else if (!strncmp(array[i], str, len)) {
            p = array[i] + len;
            if ((*p == ':') && !strcmp(p + 1, str_suffix)) {
                result |= UCS_BIT(i);
            }
        }
    }

    return result;
}

static unsigned ucp_tl_alias_count(ucp_tl_alias_t *alias)
{
    unsigned count;
    for (count = 0; alias->tls[count] != NULL; ++count);
    return count;
}

static int ucp_tls_array_is_present(const char **tls, unsigned count,
                                    const char *tl_name, const char *info,
                                    uint8_t *rsc_flags, uint64_t *tl_cfg_mask)
{
    uint64_t mask;

    if ((mask = ucp_str_array_search(tls, count, tl_name, NULL)) != 0) {
        *tl_cfg_mask |= mask;
        ucs_trace("enabling tl '%s'%s", tl_name, info);
        return 1;
    } else if ((mask = ucp_str_array_search(tls, count, tl_name, "aux")) != 0) {
        /* Search for tl names with 'aux' suffix, such tls can be
         * used for auxiliary wireup purposes only */
        *rsc_flags   |= UCP_TL_RSC_FLAG_AUX;
        *tl_cfg_mask |= mask;
        ucs_trace("enabling auxiliary tl '%s'%s", tl_name, info);
        return 1;
    } else {
        return 0;
    }
}

static int ucp_config_is_tl_enabled(const char **names, unsigned count,
                                    const char *tl_name, int is_alias,
                                    uint8_t *rsc_flags, uint64_t *tl_cfg_mask)
{
    char strict_name[UCT_TL_NAME_MAX + 1];

    snprintf(strict_name, sizeof(strict_name), "\\%s", tl_name);
    return /* strict name, with leading \\ */
           (!is_alias && ucp_tls_array_is_present(names, count, strict_name, "",
                                                  rsc_flags, tl_cfg_mask)) ||
           /* plain transport name */
           ucp_tls_array_is_present(names, count, tl_name, "", rsc_flags,
                                    tl_cfg_mask) ||
           /* all available transports */
           ucp_tls_array_is_present(names, count, UCP_RSC_CONFIG_ALL, "", rsc_flags,
                                    tl_cfg_mask);
}

static int ucp_is_resource_in_device_list(const uct_tl_resource_desc_t *resource,
                                          const ucs_config_names_array_t *devices,
                                          uint64_t *dev_cfg_mask,
                                          uct_device_type_t dev_type)
{
    uint64_t mask, exclusive_mask;

    /* go over the device list from the user and check (against the available resources)
     * which can be satisfied */
    ucs_assert_always(devices[dev_type].count <= 64); /* Using uint64_t bitmap */
    mask = ucp_str_array_search((const char**)devices[dev_type].names,
                                devices[dev_type].count, resource->dev_name,
                                NULL);
    if (!mask) {
        /* if the user's list is 'all', use all the available resources */
        mask = ucp_str_array_search((const char**)devices[dev_type].names,
                                    devices[dev_type].count, UCP_RSC_CONFIG_ALL,
                                    NULL);
    }

    /* warn if we got new device which appears more than once */
    exclusive_mask = mask & ~(*dev_cfg_mask);
    if (exclusive_mask && !ucs_is_pow2(exclusive_mask)) {
        ucs_warn("device '%s' is specified multiple times",
                 devices[dev_type].names[ucs_ilog2(exclusive_mask)]);
    }

    *dev_cfg_mask |= mask;
    return !!mask;
}

static int ucp_is_resource_in_transports_list(const char *tl_name,
                                              const char **names, unsigned count,
                                              uint8_t *rsc_flags, uint64_t *tl_cfg_mask)
{
    uint64_t dummy_mask, tmp_tl_cfg_mask;
    uint8_t tmp_rsc_flags;
    ucp_tl_alias_t *alias;
    int tl_enabled;
    char info[32];
    unsigned alias_arr_count;

    ucs_assert(count > 0);
    if (ucp_config_is_tl_enabled(names, count, tl_name, 0,
                                 rsc_flags, tl_cfg_mask)) {
        tl_enabled = 1;
    } else {
        tl_enabled = 0;

        /* check aliases */
        for (alias = ucp_tl_aliases; alias->alias != NULL; ++alias) {
            /* If an alias is enabled, and the transport is part of this alias,
             * enable the transport.
             */
            alias_arr_count = ucp_tl_alias_count(alias);
            snprintf(info, sizeof(info), "for alias '%s'", alias->alias);
            dummy_mask      = 0;
            tmp_rsc_flags   = 0;
            tmp_tl_cfg_mask = 0;
            if (ucp_config_is_tl_enabled(names, count, alias->alias, 1,
                                         &tmp_rsc_flags, &tmp_tl_cfg_mask) &&
                ucp_tls_array_is_present(alias->tls, alias_arr_count, tl_name,
                                         info, &tmp_rsc_flags, &dummy_mask)) {
                *rsc_flags   |= tmp_rsc_flags;
                *tl_cfg_mask |= tmp_tl_cfg_mask;
                tl_enabled  = 1;
                break;
            }
        }
    }

    return tl_enabled;
}

static int ucp_is_resource_enabled(const uct_tl_resource_desc_t *resource,
                                   const ucp_config_t *config, uint8_t *rsc_flags,
                                   uint64_t dev_cfg_masks[], uint64_t *tl_cfg_mask)
{
    int device_enabled, tl_enabled;

    /* Find the enabled devices */
    device_enabled = (*rsc_flags & UCP_TL_RSC_FLAG_SOCKADDR) ||
                     ucp_is_resource_in_device_list(resource, config->devices,
                                                    &dev_cfg_masks[resource->dev_type],
                                                    resource->dev_type);


    /* Find the enabled UCTs */
    tl_enabled = ucp_is_resource_in_transports_list(resource->tl_name,
                                                    (const char**)config->tls.names,
                                                    config->tls.count, rsc_flags,
                                                    tl_cfg_mask);

    ucs_trace(UCT_TL_RESOURCE_DESC_FMT " is %sabled",
              UCT_TL_RESOURCE_DESC_ARG(resource),
              (device_enabled && tl_enabled) ? "en" : "dis");
    return device_enabled && tl_enabled;
}

static void ucp_add_tl_resource_if_enabled(ucp_context_h context, ucp_tl_md_t *md,
                                           ucp_md_index_t md_index,
                                           const ucp_config_t *config,
                                           const uct_tl_resource_desc_t *resource,
                                           uint8_t rsc_flags, unsigned *num_resources_p,
                                           uint64_t dev_cfg_masks[],
                                           uint64_t *tl_cfg_mask)
{
    ucp_rsc_index_t dev_index, i;

    if (ucp_is_resource_enabled(resource, config, &rsc_flags, dev_cfg_masks,
                                tl_cfg_mask)) {
        context->tl_rscs[context->num_tls].tl_rsc       = *resource;
        context->tl_rscs[context->num_tls].md_index     = md_index;
        context->tl_rscs[context->num_tls].tl_name_csum =
                                  ucs_crc16_string(resource->tl_name);
        context->tl_rscs[context->num_tls].flags        = rsc_flags;

        dev_index = 0;
        for (i = 0; i < context->num_tls; ++i) {
            if (!strcmp(context->tl_rscs[i].tl_rsc.dev_name, resource->dev_name)) {
                dev_index = context->tl_rscs[i].dev_index;
                break;
            } else {
                dev_index = ucs_max(context->tl_rscs[i].dev_index + 1, dev_index);
            }
        }
        context->tl_rscs[context->num_tls].dev_index = dev_index;

        ++context->num_tls;
        ++(*num_resources_p);
    }
}

static ucs_status_t ucp_add_tl_resources(ucp_context_h context,
                                         ucp_md_index_t md_index,
                                         const ucp_config_t *config,
                                         unsigned *num_resources_p,
                                         ucs_string_set_t avail_devices[],
                                         ucs_string_set_t *avail_tls,
                                         uint64_t dev_cfg_masks[],
                                         uint64_t *tl_cfg_mask)
{
    ucp_tl_md_t *md = &context->tl_mds[md_index];
    uct_tl_resource_desc_t *tl_resources;
    uct_tl_resource_desc_t sa_rsc;
    ucp_tl_resource_desc_t *tmp;
    unsigned num_tl_resources;
    unsigned num_sa_resources;
    ucs_status_t status;
    ucp_rsc_index_t i;

    *num_resources_p = 0;

    /* check what are the available uct resources */
    status = uct_md_query_tl_resources(md->md, &tl_resources, &num_tl_resources);
    if (status != UCS_OK) {
        ucs_error("Failed to query resources: %s", ucs_status_string(status));
        goto err;
    }

    /* If the md supports client-server connection establishment via sockaddr,
       add a new tl resource here for the client side iface. */
    num_sa_resources = !!(md->attr.cap.flags & UCT_MD_FLAG_SOCKADDR);

    if ((num_tl_resources == 0) && (!num_sa_resources)) {
        ucs_debug("No tl resources found for md %s", md->rsc.md_name);
        goto out_free_resources;
    }

    tmp = ucs_realloc(context->tl_rscs,
                      sizeof(*context->tl_rscs) *
                      (context->num_tls + num_tl_resources + num_sa_resources),
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
        if (!(md->attr.cap.flags & UCT_MD_FLAG_SOCKADDR)) {
            ucs_string_set_addf(&avail_devices[tl_resources[i].dev_type],
                                "'%s'(%s)", tl_resources[i].dev_name,
                                context->tl_cmpts[md->cmpt_index].attr.name);
            ucs_string_set_add(avail_tls, tl_resources[i].tl_name);
        }
        ucp_add_tl_resource_if_enabled(context, md, md_index, config,
                                       &tl_resources[i], 0, num_resources_p,
                                       dev_cfg_masks, tl_cfg_mask);
    }

    /* add sockaddr dummy resource, if md supports it */
    if (md->attr.cap.flags & UCT_MD_FLAG_SOCKADDR) {
        sa_rsc.dev_type   = UCT_DEVICE_TYPE_NET;
        sa_rsc.sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
        ucs_snprintf_zero(sa_rsc.tl_name, UCT_TL_NAME_MAX, "%s", md->rsc.md_name);
        ucs_snprintf_zero(sa_rsc.dev_name, UCT_DEVICE_NAME_MAX, "sockaddr");
        ucp_add_tl_resource_if_enabled(context, md, md_index, config, &sa_rsc,
                                       UCP_TL_RSC_FLAG_SOCKADDR, num_resources_p,
                                       dev_cfg_masks, tl_cfg_mask);
    }

out_free_resources:
    uct_release_tl_resource_list(tl_resources);
    return UCS_OK;

err_free_resources:
    uct_release_tl_resource_list(tl_resources);
err:
    return status;
}

static void ucp_get_aliases_set(ucs_string_set_t *avail_tls)
{
    ucp_tl_alias_t *alias;
    const char **tl_name;

    for (alias = ucp_tl_aliases; alias->alias != NULL; ++alias) {
        for (tl_name = alias->tls; *tl_name != NULL; ++tl_name) {
            if (ucs_string_set_contains(avail_tls, *tl_name)) {
                ucs_string_set_add(avail_tls, alias->alias);
                break;
            }
        }
    }
}

static void ucp_report_unavailable(const ucs_config_names_array_t* cfg,
                                   uint64_t mask, const char *title1,
                                   const char *title2,
                                   const ucs_string_set_t *avail_names)
{
    UCS_STRING_BUFFER_ONSTACK(avail_strb,   256);
    UCS_STRING_BUFFER_ONSTACK(unavail_strb, 256);
    unsigned i;
    int found;

    found = 0;
    for (i = 0; i < cfg->count; i++) {
        if (!(mask & UCS_BIT(i)) && strcmp(cfg->names[i], UCP_RSC_CONFIG_ALL) &&
            !ucs_string_set_contains(avail_names, cfg->names[i])) {
            ucs_string_buffer_appendf(&unavail_strb, "%s'%s'",
                                      found++ ? "," : "",
                                      cfg->names[i]);
        }
    }

    if (found) {
        ucs_string_set_print_sorted(avail_names, &avail_strb, ", ");
        ucs_warn("%s%s%s %s %s not available, please use one or more of: %s",
                 title1, title2,
                 (found > 1) ? "s" : "",
                 ucs_string_buffer_cstr(&unavail_strb),
                 (found > 1) ? "are" : "is",
                 ucs_string_buffer_cstr(&avail_strb));
    }
}

const char * ucp_find_tl_name_by_csum(ucp_context_t *context, uint16_t tl_name_csum)
{
    ucp_tl_resource_desc_t *rsc;

    for (rsc = context->tl_rscs; rsc < context->tl_rscs + context->num_tls; ++rsc) {
        if (!(rsc->flags & UCP_TL_RSC_FLAG_SOCKADDR) && (rsc->tl_name_csum == tl_name_csum)) {
            return rsc->tl_rsc.tl_name;
        }
    }
    return NULL;
}

static ucs_status_t ucp_check_tl_names(ucp_context_t *context)
{
    ucp_tl_resource_desc_t *rsc;
    const char *tl_name;

    /* Make sure there we don't have two different transports with same checksum. */
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

const char* ucp_tl_bitmap_str(ucp_context_h context, uint64_t tl_bitmap,
                              char *str, size_t max_str_len)
{
    ucp_rsc_index_t i;
    char *p, *endp;

    p    = str;
    endp = str + max_str_len;

    ucs_for_each_bit(i, tl_bitmap) {
        ucs_snprintf_zero(p, endp - p, "%s ",
                          context->tl_rscs[i].tl_rsc.tl_name);
        p += strlen(p);
    }

    return str;
}


static void ucp_free_resources(ucp_context_t *context)
{
    ucp_rsc_index_t i;

    if (context->memtype_cache != NULL) {
        ucs_memtype_cache_destroy(context->memtype_cache);
    }

    ucs_free(context->tl_rscs);
    for (i = 0; i < context->num_mds; ++i) {
        uct_md_close(context->tl_mds[i].md);
    }
    ucs_free(context->tl_mds);
    ucs_free(context->tl_cmpts);
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

static ucs_status_t ucp_fill_tl_md(ucp_context_h context,
                                   ucp_rsc_index_t cmpt_index,
                                   const uct_md_resource_desc_t *md_rsc,
                                   ucp_tl_md_t *tl_md)
{
    uct_md_config_t *md_config;
    ucs_status_t status;

    /* Initialize tl_md structure */
    tl_md->cmpt_index = cmpt_index;
    tl_md->rsc        = *md_rsc;

    /* Read MD configuration */
    status = uct_md_config_read(context->tl_cmpts[cmpt_index].cmpt, NULL, NULL,
                                &md_config);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_md_open(context->tl_cmpts[cmpt_index].cmpt, md_rsc->md_name,
                         md_config, &tl_md->md);
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

    if (ucp_str_array_search((const char**)array->names, array->count,
                             UCP_RSC_CONFIG_ALL, NULL)) {
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

static void ucp_fill_sockaddr_aux_tls_config(ucp_context_h context,
                                             const ucp_config_t *config)
{
    const char **tl_names = (const char**)config->sockaddr_aux_tls.aux_tls;
    unsigned count        = config->sockaddr_aux_tls.count;
    uint8_t dummy_flags   = 0;
    uint64_t dummy_mask   = 0;
    ucp_rsc_index_t tl_id;

    context->config.sockaddr_aux_rscs_bitmap = 0;

    /* Check if any of the context's resources are present in the sockaddr
     * auxiliary transports for the client-server flow */
    ucs_for_each_bit(tl_id, context->tl_bitmap) {
        if (ucp_is_resource_in_transports_list(context->tl_rscs[tl_id].tl_rsc.tl_name,
                                               tl_names, count, &dummy_flags,
                                               &dummy_mask)) {
            context->config.sockaddr_aux_rscs_bitmap |= UCS_BIT(tl_id);
        }
    }
}

static void ucp_fill_sockaddr_tls_prio_list(ucp_context_h context,
                                            const char **sockaddr_tl_names,
                                            ucp_rsc_index_t num_sockaddr_tls)
{
    uint64_t sa_tls_bitmap = 0;
    ucp_rsc_index_t idx    = 0;
    ucp_tl_resource_desc_t *resource;
    ucp_rsc_index_t tl_id;
    ucp_tl_md_t *tl_md;
    ucp_rsc_index_t j;

    /* Set a bitmap of sockaddr transports */
    for (j = 0; j < context->num_tls; ++j) {
        resource = &context->tl_rscs[j];
        tl_md    = &context->tl_mds[resource->md_index];
        if (tl_md->attr.cap.flags & UCT_MD_FLAG_SOCKADDR) {
            sa_tls_bitmap |= UCS_BIT(j);
        }
    }

    /* Parse the sockaddr transports priority list */
    for (j = 0; j < num_sockaddr_tls; j++) {
        /* go over the priority list and find the transport's tl_id in the
         * sockaddr tls bitmap. save the tl_id's for the client/server usage
         * later */
        ucs_for_each_bit(tl_id, sa_tls_bitmap) {
            resource = &context->tl_rscs[tl_id];

            if (!strcmp(sockaddr_tl_names[j], "*") ||
                !strncmp(sockaddr_tl_names[j], resource->tl_rsc.tl_name,
                         UCT_TL_NAME_MAX)) {
                context->config.sockaddr_tl_ids[idx] = tl_id;
                idx++;
                sa_tls_bitmap &= ~UCS_BIT(tl_id);
            }
        }
    }

    context->config.num_sockaddr_tls = idx;
}

static void ucp_fill_sockaddr_cms_prio_list(ucp_context_h context,
                                            const char **sockaddr_cm_names,
                                            ucp_rsc_index_t num_sockaddr_cms,
                                            int sockaddr_cm_enable)
{
    uint64_t cm_cmpts_bitmap = context->config.cm_cmpts_bitmap;
    uint64_t cm_cmpts_bitmap_safe;
    ucp_rsc_index_t cmpt_idx, cm_idx;

    memset(&context->config.cm_cmpt_idxs, UCP_NULL_RESOURCE, UCP_MAX_RESOURCES);
    context->config.num_cm_cmpts = 0;

    if (!sockaddr_cm_enable) {
        return;
    }

    /* Parse the sockaddr CMs priority list */
    for (cm_idx = 0; cm_idx < num_sockaddr_cms; ++cm_idx) {
        /* go over the priority list and find the CM's cm_idx in the
         * sockaddr CMs bitmap. Save the cmpt_idx for the client/server usage
         * later */
        cm_cmpts_bitmap_safe = cm_cmpts_bitmap;
        ucs_for_each_bit(cmpt_idx, cm_cmpts_bitmap_safe) {
            if (!strcmp(sockaddr_cm_names[cm_idx], "*") ||
                !strncmp(sockaddr_cm_names[cm_idx],
                         context->tl_cmpts[cmpt_idx].attr.name,
                         UCT_COMPONENT_NAME_MAX)) {
                context->config.cm_cmpt_idxs[context->config.num_cm_cmpts++] = cmpt_idx;
                cm_cmpts_bitmap &= ~UCS_BIT(cmpt_idx);
            }
        }
    }
}

static ucs_status_t ucp_fill_sockaddr_prio_list(ucp_context_h context,
                                                const ucp_config_t *config)
{
    const char **sockaddr_tl_names = (const char**)config->sockaddr_cm_tls.cm_tls;
    unsigned num_sockaddr_tls      = config->sockaddr_cm_tls.count;
    int sockaddr_cm_enable         = context->config.ext.sockaddr_cm_enable !=
                                     UCS_NO;

    /* Check if a list of sockaddr transports/CMs has valid length */
    if (num_sockaddr_tls > UCP_MAX_RESOURCES) {
        ucs_warn("sockaddr transports or connection managers list is too long, "
                 "only first %d entries will be used", UCP_MAX_RESOURCES);
        num_sockaddr_tls = UCP_MAX_RESOURCES;
    }

    ucp_fill_sockaddr_tls_prio_list(context, sockaddr_tl_names,
                                    num_sockaddr_tls);
    ucp_fill_sockaddr_cms_prio_list(context, sockaddr_tl_names,
                                    num_sockaddr_tls, sockaddr_cm_enable);
    if ((context->config.ext.sockaddr_cm_enable == UCS_YES) &&
        (context->config.num_cm_cmpts == 0)) {
        ucs_error("UCX_SOCKADDR_CM_ENABLE is set to yes but none of the available components supports SOCKADDR_CM");
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static ucs_status_t ucp_check_resources(ucp_context_h context,
                                        const ucp_config_t *config)
{
    char info_str[128];
    ucp_rsc_index_t tl_id;
    ucp_tl_resource_desc_t *resource;
    unsigned num_usable_tls;

    /* Error check: Make sure there is at least one transport that is not
     * sockaddr or auxiliary */
    num_usable_tls = 0;
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        ucs_assert(context->tl_rscs != NULL);
        resource = &context->tl_rscs[tl_id];
        if (!(resource->flags & (UCP_TL_RSC_FLAG_AUX|UCP_TL_RSC_FLAG_SOCKADDR))) {
            num_usable_tls++;
        }
    }

    if (num_usable_tls == 0) {
        ucp_resource_config_str(config, info_str, sizeof(info_str));
        ucs_error("no usable transports/devices (asked %s)", info_str);
        return UCS_ERR_NO_DEVICE;
    }

    /* Error check: Make sure there are not too many transports */
    if (context->num_tls >= UCP_MAX_RESOURCES) {
        ucs_error("exceeded transports/devices limit "
                  "(%u requested, up to %d are supported)",
                  context->num_tls, UCP_MAX_RESOURCES);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    return ucp_check_tl_names(context);
}

static ucs_status_t ucp_add_component_resources(ucp_context_h context,
                                                ucp_rsc_index_t cmpt_index,
                                                ucs_string_set_t avail_devices[],
                                                ucs_string_set_t *avail_tls,
                                                uint64_t dev_cfg_masks[],
                                                uint64_t *tl_cfg_mask,
                                                const ucp_config_t *config)
{
    const ucp_tl_cmpt_t *tl_cmpt = &context->tl_cmpts[cmpt_index];
    uct_component_attr_t uct_component_attr;
    unsigned num_tl_resources;
    ucs_status_t status;
    ucp_rsc_index_t i;
    unsigned md_index;
    uint64_t mem_type_mask;
    uint64_t mem_type_bitmap;


    /* List memory domain resources */
    uct_component_attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    uct_component_attr.md_resources =
                    ucs_alloca(tl_cmpt->attr.md_resource_count *
                               sizeof(*uct_component_attr.md_resources));
    status = uct_component_query(tl_cmpt->cmpt, &uct_component_attr);
    if (status != UCS_OK) {
        goto out;
    }

    /* Open all memory domains */
    mem_type_mask = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    for (i = 0; i < tl_cmpt->attr.md_resource_count; ++i) {
        md_index = context->num_mds;
        status = ucp_fill_tl_md(context, cmpt_index,
                                &uct_component_attr.md_resources[i],
                                &context->tl_mds[md_index]);
        if (status != UCS_OK) {
            continue;
        }

        /* Add communication resources of each MD */
        status = ucp_add_tl_resources(context, md_index, config,
                                      &num_tl_resources, avail_devices,
                                      avail_tls, dev_cfg_masks, tl_cfg_mask);
        if (status != UCS_OK) {
            uct_md_close(context->tl_mds[md_index].md);
            goto out;
        }

        /* If the MD does not have transport resources (device or sockaddr),
         * don't use it */
        if (num_tl_resources > 0) {
            /* List of memory type MDs */
            mem_type_bitmap = context->tl_mds[md_index].attr.cap.detect_mem_types;
            if (~mem_type_mask & mem_type_bitmap) {
                context->mem_type_detect_mds[context->num_mem_type_detect_mds] = md_index;
                ++context->num_mem_type_detect_mds;
                mem_type_mask |= mem_type_bitmap;
            }
            ++context->num_mds;
        } else {
            ucs_debug("closing md %s because it has no selected transport resources",
                      context->tl_mds[md_index].rsc.md_name);
            uct_md_close(context->tl_mds[md_index].md);
        }
    }

    context->mem_type_mask |= mem_type_mask;

    status = UCS_OK;
out:
    return status;
}

static ucs_status_t ucp_fill_resources(ucp_context_h context,
                                       const ucp_config_t *config)
{
    uint64_t dev_cfg_masks[UCT_DEVICE_TYPE_LAST] = {};
    uint64_t tl_cfg_mask                         = 0;
    ucs_string_set_t avail_devices[UCT_DEVICE_TYPE_LAST];
    ucs_string_set_t avail_tls;
    uct_component_h *uct_components;
    unsigned i, num_uct_components;
    uct_device_type_t dev_type;
    ucs_status_t status;
    unsigned max_mds;

    context->tl_cmpts         = NULL;
    context->num_cmpts        = 0;
    context->tl_mds           = NULL;
    context->num_mds          = 0;
    context->tl_rscs          = NULL;
    context->num_tls          = 0;
    context->memtype_cache    = NULL;
    context->mem_type_mask    = 0;
    context->num_mem_type_detect_mds = 0;

    for (i = 0; i < UCS_MEMORY_TYPE_LAST; ++i) {
        context->mem_type_access_tls[i] = 0;
    }

    ucs_string_set_init(&avail_tls);
    UCS_STATIC_ASSERT(UCT_DEVICE_TYPE_NET == 0);
    for (dev_type = UCT_DEVICE_TYPE_NET; dev_type < UCT_DEVICE_TYPE_LAST; ++dev_type) {
        ucs_string_set_init(&avail_devices[dev_type]);
    }

    status = ucp_check_resource_config(config);
    if (status != UCS_OK) {
        goto out_cleanup_avail_devices;
    }

    status = uct_query_components(&uct_components, &num_uct_components);
    if (status != UCS_OK) {
        goto out_cleanup_avail_devices;
    }

    if (num_uct_components > UCP_MAX_RESOURCES) {
        ucs_error("too many components: %u, max: %u", num_uct_components,
                  UCP_MAX_RESOURCES);
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto out_release_components;
    }

    context->num_cmpts = num_uct_components;
    context->tl_cmpts  = ucs_calloc(context->num_cmpts,
                                    sizeof(*context->tl_cmpts), "ucp_tl_cmpts");
    if (context->tl_cmpts == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_release_components;
    }

    context->config.cm_cmpts_bitmap = 0;

    max_mds = 0;
    for (i = 0; i < context->num_cmpts; ++i) {
        memset(&context->tl_cmpts[i].attr, 0, sizeof(context->tl_cmpts[i].attr));
        context->tl_cmpts[i].cmpt = uct_components[i];
        context->tl_cmpts[i].attr.field_mask =
                        UCT_COMPONENT_ATTR_FIELD_NAME              |
                        UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT |
                        UCT_COMPONENT_ATTR_FIELD_FLAGS;
        status = uct_component_query(context->tl_cmpts[i].cmpt,
                                     &context->tl_cmpts[i].attr);
        if (status != UCS_OK) {
            goto err_free_resources;
        }

        if (context->tl_cmpts[i].attr.flags & UCT_COMPONENT_FLAG_CM) {
            context->config.cm_cmpts_bitmap |= UCS_BIT(i);
        }

        max_mds += context->tl_cmpts[i].attr.md_resource_count;
    }

    if ((context->config.ext.sockaddr_cm_enable == UCS_YES) &&
        (context->config.cm_cmpts_bitmap == 0)) {
        ucs_error("there are no UCT components with CM capability");
        status = UCS_ERR_UNSUPPORTED;
        goto err_free_resources;
    }

    /* Allocate actual array of MDs */
    context->tl_mds = ucs_malloc(max_mds * sizeof(*context->tl_mds),
                                 "ucp_tl_mds");
    if (context->tl_mds == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_resources;
    }

    /* Collect resources of each component */
    for (i = 0; i < context->num_cmpts; ++i) {
        status = ucp_add_component_resources(context, i, avail_devices,
                                             &avail_tls, dev_cfg_masks,
                                             &tl_cfg_mask, config);
        if (status != UCS_OK) {
            goto err_free_resources;
        }
    }

    /* Create memtype cache if we have memory type MDs, and it's enabled by
     * configuration
     */
    if (context->num_mem_type_detect_mds && context->config.ext.enable_memtype_cache) {
        status = ucs_memtype_cache_create(&context->memtype_cache);
        if (status != UCS_OK) {
            ucs_debug("could not create memtype cache for mem_type allocations");
            goto err_free_resources;
        }
    }

    /* If unified mode is enabled, initialize tl_bitmap to 0.
     * Then the worker will open all available transport resources and will
     * select only the best ones for each particular device.
     */
    context->tl_bitmap = config->ctx.unified_mode ? 0 : UCS_MASK(context->num_tls);

    /* Warn about devices and transports which were specified explicitly in the
     * configuration, but are not available
     */
    if (config->warn_invalid_config) {
        UCS_STATIC_ASSERT(UCT_DEVICE_TYPE_NET == 0);
        for (dev_type = UCT_DEVICE_TYPE_NET; dev_type < UCT_DEVICE_TYPE_LAST; ++dev_type) {
            ucp_report_unavailable(&config->devices[dev_type],
                                   dev_cfg_masks[dev_type],
                                   ucp_device_type_names[dev_type], " device",
                                   &avail_devices[dev_type]);
        }

        ucp_get_aliases_set(&avail_tls);
        ucp_report_unavailable(&config->tls, tl_cfg_mask, "", "transport",
                               &avail_tls);
    }

    /* Validate context resources */
    status = ucp_check_resources(context, config);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    ucp_fill_sockaddr_aux_tls_config(context, config);
    status = ucp_fill_sockaddr_prio_list(context, config);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    goto out_release_components;

err_free_resources:
    ucp_free_resources(context);
out_release_components:
    uct_release_component_list(uct_components);
out_cleanup_avail_devices:
    UCS_STATIC_ASSERT(UCT_DEVICE_TYPE_NET == 0);
    for (dev_type = UCT_DEVICE_TYPE_NET; dev_type < UCT_DEVICE_TYPE_LAST; ++dev_type) {
        ucs_string_set_cleanup(&avail_devices[dev_type]);
    }
    ucs_string_set_cleanup(&avail_tls);
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

    if (params->field_mask & UCP_PARAM_FIELD_ESTIMATED_NUM_PPN) {
        context->config.est_num_ppn = params->estimated_num_ppn;
    } else {
        context->config.est_num_ppn = 1;
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

    if (context->config.ext.estimated_num_eps != UCS_ULUNITS_AUTO) {
        /* num_eps was set via the env variable. Override current value */
        context->config.est_num_eps = context->config.ext.estimated_num_eps;
    }
    ucs_debug("estimated number of endpoints is %d",
              context->config.est_num_eps);

    if (context->config.ext.estimated_num_ppn != UCS_ULUNITS_AUTO) {
        /* num_ppn was set via the env variable. Override current value */
        context->config.est_num_ppn = context->config.ext.estimated_num_ppn;
    }
    ucs_debug("estimated number of endpoints per node is %d",
              context->config.est_num_ppn);

    if (UCS_CONFIG_BW_IS_AUTO(context->config.ext.bcopy_bw)) {
        /* bcopy_bw wasn't set via the env variable. Calculate the value */
        context->config.ext.bcopy_bw = ucs_cpu_get_memcpy_bw();
    }
    ucs_debug("estimated bcopy bandwidth is %f",
              context->config.ext.bcopy_bw);

    /* always init MT lock in context even though it is disabled by user,
     * because we need to use context lock to protect ucp_mm_ and ucp_rkey_
     * routines */
    UCP_THREAD_LOCK_INIT(&context->mt_lock);

    /* save comparison MD for iface_attr adjustment */
    context->config.selection_cmp = ucs_strdup(config->selection_cmp,
                                               "selection cmp");
    if (context->config.selection_cmp == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* save environment prefix to later notify user for unused variables */
    context->config.env_prefix = ucs_strdup(config->env_prefix, "ucp config");
    if (context->config.env_prefix == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_selection_cmp;
    }

    /* Get allocation alignment from configuration, make sure it's valid */
    if (config->alloc_prio.count == 0) {
        ucs_error("No allocation methods specified - aborting");
        status = UCS_ERR_INVALID_PARAM;
        goto err_free_env_prefix;
    }

    num_alloc_methods = config->alloc_prio.count;
    context->config.num_alloc_methods = num_alloc_methods;

    /* Allocate an array to hold the allocation methods configuration */
    context->config.alloc_methods = ucs_calloc(num_alloc_methods,
                                               sizeof(*context->config.alloc_methods),
                                               "ucp_alloc_methods");
    if (context->config.alloc_methods == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_env_prefix;
    }

    /* Parse the allocation methods specified in the configuration */
    for (i = 0; i < num_alloc_methods; ++i) {
        method_name = config->alloc_prio.methods[i];
        if (!strncasecmp(method_name, "md:", 3)) {
            /* If the method name begins with 'md:', treat it as memory domain
             * component name.
             */
            context->config.alloc_methods[i].method = UCT_ALLOC_METHOD_MD;
            ucs_strncpy_zero(context->config.alloc_methods[i].cmpt_name,
                             method_name + 3, UCT_COMPONENT_NAME_MAX);
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
                    context->config.alloc_methods[i].method = (uct_alloc_method_t)method;
                    strcpy(context->config.alloc_methods[i].cmpt_name, "");
                    ucs_debug("allocation method[%d] is '%s'", i, method_name);
                    break;
                }
            }
            if (context->config.alloc_methods[i].method == UCT_ALLOC_METHOD_LAST) {
                ucs_error("Invalid allocation method: %s", method_name);
                status = UCS_ERR_INVALID_PARAM;
                goto err_free_alloc_methods;
            }
        }
    }

    /* Need to check TM_SEG_SIZE value if it is enabled only */
    if (context->config.ext.tm_max_bb_size > context->config.ext.tm_thresh) {
        if (context->config.ext.tm_max_bb_size < sizeof(ucp_request_hdr_t)) {
            /* In case of expected SW RNDV message, the header (ucp_request_hdr_t) is
             * scattered to UCP user buffer. Make sure that bounce buffer is used for
             * messages which can not fit SW RNDV hdr. */
            context->config.ext.tm_max_bb_size = sizeof(ucp_request_hdr_t);
            ucs_info("UCX_TM_MAX_BB_SIZE value: %zu, adjusted to: %zu",
                     context->config.ext.tm_max_bb_size, sizeof(ucp_request_hdr_t));
        }

        if (context->config.ext.tm_max_bb_size > context->config.ext.seg_size) {
            context->config.ext.tm_max_bb_size = context->config.ext.seg_size;
            ucs_info("Wrong UCX_TM_MAX_BB_SIZE value: %zu, adjusted to: %zu",
                     context->config.ext.tm_max_bb_size,
                     context->config.ext.seg_size);
        }
    }

    if (context->config.ext.keepalive_num_eps == 0) {
        ucs_error("UCX_KEEPALIVE_NUM_EPS value must be greater than 0");
        status = UCS_ERR_INVALID_PARAM;
        goto err_free_alloc_methods;
    }

    context->config.keepalive_interval = ucs_time_from_sec(context->config.ext.keepalive_interval);
    return UCS_OK;

err_free_alloc_methods:
    ucs_free(context->config.alloc_methods);
err_free_env_prefix:
    ucs_free(context->config.env_prefix);
err_free_selection_cmp:
    ucs_free(context->config.selection_cmp);
err:
    UCP_THREAD_LOCK_FINALIZE(&context->mt_lock);
    return status;
}

static void ucp_free_config(ucp_context_h context)
{
    ucs_free(context->config.alloc_methods);
    ucs_free(context->config.env_prefix);
    ucs_free(context->config.selection_cmp);
}

ucs_status_t ucp_init_version(unsigned api_major_version, unsigned api_minor_version,
                              const ucp_params_t *params, const ucp_config_t *config,
                              ucp_context_h *context_p)
{
    unsigned major_version, minor_version, release_number;
    ucp_config_t *dfl_config = NULL;
    ucp_context_t *context;
    ucs_status_t status;
    ucs_debug_address_info_t addr_info;

    ucp_get_version(&major_version, &minor_version, &release_number);

    if ((api_major_version != major_version) ||
        ((api_major_version == major_version) && (api_minor_version > minor_version))) {
        status = ucs_debug_lookup_address(ucp_init_version, &addr_info);
        ucs_warn("UCP version is incompatible, required: %d.%d, actual: %d.%d (release %d %s)",
                  api_major_version, api_minor_version,
                  major_version, minor_version, release_number,
                  status == UCS_OK ? addr_info.file.path : "");
    }

    if (config == NULL) {
        status = ucp_config_read(NULL, NULL, &dfl_config);
        if (status != UCS_OK) {
            goto err;
        }
        config = dfl_config;
    }

    /* allocate a ucp context */
    context = ucs_calloc(1, sizeof(*context), "ucp context");
    if (context == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_release_config;
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

    if (dfl_config != NULL) {
        ucp_config_release(dfl_config);
    }

    ucs_debug("created ucp context %p [%d mds %d tls] features 0x%"PRIx64
              " tl bitmap 0x%"PRIx64, context, context->num_mds,
              context->num_tls, context->config.features, context->tl_bitmap);

    *context_p = context;
    return UCS_OK;

err_free_config:
    ucp_free_config(context);
err_free_ctx:
    ucs_free(context);
err_release_config:
    if (dfl_config != NULL) {
        ucp_config_release(dfl_config);
    }
err:
    return status;
}

void ucp_cleanup(ucp_context_h context)
{
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

void ucp_context_uct_atomic_iface_flags(ucp_context_h context,
                                        ucp_tl_iface_atomic_flags_t *atomic)
{
    if (context->config.features & UCP_FEATURE_AMO32) {
        atomic->atomic32.op_flags  = UCP_ATOMIC_OP_MASK;
        atomic->atomic32.fop_flags = UCP_ATOMIC_FOP_MASK;
    } else {
        atomic->atomic32.op_flags  = 0;
        atomic->atomic32.fop_flags = 0;
    }

    if (context->config.features & UCP_FEATURE_AMO64) {
        atomic->atomic64.op_flags  = UCP_ATOMIC_OP_MASK;
        atomic->atomic64.fop_flags = UCP_ATOMIC_FOP_MASK;
    } else {
        atomic->atomic64.op_flags  = 0;
        atomic->atomic64.fop_flags = 0;
    }
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

    if (attr->field_mask & UCP_ATTR_FIELD_MEMORY_TYPES) {
        attr->memory_types = context->mem_type_mask;
    }

    return UCS_OK;
}

void ucp_context_print_info(ucp_context_h context, FILE *stream)
{
    ucp_rsc_index_t cmpt_index, md_index, rsc_index;

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP context\n");
    fprintf(stream, "#\n");

    for (cmpt_index = 0; cmpt_index < context->num_cmpts; ++cmpt_index) {
        fprintf(stream, "#     component %-2d :  %s\n",
                cmpt_index, context->tl_cmpts[cmpt_index].attr.name);
    }
    fprintf(stream, "#\n");

    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        fprintf(stream, "#            md %-2d :  component %-2d %s \n",
                md_index, context->tl_mds[md_index].cmpt_index,
                context->tl_mds[md_index].rsc.md_name);
    }

    fprintf(stream, "#\n");

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        ucp_tl_resource_desc_t *rsc = &context->tl_rscs[rsc_index];
        fprintf(stream, "#      resource %-2d :  md %-2d dev %-2d flags %c%c "
                UCT_TL_RESOURCE_DESC_FMT"\n",
                rsc_index, rsc->md_index, rsc->dev_index,
                (rsc->flags & UCP_TL_RSC_FLAG_AUX)      ? 'a' : '-',
                (rsc->flags & UCP_TL_RSC_FLAG_SOCKADDR) ? 's' : '-',
                UCT_TL_RESOURCE_DESC_ARG(&rsc->tl_rsc));
    }

    fprintf(stream, "#\n");
}

uct_md_h ucp_context_find_tl_md(ucp_context_h context, const char *md_name)
{
    ucp_rsc_index_t rsc_index;

    for (rsc_index = 0; rsc_index < context->num_mds; ++rsc_index) {
        if (strstr(context->tl_mds[rsc_index].rsc.md_name, md_name)) {
            return context->tl_mds[rsc_index].md;
        }
    }

    return NULL;
}

ucs_memory_type_t
ucp_memory_type_detect_mds(ucp_context_h context, const void *address, size_t size)
{
    ucs_memory_type_t mem_type;
    unsigned i, md_index;
    ucs_status_t status;

    for (i = 0; i < context->num_mem_type_detect_mds; ++i) {
        md_index = context->mem_type_detect_mds[i];
        status   = uct_md_detect_memory_type(context->tl_mds[md_index].md,
                                             address, size, &mem_type);
        if (status == UCS_OK) {
            if (context->memtype_cache != NULL) {
                ucs_memtype_cache_update(context->memtype_cache, address, size,
                                         mem_type);
            }
            return mem_type;
        }
    }

    /* Memory type not detected by any memtype MD - assume it is host memory */
    return UCS_MEMORY_TYPE_HOST;
}

uint64_t ucp_context_dev_tl_bitmap(ucp_context_h context, const char *dev_name)
{
    uint64_t        tl_bitmap;
    ucp_rsc_index_t tl_idx;

    tl_bitmap = 0;

    ucs_for_each_bit(tl_idx, context->tl_bitmap) {
        if (strcmp(context->tl_rscs[tl_idx].tl_rsc.dev_name, dev_name)) {
            continue;
        }

        tl_bitmap |= UCS_BIT(tl_idx);
    }

    return tl_bitmap;
}

uint64_t ucp_context_dev_idx_tl_bitmap(ucp_context_h context,
                                       ucp_rsc_index_t dev_idx)
{
    uint64_t        tl_bitmap;
    ucp_rsc_index_t tl_idx;

    tl_bitmap = 0;

    ucs_for_each_bit(tl_idx, context->tl_bitmap) {
        if (context->tl_rscs[tl_idx].dev_index == dev_idx) {
            tl_bitmap |= UCS_BIT(tl_idx);
        }
    }

    return tl_bitmap;
}

const char* ucp_context_cm_name(ucp_context_h context, ucp_rsc_index_t cm_idx)
{
    ucs_assert(cm_idx != UCP_NULL_RESOURCE);
    return context->tl_cmpts[context->config.cm_cmpt_idxs[cm_idx]].attr.name;
}
