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
#include <ucs/debug/debug_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <string.h>
#include <dlfcn.h>


#define UCP_RSC_CONFIG_ALL    "all"

#define UCP_AM_HANDLER_FOREACH(_macro) \
    _macro(UCP_AM_ID_WIREUP) \
    _macro(UCP_AM_ID_EAGER_ONLY) \
    _macro(UCP_AM_ID_EAGER_FIRST) \
    _macro(UCP_AM_ID_EAGER_MIDDLE) \
    _macro(UCP_AM_ID_EAGER_SYNC_ONLY) \
    _macro(UCP_AM_ID_EAGER_SYNC_FIRST) \
    _macro(UCP_AM_ID_EAGER_SYNC_ACK) \
    _macro(UCP_AM_ID_RNDV_RTS) \
    _macro(UCP_AM_ID_RNDV_ATS) \
    _macro(UCP_AM_ID_RNDV_RTR) \
    _macro(UCP_AM_ID_RNDV_DATA) \
    _macro(UCP_AM_ID_OFFLOAD_SYNC_ACK) \
    _macro(UCP_AM_ID_STREAM_DATA) \
    _macro(UCP_AM_ID_RNDV_ATP) \
    _macro(UCP_AM_ID_PUT) \
    _macro(UCP_AM_ID_GET_REQ) \
    _macro(UCP_AM_ID_GET_REP) \
    _macro(UCP_AM_ID_ATOMIC_REQ) \
    _macro(UCP_AM_ID_ATOMIC_REP) \
    _macro(UCP_AM_ID_CMPL) \
    _macro(UCP_AM_ID_AM_SINGLE) \
    _macro(UCP_AM_ID_AM_FIRST) \
    _macro(UCP_AM_ID_AM_MIDDLE) \
    _macro(UCP_AM_ID_AM_SINGLE_REPLY)

#define UCP_AM_HANDLER_DECL(_id) extern ucp_am_handler_t ucp_am_handler_##_id;

#define UCP_AM_HANDLER_ENTRY(_id) [_id] = &ucp_am_handler_##_id,


/* Declare all am handlers */
UCP_AM_HANDLER_FOREACH(UCP_AM_HANDLER_DECL)

ucp_am_handler_t *ucp_am_handlers[UCP_AM_ID_LAST] = {
    UCP_AM_HANDLER_FOREACH(UCP_AM_HANDLER_ENTRY)
};

static const char *ucp_atomic_modes[] = {
    [UCP_ATOMIC_MODE_CPU]    = "cpu",
    [UCP_ATOMIC_MODE_DEVICE] = "device",
    [UCP_ATOMIC_MODE_GUESS]  = "guess",
    [UCP_ATOMIC_MODE_LAST]   = NULL,
};

static const char *ucp_rndv_modes[] = {
    [UCP_RNDV_MODE_AUTO]         = "auto",
    [UCP_RNDV_MODE_GET_ZCOPY]    = "get_zcopy",
    [UCP_RNDV_MODE_PUT_ZCOPY]    = "put_zcopy",
    [UCP_RNDV_MODE_GET_PIPELINE] = "get_ppln",
    [UCP_RNDV_MODE_PUT_PIPELINE] = "put_ppln",
    [UCP_RNDV_MODE_AM]           = "am",
    [UCP_RNDV_MODE_RKEY_PTR]     = "rkey_ptr",
    [UCP_RNDV_MODE_LAST]         = NULL,
};

static size_t ucp_rndv_frag_default_sizes[] = {
    [UCS_MEMORY_TYPE_HOST]         = 512 * UCS_KBYTE,
    [UCS_MEMORY_TYPE_CUDA]         = 4 * UCS_MBYTE,
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = 4 * UCS_MBYTE,
    [UCS_MEMORY_TYPE_ROCM]         = 4 * UCS_MBYTE,
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = 4 * UCS_MBYTE,
    [UCS_MEMORY_TYPE_LAST]         = 0
};

static size_t ucp_rndv_frag_default_num_elems[] = {
    [UCS_MEMORY_TYPE_HOST]         = 128,
    [UCS_MEMORY_TYPE_CUDA]         = 128,
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = 128,
    [UCS_MEMORY_TYPE_ROCM]         = 128,
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = 128,
    [UCS_MEMORY_TYPE_LAST]         = 0
};

const char *ucp_object_versions[] = {
    [UCP_OBJECT_VERSION_V1]   = "v1",
    [UCP_OBJECT_VERSION_V2]   = "v2",
    [UCP_OBJECT_VERSION_LAST] = NULL
};

static UCS_CONFIG_DEFINE_ARRAY(memunit_sizes, sizeof(size_t),
                               UCS_CONFIG_TYPE_MEMUNITS);

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
   " and disables aliasing.",
   ucs_offsetof(ucp_config_t, tls), UCS_CONFIG_TYPE_ALLOW_LIST},

  {"PROTOS", UCP_RSC_CONFIG_ALL,
   "Comma-separated list of glob patterns specifying protocols to use.\n"
   "The order is not meaningful.\n"
   "Each expression in the list may contain any of the following wildcard:\n"
   "  *     - matches any number of any characters including none.\n"
   "  ?     - matches any single character.\n"
   "  [abc] - matches one character given in the bracket.\n"
   "  [a-z] - matches one character from the range given in the bracket.",
   ucs_offsetof(ucp_config_t, protos), UCS_CONFIG_TYPE_ALLOW_LIST},

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

  {"SOCKADDR_AUX_TLS", "",
   "The configuration parameter is deprecated. UCX_TLS should be used to\n"
   "specify the transport for client/server connection establishment.",
   UCS_CONFIG_DEPRECATED_FIELD_OFFSET, UCS_CONFIG_TYPE_DEPRECATED},

  {"SELECT_DISTANCE_MD", "cuda_cpy",
   "MD whose distance is queried when evaluating transport selection score",
   ucs_offsetof(ucp_config_t, selection_cmp), UCS_CONFIG_TYPE_STRING},

  {"MEMTYPE_REG_WHOLE_ALLOC_TYPES", "cuda",
   "Memory types which have whole allocations registered.\n"
   "Allowed memory types: cuda, rocm, rocm-managed",
   ucs_offsetof(ucp_config_t, ctx.reg_whole_alloc_bitmap),
   UCS_CONFIG_TYPE_BITMAP(ucs_memory_type_names)},

  {"RNDV_MEMTYPE_DIRECT_SIZE", "inf",
   "Maximum size for mem type direct in rendezvous protocol\n",
   ucs_offsetof(ucp_config_t, ctx.rndv_memtype_direct_size),
   UCS_CONFIG_TYPE_MEMUNITS},

  {"WARN_INVALID_CONFIG", "y",
   "Issue a warning in case of invalid device and/or transport configuration.",
   ucs_offsetof(ucp_config_t, warn_invalid_config), UCS_CONFIG_TYPE_BOOL},

  {"BCOPY_THRESH", "auto",
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
   "Message size to start using the rendezvous protocol in case the calculated threshold\n"
   "is zero or negative",
   ucs_offsetof(ucp_config_t, ctx.rndv_thresh_fallback), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_PERF_DIFF", "1",
   "The percentage allowed for performance difference between rendezvous and "
   "the eager_zcopy protocol",
   ucs_offsetof(ucp_config_t, ctx.rndv_perf_diff), UCS_CONFIG_TYPE_DOUBLE},

  {"MULTI_LANE_MAX_RATIO", "4",
   "Maximal allowed ratio between slowest and fastest lane in a multi-lane\n"
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
   " rkey_ptr  - use rket_ptr in RNDV protocol.\n"
   " auto      - runtime automatically chooses optimal scheme to use.",
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

  {"MAX_WORKER_NAME", NULL, "",
   ucs_offsetof(ucp_config_t, ctx.max_worker_address_name),
   UCS_CONFIG_TYPE_UINT},

  {"MAX_WORKER_ADDRESS_NAME", UCS_PP_MAKE_STRING(UCP_WORKER_ADDRESS_NAME_MAX),
   "Maximal length of worker address name. Sent to remote peer as part of\n"
   "worker address if UCX_ADDRESS_DEBUG_INFO is set to 'yes'",
   ucs_offsetof(ucp_config_t, ctx.max_worker_address_name),
   UCS_CONFIG_TYPE_UINT},

  {"USE_MT_MUTEX", "n", "Use mutex for multithreading support in UCP.\n"
   "n      - Not use mutex for multithreading support in UCP (use spinlock by default).\n"
   "y      - Use mutex for multithreading support in UCP.",
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
   "and the resulting performance.",
   ucs_offsetof(ucp_config_t, ctx.estimated_num_ppn), UCS_CONFIG_TYPE_ULUNITS},

  {"RNDV_FRAG_SIZE", "host:512K,cuda:4M",
   "Comma-separated list of memory types and associated fragment sizes.\n"
   "The memory types in the list is used for rendezvous bounce buffers.",
   ucs_offsetof(ucp_config_t, rndv_frag_sizes), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"RNDV_FRAG_ALLOC_COUNT", "host:128,cuda:128",
   "Comma separated list of memory pool allocation granularity per memory type.",
   ucs_offsetof(ucp_config_t, rndv_frag_elems), UCS_CONFIG_TYPE_STRING_ARRAY},

  {"RNDV_FRAG_MEM_TYPE", "host",
   "Memory type of fragments used for RNDV pipeline protocol.\n"
   "Allowed memory types is one of: host, cuda, rocm",
   ucs_offsetof(ucp_config_t, ctx.rndv_frag_mem_type),
   UCS_CONFIG_TYPE_ENUM(ucs_memory_type_names)},

  {"RNDV_PIPELINE_SEND_THRESH", "inf",
   "RNDV size threshold to enable sender side pipeline for mem type",
   ucs_offsetof(ucp_config_t, ctx.rndv_pipeline_send_thresh), UCS_CONFIG_TYPE_MEMUNITS},

  {"RNDV_PIPELINE_SHM_ENABLE", "y",
   "Use two stage pipeline rendezvous protocol for intra-node GPU to GPU transfers",
   ucs_offsetof(ucp_config_t, ctx.rndv_shm_ppln_enable), UCS_CONFIG_TYPE_BOOL},

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

  {"CM_USE_ALL_DEVICES", "y",
   "When creating client/server endpoints, use all available devices.\n"
   "If disabled, use only the one device on which the connection\n"
   "establishment is done",
   ucs_offsetof(ucp_config_t, ctx.cm_use_all_devices), UCS_CONFIG_TYPE_BOOL},

  {"LISTENER_BACKLOG", "auto",
   "'auto' means that each transport would use its maximal allowed value.\n"
   "If a value larger than what a transport supports is set, the backlog value\n"
   "would be cut to that maximal value.",
   ucs_offsetof(ucp_config_t, ctx.listener_backlog), UCS_CONFIG_TYPE_ULUNITS},

  {"PROTO_ENABLE", "n",
   "Experimental: enable new protocol selection logic",
   ucs_offsetof(ucp_config_t, ctx.proto_enable), UCS_CONFIG_TYPE_BOOL},

  {"KEEPALIVE_INTERVAL", "20s",
   "Time interval between keepalive rounds.",
   ucs_offsetof(ucp_config_t, ctx.keepalive_interval),
   UCS_CONFIG_TYPE_TIME_UNITS},

  {"KEEPALIVE_NUM_EPS", "128",
   "Maximal number of endpoints to check on every keepalive round\n"
   "(inf - check all endpoints on every round, must be greater than 0)",
   ucs_offsetof(ucp_config_t, ctx.keepalive_num_eps), UCS_CONFIG_TYPE_UINT},

  {"RESOLVE_REMOTE_EP_ID", "n",
   "Defines whether resolving remote endpoint ID is required or not when\n"
   "creating a local endpoint. 'auto' means resolving remote endpint ID only\n"
   "in case of error handling and keepalive enabled.",
   ucs_offsetof(ucp_config_t, ctx.resolve_remote_ep_id),
   UCS_CONFIG_TYPE_ON_OFF_AUTO},

  {"PROTO_INDIRECT_ID", "auto",
   "Enable indirect IDs to object pointers (endpoint, request) in wire protocols.\n"
   "A value of 'auto' means to enable only if error handling is enabled on the\n"
   "endpoint.",
   ucs_offsetof(ucp_config_t, ctx.proto_indirect_id), UCS_CONFIG_TYPE_ON_OFF_AUTO},

  {"RNDV_PUT_FORCE_FLUSH", "n",
   "When using rendezvous put protocol, force using a flush operation to ensure\n"
   "remote data delivery before sending ATP message.\n"
   "If flush mode is not forced, and the underlying transport supports both active\n"
   "messages and put operations, the protocol will do {put,fence,ATP} on the same\n"
   "lane without waiting for remote completion.",
   ucs_offsetof(ucp_config_t, ctx.rndv_put_force_flush), UCS_CONFIG_TYPE_BOOL},

  {"SA_DATA_VERSION", "v1",
   "Defines the minimal header version the client will use for establishing\n"
   "client/server connection",
   ucs_offsetof(ucp_config_t, ctx.sa_client_min_hdr_version),
   UCS_CONFIG_TYPE_ENUM(ucp_object_versions)},

  {"RKEY_MPOOL_MAX_MD", "2",
   "Maximum number of UCP rkey MDs which can be unpacked into memory pool\n"
   "element. UCP rkeys containing larger number of MDs will be unpacked to\n"
   "dynamically allocated memory.",
   ucs_offsetof(ucp_config_t, ctx.rkey_mpool_max_md), UCS_CONFIG_TYPE_INT},

  {"RX_MPOOL_SIZES", "64,1kb",
   "List of worker mpool sizes separated by comma. The values must be power of 2\n"
   "Values larger than the maximum UCT transport segment size will be ignored.\n"
   "These pools are used for UCP AM and unexpected TAG messages. When assigning\n"
   "pool sizes, note that the data may be stored with some header.",
   ucs_offsetof(ucp_config_t, mpool_sizes), UCS_CONFIG_TYPE_ARRAY(memunit_sizes)},

  {"ADDRESS_VERSION", "v1",
   "Defines UCP worker address format obtained with ucp_worker_get_address() or\n"
   "ucp_worker_query() routines.",
   ucs_offsetof(ucp_config_t, ctx.worker_addr_version),
   UCS_CONFIG_TYPE_ENUM(ucp_object_versions)},

  {"RCACHE_ENABLE", "try", "Use user space memory registration cache.",
   ucs_offsetof(ucp_config_t, enable_rcache), UCS_CONFIG_TYPE_TERNARY},

  {"PROTO_INFO", "n", "Enable printing protocols information.",
   ucs_offsetof(ucp_config_t, ctx.proto_info), UCS_CONFIG_TYPE_BOOL},

   {NULL}
};
UCS_CONFIG_DECLARE_TABLE(ucp_config_table, "UCP context", NULL, ucp_config_t)


static ucp_tl_alias_t ucp_tl_aliases[] = {
  { "mm",    { "posix", "sysv", "xpmem", NULL } }, /* for backward compatibility */
  { "sm",    { "posix", "sysv", "xpmem", "knem", "cma", NULL } },
  { "shm",   { "posix", "sysv", "xpmem", "knem", "cma", NULL } },
  { "ib",    { "rc_verbs", "ud_verbs", "rc_mlx5", "ud_mlx5", "dc_mlx5", NULL } },
  { "ud_v",  { "ud_verbs", NULL } },
  { "ud_x",  { "ud_mlx5", NULL } },
  { "ud",    { "ud_mlx5", "ud_verbs", NULL } },
  { "rc_v",  { "rc_verbs", "ud_verbs:aux", NULL } },
  { "rc_x",  { "rc_mlx5", "ud_mlx5:aux", NULL } },
  { "rc",    { "rc_mlx5", "ud_mlx5:aux", "rc_verbs", "ud_verbs:aux", NULL } },
  { "dc",    { "dc_mlx5", "ud_mlx5:aux", NULL } },
  { "dc_x",  { "dc_mlx5", "ud_mlx5:aux", NULL } },
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


const ucp_tl_bitmap_t ucp_tl_bitmap_max = {{UINT64_MAX, UINT64_MAX}};
const ucp_tl_bitmap_t ucp_tl_bitmap_min = UCS_BITMAP_ZERO;


ucs_status_t ucp_config_read(const char *env_prefix, const char *filename,
                             ucp_config_t **config_p)
{
    unsigned full_prefix_len = sizeof(UCS_DEFAULT_ENV_PREFIX);
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
        /* Extra one byte for underscore _ character */
        full_prefix_len += env_prefix_len + 1;
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

    ucs_list_head_init(&config->cached_key_list);

    *config_p = config;
    return UCS_OK;

err_free_prefix:
    ucs_free(config->env_prefix);
err_free_config:
    ucs_free(config);
err:
    return status;
}

static void ucp_cached_key_release(ucs_config_cached_key_t *key_val)
{
    ucs_assert(key_val != NULL);

    ucs_free(key_val->key);
    ucs_free(key_val->value);
    ucs_free(key_val);
}

static void ucp_cached_key_list_release(ucs_list_link_t *list)
{
    ucs_config_cached_key_t *key_val;

    while (!ucs_list_is_empty(list)) {
        key_val = ucs_list_extract_head(list, typeof(*key_val), list);
        ucp_cached_key_release(key_val);
    }
}

static ucs_status_t
ucp_config_cached_key_add(ucs_list_link_t *list,
                          const char *key, const char *value)
{
    ucs_config_cached_key_t *cached_key;

    cached_key = ucs_malloc(sizeof(*cached_key), "cached config key/value");
    if (cached_key == NULL) {
        goto err;
    }

    cached_key->key   = ucs_strdup(key, "cached config key");
    cached_key->value = ucs_strdup(value, "cached config value");
    cached_key->used  = 0;
    if ((cached_key->key == NULL) || (cached_key->value == NULL)) {
        goto err_free_key;
    }

    ucs_list_add_tail(list, &cached_key->list);
    return UCS_OK;

err_free_key:
    ucp_cached_key_release(cached_key);
err:
    return UCS_ERR_NO_MEMORY;
}

void ucp_config_release(ucp_config_t *config)
{
    ucp_cached_key_list_release(&config->cached_key_list);
    ucs_config_parser_release_opts(config, ucp_config_table);
    ucs_free(config->env_prefix);
    ucs_free(config);
}

ucs_status_t ucp_config_modify_internal(ucp_config_t *config, const char *name,
                                        const char *value)
{
    return ucs_config_parser_set_value(config, ucp_config_table, name, value);
}

ucs_status_t ucp_config_modify(ucp_config_t *config, const char *name,
                               const char *value)
{
    ucs_status_t status;

    status = ucp_config_modify_internal(config, name, value);
    if (status != UCS_ERR_NO_ELEM) {
        return status;
    }

    status = ucs_global_opts_set_value_modifiable(name, value);
    if (status != UCS_ERR_NO_ELEM) {
        return status;
    }

    return ucp_config_cached_key_add(&config->cached_key_list, name, value);
}

static
void ucp_config_print_cached_uct(const ucp_config_t *config, FILE *stream,
                                 const char *title,
                                 ucs_config_print_flags_t flags)
{
    ucs_config_cached_key_t *key_val;

    if (flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "\n");
        fprintf(stream, "#\n");
        fprintf(stream, "# Cached UCT %s\n", title);
        fprintf(stream, "#\n");
        fprintf(stream, "\n");
    }

    if (flags & UCS_CONFIG_PRINT_CONFIG) {
        ucs_list_for_each(key_val, &config->cached_key_list, list) {
            fprintf(stream, "%s=%s\n", key_val->key, key_val->value);
        }
    }

    if (flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "\n");
    }
}

void ucp_config_print(const ucp_config_t *config, FILE *stream,
                      const char *title, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, ucp_config_table,
                                 NULL, UCS_DEFAULT_ENV_PREFIX, print_flags);
    ucp_config_print_cached_uct(config, stream, title, print_flags);
}

void ucp_apply_uct_config_list(ucp_context_h context, void *config)
{
    ucs_config_cached_key_t *key_val;
    ucs_status_t status;

    ucs_list_for_each(key_val, &context->cached_key_list, list) {
        status = uct_config_modify(config, key_val->key, key_val->value);
        if (status == UCS_OK) {
            ucs_debug("apply UCT configuration %s=%s", key_val->key,
                      key_val->value);
            key_val->used = 1;
        }
    }
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

static int
ucp_config_is_tl_name_present(const ucs_config_allow_list_t *allow_list,
                              const char *tl_name, int is_alias,
                              uint8_t *rsc_flags, uint64_t *tl_cfg_mask)
{
    char strict_name[UCT_TL_NAME_MAX + 1];

    snprintf(strict_name, sizeof(strict_name), "\\%s", tl_name);

    return /* strict name, with leading \\ */
            (!is_alias &&
             (ucp_tls_array_is_present((const char**)allow_list->array.names,
                                       allow_list->array.count, strict_name, "",
                                       rsc_flags, tl_cfg_mask))) ||
            /* plain transport name */
            (ucp_tls_array_is_present((const char**)allow_list->array.names,
                                      allow_list->array.count, tl_name, "",
                                      rsc_flags, tl_cfg_mask));
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

static int
ucp_is_resource_in_transports_list(const char *tl_name,
                                   const ucs_config_allow_list_t *allow_list,
                                   uint8_t *rsc_flags, uint64_t *tl_cfg_mask)
{
    uint64_t dummy_mask, tmp_tl_cfg_mask;
    uint8_t tmp_rsc_flags;
    ucp_tl_alias_t *alias;
    char info[32];
    unsigned alias_arr_count;

    if (allow_list->mode == UCS_CONFIG_ALLOW_LIST_ALLOW_ALL) {
        return 1;
    }

    ucs_assert(allow_list->array.count > 0);
    if (ucp_config_is_tl_name_present(allow_list, tl_name, 0, rsc_flags,
                                      tl_cfg_mask)) {
        /* If the TL was found by its strict name - the result is known,
           otherwise checking aliases is required */
        return (allow_list->mode == UCS_CONFIG_ALLOW_LIST_ALLOW);
    }

    /* check aliases */
    for (alias = ucp_tl_aliases; alias->alias != NULL; ++alias) {
        /* If an alias is in the list and the transport belongs this alias,
         * enable/disable the transport (according to the list mode)
         */
        alias_arr_count = ucp_tl_alias_count(alias);
        snprintf(info, sizeof(info), " for alias '%s'", alias->alias);
        dummy_mask      = 0;
        tmp_rsc_flags   = 0;
        tmp_tl_cfg_mask = 0;
        if (ucp_tls_array_is_present(alias->tls, alias_arr_count, tl_name, info,
                                     &tmp_rsc_flags, &dummy_mask)) {
            if (ucp_config_is_tl_name_present(allow_list, alias->alias, 1,
                                              &tmp_rsc_flags,
                                              &tmp_tl_cfg_mask)) {
                if (allow_list->mode == UCS_CONFIG_ALLOW_LIST_ALLOW) {
                    *rsc_flags   |= tmp_rsc_flags;
                    *tl_cfg_mask |= tmp_tl_cfg_mask;
                    return 1;
                } else {
                    return 0;
                }
            }
        }
    }

    return allow_list->mode == UCS_CONFIG_ALLOW_LIST_NEGATE;
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
                                                    &config->tls, rsc_flags,
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
        if ((resource->sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) &&
            (resource->sys_device >= UCP_MAX_SYS_DEVICES)) {
            ucs_diag(UCT_TL_RESOURCE_DESC_FMT
                     " system device is %d, which exceeds the maximal "
                     "supported (%d), system locality may be ignored",
                     UCT_TL_RESOURCE_DESC_ARG(resource), resource->sys_device,
                     UCP_MAX_SYS_DEVICES);
        }
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
                      sizeof(*context->tl_rscs) *
                      (context->num_tls + num_tl_resources),
                      "ucp resources");
    if (tmp == NULL) {
        ucs_error("Failed to allocate resources");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_resources;
    }

    /* print configuration */
    for (i = 0; i < config->tls.array.count; ++i) {
        ucs_trace("allowed transport %d : '%s'", i, config->tls.array.names[i]);
    }

    /* copy only the resources enabled by user configuration */
    context->tl_rscs = tmp;
    for (i = 0; i < num_tl_resources; ++i) {
        ucs_string_set_addf(&avail_devices[tl_resources[i].dev_type],
                            "'%s'(%s)", tl_resources[i].dev_name,
                            context->tl_cmpts[md->cmpt_index].attr.name);
        ucs_string_set_add(avail_tls, tl_resources[i].tl_name);
        ucp_add_tl_resource_if_enabled(context, md, md_index, config,
                                       &tl_resources[i], 0, num_resources_p,
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

const char *ucp_tl_bitmap_str(ucp_context_h context,
                              const ucp_tl_bitmap_t *tl_bitmap, char *str,
                              size_t max_str_len)
{
    ucp_rsc_index_t i;
    char *p, *endp;

    p    = str;
    endp = str + max_str_len;

    UCS_BITMAP_FOR_EACH_BIT(*tl_bitmap, i) {
        ucs_snprintf_zero(p, endp - p, "%s ",
                          context->tl_rscs[i].tl_rsc.tl_name);
        p += strlen(p);
    }

    return str;
}


static void ucp_free_resources(ucp_context_t *context)
{
    ucp_rsc_index_t i;

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
     if ((0 == config->tls.array.count) &&
         (config->tls.mode != UCS_CONFIG_ALLOW_LIST_ALLOW_ALL)) {
         ucs_error("The TLs list is empty. Please specify the transports you "
                   "would like to allow/forbid "
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

    ucp_apply_uct_config_list(context, md_config);

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

    tl_md->pack_flags_mask = (tl_md->attr.cap.flags & UCT_MD_FLAG_INVALIDATE) ?
                             UCT_MD_MKEY_PACK_FLAG_INVALIDATE : 0;
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

    ucp_resource_config_array_str(&config->tls.array, "", p, endp - p);

    if (strlen(p)) {
        p += strlen(p);
        snprintf(p, endp - p, "on ");
        p += strlen(p);
    }

    devs_p = p;
    for (dev_type_idx = 0; dev_type_idx < UCT_DEVICE_TYPE_LAST; ++dev_type_idx) {
        ucp_resource_config_array_str(&config->devices[dev_type_idx],
                                      uct_device_type_names[dev_type_idx], p,
                                      endp - p);
        p += strlen(p);
    }

    if (devs_p == p) {
        snprintf(p, endp - p, "all devices");
    }
}

static void ucp_fill_sockaddr_cms_prio_list(ucp_context_h context,
                                            const char **sockaddr_cm_names,
                                            ucp_rsc_index_t num_sockaddr_cms)
{
    ucp_tl_bitmap_t cm_cmpts_bitmap = context->config.cm_cmpts_bitmap;
    ucp_tl_bitmap_t cm_cmpts_bitmap_safe;
    ucp_rsc_index_t cmpt_idx, cm_idx;

    memset(&context->config.cm_cmpt_idxs, UCP_NULL_RESOURCE, UCP_MAX_RESOURCES);
    context->config.num_cm_cmpts = 0;

    /* Parse the sockaddr CMs priority list */
    for (cm_idx = 0; cm_idx < num_sockaddr_cms; ++cm_idx) {
        /* go over the priority list and find the CM's cm_idx in the
         * sockaddr CMs bitmap. Save the cmpt_idx for the client/server usage
         * later */
        cm_cmpts_bitmap_safe = cm_cmpts_bitmap;
        UCS_BITMAP_FOR_EACH_BIT(cm_cmpts_bitmap_safe, cmpt_idx) {
            if (!strcmp(sockaddr_cm_names[cm_idx], "*") ||
                !strncmp(sockaddr_cm_names[cm_idx],
                         context->tl_cmpts[cmpt_idx].attr.name,
                         UCT_COMPONENT_NAME_MAX)) {
                context->config.cm_cmpt_idxs[context->config.num_cm_cmpts++] = cmpt_idx;
                UCS_BITMAP_UNSET(cm_cmpts_bitmap, cmpt_idx);
            }
        }
    }
}

static void
ucp_fill_sockaddr_prio_list(ucp_context_h context, const ucp_config_t *config)
{
    const char **sockaddr_tl_names = (const char**)config->sockaddr_cm_tls.cm_tls;
    unsigned num_sockaddr_tls      = config->sockaddr_cm_tls.count;

    /* Check if a list of sockaddr transports/CMs has valid length */
    if (num_sockaddr_tls > UCP_MAX_RESOURCES) {
        ucs_warn("sockaddr transports or connection managers list is too long, "
                 "only first %d entries will be used", UCP_MAX_RESOURCES);
        num_sockaddr_tls = UCP_MAX_RESOURCES;
    }

    ucp_fill_sockaddr_cms_prio_list(context, sockaddr_tl_names,
                                    num_sockaddr_tls);
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
    ucs_memory_type_t mem_type;
    const uct_md_attr_t *md_attr;

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

            md_attr = &context->tl_mds[md_index].attr;
            if (md_attr->cap.flags & UCT_MD_FLAG_REG) {
                ucs_memory_type_for_each(mem_type) {
                    if (md_attr->cap.reg_mem_types & UCS_BIT(mem_type)) {
                        context->reg_md_map[mem_type] |= UCS_BIT(md_index);
                    }
                }
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

    context->tl_cmpts                 = NULL;
    context->num_cmpts                = 0;
    context->tl_mds                   = NULL;
    context->num_mds                  = 0;
    context->alloc_md_map_initialized = 0;
    context->alloc_md_map             = 0;
    context->tl_rscs                  = NULL;
    context->num_tls                  = 0;
    context->mem_type_mask            = 0;
    context->num_mem_type_detect_mds  = 0;

    for (i = 0; i < UCS_MEMORY_TYPE_LAST; ++i) {
        context->reg_md_map[i] = 0;
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

    UCS_BITMAP_CLEAR(&context->config.cm_cmpts_bitmap);

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
            UCS_BITMAP_SET(context->config.cm_cmpts_bitmap, i);
        }

        max_mds += context->tl_cmpts[i].attr.md_resource_count;
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

    /* If unified mode is enabled, initialize tl_bitmap to 0.
     * Then the worker will open all available transport resources and will
     * select only the best ones for each particular device.
     */
    UCS_BITMAP_MASK(&context->tl_bitmap,
                    config->ctx.unified_mode ? 0 : context->num_tls);

    /* Warn about devices and transports which were specified explicitly in the
     * configuration, but are not available
     */
    if (config->warn_invalid_config) {
        UCS_STATIC_ASSERT(UCT_DEVICE_TYPE_NET == 0);
        for (dev_type = UCT_DEVICE_TYPE_NET; dev_type < UCT_DEVICE_TYPE_LAST; ++dev_type) {
            ucp_report_unavailable(&config->devices[dev_type],
                                   dev_cfg_masks[dev_type],
                                   uct_device_type_names[dev_type], " device",
                                   &avail_devices[dev_type]);
        }

        ucp_get_aliases_set(&avail_tls);
        ucp_report_unavailable(&config->tls.array, tl_cfg_mask, "", "transport",
                               &avail_tls);
    }

    /* Validate context resources */
    status = ucp_check_resources(context, config);
    if (status != UCS_OK) {
        goto err_free_resources;
    }

    ucp_fill_sockaddr_prio_list(context, config);

out_release_components:
    uct_release_component_list(uct_components);
out_cleanup_avail_devices:
    UCS_STATIC_ASSERT(UCT_DEVICE_TYPE_NET == 0);
    for (dev_type = UCT_DEVICE_TYPE_NET; dev_type < UCT_DEVICE_TYPE_LAST; ++dev_type) {
        ucs_string_set_cleanup(&avail_devices[dev_type]);
    }
    ucs_string_set_cleanup(&avail_tls);
    return status;

err_free_resources:
    ucp_free_resources(context);
    goto out_release_components;
}

static void ucp_apply_params(ucp_context_h context, const ucp_params_t *params,
                             ucp_mt_type_t mt_type)
{
    context->config.features = UCP_PARAM_FIELD_VALUE(params, features, FEATURES,
                                                     0);
    if (!context->config.features) {
        ucs_warn("empty features set passed to ucp context create");
    }

    context->config.tag_sender_mask = UCP_PARAM_FIELD_VALUE(params,
                                                            tag_sender_mask,
                                                            TAG_SENDER_MASK, 0);

    context->config.request.size = UCP_PARAM_FIELD_VALUE(params, request_size,
                                                         REQUEST_SIZE, 0);

    context->config.request.init = UCP_PARAM_FIELD_VALUE(params, request_init,
                                                         REQUEST_INIT, NULL);

    context->config.request.cleanup = UCP_PARAM_FIELD_VALUE(params,
                                                            request_cleanup,
                                                            REQUEST_CLEANUP, NULL);

    context->config.est_num_eps = UCP_PARAM_FIELD_VALUE(params,
                                                        estimated_num_eps,
                                                        ESTIMATED_NUM_EPS, 1);

    context->config.est_num_ppn = UCP_PARAM_FIELD_VALUE(params,
                                                        estimated_num_ppn,
                                                        ESTIMATED_NUM_PPN, 1);

    if ((params->field_mask & UCP_PARAM_FIELD_MT_WORKERS_SHARED) &&
        params->mt_workers_shared) {
        context->mt_lock.mt_type = mt_type;
    } else {
        context->mt_lock.mt_type = UCP_MT_TYPE_NONE;
    }

    if ((params->field_mask & UCP_PARAM_FIELD_NAME) && (params->name != NULL)) {
        ucs_snprintf_zero(context->name, UCP_ENTITY_NAME_MAX, "%s",
                          params->name);
    } else {
        ucs_snprintf_zero(context->name, UCP_ENTITY_NAME_MAX, "%p", context);
    }
}

static ucs_status_t
ucp_fill_rndv_frag_config(const ucp_context_config_names_t *config,
                          const size_t *default_sizes, size_t *sizes)
{
    const char *mem_type_name, *size_str;
    char config_str[128];
    ucs_status_t status;
    ssize_t mem_type;
    unsigned i;

    for (mem_type = 0; mem_type < UCS_MEMORY_TYPE_LAST; ++mem_type) {
        sizes[mem_type] = default_sizes[mem_type];
    }

    for (i = 0; i < config->count; ++i) {
        ucs_strncpy_safe(config_str, config->names[i], sizeof(config_str));
        ucs_string_split(config_str, ":", 2, &mem_type_name, &size_str);
        mem_type = ucs_string_find_in_list(mem_type_name, ucs_memory_type_names,
                                           0);
        if (mem_type < 0) {
            ucs_error("invalid memory type specifier: '%s'", mem_type_name);
            return UCS_ERR_INVALID_PARAM;
        }

        ucs_assert(mem_type < UCS_MEMORY_TYPE_LAST);
        status = ucs_str_to_memunits(size_str, &sizes[mem_type]);
        if (status != UCS_OK) {
            ucs_error("failed to parse size configuration: '%s'", size_str);
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucp_fill_config(ucp_context_h context,
                                    const ucp_params_t *params,
                                    const ucp_config_t *config)
{
    unsigned i, num_alloc_methods, method;
    const char *method_name;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    int match;
    ucs_config_cached_key_t *key_val;

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

    if (config->protos.mode == UCS_CONFIG_ALLOW_LIST_ALLOW_ALL) {
        context->proto_bitmap = UCS_MASK(ucp_protocols_count());
    } else {
        for (proto_id = 0; proto_id < ucp_protocols_count(); ++proto_id) {
            match = ucs_config_names_search(&config->protos.array,
                                            ucp_proto_id_field(proto_id, name));
            if (((config->protos.mode == UCS_CONFIG_ALLOW_LIST_ALLOW) &&
                 (match >= 0)) ||
                ((config->protos.mode == UCS_CONFIG_ALLOW_LIST_NEGATE) &&
                 (match == -1))) {
                context->proto_bitmap |= UCS_BIT(proto_id);
            }
        }
    }

    /* always init MT lock in context even though it is disabled by user,
     * because we need to use context lock to protect ucp_mem_ and ucp_rkey_
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

    status = ucp_fill_rndv_frag_config(&config->rndv_frag_sizes,
                                       ucp_rndv_frag_default_sizes,
                                       context->config.ext.rndv_frag_size);
    if (status != UCS_OK) {
        goto err_free_alloc_methods;
    }

    status = ucp_fill_rndv_frag_config(&config->rndv_frag_elems,
                                       ucp_rndv_frag_default_num_elems,
                                       context->config.ext.rndv_num_frags);
    if (status != UCS_OK) {
        goto err_free_alloc_methods;
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

    ucs_list_for_each(key_val, &config->cached_key_list, list) {
        status = ucp_config_cached_key_add(&context->cached_key_list,
                                           key_val->key, key_val->value);
        if (status != UCS_OK) {
            goto err_free_key_list;
        }
    }

    context->config.am_mpools.count = config->mpool_sizes.count;
    context->config.am_mpools.sizes = ucs_malloc(sizeof(size_t) *
                                                 config->mpool_sizes.count,
                                                 "am_mpool_sizes");
    if (context->config.am_mpools.sizes == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_key_list;
    }
    memcpy(context->config.am_mpools.sizes, config->mpool_sizes.memunits,
           config->mpool_sizes.count * sizeof(size_t));

    return UCS_OK;

err_free_key_list:
    ucp_cached_key_list_release(&context->cached_key_list);
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
    ucp_cached_key_list_release(&context->cached_key_list);
    ucs_free(context->config.am_mpools.sizes);
}

static void ucp_context_create_vfs(ucp_context_h context)
{
    ucs_vfs_obj_add_dir(NULL, context, "ucp/context/%s", context->name);
    ucs_vfs_obj_add_ro_file(context, ucs_vfs_show_memory_address, NULL, 0,
                            "memory_address");
}

static void
ucp_version_check(unsigned api_major_version, unsigned api_minor_version)
{
    UCS_STRING_BUFFER_ONSTACK(strb, 256);
    unsigned major_version, minor_version, release_number;
    ucs_log_level_t log_level;
    Dl_info dl_info;
    int ret;

    ucp_get_version(&major_version, &minor_version, &release_number);

    if ((major_version == api_major_version) &&
        (minor_version >= api_minor_version)) {
        /* API version is compatible: same major, same or higher minor */
        ucs_string_buffer_appendf(&strb, "Version %s",
                                  ucp_get_version_string());
        log_level = UCS_LOG_LEVEL_INFO;
    } else {
        ucs_string_buffer_appendf(
                &strb,
                "UCP API version is incompatible: required >= %d.%d, actual %s",
                api_major_version, api_minor_version, ucp_get_version_string());
        log_level = UCS_LOG_LEVEL_WARN;
    }

    if (ucs_log_is_enabled(log_level)) {
        ret = dladdr(ucp_init_version, &dl_info);
        if (ret != 0) {
            ucs_string_buffer_appendf(&strb, " (loaded from %s)",
                                      dl_info.dli_fname);
        }
        ucs_log(log_level, "%s", ucs_string_buffer_cstr(&strb));
    }
}

ucs_status_t ucp_init_version(unsigned api_major_version, unsigned api_minor_version,
                              const ucp_params_t *params, const ucp_config_t *config,
                              ucp_context_h *context_p)
{
    ucp_config_t *dfl_config = NULL;
    ucp_context_t *context;
    ucs_status_t status;

    ucp_version_check(api_major_version, api_minor_version);

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

    ucs_list_head_init(&context->cached_key_list);

    status = ucp_fill_config(context, params, config);
    if (status != UCS_OK) {
        goto err_free_ctx;
    }

    /* fill resources we should use */
    status = ucp_fill_resources(context, config);
    if (status != UCS_OK) {
        goto err_free_config;
    }

    if (config->enable_rcache != UCS_NO) {
        status = ucp_mem_rcache_init(context);
        if (status != UCS_OK) {
            if (config->enable_rcache == UCS_YES) {
                ucs_error("could not create UCP registration cache: %s",
                          ucs_status_string(status));
                goto err_free_res;
            } else {
                ucs_diag("could not create UCP registration cache: %s",
                         ucs_status_string(status));
            }
        }
    } else {
        context->rcache = NULL;
    }

    if (dfl_config != NULL) {
        ucp_config_release(dfl_config);
    }

    ucp_context_create_vfs(context);

    ucs_debug("created ucp context %s %p [%d mds %d tls] features 0x%" PRIx64
              " tl bitmap " UCT_TL_BITMAP_FMT,
              context->name, context, context->num_mds, context->num_tls,
              context->config.features, UCT_TL_BITMAP_ARG(&context->tl_bitmap));

    *context_p = context;
    return UCS_OK;

err_free_res:
    ucp_free_resources(context);
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
    ucs_vfs_obj_remove(context);
    ucp_mem_rcache_cleanup(context);
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

ucs_status_t ucp_lib_query(ucp_lib_attr_t *attr)
{
    if (attr->field_mask & UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL) {
#if ENABLE_MT
        attr->max_thread_level = UCS_THREAD_MODE_MULTI;
#else
        attr->max_thread_level = UCS_THREAD_MODE_SERIALIZED;
#endif
    }

    return UCS_OK;
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

    if (attr->field_mask & UCP_ATTR_FIELD_NAME) {
        ucs_strncpy_safe(attr->name, context->name, UCP_ENTITY_NAME_MAX);
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

void ucp_memory_detect_slowpath(ucp_context_h context, const void *address,
                                size_t length, ucs_memory_info_t *mem_info)
{
    uct_md_mem_attr_t mem_attr;
    ucs_status_t status;
    ucp_tl_md_t *tl_md;
    ucp_md_index_t i;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE |
                          UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS |
                          UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH |
                          UCT_MD_MEM_ATTR_FIELD_SYS_DEV;

    for (i = 0; i < context->num_mem_type_detect_mds; ++i) {
        tl_md  = &context->tl_mds[context->mem_type_detect_mds[i]];
        status = uct_md_mem_query(tl_md->md, address, length, &mem_attr);
        if (status != UCS_OK) {
            continue;
        }

        ucs_trace_req("address %p length %zu: md %s detected as type '%s' %s",
                      address, length, tl_md->rsc.md_name,
                      ucs_memory_type_names[mem_attr.mem_type],
                      ucs_topo_sys_device_get_name(mem_attr.sys_dev));
        mem_info->type         = mem_attr.mem_type;
        mem_info->sys_dev      = mem_attr.sys_dev;
        mem_info->base_address = mem_attr.base_address;
        mem_info->alloc_length = mem_attr.alloc_length;
        return;
    }

    /* Memory type not detected by any memtype MD - assume it is host memory */
    ucs_trace_req("address %p length %zu: not detected by any md (have: %d), "
                  "assuming host memory",
                  address, length, context->num_mem_type_detect_mds);
    ucs_memory_info_set_host(mem_info);
}

void
ucp_context_dev_tl_bitmap(ucp_context_h context, const char *dev_name,
                          ucp_tl_bitmap_t *tl_bitmap)
{
    ucp_rsc_index_t tl_idx;

    UCS_BITMAP_CLEAR(tl_bitmap);
    UCS_BITMAP_FOR_EACH_BIT(context->tl_bitmap, tl_idx) {
        if (strcmp(context->tl_rscs[tl_idx].tl_rsc.dev_name, dev_name)) {
            continue;
        }

        UCS_BITMAP_SET(*tl_bitmap, tl_idx);
    }
}

void
ucp_context_dev_idx_tl_bitmap(ucp_context_h context, ucp_rsc_index_t dev_idx,
                              ucp_tl_bitmap_t *tl_bitmap)
{
    ucp_rsc_index_t tl_idx;

    UCS_BITMAP_CLEAR(tl_bitmap);
    UCS_BITMAP_FOR_EACH_BIT(context->tl_bitmap, tl_idx) {
        if (context->tl_rscs[tl_idx].dev_index == dev_idx) {
            UCS_BITMAP_SET(*tl_bitmap, tl_idx);
        }
    }
}

void ucp_tl_bitmap_validate(const ucp_tl_bitmap_t *tl_bitmap,
                            const ucp_tl_bitmap_t *tl_bitmap_super)
{
    ucs_assert_always(UCS_BITMAP_IS_ZERO(UCP_TL_BITMAP_AND_NOT(*tl_bitmap,
                                                               *tl_bitmap_super),
                                         UCP_MAX_RESOURCES));
}

const char* ucp_context_cm_name(ucp_context_h context, ucp_rsc_index_t cm_idx)
{
    ucs_assert(cm_idx != UCP_NULL_RESOURCE);
    return context->tl_cmpts[context->config.cm_cmpt_idxs[cm_idx]].attr.name;
}

void ucp_context_get_mem_access_tls(ucp_context_h context,
                                    ucs_memory_type_t mem_type,
                                    ucp_tl_bitmap_t *tl_bitmap)
{
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;
    ucp_rsc_index_t tl_id;

    UCS_BITMAP_CLEAR(tl_bitmap);
    UCS_BITMAP_FOR_EACH_BIT(context->tl_bitmap, tl_id) {
        md_index = context->tl_rscs[tl_id].md_index;
        md_attr  = &context->tl_mds[md_index].attr;
        if (md_attr->cap.access_mem_types & UCS_BIT(mem_type)) {
            UCS_BITMAP_SET(*tl_bitmap, tl_id);
        }
    }
}

UCS_F_CTOR void ucp_global_init(void)
{
    UCS_CONFIG_ADD_TABLE(ucp_config_table, &ucs_config_global_list);
}

UCS_F_DTOR static void ucp_global_cleanup(void)
{
    UCS_CONFIG_REMOVE_TABLE(ucp_config_table);
}
