/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_CONTEXT_H_
#define UCP_CONTEXT_H_

#include "ucp_types.h"
#include "ucp_thread.h"

#include <ucp/api/ucp.h>
#include <ucp/dt/dt.h>
#include <ucp/proto/proto.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/datastruct/bitmap.h>
#include <ucs/datastruct/conn_match.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/spinlock.h>
#include <ucs/sys/string.h>
#include <ucs/type/param.h>


enum {
    /* The flag indicates that the resource may be used for auxiliary
     * wireup communications only */
    UCP_TL_RSC_FLAG_AUX      = UCS_BIT(0),
    /* The flag indicates that the resource may be used for client-server
     * connection establishment with a sockaddr */
    UCP_TL_RSC_FLAG_SOCKADDR = UCS_BIT(1)
};


typedef struct ucp_context_config {
    /** Threshold for switching UCP to buffered copy(bcopy) protocol */
    size_t                                 bcopy_thresh;
    /** Threshold for switching UCP to rendezvous protocol */
    size_t                                 rndv_thresh;
    /** Threshold for switching UCP to rendezvous protocol
     *  in ucp_tag_send_nbr() */
    size_t                                 rndv_send_nbr_thresh;
    /** Threshold for switching UCP to rendezvous protocol in case the calculated
     *  threshold is zero or negative */
    size_t                                 rndv_thresh_fallback;
    /** The percentage allowed for performance difference between rendezvous
     *  and the eager_zcopy protocol */
    double                                 rndv_perf_diff;
    /** Maximal allowed ratio between slowest and fastest lane in a multi-lane
     *  protocol. Lanes slower than the specified ratio will not be used */
    double                                 multi_lane_max_ratio;
    /** Threshold for switching UCP to zero copy protocol */
    size_t                                 zcopy_thresh;
    /** Communication scheme in RNDV protocol */
    ucp_rndv_mode_t                        rndv_mode;
    /** RKEY PTR segment size */
    size_t                                 rkey_ptr_seg_size;
    /** Estimation of bcopy bandwidth */
    double                                 bcopy_bw;
    /** Segment size in the worker pre-registered memory pool */
    size_t                                 seg_size;
    /** RNDV pipeline fragment size */
    size_t                                 rndv_frag_size[UCS_MEMORY_TYPE_LAST];
    /** Number of RNDV pipeline fragments per allocation */
    size_t                                 rndv_num_frags[UCS_MEMORY_TYPE_LAST];
    /** Memory type of fragments used for RNDV pipeline protocol */
    ucs_memory_type_t                      rndv_frag_mem_type;
    /** RNDV pipeline send threshold */
    size_t                                 rndv_pipeline_send_thresh;
    /** Enabling 2-stage pipeline rndv protocol */
    int                                    rndv_shm_ppln_enable;
    /** Threshold for using tag matching offload capabilities. Smaller buffers
     *  will not be posted to the transport. */
    size_t                                 tm_thresh;
    /** Threshold for forcing tag matching offload capabilities */
    size_t                                 tm_force_thresh;
    /** Upper bound for posting tm offload receives with internal UCP
     *  preregistered bounce buffers. */
    size_t                                 tm_max_bb_size;
    /** Enabling SW rndv protocol with tag offload mode */
    int                                    tm_sw_rndv;
    /** Pack debug information in worker address */
    int                                    address_debug_info;
    /** Maximal size of worker address name for debugging */
    unsigned                               max_worker_address_name;
    /** Atomic mode */
    ucp_atomic_mode_t                      atomic_mode;
    /** If use mutex for MT support or not */
    int                                    use_mt_mutex;
    /** On-demand progress */
    int                                    adaptive_progress;
    /** Eager-am multi-lane support */
    unsigned                               max_eager_lanes;
    /** Rendezvous-get multi-lane support */
    unsigned                               max_rndv_lanes;
    /** Estimated number of endpoints */
    size_t                                 estimated_num_eps;
    /** Estimated number of processes per node */
    size_t                                 estimated_num_ppn;
    /** Enable flushing endpoints while flushing a worker */
    int                                    flush_worker_eps;
    /** Enable optimizations suitable for homogeneous systems */
    int                                    unified_mode;
    /** Enable cm wireup message exchange to select the best transports
     *  for all lanes after cm phase is done */
    int                                    cm_use_all_devices;
    /** Maximal number of pending connection requests for a listener */
    size_t                                 listener_backlog;
    /** Enable new protocol selection logic */
    int                                    proto_enable;
    /** Time period between keepalive rounds */
    ucs_time_t                             keepalive_interval;
    /** Maximal number of endpoints to check on every keepalive round
     * (0 - disabled, inf - check all endpoints on every round) */
    unsigned                               keepalive_num_eps;
    /** Defines whether resolving remote endpoint ID is required or not when
     *  creating a local endpoint */
    ucs_on_off_auto_value_t                resolve_remote_ep_id;
    /** Enable indirect IDs to object pointers in wire protocols */
    ucs_on_off_auto_value_t                proto_indirect_id;
    /** Bitmap of memory types whose allocations are registered fully */
    unsigned                               reg_whole_alloc_bitmap;
    /** Always use flush operation in rendezvous put */
    int                                    rndv_put_force_flush;
    /** Maximum size of mem type direct rndv*/
    size_t                                 rndv_memtype_direct_size;
    /** UCP sockaddr private data format version */
    ucp_object_version_t                   sa_client_min_hdr_version;
    /** Remote keys with that many remote MDs or less would be allocated from a
      * memory pool.*/
    int                                    rkey_mpool_max_md;
    /** Worker address format version */
    ucp_object_version_t                   worker_addr_version;
    /** Print protocols information */
    int                                    proto_info;
} ucp_context_config_t;


typedef UCS_CONFIG_STRING_ARRAY_FIELD(names) ucp_context_config_names_t;


struct ucp_config {
    /** Array of device lists names to use.
     *  This array holds four lists - network devices, shared memory devices,
     *  acceleration devices and loop-back devices */
    ucs_config_names_array_t               devices[UCT_DEVICE_TYPE_LAST];
    /** Array of transport names to use */
    ucs_config_allow_list_t                tls;
    /** Array of protocol names to use */
    ucs_config_allow_list_t                protos;
    /** Array of memory allocation methods */
    UCS_CONFIG_STRING_ARRAY_FIELD(methods) alloc_prio;
    /** Array of rendezvous fragment sizes */
    ucp_context_config_names_t             rndv_frag_sizes;
    /** Array of rendezvous fragment elems per allocation */
    ucp_context_config_names_t             rndv_frag_elems;
    /** Array of transports for client-server transports and port selection */
    UCS_CONFIG_STRING_ARRAY_FIELD(cm_tls)  sockaddr_cm_tls;
    /** Warn on invalid configuration */
    int                                    warn_invalid_config;
    /** This config environment prefix */
    char                                   *env_prefix;
    /** MD to compare for transport selection scores */
    char                                   *selection_cmp;
    /** Configuration saved directly in the context */
    ucp_context_config_t                   ctx;
    /** Save ucx configurations not listed in ucp_config_table **/
    ucs_list_link_t                        cached_key_list;
    /** Array of worker memory pool sizes */
    UCS_CONFIG_ARRAY_FIELD(size_t, memunits) mpool_sizes;
    /** Memory registration cache */
    ucs_ternary_auto_value_t               enable_rcache;
};


/**
 * UCP communication resource descriptor
 */
typedef struct ucp_tl_resource_desc {
    uct_tl_resource_desc_t        tl_rsc;       /* UCT resource descriptor */
    uint16_t                      tl_name_csum; /* Checksum of transport name */
    ucp_md_index_t                md_index;     /* Memory domain index (within the context) */
    ucp_rsc_index_t               dev_index;    /* Arbitrary device index. Resources
                                                   with same index have same device name. */
    uint8_t                       flags;        /* Flags that describe resource specifics */
} ucp_tl_resource_desc_t;


/**
 * Transport aliases.
 */
typedef struct ucp_tl_alias {
    const char                    *alias;   /* Alias name */
    const char*                   tls[8];   /* Transports which are selected by the alias */
} ucp_tl_alias_t;


/**
 * UCT component
 */
typedef struct ucp_tl_cmpt {
    uct_component_h               cmpt;      /* UCT component handle */
    uct_component_attr_t          attr;      /* UCT component attributes */
} ucp_tl_cmpt_t;


/**
 * Memory domain.
 */
typedef struct ucp_tl_md {
    /**
     * Memory domain handle
     */
    uct_md_h               md;

    /**
     * Index of owning component
     */
    ucp_rsc_index_t        cmpt_index;

    /**
     * Memory domain resource
     */
    uct_md_resource_desc_t rsc;

    /**
     * Memory domain attributes
     */
    uct_md_attr_t          attr;

    /**
     * Flags mask parameter for @ref uct_md_mkey_pack_v2
     */
    unsigned               pack_flags_mask;
} ucp_tl_md_t;


/**
 * UCP context
 */
typedef struct ucp_context {
    ucp_tl_cmpt_t                 *tl_cmpts;  /* UCT components */
    ucp_rsc_index_t               num_cmpts;  /* Number of UCT components */

    ucp_tl_md_t                   *tl_mds;    /* Memory domain resources */
    ucp_md_index_t                num_mds;    /* Number of memory domains */

    /* Map of memory domains that have valid memory keys for host memory
       allocated with ucp_mem_mmap_common().
       It's initialized on first access. */
    int                           alloc_md_map_initialized;
    ucp_md_map_t                  alloc_md_map;

    /* Map of MDs that provide registration for given memory type,
       ucp_mem_map() will register memory for all those domains. */
    ucp_md_map_t                  reg_md_map[UCS_MEMORY_TYPE_LAST];

    /* List of MDs that detect non host memory type */
    ucp_md_index_t                mem_type_detect_mds[UCS_MEMORY_TYPE_LAST];
    ucp_md_index_t                num_mem_type_detect_mds;  /* Number of mem type MDs */
    uint64_t                      mem_type_mask;            /* Supported mem type mask */

    ucp_tl_resource_desc_t        *tl_rscs;   /* Array of communication resources */
    ucp_tl_bitmap_t               tl_bitmap;  /* Cached map of tl resources used by workers.
                                               * Not all resources may be used if unified
                                               * mode is enabled. */
    ucp_rsc_index_t               num_tls;    /* Number of resources in the array */
    ucp_proto_id_mask_t           proto_bitmap;  /* Enabled protocols */

    /* Mem handle registration cache */
    ucs_rcache_t                  *rcache;

    struct {

        /* Bitmap of features supported by the context */
        uint64_t                  features;
        uint64_t                  tag_sender_mask;

        /* How many endpoints are expected to be created */
        int                       est_num_eps;

        /* How many endpoints are expected to be created on single node */
        int                       est_num_ppn;

        struct {
            size_t                         size;    /* Request size for user */
            ucp_request_init_callback_t    init;    /* Initialization user callback */
            ucp_request_cleanup_callback_t cleanup; /* Cleanup user callback */
        } request;

        /* Array of allocation methods, a mix of MD allocation methods and non-MD */
        struct {
            /* Allocation method */
            uct_alloc_method_t    method;

            /* Component name to use, if method is MD */
            char                  cmpt_name[UCT_COMPONENT_NAME_MAX];
        } *alloc_methods;
        unsigned                  num_alloc_methods;

        /* Cached map of components which support CM capability */
        ucp_tl_bitmap_t           cm_cmpts_bitmap;

        /* Array of CMs indexes. The indexes appear in the configured priority
         * order. */
        ucp_rsc_index_t           cm_cmpt_idxs[UCP_MAX_RESOURCES];
        ucp_rsc_index_t           num_cm_cmpts;

        /* Configuration supplied by the user */
        ucp_context_config_t      ext;

        /* Config environment prefix used to create the context */
        char                      *env_prefix;

        /* MD to compare for transport selection scores */
        char                      *selection_cmp;
        struct {
           unsigned               count;
           size_t                 *sizes;
        } am_mpools;
    } config;

    /* Configuration of multi-threading support */
    ucp_mt_lock_t                 mt_lock;

    char                          name[UCP_ENTITY_NAME_MAX];

    /* Save cached uct configurations */
    ucs_list_link_t               cached_key_list;
} ucp_context_t;


typedef struct ucp_am_handler {
    uint64_t                      features;
    uct_am_callback_t             cb;
    ucp_am_tracer_t               tracer;
    uint32_t                      flags;
    uct_am_callback_t             proxy_cb;
} ucp_am_handler_t;

typedef struct ucp_tl_iface_atomic_flags {
    struct {
        uint64_t                  op_flags;  /**< Attributes for atomic-post operations */
        uint64_t                  fop_flags; /**< Attributes for atomic-fetch operations */
    } atomic32, atomic64;
} ucp_tl_iface_atomic_flags_t;

#define UCP_ATOMIC_OP_MASK  (UCS_BIT(UCT_ATOMIC_OP_ADD)  | \
                             UCS_BIT(UCT_ATOMIC_OP_AND)  | \
                             UCS_BIT(UCT_ATOMIC_OP_OR)   | \
                             UCS_BIT(UCT_ATOMIC_OP_XOR))

#define UCP_ATOMIC_FOP_MASK (UCS_BIT(UCT_ATOMIC_OP_ADD)  | \
                             UCS_BIT(UCT_ATOMIC_OP_AND)  | \
                             UCS_BIT(UCT_ATOMIC_OP_OR)   | \
                             UCS_BIT(UCT_ATOMIC_OP_XOR)  | \
                             UCS_BIT(UCT_ATOMIC_OP_SWAP) | \
                             UCS_BIT(UCT_ATOMIC_OP_CSWAP))


/*
 * Define UCP active message handler helper macro.
 */
#define _UCP_DEFINE_AM(_features, _id, _cb, _tracer, _flags, _proxy) \
    ucp_am_handler_t ucp_am_handler_##_id  = { \
        .features = _features, \
        .cb       = _cb, \
        .tracer   = _tracer, \
        .flags    = _flags, \
        .proxy_cb = _proxy \
    }


/*
 * Define UCP active message handler.
 */
#define UCP_DEFINE_AM(_features, _id, _cb, _tracer, _flags) \
    _UCP_DEFINE_AM(_features, _id, _cb, _tracer, _flags, NULL)


/**
 * Defines UCP active message handler with proxy handler which counts received
 * messages on ucp_worker_iface_t context. It's used to determine if there is
 * activity on a transport interface.
 */
#define UCP_DEFINE_AM_WITH_PROXY(_features, _id, _cb, _tracer, _flags) \
    \
    static ucs_status_t \
    ucp_am_##_id##_counting_proxy(void *arg, void *data, size_t length, \
                                  unsigned flags) \
    { \
        ucp_worker_iface_t *wiface = arg; \
        wiface->proxy_recv_count++; \
        return _cb(wiface->worker, data, length, flags); \
    } \
    \
    _UCP_DEFINE_AM(_features, _id, _cb, _tracer, _flags, \
                   ucp_am_##_id##_counting_proxy)


#define UCP_CHECK_PARAM_NON_NULL(_param, _status, _action) \
    if ((_param) == NULL) { \
        ucs_error("the parameter %s must not be NULL", #_param); \
        (_status) = UCS_ERR_INVALID_PARAM; \
        _action; \
    };


/**
 * Check if at least one feature flag from @a _flags is initialized.
 */
#define UCP_CONTEXT_CHECK_FEATURE_FLAGS(_context, _flags, _action) \
    do { \
        if (ENABLE_PARAMS_CHECK && \
            ucs_unlikely(!((_context)->config.features & (_flags)))) {  \
            size_t feature_list_str_max = 512; \
            char *feature_list_str = ucs_alloca(feature_list_str_max);  \
            ucs_error("feature flags %s were not set for ucp_init()", \
                      ucs_flags_str(feature_list_str, feature_list_str_max,  \
                                    (_flags) & ~(_context)->config.features, \
                                    ucp_feature_str)); \
            _action; \
        } \
    } while (0)


#define UCP_PARAM_VALUE(_obj, _params, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCS_PP_TOKENPASTE3(UCP_, _obj, _PARAM_FIELD), _params, \
                    _name, _flag, _default)


#define UCP_PARAM_FIELD_VALUE(_params, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCP_PARAM_FIELD, _params, _name, _flag, _default)


#define UCP_ATTR_VALUE(_obj, _attrs, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCS_PP_TOKENPASTE3(UCP_, _obj, _ATTR_FIELD), _attrs, \
                    _name, _flag, _default)


#define ucp_assert_memtype(_context, _buffer, _length, _mem_type) \
    ucs_assert(ucp_memory_type_detect(_context, _buffer, _length) == (_mem_type))


extern ucp_am_handler_t *ucp_am_handlers[];
extern const char       *ucp_feature_str[];


void ucp_dump_payload(ucp_context_h context, char *buffer, size_t max,
                      const void *data, size_t length);

void ucp_context_tag_offload_enable(ucp_context_h context);

void ucp_context_uct_atomic_iface_flags(ucp_context_h context,
                                        ucp_tl_iface_atomic_flags_t *atomic);

const char * ucp_find_tl_name_by_csum(ucp_context_t *context, uint16_t tl_name_csum);

const char *ucp_tl_bitmap_str(ucp_context_h context,
                              const ucp_tl_bitmap_t *tl_bitmap, char *str,
                              size_t max_str_len);

const char* ucp_feature_flags_str(unsigned feature_flags, char *str,
                                  size_t max_str_len);

void ucp_memory_detect_slowpath(ucp_context_h context, const void *address,
                                size_t length, ucs_memory_info_t *mem_info);

/**
 * Calculate a small value to overcome float imprecision
 * between two float values
 */
static UCS_F_ALWAYS_INLINE
double ucp_calc_epsilon(double val1, double val2)
{
    return (val1 + val2) * (1e-6);
}

/**
 * Compare two scores and return:
 * - `-1` if score1 < score2
 * -  `0` if score1 == score2
 * -  `1` if score1 > score2
 */
static UCS_F_ALWAYS_INLINE
int ucp_score_cmp(double score1, double score2)
{
    double diff = score1 - score2;
    return ((fabs(diff) < ucp_calc_epsilon(score1, score2)) ?
            0 : ucs_signum(diff));
}

/**
 * Compare two scores taking into account priorities if scores are equal
 */
static UCS_F_ALWAYS_INLINE
int ucp_score_prio_cmp(double score1, int prio1, double score2, int prio2)
{
    int score_res = ucp_score_cmp(score1, score2);

    return score_res ? score_res : ucs_signum(prio1 - prio2);
}

static UCS_F_ALWAYS_INLINE
int ucp_is_scalable_transport(ucp_context_h context, size_t max_num_eps)
{
    return (max_num_eps >= (size_t)context->config.est_num_eps);
}

static UCS_F_ALWAYS_INLINE double
ucp_tl_iface_latency(ucp_context_h context, const ucs_linear_func_t *latency)
{
    return ucs_linear_func_apply(*latency, context->config.est_num_eps);
}

static UCS_F_ALWAYS_INLINE double
ucp_tl_iface_bandwidth(ucp_context_h context, const uct_ppn_bandwidth_t *bandwidth)
{
    return bandwidth->dedicated +
           (bandwidth->shared / context->config.est_num_ppn);
}

static UCS_F_ALWAYS_INLINE void
ucp_memory_info_set_host(ucp_memory_info_t *mem_info)
{
    mem_info->type    = UCS_MEMORY_TYPE_HOST;
    mem_info->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
}

static UCS_F_ALWAYS_INLINE void
ucp_memory_detect_internal(ucp_context_h context, const void *address,
                           size_t length, ucs_memory_info_t *mem_info)
{
    ucs_status_t status;

    if (ucs_likely(context->num_mem_type_detect_mds == 0)) {
        goto out_host_mem;
    }

    status = ucs_memtype_cache_lookup(address, length, mem_info);
    if (ucs_likely(status == UCS_ERR_NO_ELEM)) {
        ucs_trace_req("address %p length %zu: not found in memtype cache, "
                      "assuming host memory",
                      address, length);
        goto out_host_mem;
    } else if (ucs_likely(status == UCS_OK)) {
        if (ucs_unlikely(mem_info->type == UCS_MEMORY_TYPE_UNKNOWN)) {
            ucs_trace_req(
                    "address %p length %zu: memtype cache returned 'unknown'",
                    address, length);
            ucp_memory_detect_slowpath(context, address, length, mem_info);
        } else {
            ucs_trace_req(
                    "address %p length %zu: memtype cache returned '%s' %s",
                    address, length, ucs_memory_type_names[mem_info->type],
                    ucs_topo_sys_device_get_name(mem_info->sys_dev));
        }
    } else {
        ucp_memory_detect_slowpath(context, address, length, mem_info);
    }

    /* Memory type and system device was detected successfully */
    return;

out_host_mem:
    /* Memory type cache lookup failed - assume it is host memory */
    ucs_memory_info_set_host(mem_info);
}

static UCS_F_ALWAYS_INLINE void
ucp_memory_detect(ucp_context_h context, const void *address, size_t length,
                  ucp_memory_info_t *mem_info)
{
    ucs_memory_info_t mem_info_internal;

    ucp_memory_detect_internal(context, address, length, &mem_info_internal);

    mem_info->type    = mem_info_internal.type;
    mem_info->sys_dev = mem_info_internal.sys_dev;
}


void
ucp_context_dev_tl_bitmap(ucp_context_h context, const char *dev_name,
                          ucp_tl_bitmap_t *tl_bitmap);


void
ucp_context_dev_idx_tl_bitmap(ucp_context_h context, ucp_rsc_index_t dev_idx,
                              ucp_tl_bitmap_t *tl_bitmap);


void ucp_tl_bitmap_validate(const ucp_tl_bitmap_t *tl_bitmap,
                            const ucp_tl_bitmap_t *tl_bitmap_super);


const char* ucp_context_cm_name(ucp_context_h context, ucp_rsc_index_t cm_idx);


ucs_status_t
ucp_config_modify_internal(ucp_config_t *config, const char *name,
                           const char *value);


void ucp_apply_uct_config_list(ucp_context_h context, void *config);


void ucp_context_get_mem_access_tls(ucp_context_h context,
                                    ucs_memory_type_t mem_type,
                                    ucp_tl_bitmap_t *tl_bitmap);

#endif
