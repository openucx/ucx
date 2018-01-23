/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_CONTEXT_H_
#define UCP_CONTEXT_H_

#include "ucp_types.h"
#include "ucp_thread.h"

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/type/component.h>
#include <ucs/type/spinlock.h>


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
    /** Threshold for switching UCP to zero copy protocol */
    size_t                                 zcopy_thresh;
    /** Communication scheme in RNDV protocol */
    ucp_rndv_mode_t                        rndv_mode;
    /** Estimation of bcopy bandwidth */
    size_t                                 bcopy_bw;
    /** Segment size in the worker pre-registered memory pool */
    size_t                                 seg_size;
    /** RNDV pipeline fragment size */
    size_t                                 rndv_frag_size;
    /** Threshold for using tag matching offload capabilities. Smaller buffers
     *  will not be posted to the transport. */
    size_t                                 tm_thresh;
    /** Threshold for forcing tag matching offload capabilities */
    size_t                                 tm_force_thresh;
    /** Upper bound for posting tm offload receives with internal UCP
     *  preregistered bounce buffers. */
    size_t                                 tm_max_bcopy;
    /** Maximal size of worker name for debugging */
    unsigned                               max_worker_name;
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
} ucp_context_config_t;


struct ucp_config {
    /** Array of device lists names to use.
     *  This array holds three lists - network devices, shared memory devices
     *  and acceleration devices */
    ucs_config_names_array_t               devices[UCT_DEVICE_TYPE_LAST];
    /** Array of transport names to use */
    ucs_config_names_array_t               tls;
    /** Array of memory allocation methods */
    UCS_CONFIG_STRING_ARRAY_FIELD(methods) alloc_prio;
    /** Configuration saved directly in the context */
    ucp_context_config_t                   ctx;
};


/**
 * UCP communication resource descriptor
 */
typedef struct ucp_tl_resource_desc {
    uct_tl_resource_desc_t        tl_rsc;     /* UCT resource descriptor */
    uint16_t                      tl_name_csum; /* Checksum of transport name */
    ucp_rsc_index_t               md_index;   /* Memory domain index (within the context) */
    ucp_rsc_index_t               dev_index;  /* Arbitrary device index. Resources
                                                 with same index have same device name. */
    uint8_t                       flags;      /* Flags that describe resource specifics */
} ucp_tl_resource_desc_t;


/**
 * Transport aliases.
 */
typedef struct ucp_tl_alias {
    const char                    *alias;   /* Alias name */
    const char*                   tls[8];   /* Transports which are selected by the alias */
} ucp_tl_alias_t;


/**
 * Memory domain.
 */
typedef struct ucp_tl_md {
    uct_md_h                      md;       /* Memory domain handle */
    uct_md_resource_desc_t        rsc;      /* Memory domain resource */
    uct_md_attr_t                 attr;     /* Memory domain attributes */
} ucp_tl_md_t;


/**
 * UCP context
 */
typedef struct ucp_context {
    ucp_tl_md_t                   *tl_mds;    /* Memory domain resources */
    ucp_rsc_index_t               num_mds;    /* Number of memory domains */

    /* List of MDs which detect non host memory type */
    ucp_rsc_index_t               mem_type_tl_mds[UCT_MD_MEM_TYPE_LAST];
    ucp_rsc_index_t               num_mem_type_mds;  /* Number of mem type MDs */

    ucp_tl_resource_desc_t        *tl_rscs;   /* Array of communication resources */
    ucp_rsc_index_t               num_tls;    /* Number of resources in the array*/

    /* Mask of memory type communication resources */
    uint64_t                      mem_type_tls[UCT_MD_MEM_TYPE_LAST];

    ucs_mpool_t                   rkey_mp;    /* Pool for memory keys */

    struct {

        /* Bitmap of features supported by the context */
        uint64_t                  features;
        uint64_t                  tag_sender_mask;

        /* How many endpoints are expected to be created */
        int                       est_num_eps;

        struct {
            size_t                         size;    /* Request size for user */
            ucp_request_init_callback_t    init;    /* Initialization user callback */
            ucp_request_cleanup_callback_t cleanup; /* Cleanup user callback */
        } request;

        /* Array of allocation methods, a mix of MD allocation methods and non-MD */
        struct {
            /* Allocation method */
            uct_alloc_method_t    method;

            /* MD name to use, if method is MD */
            char                  mdc_name[UCT_MD_COMPONENT_NAME_MAX];
        } *alloc_methods;
        unsigned                  num_alloc_methods;

        /* Configuration supplied by the user */
        ucp_context_config_t      ext;

    } config;

    /* All configurations about multithreading support */
    ucp_mt_lock_t                 mt_lock;

} ucp_context_t;


typedef struct ucp_am_handler {
    uint64_t                      features;
    uct_am_callback_t             cb;
    ucp_am_tracer_t               tracer;
    uint32_t                      flags;
    uct_am_callback_t             proxy_cb;
} ucp_am_handler_t;


/*
 * Define UCP active message handler.
 */
#define UCP_DEFINE_AM(_features, _id, _cb, _tracer, _flags) \
    UCS_STATIC_INIT { \
        ucp_am_handlers[_id].features = _features; \
        ucp_am_handlers[_id].cb       = _cb; \
        ucp_am_handlers[_id].tracer   = _tracer; \
        ucp_am_handlers[_id].flags    = _flags; \
    }


/**
 * Defines a proxy handler which counts received messages on ucp_worker_iface_t
 * context. It's used to determine if there is activity on a transport interface.
 */
#define UCP_DEFINE_AM_PROXY(_id) \
    \
    static ucs_status_t \
    ucp_am_##_id##_counting_proxy(void *arg, void *data, size_t length, \
                                  unsigned flags) \
    { \
        ucp_worker_iface_t *wiface = arg; \
        wiface->proxy_recv_count++; \
        return ucp_am_handlers[_id].cb(wiface->worker, data, length, flags); \
    } \
    \
    UCS_STATIC_INIT { \
        ucp_am_handlers[_id].proxy_cb = ucp_am_##_id##_counting_proxy; \
    }


#define UCP_CHECK_PARAM_NON_NULL(_param, _status, _action) \
    if ((_param) == NULL) { \
        ucs_error("the parameter %s must not be NULL", #_param); \
        (_status) = UCS_ERR_INVALID_PARAM; \
        _action; \
    };


#define UCP_PARAM_VALUE(_obj, _params, _name, _flag, _default) \
    (((_params)->field_mask & (UCP_##_obj##_PARAM_FIELD_##_flag)) ? \
                    (_params)->_name : (_default))


extern ucp_am_handler_t ucp_am_handlers[];


void ucp_dump_payload(ucp_context_h context, char *buffer, size_t max,
                      const void *data, size_t length);

void ucp_context_tag_offload_enable(ucp_context_h context);

uint64_t ucp_context_uct_atomic_iface_flags(ucp_context_h context);

const char * ucp_find_tl_name_by_csum(ucp_context_t *context, uint16_t tl_name_csum);

static UCS_F_ALWAYS_INLINE double
ucp_tl_iface_latency(ucp_context_h context, const uct_iface_attr_t *iface_attr)
{
    return iface_attr->latency.overhead +
           (iface_attr->latency.growth * context->config.est_num_eps);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_memory_type_detect_mds(ucp_context_h context, void *addr, size_t length,
                           uct_memory_type_t *mem_type_p)
{
    unsigned i, md_index;

    for (i = 0; i < context->num_mem_type_mds; ++i) {
        md_index = context->mem_type_tl_mds[i];
        if (uct_md_is_mem_type_owned(context->tl_mds[md_index].md, addr, length)) {
            *mem_type_p = context->tl_mds[md_index].attr.cap.mem_type;
            return UCS_OK;
        }
    }
    *mem_type_p = UCT_MD_MEM_TYPE_HOST;
    return UCS_OK;
}

#endif
