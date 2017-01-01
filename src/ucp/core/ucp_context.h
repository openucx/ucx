/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_CONTEXT_H_
#define UCP_CONTEXT_H_

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/type/component.h>
#include <ucs/type/spinlock.h>
#include "config.h"

typedef enum ucp_mt_type {
    UCP_MT_TYPE_NONE = 0,
    UCP_MT_TYPE_SPINLOCK,
    UCP_MT_TYPE_MUTEX
} ucp_mt_type_t;

typedef struct ucp_mt_lock {
    ucp_mt_type_t                 mt_type;
    union {
        /* Lock for multithreading support. Either spinlock or mutex is used at
           at one time. Spinlock is the default option. */
        pthread_mutex_t           mt_mutex;
        ucs_spinlock_t            mt_spinlock;
    } lock;
} ucp_mt_lock_t;

#if ENABLE_MT
#define UCP_THREAD_IS_REQUIRED(_lock_ptr) ((_lock_ptr)->mt_type)
#define UCP_THREAD_LOCK_INIT(_lock_ptr)                                 \
    {                                                                   \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) {                \
            pthread_mutex_init(&((_lock_ptr)->lock.mt_mutex), NULL);    \
        } else {                                                        \
            ucs_spinlock_init(&((_lock_ptr)->lock.mt_spinlock));        \
        }                                                               \
    }
#define UCP_THREAD_LOCK_FINALIZE(_lock_ptr)                             \
    {                                                                   \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) {                \
            pthread_mutex_destroy(&((_lock_ptr)->lock.mt_mutex));       \
        } else {                                                        \
            ucs_spinlock_destroy(&((_lock_ptr)->lock.mt_spinlock));     \
        }                                                               \
    }
#define UCP_THREAD_CS_ENTER(_lock_ptr)                                  \
    {                                                                   \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) {                \
            pthread_mutex_lock(&((_lock_ptr)->lock.mt_mutex));          \
        } else {                                                        \
            ucs_spin_lock(&((_lock_ptr)->lock.mt_spinlock));            \
        }                                                               \
    }
#define UCP_THREAD_CS_EXIT(_lock_ptr)                                   \
    {                                                                   \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) {                \
            pthread_mutex_unlock(&((_lock_ptr)->lock.mt_mutex));        \
        } else {                                                        \
            ucs_spin_unlock(&((_lock_ptr)->lock.mt_spinlock));          \
        }                                                               \
    }
#define UCP_THREAD_CS_YIELD(_lock_ptr)                                  \
    {                                                                   \
        UCP_THREAD_CS_EXIT(_lock_ptr);                                  \
        sched_yield();                                                  \
        UCP_THREAD_CS_ENTER(_lock_ptr);                                 \
    }
#define UCP_THREAD_LOCK_INIT_CONDITIONAL(_lock_ptr)                     \
    {                                                                   \
        if (UCP_THREAD_IS_REQUIRED(_lock_ptr)) {                        \
            UCP_THREAD_LOCK_INIT(_lock_ptr);                            \
        }                                                               \
    }
#define UCP_THREAD_LOCK_FINALIZE_CONDITIONAL(_lock_ptr)                 \
    {                                                                   \
        if (UCP_THREAD_IS_REQUIRED(_lock_ptr)) {                        \
            UCP_THREAD_LOCK_FINALIZE(_lock_ptr);                        \
        }                                                               \
    }
#define UCP_THREAD_CS_ENTER_CONDITIONAL(_lock_ptr)                      \
    {                                                                   \
        if (UCP_THREAD_IS_REQUIRED(_lock_ptr)) {                        \
            UCP_THREAD_CS_ENTER(_lock_ptr);                             \
        }                                                               \
    }
#define UCP_THREAD_CS_EXIT_CONDITIONAL(_lock_ptr)                       \
    {                                                                   \
        if (UCP_THREAD_IS_REQUIRED(_lock_ptr)) {                        \
            UCP_THREAD_CS_EXIT(_lock_ptr);                              \
        }                                                               \
    }
#define UCP_THREAD_CS_YIELD_CONDITIONAL(_lock_ptr)                      \
    {                                                                   \
        if (UCP_THREAD_IS_REQUIRED(_lock_ptr)) {                        \
            UCP_THREAD_CS_EXIT_CONDITIONAL(_lock_ptr);                  \
            sched_yield();                                              \
            UCP_THREAD_CS_ENTER_CONDITIONAL(_lock_ptr);                 \
        }                                                               \
    }
#else
#define UCP_THREAD_IS_REQUIRED(_lock_ptr)                0
#define UCP_THREAD_LOCK_INIT(_lock_ptr)                  {}
#define UCP_THREAD_LOCK_FINALIZE(_lock_ptr)              {}
#define UCP_THREAD_CS_ENTER(_lock_ptr)                   {}
#define UCP_THREAD_CS_EXIT(_lock_ptr)                    {}
#define UCP_THREAD_CS_YIELD(_lock_ptr)                   {}
#define UCP_THREAD_LOCK_INIT_CONDITIONAL(_lock_ptr)      {}
#define UCP_THREAD_LOCK_FINALIZE_CONDITIONAL(_lock_ptr)  {}
#define UCP_THREAD_CS_ENTER_CONDITIONAL(_lock_ptr)       {}
#define UCP_THREAD_CS_EXIT_CONDITIONAL(_lock_ptr)        {}
#define UCP_THREAD_CS_YIELD_CONDITIONAL(_lock_ptr)       {}
#endif

#define UCP_WORKER_NAME_MAX          32   /* Worker name for debugging */
#define UCP_MIN_BCOPY                64   /* Minimal size for bcopy */

/* Resources */
#define UCP_MAX_RESOURCES            UINT8_MAX
#define UCP_NULL_RESOURCE            ((ucp_rsc_index_t)-1)
typedef uint8_t                      ucp_rsc_index_t;

/* MDs */
#define UCP_UINT_TYPE(_bits)         typedef UCS_PP_TOKENPASTE(UCS_PP_TOKENPASTE(uint, _bits), _t)
#define UCP_MD_INDEX_BITS            8  /* How many bits are in MD index */
#define UCP_MAX_MDS                  UCP_MAX_RESOURCES
UCP_UINT_TYPE(UCP_MD_INDEX_BITS)     ucp_md_map_t;

/* Lanes */
#define UCP_MAX_LANES                8
#define UCP_NULL_LANE                ((ucp_lane_index_t)-1)
typedef uint8_t                      ucp_lane_index_t;
UCP_UINT_TYPE(UCP_MAX_LANES)         ucp_lane_map_t;

/* MD-lane map */
#define UCP_MD_LANE_MAP_BITS         64 /* should be UCP_MD_INDEX_BITS * UCP_MAX_LANES */
UCP_UINT_TYPE(UCP_MD_LANE_MAP_BITS)  ucp_md_lane_map_t;

/* Forward declarations */
typedef struct ucp_request              ucp_request_t;
typedef struct ucp_address_iface_attr   ucp_address_iface_attr_t;
typedef struct ucp_address_entry        ucp_address_entry_t;
typedef struct ucp_stub_ep              ucp_stub_ep_t;


/**
 * Active message codes
 */
enum {
    UCP_AM_ID_WIREUP            =  1, /* Connection establishment */

    UCP_AM_ID_EAGER_ONLY        =  2, /* Single packet eager */
    UCP_AM_ID_EAGER_FIRST       =  3, /* First eager fragment */
    UCP_AM_ID_EAGER_MIDDLE      =  4, /* Middle eager fragment */
    UCP_AM_ID_EAGER_LAST        =  5, /* Last eager fragment */

    UCP_AM_ID_EAGER_SYNC_ONLY   =  6, /* Single packet eager-sync */
    UCP_AM_ID_EAGER_SYNC_FIRST  =  7, /* First eager-sync fragment */
    UCP_AM_ID_EAGER_SYNC_ACK    =  8, /* Eager-sync acknowledge */

    UCP_AM_ID_RNDV_RTS          =  9, /* Ready-to-Send to init rendezvous */
    UCP_AM_ID_RNDV_ATS          =  10, /* Ack-to-Send after finishing a get operation */
    UCP_AM_ID_RNDV_RTR          =  11, /* Ready-to-Receive rendezvous for a receiver
                                          with a generic datatype */
    UCP_AM_ID_RNDV_DATA         =  12, /* Rndv data fragments when using software
                                          rndv (bcopy) */
    UCP_AM_ID_RNDV_DATA_LAST    =  13, /* The last rndv data fragment when using
                                          software rndv (bcopy) */
    UCP_AM_ID_LAST
};


/**
 * Atomic operations mode.
 */
typedef enum {
    UCP_ATOMIC_MODE_CPU,     /* Use CPU-based atomics */
    UCP_ATOMIC_MODE_DEVICE,  /* Use device-based atomics */
    UCP_ATOMIC_MODE_GUESS,   /* If all transports support CPU AMOs only (no DEVICE),
                              * the CPU is selected, otherwise DEVICE is selected */
    UCP_ATOMIC_MODE_LAST
} ucp_atomic_mode_t;


typedef struct ucp_context_config {
    /** Threshold for switching UCP to buffered copy(bcopy) protocol */
    size_t                                 bcopy_thresh;
    /** Threshold for switching UCP to rendezvous protocol */
    size_t                                 rndv_thresh;
    /** Threshold for switching UCP to rendezvous protocol in case the calculated
     *  threshold is zero or negative */
    size_t                                 rndv_thresh_fallback;
    /** Threshold for switching UCP to zero copy protocol */
    size_t                                 zcopy_thresh;
    /** Estimation of bcopy bandwidth */
    size_t                                 bcopy_bw;
    /** Size of packet data that is dumped to the log system in debug mode */
    size_t                                 log_data_size;
    /** Maximal size of worker name for debugging */
    unsigned                               max_worker_name;
    /** Atomic mode */
    ucp_atomic_mode_t                      atomic_mode;
    /** If use mutex for MT support or not */
    int                                    use_mt_mutex;
} ucp_context_config_t;


struct ucp_config {
    /** Array of device lists names to use.
     *  This array holds three lists - network devices, shared memory devices
     *  and acceleration devices */
    str_names_array_t                      devices[UCT_DEVICE_TYPE_LAST];
    /** Array of transport names to use */
    str_names_array_t                      tls;
    /** Array of memory allocation methods */
    UCS_CONFIG_STRING_ARRAY_FIELD(methods) alloc_prio;
    /** Configuration saved directly in the context */
    ucp_context_config_t                   ctx;
};


/**
 * Active message tracer.
 */
typedef void (*ucp_am_tracer_t)(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max);


/**
 * UCP communication resource descriptor
 */
typedef struct ucp_tl_resource_desc {
    uct_tl_resource_desc_t        tl_rsc;   /* UCT resource descriptor */
    ucp_rsc_index_t               md_index; /* Memory domain index (within the context) */
    uint16_t                      tl_name_csum; /* Checksum of transport name */
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
    ucp_rsc_index_t               max_rkey_md;/* Maximal MD index with rkey */

    ucp_tl_resource_desc_t        *tl_rscs;   /* Array of communication resources */
    ucp_rsc_index_t               num_tls;    /* Number of resources in the array*/

    struct {
        ucs_queue_head_t          expected;   /* Expected requests */
        ucs_queue_head_t          unexpected; /* Unexpected received descriptors */
    } tag;

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


extern ucp_am_handler_t ucp_am_handlers[];


void ucp_dump_payload(ucp_context_h context, char *buffer, size_t max,
                      const void *data, size_t length);

uint64_t ucp_context_uct_atomic_iface_flags(ucp_context_h context);


static inline double ucp_tl_iface_latency(ucp_context_h context,
                                          const uct_iface_attr_t *iface_attr)
{
    return iface_attr->latency.overhead +
           (iface_attr->latency.growth * context->config.est_num_eps);
}

#endif
