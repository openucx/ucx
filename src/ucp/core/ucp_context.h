/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_CONTEXT_H_
#define UCP_CONTEXT_H_

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/type/component.h>


#define UCP_WORKER_NAME_MAX          32   /* Worker name for debugging */
#define UCP_MIN_BCOPY                64   /* Minimal size for bcopy */

/* Resources */
#define UCP_MAX_RESOURCES            UINT8_MAX
#define UCP_NULL_RESOURCE            ((ucp_rsc_index_t)-1)
typedef uint8_t                      ucp_rsc_index_t;

/* MDs */
#define UCP_UINT_TYPE(_bits)         typedef UCS_PP_TOKENPASTE(UCS_PP_TOKENPASTE(uint, _bits), _t)
#define UCP_MD_INDEX_BITS            8  /* How many bits are in MD index */
#define UCP_MAX_MDS                  (1ul << UCP_MD_INDEX_BITS)
UCP_UINT_TYPE(UCP_MD_INDEX_BITS)     ucp_md_map_t;

/* Lanes */
#define UCP_MAX_LANES                8
#define UCP_NULL_LANE                ((ucp_lane_index_t)-1)
typedef uint8_t                      ucp_lane_index_t;

/* MD-lane map */
#define UCP_MD_LANE_MAP_BITS         64 /* should be UCP_MD_INDEX_BITS * UCP_MAX_LANES */
UCP_UINT_TYPE(UCP_MD_LANE_MAP_BITS)  ucp_md_lane_map_t;

/* Forward declarations */
typedef struct ucp_request           ucp_request_t;
typedef struct ucp_wireup_tl_info    ucp_wireup_iface_attr_t;
typedef struct ucp_address_entry     ucp_address_entry_t;
typedef struct ucp_stub_ep           ucp_stub_ep_t;


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
    UCP_AM_ID_EAGER_SYNC_ACK    =  8, /* Eager-sync acknowldge */

    UCP_AM_ID_LAST
};


typedef struct ucp_context_config {
    /** Threshold for switching UCP to buffered copy(bcopy) protocol */
    size_t                                 bcopy_thresh;
    /** Threshold for switching UCP to rendezvous protocol */
    size_t                                 rndv_thresh;
    /** Threshold for switching UCP to zero copy protocol */
    size_t                                 zcopy_thresh;
    /** Estimation of bcopy bandwidth */
    size_t                                 bcopy_bw;
    /** Size of packet data that is dumped to the log system in debug mode */
    size_t                                 log_data_size;
    /** Maximal size of worker name for debugging */
    unsigned                               max_worker_name;
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
 * UCP context
 */
typedef struct ucp_context {
    uct_md_resource_desc_t        *md_rscs;   /* Memory domain resources */
    uct_md_h                      *mds;       /* Memory domain handles */
    uct_md_attr_t                 *md_attrs;  /* Memory domain attributes */
    ucp_rsc_index_t               num_mds;    /* Number of memory domains */

    ucp_tl_resource_desc_t        *tl_rscs;   /* Array of communication resources */
    ucp_rsc_index_t               num_tls;    /* Number of resources in the array*/

    struct {
        ucs_queue_head_t          expected;   /* Expected requests */
        ucs_queue_head_t          unexpected; /* Unexpected received descriptors */
    } tag;

    struct {

        /* Bitmap of features supported by the context */
        uint64_t                  features;

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

#endif
