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


#define UCP_MAX_RESOURCES       UINT8_MAX
#define UCP_MAX_PDS             (sizeof(uint64_t) * 8)
typedef uint8_t                 ucp_rsc_index_t;


/**
 * Active message codes
 */
enum {
    UCP_AM_ID_WIREUP            =  1, /* Connection establishment */

    UCP_AM_ID_EAGER_ONLY        =  2, /* Single packet eager */
    UCP_AM_ID_EAGER_FIRST       =  3, /* First eager fragment */
    UCP_AM_ID_EAGER_MIDDLE      =  4, /* Middle eager fragment */
    UCP_AM_ID_EAGER_LAST        =  5, /* Last eager fragment */

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
    ucp_rsc_index_t               pd_index; /* Protection domain index (within the context) */
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
    uct_pd_resource_desc_t        *pd_rscs;   /* Protection domain resources */
    uct_pd_h                      *pds;       /* Protection domain handles */
    uct_pd_attr_t                 *pd_attrs;  /* Protection domain attributes */
    ucp_rsc_index_t               num_pds;    /* Number of protection domains */

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

        /* Array of allocation methods, a mix of PD allocation methods and non-PD */
        struct {
            /* Allocation method */
            uct_alloc_method_t    method;

            /* PD name to use, if method is PD */
            char                  pdc_name[UCT_PD_COMPONENT_NAME_MAX];
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
