/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_H_
#define UCP_PROTO_H_

#include "lane_type.h"

#include <ucp/core/ucp_types.h>
#include <ucs/datastruct/linear_func.h>
#include <ucs/datastruct/string_buffer.h>


/* Maximal number of lanes per protocol */
#define UCP_PROTO_MAX_LANES         UCP_MAX_LANES


/* Maximal number of protocol performance ranges */
#define UCP_PROTO_MAX_PERF_RANGES   32


/* Maximal size of protocol private data */
#define UCP_PROTO_PRIV_MAX          1024


/* Maximal number of protocols in total */
#define UCP_PROTO_MAX_COUNT         64


/* Special value for non-existent protocol */
#define UCP_PROTO_ID_INVALID        ((ucp_proto_id_t)-1)


/** Maximal length of ucp_proto_config_str_func_t output */
#define UCP_PROTO_CONFIG_STR_MAX 128


/* Protocol identifier */
typedef unsigned ucp_proto_id_t;


/* Bitmap of protocols */
typedef uint64_t ucp_proto_id_mask_t;


/**
 * Protocol flags for internal usage, to allow searching for specific protocols
 */
enum {
    UCP_PROTO_FLAG_AM_SHORT  = UCS_BIT(0), /* The protocol uses only uct_ep_am_short() */
    UCP_PROTO_FLAG_PUT_SHORT = UCS_BIT(1), /* The protocol uses only uct_ep_put_short() */
    UCP_PROTO_FLAG_TAG_SHORT = UCS_BIT(2), /* The protocol uses only
                                              uct_ep_tag_eager_short() */
    UCP_PROTO_FLAG_INVALID   = UCS_BIT(3)  /* The protocol is a placeholder */
};


/**
 * Key for looking up protocol configuration by operation parameters
 */
typedef struct {
    uint8_t                 op_id;      /* Operation ID */
    uint8_t                 op_flags;   /* Operation flags */
    uint8_t                 dt_class;   /* Datatype */
    uint8_t                 mem_type;   /* Memory type */
    uint8_t                 sys_dev;    /* System device */
    uint8_t                 sg_count;   /* Number of non-contig scatter/gather
                                           entries. If the actual number is larger
                                           than UINT8_MAX, UINT8_MAX is used. */
    uint8_t                 padding[2]; /* Make structure size be sizeof(uint64_t) */
} UCS_S_PACKED ucp_proto_select_param_t;


/**
 * Protocol and its private configuration
 */
typedef struct {
    const ucp_proto_t        *proto;       /* Protocol definition */
    const void               *priv;        /* Protocol private configuration space */
    ucp_worker_cfg_index_t   ep_cfg_index; /* Endpoint configuration index this
                                              protocol was selected on */
    ucp_worker_cfg_index_t   rkey_cfg_index; /* Remote key configuration index
                                                this protocol was elected on
                                                (can be UCP_WORKER_CFG_INDEX_NULL) */
    ucp_proto_select_param_t select_param; /* Copy of protocol selection parameters,
                                              used to re-select protocol for existing
                                              in-progress request */
} ucp_proto_config_t;


/*
 * Performance estimation for a range of message sizes
 */
typedef struct {
    size_t                  max_length; /* Maximal message size */
    ucs_linear_func_t       perf;       /* Estimated time in seconds, as a
                                           function of message size in bytes */
} ucp_proto_perf_range_t;


/**
 * UCP protocol capabilities (per operation parameters)
 */
typedef struct {
    size_t                  cfg_thresh;   /* Configured protocol threshold */
    unsigned                cfg_priority; /* Priority of configuration */
    size_t                  min_length;   /* Minimal message size */
    unsigned                num_ranges;   /* Number of entries in 'ranges' */

    /* Performance estimation function for different message sizes */
    ucp_proto_perf_range_t  ranges[UCP_PROTO_MAX_PERF_RANGES];

} ucp_proto_caps_t;


/**
 * Parameters for protocol initialization function
 */
typedef struct {
    /* Input parameters */
    ucp_worker_h                   worker;           /* Worker to initialize on */
    const ucp_proto_select_param_t *select_param;    /* Operation parameters */
    ucp_worker_cfg_index_t         ep_cfg_index;     /* Endpoint configuration index */
    const ucp_ep_config_key_t      *ep_config_key;   /* Endpoint configuration */
    const ucp_rkey_config_key_t    *rkey_config_key; /* Remote key configuration,
                                                        may be NULL */
    const char                     *proto_name;      /* Name of the initialized
                                                        protocol, for debugging */

    /* Output parameters */
    void                           *priv;       /* Pointer to priv buffer */
    size_t                         *priv_size;  /* Occupied size in priv buffer */
    ucp_proto_caps_t               *caps;       /* Protocol capabilities */
} ucp_proto_init_params_t;


/**
 * Initialize protocol-specific configuration and estimate protocol performance
 * as function of message size.
 *
 * @param [in]  params   Protocol initialization parameters.
 *
 * @return UCS_OK              - if successful.
 *         UCS_ERR_UNSUPPORTED - if the protocol is not supported on the key.
 */
typedef ucs_status_t
(*ucp_proto_init_func_t)(const ucp_proto_init_params_t *params);


/**
 * Dump protocol-specific configuration.
 *
 * @param [in]  min_length  Return information starting from this message length.
 * @param [in]  max_length  Return information up to this message length (inclusive).
 * @param [in]  priv        Protocol private data, which was previously filled by
 *                          @ref ucp_proto_init_func_t.
 * @param [out] strb        Protocol configuration text should be written to this
 *                          string buffer. This function should only **append**
 *                          data to the buffer, and should not initialize, release
 *                          or erase any data already in the buffer.
 */
typedef void (*ucp_proto_config_str_func_t)(size_t min_length,
                                            size_t max_length, const void *priv,
                                            ucs_string_buffer_t *strb);


/**
 * UCP base protocol definition
 */
struct ucp_proto {
    const char                      *name;      /* Protocol name */
    unsigned                        flags;      /* Protocol flags for special handling */
    ucp_proto_init_func_t           init;       /* Initialization function */
    ucp_proto_config_str_func_t     config_str; /* Configuration dump function */

    /* Initial UCT progress function, can be changed during the protocol
     * request lifetime to implement different stages
     */
    uct_pending_callback_t          progress;
};


/**
 * Register a protocol definition.
 */
#define UCP_PROTO_REGISTER(_proto) \
    UCS_STATIC_INIT { \
        ucs_assert_always(ucp_protocols_count < UCP_PROTO_MAX_COUNT); \
        ucp_protocols[ucp_protocols_count++] = (_proto); \
    }


/**
 * Retrieve a protocol field by protocol id.
 */
#define ucp_proto_id_field(_proto_id, _field) \
    (ucp_protocols[(_proto_id)]->_field)


/**
 * Call a protocol method by protocol id.
 */
#define ucp_proto_id_call(_proto_id, _func, ...) \
    ucp_proto_id_field(_proto_id, _func)(__VA_ARGS__)


/* Global array of all registered protocols */
extern const ucp_proto_t *ucp_protocols[UCP_PROTO_MAX_COUNT];

/* Number of globally registered protocols */
extern unsigned ucp_protocols_count;


#endif
