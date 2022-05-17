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
#define UCP_PROTO_MAX_PERF_RANGES 24


/* Maximal size of protocol private data */
#define UCP_PROTO_PRIV_MAX          1024


/* Maximal number of protocols in total */
#define UCP_PROTO_MAX_COUNT         64


/* Special value for non-existent protocol */
#define UCP_PROTO_ID_INVALID        ((ucp_proto_id_t)-1)


/* Maximal length of protocol description string */
#define UCP_PROTO_DESC_STR_MAX      64


/* Maximal length of protocol configuration string */
#define UCP_PROTO_CONFIG_STR_MAX    128


/* Protocol identifier */
typedef unsigned ucp_proto_id_t;


/* Bitmap of protocols */
typedef uint64_t ucp_proto_id_mask_t;


/* Protocol stage ID */
enum {
    /* Initial stage. All protocols start from this stage. */
    UCP_PROTO_STAGE_START = 0,

    /* Stage ID must be lower than this value */
    UCP_PROTO_STAGE_LAST  = 8
};


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
    uint16_t                op_flags;   /* Operation flags */
    uint8_t                 dt_class;   /* Datatype */
    uint8_t                 mem_type;   /* Memory type */
    uint8_t                 sys_dev;    /* System device */
    uint8_t                 sg_count;   /* Number of non-contig scatter/gather
                                           entries. If the actual number is larger
                                           than UINT8_MAX, UINT8_MAX is used. */
    uint8_t                 padding;    /* Make structure size be
                                           sizeof(uint64_t) */
} UCS_S_PACKED ucp_proto_select_param_t;


/*
 * Some protocols can be pipelined, so the time they consume when multiple
 * such operations are issued is less than their cumulative time. Therefore we
 * define two metrics: "single" operation time and "multi" operation time.
 *
 * -------time---------->
 *
 *        +-------------------------+
 * op1:   |   "single" time         |
 *        +---------------+---------+---------------+
 *                op2:    | overlap | "multi" time  |
 *                        +---------+-----+---------+---------------+
 *                                op3:    | overlap | "multi" time  |
 *                                        +---------+---------------+
 */
typedef enum {
    /* Time to complete this operation assuming it's the only one. */
    UCP_PROTO_PERF_TYPE_SINGLE,

    /* Time to complete this operation after all previous ones complete. */
    UCP_PROTO_PERF_TYPE_MULTI,

    UCP_PROTO_PERF_TYPE_LAST
} ucp_proto_perf_type_t;


/*
 * Performance estimation for a range of message sizes.
 */
typedef struct {
    /* Protocol name */
    const char        *name;

    /* Maximal payload size for this range */
    size_t            max_length;

    /* Estimated time in seconds, as a function of message size in bytes, to
     * complete the operation. See @ref ucp_proto_perf_type_t for details
     */
    ucs_linear_func_t perf[UCP_PROTO_PERF_TYPE_LAST];
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
    ucp_worker_cfg_index_t         rkey_cfg_index;   /* Remote key configuration index,
                                                        may be UCP_WORKER_CFG_INDEX_NULL */
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


typedef struct {
    /* Protocol definition */
    const ucp_proto_t              *proto;

    /* Protocol private configuration area */
    const void                     *priv;

    /* Worker on which the protocol was initialized */
    ucp_worker_h                   worker;

    /* Protocol selection parameters */
    const ucp_proto_select_param_t *select_param;

    /* Endpoint configuration */
    const ucp_ep_config_key_t      *ep_config_key;

    /* Get information about this message length */
    size_t                         msg_length;
} ucp_proto_query_params_t;


typedef struct {
    /* Maximal message size of the range started from 'msg_length' for
       which the description and configuration information is relevant.
       It must be > msg_length. */
    size_t max_msg_length;

    /* Whether the reported information is not definitive, and the actual used
       protocol depends on remote side decision as well. */
    int    is_estimation;

    /* High-level description of what the protocol is doing in this range */
    char   desc[UCP_PROTO_DESC_STR_MAX];

    /* Protocol configuration in the range, such as devices and transports */
    char   config[UCP_PROTO_CONFIG_STR_MAX];
} ucp_proto_query_attr_t;


/**
 * Query protocol-specific information.
 *
 * @param [in]  params  Protocol information query parameters.
 * @param [out] attr    Protocol information query output.
 *
 * @return Maximal message size for which the returned information is relevant.
 */
typedef void (*ucp_proto_query_func_t)(const ucp_proto_query_params_t *params,
                                       ucp_proto_query_attr_t *attr);


/**
 * Abort UCP request at any stage with error status.
 *
 * @param [in]  request Request to abort.
 * @param [in]  status  Error completion status.
 */
typedef void (*ucp_request_abort_func_t)(ucp_request_t *request,
                                         ucs_status_t status);


/**
 * UCP base protocol definition
 */
struct ucp_proto {
    const char               *name; /* Protocol name */
    const char               *desc; /* Protocol description */
    unsigned                 flags; /* Protocol flags for special handling */
    ucp_proto_init_func_t    init;  /* Initialization function */
    ucp_proto_query_func_t   query; /* Query protocol information */

    /* Initial UCT progress function, can be changed during the protocol
     * request lifetime to implement different stages
     */
    uct_pending_callback_t   progress[UCP_PROTO_STAGE_LAST];

    /*
     * Abort a request (which is currently not scheduled to a pending queue).
     * The method should wait for UCT completions and release associated
     * resources, such as memory handles, remote keys, request ID, etc.
     */
    ucp_request_abort_func_t abort;
};


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
extern const ucp_proto_t *ucp_protocols[];


/* Operations names and descriptions */
extern const char *ucp_operation_names[];
extern const char *ucp_operation_descs[];


/* Performance types names */
extern const char *ucp_proto_perf_types[];


/* Get number of globally registered protocols */
unsigned ucp_protocols_count(void);


/* Default protocol query function: set max_msg_length to SIZE_MAX, take
   description from proto->desc, and set config to an empty string. */
void ucp_proto_default_query(const ucp_proto_query_params_t *params,
                             ucp_proto_query_attr_t *attr);

#endif
