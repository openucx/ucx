/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
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


/* Maximal number of protocols in total */
#define UCP_PROTO_MAX_COUNT         64


/* Special value for non-existent protocol */
#define UCP_PROTO_ID_INVALID        ((ucp_proto_id_t)-1)


/* Threshold for considering two performance values as equal */
#define UCP_PROTO_PERF_EPSILON     1e-15


/* Maximal length of protocol description string */
#define UCP_PROTO_DESC_STR_MAX      64


/* Maximal length of protocol configuration string */
#define UCP_PROTO_CONFIG_STR_MAX    128


/* Protocol identifier */
typedef unsigned ucp_proto_id_t;


/* Bitmap of protocols */
typedef uint64_t ucp_proto_id_mask_t;


/* Performance calculation tree node */
typedef struct ucp_proto_perf_node ucp_proto_perf_node_t;


/* Key for selecting a protocol */
typedef struct ucp_proto_select_param ucp_proto_select_param_t;


/* Context for probing a protocol for a specific selection key */
typedef struct ucp_proto_probe_ctx ucp_proto_probe_ctx_t;


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
    ucp_proto_id_t                 proto_id;         /* Initial protocol ID */
    ucp_proto_probe_ctx_t          *ctx;             /* Context for adding caps */
} ucp_proto_init_params_t;


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

    /* Map of used lanes */
    ucp_lane_map_t lane_map;
} ucp_proto_query_attr_t;


/**
 * Initialize protocol-specific configuration and estimate protocol performance
 * as function of message size.
 *
 * @param [in]  params  Protocol initialization parameters.
 */
typedef void (*ucp_proto_probe_func_t)(const ucp_proto_init_params_t *params);


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
 * Reset UCP request to its initial state and release any resources related to
 * the specific protocol. Used to switch a send request to a different protocol.
 *
 * @param [in]  request Request to reset.
 *
 * @return UCS_OK           - The request was reset successfully and can be used
 *                            to resend the operation.
 *         UCS_ERR_CANCELED - The request was canceled and released, since it's
 *                            a part of a higher-level operation and should not
 *                            be reused.
 */
typedef ucs_status_t (*ucp_request_reset_func_t)(ucp_request_t *request);


/**
 * UCP base protocol definition
 */
struct ucp_proto {
    const char               *name; /* Protocol name */
    const char               *desc; /* Protocol description */
    unsigned                 flags; /* Protocol flags for special handling */

    /* Probe and add protocol instances */
    ucp_proto_probe_func_t   probe;

    /* Query protocol information */
    ucp_proto_query_func_t   query;

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

    /*
     * Reset a request (which is scheduled to a pending queue).
     * The method should release associated resources, such as memory handles,
     * remote keys, request ID, etc.
     */
    ucp_request_reset_func_t reset;
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
extern const char *ucp_proto_perf_type_names[];


/* Get number of globally registered protocols */
unsigned ucp_protocols_count(void);


/* Default protocol query function: set max_msg_length to SIZE_MAX, take
   description from proto->desc, and set config to an empty string. */
void ucp_proto_default_query(const ucp_proto_query_params_t *params,
                             ucp_proto_query_attr_t *attr);

#endif
