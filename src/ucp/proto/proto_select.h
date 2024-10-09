/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_SELECT_H_
#define UCP_PROTO_SELECT_H_

#include "proto.h"
#include "proto_perf.h"

#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/array.h>


/**
 * Some flags from ucp_request_param_t.op_attr_mask can affect protocol
 * selection decision.
 */
#define UCP_PROTO_SELECT_OP_ATTR_BASE   UCP_OP_ATTR_FLAG_NO_IMM_CMPL
#define UCP_PROTO_SELECT_OP_ATTR_MASK   (UCP_OP_ATTR_FLAG_FAST_CMPL | \
                                         UCP_OP_ATTR_FLAG_MULTI_SEND)

/* Operation flags start bit */
#define UCP_PROTO_SELECT_OP_FLAGS_BASE UCS_BIT(4)

/* Select protocol which supports resume request after reset. */
#define UCP_PROTO_SELECT_OP_FLAG_RESUME (UCP_PROTO_SELECT_OP_FLAGS_BASE << 0)

/* Select a protocol for sending one fragment of a rendezvous pipeline.
 * Relevant for UCP_OP_ID_RNDV_SEND and UCP_OP_ID_RNDV_RECV. */
#define UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG (UCP_PROTO_SELECT_OP_FLAGS_BASE << 1)


/* Select eager/rendezvous protocol for Active Message sends.
 * Relevant for UCP_OP_ID_AM_SEND and UCP_OP_ID_AM_SEND_REPLY. */
#define UCP_PROTO_SELECT_OP_FLAG_AM_EAGER (UCP_PROTO_SELECT_OP_FLAGS_BASE << 1)
#define UCP_PROTO_SELECT_OP_FLAG_AM_RNDV  (UCP_PROTO_SELECT_OP_FLAGS_BASE << 2)


/** Maximal length of ucp_proto_select_param_str() */
#define UCP_PROTO_SELECT_PARAM_STR_MAX 128


typedef struct {
    ucp_proto_id_t        proto_id;
    size_t                priv_offset;
    size_t                cfg_thresh; /* Configured protocol threshold */
    unsigned              cfg_priority; /* Priority of configuration */
    ucp_proto_perf_t      *perf;
    ucp_proto_flat_perf_t *flat_perf; /* Flat performance considering all parts */
} ucp_proto_init_elem_t;


/* Parameters structure for initializing protocols for a selection parameter */
struct ucp_proto_probe_ctx {
    ucs_array_s(size_t, uint8_t)                 priv_buf;
    ucs_array_s(unsigned, ucp_proto_init_elem_t) protocols;
};


typedef ucp_proto_probe_ctx_t ucp_proto_select_init_protocols_t;


/**
 * Key for looking up protocol configuration by operation parameters
 */
struct ucp_proto_select_param {
    uint8_t                 op_id_flags;/* Operation ID and flags */
    uint8_t                 op_attr;    /* Operation attributes from params */
    uint8_t                 dt_class;   /* Datatype */
    uint8_t                 mem_type;   /* Memory type */
    uint8_t                 sys_dev;    /* System device */
    uint8_t                 sg_count;   /* Number of non-contig scatter/gather
                                           entries. If the actual number is larger
                                           than UINT8_MAX, UINT8_MAX is used. */
    union {
        /* Reply buffer parameters.
         * Used for UCP_OP_ID_AMO_FETCH and UCP_OP_ID_AMO_CSWAP.
         */
        struct {
            uint8_t         mem_type;   /* Reply buffer memory type */
            uint8_t         sys_dev;    /* Reply buffer system device */
        } UCS_S_PACKED reply;

        /* Align struct size to uint64_t */
        uint8_t             padding[2];

    } UCS_S_PACKED op;
} UCS_S_PACKED;


/**
 * Protocol and its private configuration
 */
typedef struct {
    /* Protocol definition */
    const ucp_proto_t        *proto;

    /* Protocol private configuration space */
    const void               *priv;

    /* Endpoint configuration index this protocol was selected on */
    ucp_worker_cfg_index_t   ep_cfg_index;

    /* Remote key configuration index this protocol was selected on (can be
     * UCP_WORKER_CFG_INDEX_NULL)
     */
    ucp_worker_cfg_index_t   rkey_cfg_index;

    /* Copy of protocol selection parameters, used to re-select protocol for
     * existing in-progress request
     */
    ucp_proto_select_param_t select_param;

    /* Pointer to the corresponding initialization data */
    const ucp_proto_init_elem_t *init_elem;
} ucp_proto_config_t;


/**
 * Entry which defines which protocol should be used for a message size range.
 */
typedef struct {
    ucp_proto_config_t          proto_config;   /* Protocol configuration to use */
    size_t                      max_msg_length; /* Max message length, inclusive */
} ucp_proto_threshold_elem_t;


/**
 * Protocol selection per a particular buffer type and operation
 */
typedef struct {
    /* Array of which protocol to use for different message sizes */
    const ucp_proto_threshold_elem_t  *thresholds;

    /* All the initialized protocols that can be chosen */
    ucp_proto_select_init_protocols_t proto_init;
} ucp_proto_select_elem_t;


/* Hash type of mapping a buffer-type (key) to a protocol selection */
KHASH_TYPE(ucp_proto_select_hash, khint64_t, ucp_proto_select_elem_t)


/**
 * Top-level data structure to select protocols for various buffer types
 */
typedef struct {
    /* Lookup from protocol selection key to thresholds array */
    khash_t(ucp_proto_select_hash)    *hash;

    /* cache the last used protocol, for fast lookup */
    struct {
        uint64_t                      key;
        const ucp_proto_select_elem_t *value;
    } cache;
} ucp_proto_select_t;


/*
 * Settings for short protocol
 */
typedef struct {
    ssize_t             max_length_host_mem;    /* max length of short protocol for
                                                   host memory buffer */
    ssize_t             max_length_unknown_mem; /* max length of short protocol
                                                   for unknown memory buffer */
    ucp_lane_index_t    lane;                   /* lane for sending short message */
    ucp_md_index_t      rkey_index;             /* uct rkey index (for put_short) */
} ucp_proto_select_short_t;


ucs_status_t ucp_proto_select_init(ucp_proto_select_t *proto_select);


void ucp_proto_select_cleanup(ucp_proto_select_t *proto_select);


void ucp_proto_select_add_proto(const ucp_proto_init_params_t *init_params,
                                size_t cfg_thresh, unsigned cfg_priority,
                                ucp_proto_perf_t *perf, const void *priv,
                                size_t priv_size);


ucp_proto_select_elem_t *
ucp_proto_select_lookup_slow(ucp_worker_h worker,
                             ucp_proto_select_t *proto_select, int internal,
                             ucp_worker_cfg_index_t ep_cfg_index,
                             ucp_worker_cfg_index_t rkey_cfg_index,
                             const ucp_proto_select_param_t *select_param);


const ucp_proto_threshold_elem_t*
ucp_proto_thresholds_search_slow(const ucp_proto_threshold_elem_t *thresholds,
                                 size_t msg_length);


void ucp_proto_select_short_disable(ucp_proto_select_short_t *proto_short);


void ucp_proto_select_short_init(ucp_worker_h worker,
                                 ucp_proto_select_t *proto_select,
                                 ucp_worker_cfg_index_t ep_cfg_index,
                                 ucp_worker_cfg_index_t rkey_cfg_index,
                                 ucp_operation_id_t op_id, unsigned proto_flags,
                                 ucp_proto_select_short_t *proto_short);


int ucp_proto_select_get_valid_range(
        const ucp_proto_threshold_elem_t *thresholds, size_t *min_length_p,
        size_t *max_length_p);


/* Get the protocol selection hash for the endpoint or remote key config */
ucp_proto_select_t *
ucp_proto_select_get(ucp_worker_h worker, ucp_worker_cfg_index_t ep_cfg_index,
                     ucp_worker_cfg_index_t rkey_cfg_index,
                     ucp_worker_cfg_index_t *new_rkey_cfg_index);


void ucp_proto_config_query(ucp_worker_h worker,
                            const ucp_proto_config_t *proto_config,
                            size_t msg_length,
                            ucp_proto_query_attr_t *proto_attr);


int ucp_proto_select_elem_query(ucp_worker_h worker,
                                const ucp_proto_select_elem_t *select_elem,
                                size_t msg_length,
                                ucp_proto_query_attr_t *proto_attr);

#endif
