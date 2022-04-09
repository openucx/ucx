/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_SELECT_H_
#define UCP_PROTO_SELECT_H_

#include "proto.h"

#include <ucs/datastruct/khash.h>


/**
 * Some flags from ucp_request_param_t.op_attr_mask can affect protocol
 * selection decision.
 */
#define UCP_PROTO_SELECT_OP_ATTR_BASE   UCP_OP_ATTR_FLAG_NO_IMM_CMPL
#define UCP_PROTO_SELECT_OP_ATTR_MASK   (UCP_OP_ATTR_FLAG_FAST_CMPL | \
                                         UCP_OP_ATTR_FLAG_MULTI_SEND)
#define UCP_PROTO_SELECT_OP_FLAGS_BASE  UCS_BIT(5)


/* Select a protocol for sending one fragment of a rendezvous pipeline */
#define UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG (UCP_PROTO_SELECT_OP_FLAGS_BASE << 0)


/* Select a protocol as part of performance estimation of another protocol,
   rather for actually sending a request */
#define UCP_PROTO_SELECT_OP_FLAG_INTERNAL (UCP_PROTO_SELECT_OP_FLAGS_BASE << 1)


/* Select eager/rendezvous protocol for Active Message sends */
#define UCP_PROTO_SELECT_OP_FLAG_AM_EAGER (UCP_PROTO_SELECT_OP_FLAGS_BASE << 2)
#define UCP_PROTO_SELECT_OP_FLAG_AM_RNDV  (UCP_PROTO_SELECT_OP_FLAGS_BASE << 3)


/** Maximal length of ucp_proto_select_param_str() */
#define UCP_PROTO_SELECT_PARAM_STR_MAX 128


typedef struct {
    ucp_proto_perf_range_t super;
    size_t                 cfg_thresh; /* Configured protocol threshold */
} ucp_proto_select_range_t;


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
    const ucp_proto_threshold_elem_t *thresholds;

    /* Estimated performance for the selected protocols */
    const ucp_proto_select_range_t   *perf_ranges;

    /* Private configuration area for the selected protocols */
    void                             *priv_buf;
} ucp_proto_select_elem_t;


/* Hash type of mapping a buffer-type (key) to a protocol selection */
KHASH_TYPE(ucp_proto_select_hash, khint64_t, ucp_proto_select_elem_t)


/**
 * Top-level data structure to select protocols for various buffer types
 */
typedef struct {
    /* Lookup from protocol selection key to thresholds array */
    khash_t(ucp_proto_select_hash)    hash;

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


ucp_proto_select_elem_t *
ucp_proto_select_lookup_slow(ucp_worker_h worker,
                             ucp_proto_select_t *proto_select,
                             ucp_worker_cfg_index_t ep_cfg_index,
                             ucp_worker_cfg_index_t rkey_cfg_index,
                             const ucp_proto_select_param_t *select_param);


const ucp_proto_threshold_elem_t*
ucp_proto_thresholds_search_slow(const ucp_proto_threshold_elem_t *thresholds,
                                 size_t msg_length);


void ucp_proto_select_short_disable(ucp_proto_select_short_t *proto_short);


void
ucp_proto_select_short_init(ucp_worker_h worker, ucp_proto_select_t *proto_select,
                            ucp_worker_cfg_index_t ep_cfg_index,
                            ucp_worker_cfg_index_t rkey_cfg_index,
                            ucp_operation_id_t op_id, uint32_t op_attr_mask,
                            unsigned proto_flags,
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
