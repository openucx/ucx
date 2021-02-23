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
#define UCP_PROTO_SELECT_OP_ATTR_MASK   UCP_OP_ATTR_FLAG_FAST_CMPL


/** Maximal length of ucp_proto_select_param_str() */
#define UCP_PROTO_SELECT_PARAM_STR_MAX 128


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
    const ucp_proto_threshold_elem_t *thresholds; /* Array of which protocol to use
                                                     for different message sizes */
    const ucp_proto_perf_range_t     *perf_ranges;/* Estimated performance for
                                                     the selected protocols */
    void                             *priv_buf;   /* Private configuration area
                                                     for the selected protocols */
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


void ucp_proto_select_dump(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_t *proto_select,
                           ucs_string_buffer_t *strb);


void ucp_proto_select_dump_short(const ucp_proto_select_short_t *select_short,
                                 const char *name, ucs_string_buffer_t *strb);


void ucp_proto_select_param_str(const ucp_proto_select_param_t *select_param,
                                ucs_string_buffer_t *strb);


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

#endif
