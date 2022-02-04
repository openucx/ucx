/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_MULTI_H_
#define UCP_PROTO_MULTI_H_

#include "proto.h"
#include "proto_common.h"

#include <ucp/dt/datatype_iter.h>


/* ucp_proto_multi_lane_priv_t.weight is shifted by this value */
#define UCP_PROTO_MULTI_WEIGHT_SHIFT 16
#define UCP_PROTO_MULTI_WEIGHT_MAX   UCS_BIT(UCP_PROTO_MULTI_WEIGHT_SHIFT)


/**
 * UCP base protocol definition for multi-fragment protocols
 */
typedef struct ucp_proto_send_multi {
    ucp_proto_t                    super;
    ptrdiff_t                      iface_fragsz_offset; /* Offset of fragment
                                                           size in iface attr */
} ucp_proto_send_multi_t;


/*
 * One lane configuration for multi-lane protocol
 */
typedef struct {
    ucp_proto_common_lane_priv_t super;

    /* Maximal fragment size on this lane */
    size_t                       max_frag;

    /* Sum of 'max_frag' on all previous lanes, inclusive */
    size_t                       max_frag_sum;

    /* Ratio of data to send on this lane.
     * This is a fixed-point numeric representation (n * 2^shift), where "n" is
     * the real value, and "shift" is defined by UCP_PROTO_MULTI_WEIGHT_SHIFT.
     */
    uint32_t                     weight;

    /* Sum of 'weight' on all previous lanes, inclusive */
    uint32_t                     weight_sum;
} ucp_proto_multi_lane_priv_t;


/*
 * Base class for protocols with fragmentation
 */
typedef struct {
    ucp_md_map_t                reg_md_map;   /* Memory domains to register on */
    size_t                      min_frag;     /* Largest minimal fragment size */
    size_t                      max_frag_sum; /* 'max_frag' sum of all lanes */
    ucp_lane_map_t              lane_map;     /* Map of used lanes */
    ucp_lane_index_t            num_lanes;    /* Number of lanes to use */
    ucp_proto_multi_lane_priv_t lanes[0];     /* Array of lanes */
} ucp_proto_multi_priv_t;


/**
 * Initialization parameters for multi-lane protocol
 */
typedef struct {
    ucp_proto_common_init_params_t super;

    /* Maximal number of lanes to select */
    ucp_lane_index_t               max_lanes;

    /* MDs on which the buffer is expected to be already registered, so no need
       to account for the overhead of registering on them */
    ucp_md_map_t                   initial_reg_md_map;
    struct {
        /* Required iface capabilities */
        uint64_t        tl_cap_flags;

        /* Required lane type */
        ucp_lane_type_t lane_type;
    } first, middle;
} ucp_proto_multi_init_params_t;


/**
 * Context for ucp_proto_multi_data_pack()
 */
typedef struct {
    ucp_request_t                  *req;
    size_t                         max_payload;
    ucp_datatype_iter_t            *next_iter;
} ucp_proto_multi_pack_ctx_t;


typedef ucs_status_t (*ucp_proto_send_multi_cb_t)(
                ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
                ucp_datatype_iter_t *next_iter);


/**
 * Send callback for lane-map multi-send protocol
 *
 * @param [in] req   Request to send.
 * @param [in] lane  Endpoint lane index to send on.
 *
 * @return Send operation status, using same semantics as returned from UCT send
 *         functions.
 */
typedef ucs_status_t (*ucp_proto_multi_lane_send_func_t)(ucp_request_t *req,
                                                         ucp_lane_index_t lane);


ucs_status_t ucp_proto_multi_init(const ucp_proto_multi_init_params_t *params,
                                  ucp_proto_multi_priv_t *mpriv,
                                  size_t *priv_size_p);


void ucp_proto_multi_query_config(const ucp_proto_query_params_t *params,
                                  ucp_proto_query_attr_t *attr);


void ucp_proto_multi_query(const ucp_proto_query_params_t *params,
                           ucp_proto_query_attr_t *attr);

#endif
