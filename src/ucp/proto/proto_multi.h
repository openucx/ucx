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

    /* Ratio of data to send on this lane.
     * This is a fixed-point numeric representation (n * 2^shift), where "n" is
     * the real value, and "shift" is defined by UCP_PROTO_MULTI_WEIGHT_SHIFT.
     */
    uint32_t                     weight;
} ucp_proto_multi_lane_priv_t;


/*
 * Base class for protocols with fragmentation
 */
typedef struct {
    ucp_lane_index_t               num_lanes;  /* Number of lanes to use */
    ucp_md_map_t                   reg_md_map; /* Memory domains to register on */
    ucp_proto_multi_lane_priv_t    lanes[0];   /* Array of lanes */
} ucp_proto_multi_priv_t;


/**
 * Initialization parameters for multi-lane protocol
 */
typedef struct {
    ucp_proto_common_init_params_t super;
    ucp_lane_index_t               max_lanes;  /* Max lanes to select */

    struct {
        uint64_t                   tl_cap_flags; /* Required iface capabilities */
        ucp_lane_type_t            lane_type;    /* Required lane type */
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


ucs_status_t ucp_proto_multi_init(const ucp_proto_multi_init_params_t *params);


void ucp_proto_multi_config_str(size_t min_length, size_t max_length,
                                const void *priv, ucs_string_buffer_t *strb);

#endif
