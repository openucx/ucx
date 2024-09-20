/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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
 * Helper macro to calculate the size of a protocol private data structure that
 * extends ucp_proto_multi_priv_t, according to the actual number of lanes.
 *
 * @param _priv   Pointer to the private data structure.
 * @param _mpriv  Name of the ucp_proto_multi_priv_t field in _priv.
 *
 * Example usage:
 *  typedef struct { int abc; ucp_proto_multi_priv_t mpriv; } my_priv_t;
 *       my_priv_t priv;
 *       ...
 *       UCP_PROTO_MULTI_EXTENDED_PRIV_SIZE(&priv, mpriv);
 */
#define UCP_PROTO_MULTI_EXTENDED_PRIV_SIZE(_priv, _mpriv) \
    ({ \
        typedef ucs_typeof(*(_priv)) _type; \
        \
        /* Make sure _mpriv is the last field in _type */ \
        UCS_STATIC_ASSERT((ucs_offsetof(_type, _mpriv) + \
                           sizeof(ucp_proto_multi_priv_t)) == sizeof(_type)); \
        \
        /* Add actual priv size to the offset it starts at */ \
        ucs_offsetof(_type, _mpriv) + \
                ucp_proto_multi_priv_size(&(_priv)->_mpriv); \
    })


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

    /* Optimal alignment for zero-copy buffer address */
    size_t                       opt_align;

    /* Minimal offset to reach, taking into account minimum RNDV chunk size */
    size_t                       min_end_offset;
} ucp_proto_multi_lane_priv_t;


/*
 * Base class for protocols with fragmentation
 * When part of a larger struct, must be the last field to allow a smaller size
 * according to the actual number of lanes. The structure size can be obtained
 * by @ref ucp_proto_multi_priv_size.
 */
typedef struct {
    ucp_md_map_t                reg_md_map;   /* Memory domains to register on */
    size_t                      min_frag;     /* Largest minimal fragment size */
    size_t                      max_frag_sum; /* 'max_frag' sum of all lanes */
    ucp_lane_map_t              lane_map;     /* Map of used lanes */
    ucp_lane_index_t            num_lanes;    /* Number of lanes to use */
    size_t                      align_thresh; /* Cached value of threshold for
                                                 enabling data split alignment */
    ucp_proto_multi_lane_priv_t lanes[UCP_MAX_LANES]; /* Array of lanes */
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

    /* Offset in uct_iface_attr_t structure of the field which specifies the
     * optimal alignment for buffer address for the UCT operation used
     * by this protocol */
    ptrdiff_t                      opt_align_offs;

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
                ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift);


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
                                  const char *perf_name,
                                  ucp_proto_perf_t **perf_p,
                                  ucp_proto_multi_priv_t *mpriv);


size_t ucp_proto_multi_priv_size(const ucp_proto_multi_priv_t *mpriv);


void ucp_proto_multi_probe(const ucp_proto_multi_init_params_t *params);


void ucp_proto_multi_query_config(const ucp_proto_query_params_t *params,
                                  ucp_proto_query_attr_t *attr);


void ucp_proto_multi_query(const ucp_proto_query_params_t *params,
                           ucp_proto_query_attr_t *attr);

#endif
