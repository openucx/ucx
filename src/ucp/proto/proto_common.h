/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_COMMON_H_
#define UCP_PROTO_COMMON_H_

#include "proto.h"
#include "proto_select.h"

#include <uct/api/v2/uct_v2.h>


/* Constant for "undefined"/"not-applicable" structure offset */
#define UCP_PROTO_COMMON_OFFSET_INVALID          PTRDIFF_MAX


typedef enum {
    /* Send buffer is used by zero-copy operations */
    UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY    = UCS_BIT(0),

    /* Receive side is not doing memory copy */
    UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    = UCS_BIT(1),

    /* One-sided remote access (implies RECV_ZCOPY) */
    UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS = UCS_BIT(2),

    /* Only the header is sent from initiator side to target side, the data
     * (without headers) arrives back from target to initiator side, and only
     * then the operation is considered completed  */
    UCP_PROTO_COMMON_INIT_FLAG_RESPONSE      = UCS_BIT(3),

    /* The protocol can send only one fragment */
    UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG   = UCS_BIT(4),

    /* The message does not contain payload from the send buffer  */
    UCP_PROTO_COMMON_INIT_FLAG_HDR_ONLY      = UCS_BIT(5)
} ucp_proto_common_init_flags_t;


/* Protocol common initialization parameters which are used to calculate
 * thresholds, performance, etc. */
typedef struct {
    ucp_proto_init_params_t super;
    double                  latency;       /* protocol added latency */
    double                  overhead;      /* protocol overhead */
    size_t                  cfg_thresh;    /* user-configured threshold */
    unsigned                cfg_priority;  /* user configuration priority */
    size_t                  min_length;    /* Minimal payload size */
    size_t                  max_length;    /* Maximal payload size */
    ptrdiff_t               min_frag_offs; /* offset in uct_iface_attr_t of the
                                              minimal size of a single fragment */
    ptrdiff_t               max_frag_offs; /* offset in uct_iface_attr_t of the
                                              maximal size of a single fragment */
    size_t                  hdr_size;      /* header size on first lane */
    uct_ep_operation_t      memtype_op;    /* operation used for memtype copy */
    unsigned                flags;         /* see ucp_proto_common_init_flags_t */
} ucp_proto_common_init_params_t;


/*
 * Lane performance characteristics
 */
typedef struct {
    /* Operation overhead */
    double overhead;

    /* Transport bandwidth (without protocol memory copies) */
    double bandwidth;

    /* Network latency */
    double latency;

    /* Latency of device to memory access */
    double sys_latency;

    /* Minimal single message length */
    size_t min_frag;

    /* Maximum single message length */
    size_t max_frag;
} ucp_proto_common_tl_perf_t;


/* Private data per lane */
typedef struct {
    ucp_lane_index_t        lane;       /* Lane index in the endpoint */
    ucp_rsc_index_t         memh_index; /* Index of UCT memory handle (for zero copy) */
    ucp_md_index_t          rkey_index; /* Remote key index (for remote access) */
} ucp_proto_common_lane_priv_t;


/**
 * Called the first time the protocol starts sending a request, and only once
 * per request.
 *
 * @param [in] req   Request which started to send.
 */
typedef void (*ucp_proto_init_cb_t)(ucp_request_t *req);


/**
 * Called when a protocol finishes sending (or queueing to the transport) all
 * its data successfully.
 *
 * @param [in] req   Request which is finished sending.
 *
 * @return Status code to be returned from the progress function.
 */
typedef ucs_status_t (*ucp_proto_complete_cb_t)(ucp_request_t *req);


void ucp_proto_common_lane_priv_init(const ucp_proto_common_init_params_t *params,
                                     ucp_md_map_t md_map, ucp_lane_index_t lane,
                                     ucp_proto_common_lane_priv_t *lane_priv);


void ucp_proto_common_lane_priv_str(const ucp_proto_common_lane_priv_t *lpriv,
                                    ucs_string_buffer_t *strb);


ucp_rsc_index_t
ucp_proto_common_get_md_index(const ucp_proto_init_params_t *params,
                              ucp_lane_index_t lane);


const uct_iface_attr_t *
ucp_proto_common_get_iface_attr(const ucp_proto_init_params_t *params,
                                ucp_lane_index_t lane);


size_t
ucp_proto_common_get_max_frag(const ucp_proto_common_init_params_t *params,
                              const uct_iface_attr_t *iface_attr);


size_t ucp_proto_common_get_iface_attr_field(const uct_iface_attr_t *iface_attr,
                                             ptrdiff_t field_offset,
                                             size_t dfl_value);


void ucp_proto_common_get_lane_perf(const ucp_proto_common_init_params_t *params,
                                    ucp_lane_index_t lane,
                                    ucp_proto_common_tl_perf_t *perf);


/* @return number of lanes found */
ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_common_init_params_t *params,
                            ucp_lane_type_t lane_type, uint64_t tl_cap_flags,
                            ucp_lane_index_t max_lanes,
                            ucp_lane_map_t exclude_map,
                            ucp_lane_index_t *lanes);


ucp_md_map_t
ucp_proto_common_reg_md_map(const ucp_proto_common_init_params_t *params,
                            ucp_lane_map_t lane_map);


ucp_lane_index_t
ucp_proto_common_find_am_bcopy_hdr_lane(const ucp_proto_init_params_t *params);


/*
 * Calculate the performance pipelining an operation with performance 'perf1'
 * with an operation with performance 'perf2' by using fragments with size
 * 'frag_size' bytes.
*/
ucs_linear_func_t ucp_proto_common_ppln_perf(ucs_linear_func_t perf1,
                                             ucs_linear_func_t perf2,
                                             double frag_size);


ucs_status_t
ucp_proto_common_init_caps(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *perf,
                           ucp_md_map_t reg_md_map);


void ucp_proto_request_zcopy_completion(uct_completion_t *self);


void ucp_proto_trace_selected(ucp_request_t *req, size_t msg_length);


void ucp_proto_request_select_error(ucp_request_t *req,
                                    ucp_proto_select_t *proto_select,
                                    ucp_worker_cfg_index_t rkey_cfg_index,
                                    const ucp_proto_select_param_t *sel_param,
                                    size_t msg_length);

void ucp_proto_request_abort(ucp_request_t *req, ucs_status_t status);

#endif
