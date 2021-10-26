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


/* Format string to display a protocol time */
#define UCP_PROTO_TIME_FMT(_time_var) " " #_time_var ": %.2f ns"
#define UCP_PROTO_TIME_ARG(_time_val) ((_time_val) * 1e9)

/* Format string to display a protocol performance function time */
#define UCP_PROTO_PERF_FUNC_TIME_FMT "%.2f+%.3f*N"
#define UCP_PROTO_PERF_FUNC_TIME_ARG(_perf_func) \
    ((_perf_func)->c * 1e9), ((_perf_func)->m * 1e9 * UCS_KBYTE)

/* Format string to display a protocol performance function bandwidth */
#define UCP_PROTO_PERF_FUNC_BW_FMT "%.2f"
#define UCP_PROTO_PERF_FUNC_BW_ARG(_perf_func) \
    (1.0 / ((_perf_func)->m * UCS_MBYTE))

/* Format string to display a protocol performance function */
#define UCP_PROTO_PERF_FUNC_FMT(_perf_var) " " #_perf_var ": " \
    UCP_PROTO_PERF_FUNC_TIME_FMT " ns/KB, " \
    UCP_PROTO_PERF_FUNC_BW_FMT " MB/s"
#define UCP_PROTO_PERF_FUNC_ARG(_perf_func) \
    UCP_PROTO_PERF_FUNC_TIME_ARG(_perf_func), \
    UCP_PROTO_PERF_FUNC_BW_ARG(_perf_func)

/* Format string to display a protocol performance estimations
 * of different types. See ucp_proto_perf_type_t */
#define UCP_PROTO_PERF_FUNC_TYPES_FMT \
    UCP_PROTO_PERF_FUNC_FMT(sigle) \
    UCP_PROTO_PERF_FUNC_FMT(multi)
#define UCP_PROTO_PERF_FUNC_TYPES_ARG(_perf_func) \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_SINGLE])), \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_MULTI]))


/* Constant for "undefined"/"not-applicable" structure offset */
#define UCP_PROTO_COMMON_OFFSET_INVALID PTRDIFF_MAX


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
    UCP_PROTO_COMMON_INIT_FLAG_HDR_ONLY      = UCS_BIT(5),

    /* Requires rkey_ptr capable MD */
    UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR      = UCS_BIT(6),
} ucp_proto_common_init_flags_t;


/* Protocol common initialization parameters which are used to calculate
 * thresholds, performance, etc. for a specific selection criteria.
 */
typedef struct {
    ucp_proto_init_params_t super;

    /* Protocol added latency */
    double                  latency;

    /* Protocol overhead */
    double                  overhead;

    /* User-configured threshold */
    size_t                  cfg_thresh;

    /* User configuration priority */
    unsigned                cfg_priority;

    /* Minimal payload size */
    size_t                  min_length;

    /* Maximal payload size */
    size_t                  max_length;

    /* Offset in uct_iface_attr_t structure of the field which specifies the
     * minimal fragment size for the UCT operation used by this protocol */
    ptrdiff_t               min_frag_offs;

    /* Offset in uct_iface_attr_t structure of the field which specifies the
     * maximal fragment size for the UCT operation used by this protocol */
    ptrdiff_t               max_frag_offs;

    /* Offset in uct_iface_attr_t structure of the field which specifies the
     * maximal number of iov elements for the UCT operation used by this
    * protocol */
    ptrdiff_t               max_iov_offs;

    /* Header size on the first lane */
    size_t                  hdr_size;

    /* UCT operation used to sending the data. Used for performance estimation */
    uct_ep_operation_t      send_op;

    /* UCT operation used for copying from memory from request buffer to a
     * bounce buffer used by the transport. If set to LAST, the protocol supports
     * only host memory copy using memcpy(). */
    uct_ep_operation_t      memtype_op;

    /* Protocol instance flags, see @ref ucp_proto_common_init_flags_t */
    unsigned                flags;
} ucp_proto_common_init_params_t;


/*
 * Lane performance characteristics
 */
typedef struct {
    /* Operation send overhead */
    double send_overhead;

    /* Operation receive overhead */
    double recv_overhead;

    /* Transport bandwidth (without protocol memory copies) */
    double bandwidth;

    /* Network latency */
    double latency;

    /* Latency of device to memory access */
    double sys_latency;

    /* Minimal total message length */
    size_t min_length;

    /* Maximum single message length */
    size_t max_frag;
} ucp_proto_common_tl_perf_t;


/* Private data per lane */
typedef struct {
    ucp_lane_index_t        lane;       /* Lane index in the endpoint */
    ucp_rsc_index_t         memh_index; /* Index of UCT memory handle (for zero copy) */
    ucp_md_index_t          rkey_index; /* Remote key index (for remote access) */
    uint8_t                 max_iov;    /* Maximal number of IOVs on this lane */
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

ucs_sys_device_t
ucp_proto_common_get_sys_dev(const ucp_proto_init_params_t *params,
                             ucp_lane_index_t lane);


void ucp_proto_common_get_lane_distance(const ucp_proto_init_params_t *params,
                                        ucp_lane_index_t lane,
                                        ucs_sys_device_t sys_dev,
                                        ucs_sys_dev_distance_t *distance);

const uct_iface_attr_t *
ucp_proto_common_get_iface_attr(const ucp_proto_init_params_t *params,
                                ucp_lane_index_t lane);


size_t
ucp_proto_common_get_max_frag(const ucp_proto_common_init_params_t *params,
                              const uct_iface_attr_t *iface_attr);


size_t ucp_proto_common_get_iface_attr_field(const uct_iface_attr_t *iface_attr,
                                             ptrdiff_t field_offset,
                                             size_t dfl_value);


ucs_status_t
ucp_proto_common_lane_perf_attr(const ucp_proto_init_params_t *params,
                                ucp_lane_index_t lane, uct_ep_operation_t op,
                                uint64_t uct_field_mask,
                                uct_perf_attr_t* perf_attr);


ucs_status_t
ucp_proto_common_get_lane_perf(const ucp_proto_common_init_params_t *params,
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


/**
 * Add a "pipelined performance" range, which represents the send time of
 * multiples fragments of 'frag_size' bytes.
 * 'frag_range' is the time to send a single fragment.
 */
void ucp_proto_common_add_ppln_range(const ucp_proto_init_params_t *init_params,
                                     const ucp_proto_perf_range_t *frag_range,
                                     size_t max_length);


void ucp_proto_common_init_base_caps(
        const ucp_proto_common_init_params_t *params, size_t min_length);


void ucp_proto_common_add_perf_range(
        const ucp_proto_common_init_params_t *params, size_t max_length,
        ucs_linear_func_t send_overhead, ucs_linear_func_t recv_overhead,
        const ucs_linear_func_t *xfer_time, ucs_linear_func_t bias);


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


ucs_linear_func_t
ucp_proto_common_memreg_time(const ucp_proto_common_init_params_t *params,
                             ucp_md_map_t reg_md_map);


ucs_status_t
ucp_proto_common_buffer_copy_time(ucp_worker_h worker, const char *title,
                                  ucs_memory_type_t local_mem_type,
                                  ucs_memory_type_t remote_mem_type,
                                  uct_ep_operation_t memtype_op,
                                  ucs_linear_func_t *copy_time);

#endif
