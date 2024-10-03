/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_COMMON_H_
#define UCP_PROTO_COMMON_H_

#include "proto.h"
#include "proto_select.h"
#include <ucp/dt/dt.h>

#include <uct/api/v2/uct_v2.h>


/* Constant for "undefined"/"not-applicable" structure offset */
#define UCP_PROTO_COMMON_OFFSET_INVALID PTRDIFF_MAX


/* Common protocol description strings */
#define UCP_PROTO_SHORT_DESC      "short"
#define UCP_PROTO_COPY_IN_DESC    "copy-in"
#define UCP_PROTO_COPY_OUT_DESC   "copy-out"
#define UCP_PROTO_ZCOPY_DESC      "zero-copy"
#define UCP_PROTO_MULTI_FRAG_DESC "multi-frag"


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

    /* Supports non-zero minimal fragment size */
    UCP_PROTO_COMMON_INIT_FLAG_MIN_FRAG      = UCS_BIT(7),

    /* Adjust maximum fragment size taking into account segment size to prevent
     * sending more than the remote side supports */
    UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE  = UCS_BIT(8),

    /* Supports error handling */
    UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING  = UCS_BIT(9),

    /* Supports starting the request when its datatype iterator offset is > 0 */
    UCP_PROTO_COMMON_INIT_FLAG_RESUME        = UCS_BIT(10),
    UCP_PROTO_COMMON_KEEP_MD_MAP             = UCS_BIT(11)
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

    /* Minimal number of iov */
    size_t                  min_iov;

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

    /* Map of unsuitable lanes */
    ucp_lane_map_t          exclude_map;

    /* Memory info of the buffer used for data transfer on the transport level.
     * If UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY flag is set, it is expected to
     * be the user buffer memory info. Alternatively, it refers to the type of
     * memory used for bounce buffers (either in the UCP or UCT layer) where
     * data needs to be copied as part of the protocol. */
    ucp_memory_info_t       reg_mem_info;
} ucp_proto_common_init_params_t;


/*
 * Lane performance characteristics
 */
typedef struct {
    /* Operation send overhead */
    double send_pre_overhead;

    /* Operation send finalization overhead */
    double send_post_overhead;

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
    ucp_rsc_index_t         md_index;   /* Index of UCT memory handle (for zero copy) */
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


ucp_proto_common_init_params_t
ucp_proto_common_init_params(const ucp_proto_init_params_t *init_params);


ucp_memory_info_t ucp_proto_common_select_param_mem_info(
                                  const ucp_proto_select_param_t *select_param);


/**
 * Check if protocol can be used according to error handling requirements.
 *
 * @param [in] init_params      Protocol initialization parameters.
 *
 * @return Nonzero if protocol can be used.
 */
int ucp_proto_common_init_check_err_handling(
        const ucp_proto_common_init_params_t *init_params);


ucp_rsc_index_t
ucp_proto_common_get_rsc_index(const ucp_proto_init_params_t *params,
                               ucp_lane_index_t lane);

void ucp_proto_common_lane_priv_init(const ucp_proto_common_init_params_t *params,
                                     ucp_md_map_t md_map, ucp_lane_index_t lane,
                                     ucp_proto_common_lane_priv_t *lane_priv);


void ucp_proto_common_lane_priv_str(const ucp_proto_query_params_t *params,
                                    const ucp_proto_common_lane_priv_t *lpriv,
                                    int show_rsc, int show_path,
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


size_t ucp_proto_common_get_iface_attr_field(const uct_iface_attr_t *iface_attr,
                                             ptrdiff_t field_offset,
                                             size_t dfl_value);


void ucp_proto_common_lane_perf_node(ucp_context_h context,
                                     ucp_rsc_index_t rsc_index,
                                     const uct_perf_attr_t *perf_attr,
                                     ucp_proto_perf_node_t **perf_node_p);

ucs_status_t
ucp_proto_common_get_lane_perf(const ucp_proto_common_init_params_t *params,
                               ucp_lane_index_t lane,
                               ucp_proto_common_tl_perf_t *perf,
                               ucp_proto_perf_node_t **perf_node_p);


/* @return number of lanes found */
ucp_lane_index_t ucp_proto_common_find_lanes_with_min_frag(
        const ucp_proto_common_init_params_t *params, ucp_lane_type_t lane_type,
        uint64_t tl_cap_flags, ucp_lane_index_t max_lanes,
        ucp_lane_map_t exclude_map, ucp_lane_index_t *lanes);


ucp_lane_index_t
ucp_proto_common_find_lanes(const ucp_proto_init_params_t *params,
                            uct_ep_operation_t memtype_op, unsigned flags,
                            ptrdiff_t max_iov_offs, size_t min_iov,
                            ucp_lane_type_t lane_type,
                            ucs_memory_type_t reg_mem_type,
                            uint64_t tl_cap_flags, ucp_lane_index_t max_lanes,
                            ucp_lane_map_t exclude_map,
                            ucp_lane_index_t *lanes);


ucp_md_map_t
ucp_proto_common_reg_md_map(const ucp_proto_common_init_params_t *params,
                            ucp_lane_map_t lane_map);


void ucp_proto_request_zcopy_completion(uct_completion_t *self);


int ucp_proto_is_short_supported(const ucp_proto_select_param_t *select_param);


void ucp_proto_trace_selected(ucp_request_t *req, size_t msg_length);


void ucp_proto_request_select_error(ucp_request_t *req,
                                    ucp_proto_select_t *proto_select,
                                    ucp_worker_cfg_index_t rkey_cfg_index,
                                    const ucp_proto_select_param_t *sel_param,
                                    size_t msg_length);


void ucp_proto_common_zcopy_adjust_min_frag_always(ucp_request_t *req,
                                                   size_t min_frag_diff,
                                                   uct_iov_t *iov,
                                                   size_t iovcnt,
                                                   size_t *offset_p);


void ucp_proto_request_abort(ucp_request_t *req, ucs_status_t status);

ucs_status_t
ucp_proto_request_init(ucp_request_t *req,
                       const ucp_proto_select_param_t *select_param);

void ucp_proto_request_check_reset_state(const ucp_request_t *req);

void ucp_proto_request_restart(ucp_request_t *req);

void ucp_proto_request_bcopy_abort(ucp_request_t *req, ucs_status_t status);

void ucp_proto_request_bcopy_id_abort(ucp_request_t *req, ucs_status_t status);

ucs_status_t ucp_proto_request_bcopy_reset(ucp_request_t *req);

ucs_status_t ucp_proto_request_bcopy_id_reset(ucp_request_t *req);

void ucp_proto_request_zcopy_abort(ucp_request_t *req, ucs_status_t status);

ucs_status_t ucp_proto_request_zcopy_reset(ucp_request_t *req);

ucs_status_t ucp_proto_request_zcopy_id_reset(ucp_request_t *req);

void ucp_proto_abort_fatal_not_implemented(ucp_request_t *req,
                                           ucs_status_t status);

void ucp_proto_reset_fatal_not_implemented(ucp_request_t *req);

void ucp_proto_fatal_invalid_stage(ucp_request_t *req, const char *func_name);

#endif
