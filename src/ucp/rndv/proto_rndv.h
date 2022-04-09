/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_RNDV_H_
#define UCP_PROTO_RNDV_H_

#include "rndv.h"

#include <ucp/proto/proto_multi.h>


/**
 * Rendezvous protocol which sends a control message to the remote peer, and not
 * actually transferring bulk data. The remote peer is expected to perform the
 * "remote_proto" protocol to complete data transfer.
 * Typically, a rendezvous protocol will have one or two control message
 * exchanges before the bulk transfer takes place.
 */
typedef struct {
    /* Memory domains to send remote keys */
    ucp_md_map_t            md_map;

    /* System devices used for communication, used to pack distance in rkey */
    ucp_sys_dev_map_t       sys_dev_map;

    /* Cached system distance from each system device */
    ucs_sys_dev_distance_t  sys_dev_distance[UCP_MAX_LANES];

    /* Total size of packed rkeys */
    size_t                  packed_rkey_size;

    /* Lane for sending the "remote_op" message */
    ucp_lane_index_t        lane;

    /* Which protocol the remote side is expected to use, for performance
       estimation and reporting purpose */
    ucp_proto_select_elem_t remote_proto;
} ucp_proto_rndv_ctrl_priv_t;


/*
 * Private data for rendezvous protocol which sends an acknowledgement packet
 */
typedef struct {
    /* Lane to send completion message (ATP, RTS, ATS) */
    ucp_lane_index_t lane;
} ucp_proto_rndv_ack_priv_t;


/*
 * Private data for rendezvous protocol which sends bulk data followed by an
 * acknowledgement packet
 */
typedef struct {
    ucp_proto_rndv_ack_priv_t super;

    /*
     * Multi-lane common part.
     * Must be the last element in this struct, since it's variable-size and
     * ends with a zero-size array.
     */
    ucp_proto_multi_priv_t mpriv;
} ucp_proto_rndv_bulk_priv_t;


/**
 * Rendezvous control-message protocol initialization parameters
 */
typedef struct {
    ucp_proto_common_init_params_t super;

    /* Which operation the remote peer is expected to perform */
    ucp_operation_id_t             remote_op_id;

    /* Time to unpack the received data */
    ucs_linear_func_t              unpack_time;

    /* Reduce estimated time by this value (for example, 0.03 means to report
       a 3% better time) */
    double                         perf_bias;

    /* Memory type of the transfer. Used as rkey memory information when
       selecting the remote protocol. */
    ucp_memory_info_t              mem_info;

} ucp_proto_rndv_ctrl_init_params_t;


ucs_status_t
ucp_proto_rndv_ctrl_init(const ucp_proto_rndv_ctrl_init_params_t *params);


ucs_status_t
ucp_proto_rndv_rts_init(const ucp_proto_init_params_t *init_params);


void ucp_proto_rndv_rts_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr);


void ucp_proto_rndv_rts_abort(ucp_request_t *req, ucs_status_t status);


ucs_status_t ucp_proto_rndv_ack_init(const ucp_proto_init_params_t *init_params,
                                     ucp_proto_rndv_ack_priv_t *apriv);


ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *init_params,
                         ucp_proto_rndv_bulk_priv_t *rpriv,
                         size_t *priv_size_p);


ucs_status_t ucp_proto_rndv_ats_progress(uct_pending_req_t *uct_req);


void ucp_proto_rndv_bulk_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr);


void ucp_proto_rndv_receive_start(ucp_worker_h worker, ucp_request_t *recv_req,
                                  const ucp_rndv_rts_hdr_t *rts,
                                  const void *rkey_buffer, size_t rkey_length);


ucs_status_t
ucp_proto_rndv_handle_rtr(void *arg, void *data, size_t length, unsigned flags);


ucs_status_t ucp_proto_rndv_rtr_handle_atp(void *arg, void *data, size_t length,
                                           unsigned flags);


ucs_status_t ucp_proto_rndv_handle_data(void *arg, void *data, size_t length,
                                        unsigned flags);


/* Initialize req->send.multi_lane_idx according to req->rndv.offset */
void ucp_proto_rndv_bulk_request_init_lane_idx(
        ucp_request_t *req, const ucp_proto_rndv_bulk_priv_t *rpriv);


void ucp_proto_rndv_ppln_send_frag_complete(ucp_request_t *freq, int send_ack);


void ucp_proto_rndv_ppln_recv_frag_complete(ucp_request_t *freq, int send_ack);

#endif
