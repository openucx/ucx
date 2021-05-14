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

    /* Reduce estimated time by this value (for example, 0.03 means to report
       a 3% better time) */
    double                         perf_bias;

    /* Memory type of the transfer */
    ucs_memory_info_t              mem_info;

    /* Minimal data length */
    size_t                         min_length;
} ucp_proto_rndv_ctrl_init_params_t;


ucs_status_t
ucp_proto_rndv_ctrl_init(const ucp_proto_rndv_ctrl_init_params_t *params);


void ucp_proto_rndv_ctrl_config_str(size_t min_length, size_t max_length,
                                    const void *priv,
                                    ucs_string_buffer_t *strb);


ucs_status_t
ucp_proto_rndv_rts_init(const ucp_proto_init_params_t *init_params);


ucs_status_t
ucp_proto_rndv_ack_init(const ucp_proto_init_params_t *init_params);


ucs_linear_func_t
ucp_proto_rndv_ack_time(const ucp_proto_init_params_t *init_params);


void ucp_proto_rndv_ack_config_str(size_t min_length, size_t max_length,
                                   const void *priv, ucs_string_buffer_t *strb);


ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *init_params);


size_t ucp_proto_rndv_pack_ack(void *dest, void *arg);


void ucp_proto_rndv_bulk_config_str(size_t min_length, size_t max_length,
                                    const void *priv,
                                    ucs_string_buffer_t *strb);


void ucp_proto_rndv_receive(ucp_worker_h worker, ucp_request_t *recv_req,
                            const ucp_rndv_rts_hdr_t *rts,
                            const void *rkey_buffer, size_t rkey_length);


ucs_status_t
ucp_proto_rndv_handle_rtr(void *arg, void *data, size_t length, unsigned flags);


ucs_status_t ucp_proto_rndv_handle_data(void *arg, void *data, size_t length,
                                        unsigned flags);

#endif
