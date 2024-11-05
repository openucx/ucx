/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_RNDV_H_
#define UCP_PROTO_RNDV_H_

#include "rndv.h"

#include <ucp/proto/proto_multi.h>


/* Names of rendezvous control messages */
#define UCP_PROTO_RNDV_RTS_NAME "RTS"
#define UCP_PROTO_RNDV_RTR_NAME "RTR"
#define UCP_PROTO_RNDV_ATS_NAME "ATS"
#define UCP_PROTO_RNDV_ATP_NAME "ATP"


/* Mask of rendezvous operations */
#define UCP_PROTO_RNDV_OP_ID_MASK \
    (UCS_BIT(UCP_OP_ID_RNDV_SEND) | UCS_BIT(UCP_OP_ID_RNDV_RECV))


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

    /* Config of the remote protocol, which is expected to be selected by peer.
       Used for performance estimation and reporting purpose */
    ucp_proto_config_t      remote_proto_config;
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

    /* Memory type of fragment buffers which are used by get/mtype and put/mtype
     * protocols.
     * TODO: Create a separate struct for mtype protocols and move it there. */
    ucs_memory_type_t         frag_mem_type;

    /* Multi-lane common part. Must be the last field, see
       @ref ucp_proto_multi_priv_t */
    ucp_proto_multi_priv_t    mpriv;
} ucp_proto_rndv_bulk_priv_t;


/**
 * Rendezvous control-message protocol initialization parameters
 */
typedef struct {
    ucp_proto_common_init_params_t super;

    /* Which operation the remote peer is expected to perform */
    ucp_operation_id_t             remote_op_id;

    /* Lane to send control message */
    ucp_lane_index_t               lane;

    /* Performance data to unpack the received data */
    ucp_proto_perf_t               *unpack_perf;

    /* Reduce estimated time by this value (for example, 0.03 means to report
       a 3% better time) */
    double                         perf_bias;

    /* Name of the control message, e.g "RTS" */
    const char                     *ctrl_msg_name;

    /* Map of mandatory mds which keys should be packed to the rkey */
    ucp_md_map_t                   md_map;
} ucp_proto_rndv_ctrl_init_params_t;

/* Return rendezvous threshold for the provided configuration */
size_t ucp_proto_rndv_thresh(const ucp_proto_init_params_t *init_params);

/* rndv_put stages */
enum {
    /* Initial stage for put zcopy is sending the data */
    UCP_PROTO_RNDV_PUT_ZCOPY_STAGE_SEND = UCP_PROTO_STAGE_START,

    /* Initial stage for put memtype is copy the data to the fragment */
    UCP_PROTO_RNDV_PUT_MTYPE_STAGE_COPY = UCP_PROTO_STAGE_START,

    /* Flush all lanes to ensure remote delivery */
    UCP_PROTO_RNDV_PUT_STAGE_FLUSH,

    /* Send ATP without fence (could be done after a flush) */
    UCP_PROTO_RNDV_PUT_STAGE_ATP,

    /* Send ATP with fence (could be done if using send lanes for ATP) */
    UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP,

    /* Memtype only: send the fragment to the remote side */
    UCP_PROTO_RNDV_PUT_MTYPE_STAGE_SEND
};


/* rndv_get stages */
enum {
    UCP_PROTO_RNDV_GET_STAGE_FETCH = UCP_PROTO_STAGE_START,
    UCP_PROTO_RNDV_GET_STAGE_ATS
};


/* Initializes protocol which sends rendezvous control message using AM lane
 * (e.g. RTS and ATS). */
void ucp_proto_rndv_ctrl_probe(const ucp_proto_rndv_ctrl_init_params_t *params,
                               void *priv, size_t priv_size);


ucp_lane_index_t
ucp_proto_rndv_find_ctrl_lane(const ucp_proto_init_params_t *params);


void ucp_proto_rndv_rts_probe(const ucp_proto_init_params_t *init_params);


void ucp_proto_rndv_set_variant_config(
        const ucp_proto_init_params_t *init_params,
        const ucp_proto_init_elem_t *proto,
        const ucp_proto_select_param_t *select_param, const void *priv,
        ucp_proto_config_t *cfg);


void ucp_proto_rndv_rts_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr);


void ucp_proto_rndv_rts_abort(ucp_request_t *req, ucs_status_t status);

ucs_status_t ucp_proto_rndv_rts_reset(ucp_request_t *req);


ucs_status_t
ucp_proto_rndv_ack_init(const ucp_proto_common_init_params_t *init_params,
                        const char *name, double overhead,
                        ucp_proto_perf_t **perf_p,
                        ucp_proto_rndv_ack_priv_t *apriv);


ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *init_params,
                         const char *name, const char *ack_name,
                         ucp_proto_perf_t **perf_p,
                         ucp_proto_rndv_bulk_priv_t *rpriv);


ucs_status_t ucp_proto_rndv_ats_progress(uct_pending_req_t *uct_req);


size_t ucp_proto_rndv_common_pack_ack(void *dest, void *arg);

ucs_status_t ucp_proto_rndv_ats_complete(ucp_request_t *req);

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


void ucp_proto_rndv_ppln_recv_frag_complete(ucp_request_t *freq, int send_ack,
                                            int abort);


void ucp_proto_rndv_stub_abort(ucp_request_t *req, ucs_status_t status);

#endif
