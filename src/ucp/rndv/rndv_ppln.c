/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_rndv.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_multi.inl>


enum {
    /* Send the pipelined requests */
    UCP_PROTO_RNDV_PPLN_STAGE_SEND = UCP_PROTO_STAGE_START,

    /* Send ATS/ATP */
    UCP_PROTO_RNDV_PPLN_STAGE_ACK,
};


/* Private data for pipeline protocol */
typedef struct {
    ucp_proto_rndv_ack_priv_t ack;        /* Ack configuration */
    size_t                    frag_size;  /* Fragment size */
    ucp_proto_select_elem_t   frag_proto; /* Protocol for fragments */
} ucp_proto_rndv_ppln_priv_t;


static ucs_status_t
ucp_proto_rndv_ppln_init(const ucp_proto_init_params_t *init_params)
{
    static const double ppln_frag_overhead       = 30e-9;
    ucp_worker_h worker                          = init_params->worker;
    ucp_proto_rndv_ppln_priv_t *rpriv            = init_params->priv;
    ucp_proto_caps_t *caps                       = init_params->caps;
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    ucs_linear_func_t ack_perf[UCP_PROTO_PERF_TYPE_LAST];
    const ucp_proto_select_range_t *frag_range;
    const ucp_proto_select_elem_t *select_elem;
    size_t frag_min_length, frag_max_length;
    ucp_proto_select_param_t sel_param;
    ucs_linear_func_t ppln_overhead;
    ucp_proto_perf_type_t perf_type;
    ucp_rkey_config_t *rkey_config;
    ucs_linear_func_t *ppln_perf;
    char frag_size_str[32];
    ucs_status_t status;

    if ((init_params->rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) ||
        (select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        ((select_param->op_id != UCP_OP_ID_RNDV_SEND) &&
         (select_param->op_id != UCP_OP_ID_RNDV_RECV)) ||
        (init_params->select_param->op_flags &
         UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_rndv_ack_init(init_params, &rpriv->ack, ack_perf);
    if (status != UCS_OK) {
        return status;
    }

    /* Select a protocol for rndv recv */
    sel_param          = *init_params->select_param;
    sel_param.op_flags = UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG |
                         ucp_proto_select_op_attr_to_flags(
                                 UCP_OP_ATTR_FLAG_MULTI_SEND);
    rkey_config        = &worker->rkey_config[init_params->rkey_cfg_index];

    select_elem = ucp_proto_select_lookup_slow(worker,
                                               &rkey_config->proto_select,
                                               init_params->ep_cfg_index,
                                               init_params->rkey_cfg_index,
                                               &sel_param);
    if (select_elem == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Find the performance range of sending one fragment */
    if (!ucp_proto_select_get_valid_range(select_elem->thresholds,
                                          &frag_min_length, &frag_max_length)) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Take the last protocol in the list */
    for (frag_range = select_elem->perf_ranges;
         frag_range->super.max_length < frag_max_length; ++frag_range)
        ;

    /* Initialize private data */
    *init_params->priv_size = sizeof(*rpriv);
    rpriv->frag_proto       = *select_elem;
    rpriv->frag_size        = frag_max_length;
    caps->cfg_thresh        = frag_range->cfg_thresh;
    caps->cfg_priority      = 0;
    caps->min_length        = frag_max_length + 1;
    caps->num_ranges        = 0;

    ucs_trace("rndv_ppln frag %s"
              " single:" UCP_PROTO_PERF_FUNC_FMT
              " multi:" UCP_PROTO_PERF_FUNC_FMT,
              ucs_memunits_to_str(rpriv->frag_size, frag_size_str,
                                  sizeof(frag_size_str)),
              UCP_PROTO_PERF_FUNC_ARG(
                      &frag_range->super.perf[UCP_PROTO_PERF_TYPE_SINGLE]),
              UCP_PROTO_PERF_FUNC_ARG(
                      &frag_range->super.perf[UCP_PROTO_PERF_TYPE_MULTI]));

    /* Add the single range of the pipeline protocol */
    ucp_proto_common_add_ppln_range(init_params, &frag_range->super, SIZE_MAX);

    /* Add overheads: PPLN overhead and ack time */
    ppln_overhead = ucs_linear_func_make(ppln_frag_overhead,
                                         ppln_frag_overhead / rpriv->frag_size);
    for (perf_type = 0; perf_type < UCP_PROTO_PERF_TYPE_LAST; ++perf_type) {
        ppln_perf = &caps->ranges[0].perf[perf_type];
        ucs_linear_func_add_inplace(ppln_perf, ppln_overhead);
        ucs_linear_func_add_inplace(ppln_perf, ack_perf[perf_type]);
    }

    return UCS_OK;
}

static void
ucp_proto_rndv_ppln_frag_complete(ucp_request_t *freq, int send_ack,
                                  ucp_proto_complete_cb_t complete_func,
                                  const char *title)
{
    ucp_request_t *req = ucp_request_get_super(freq);

    req->send.rndv.ppln.send_ack |= send_ack;
    if (!ucp_proto_rndv_frag_complete(req, freq, title)) {
        return;
    }

    if (req->send.rndv.rkey != NULL) {
        ucp_proto_rndv_rkey_destroy(req);
    }

    if (req->send.rndv.ppln.send_ack) {
        ucp_proto_request_set_stage(req, UCP_PROTO_RNDV_PPLN_STAGE_ACK);
        ucp_request_send(req);
    } else {
        complete_func(req);
    }
}

void ucp_proto_rndv_ppln_send_frag_complete(ucp_request_t *freq, int send_ack)
{
    ucp_proto_rndv_ppln_frag_complete(freq, send_ack,
                                      ucp_proto_request_complete_success,
                                      "ppln_send");
}

void ucp_proto_rndv_ppln_recv_frag_complete(ucp_request_t *freq, int send_ack)
{
    ucp_proto_rndv_ppln_frag_complete(freq, send_ack,
                                      ucp_proto_rndv_recv_complete,
                                      "ppln_recv");
}

static ucs_status_t ucp_proto_rndv_ppln_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req  = ucs_container_of(uct_req, ucp_request_t, send.uct);
    ucp_worker_h worker = req->send.ep->worker;
    const ucp_proto_rndv_ppln_priv_t *rpriv;
    ucp_datatype_iter_t next_iter;
    ucs_status_t status;
    ucp_request_t *freq;
    uint8_t sg_count;

    /* Nested pipeline is prevented during protocol selection */
    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_RNDV_FRAG));

    req->send.state.completed_size = 0;
    req->send.rndv.ppln.send_ack   = 0;
    rpriv                          = req->send.proto_config->priv;

    while (!ucp_datatype_iter_is_end(&req->send.state.dt_iter)) {
        status = ucp_proto_rndv_frag_request_alloc(worker, req, &freq);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        /* Initialize datatype for the fragment */
        ucp_datatype_iter_next_slice(&req->send.state.dt_iter, rpriv->frag_size,
                                     &freq->send.state.dt_iter, &next_iter,
                                     &sg_count);

        /* Initialize rendezvous parameters */
        freq->send.rndv.remote_req_id  = req->send.rndv.remote_req_id;
        freq->send.rndv.remote_address = req->send.rndv.remote_address +
                                         req->send.state.dt_iter.offset;
        freq->send.rndv.rkey           = req->send.rndv.rkey;
        freq->send.rndv.offset         = req->send.rndv.offset +
                                         req->send.state.dt_iter.offset;
        ucp_proto_request_select_proto(freq, &rpriv->frag_proto,
                                       freq->send.state.dt_iter.length);

        ucp_trace_req(req, "send fragment request %p", freq);
        ucp_request_send(freq);

        ucp_datatype_iter_copy_position(&req->send.state.dt_iter, &next_iter,
                                        UCS_BIT(UCP_DATATYPE_CONTIG));
    }

    return UCS_OK;
}

static void ucp_proto_rndv_ppln_config_str(size_t min_length, size_t max_length,
                                           const void *priv,
                                           ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_ppln_priv_t *rpriv = priv;
    char str[128];

    ucs_memunits_to_str(rpriv->frag_size, str, sizeof(str));
    ucs_string_buffer_appendf(strb, "fr:%s ", str);
    ucp_proto_threshold_elem_str(rpriv->frag_proto.thresholds, rpriv->frag_size,
                                 rpriv->frag_size, strb);
}

static ucs_status_t
ucp_proto_rndv_send_ppln_init(const ucp_proto_init_params_t *init_params)
{
    if (init_params->select_param->op_id != UCP_OP_ID_RNDV_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_ppln_init(init_params);
}

static size_t ucp_proto_rndv_send_ppln_pack_atp(void *dest, void *arg)
{
    return ucp_proto_rndv_send_pack_atp(arg, dest, 1);
}

static ucs_status_t
ucp_proto_rndv_send_ppln_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_ppln_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_RNDV_ATP, rpriv->ack.lane,
            ucp_proto_rndv_send_ppln_pack_atp, req, sizeof(ucp_rndv_atp_hdr_t),
            ucp_proto_request_zcopy_complete_success);
}

static ucp_proto_t ucp_rndv_send_ppln_proto = {
    .name       = "rndv/send/ppln",
    .flags      = 0,
    .init       = ucp_proto_rndv_send_ppln_init,
    .config_str = ucp_proto_rndv_ppln_config_str,
    .progress   = {
        [UCP_PROTO_RNDV_PPLN_STAGE_SEND] = ucp_proto_rndv_ppln_progress,
        [UCP_PROTO_RNDV_PPLN_STAGE_ACK]  = ucp_proto_rndv_send_ppln_atp_progress,
    },
};
UCP_PROTO_REGISTER(&ucp_rndv_send_ppln_proto);

static ucs_status_t
ucp_proto_rndv_recv_ppln_init(const ucp_proto_init_params_t *init_params)
{
    if (init_params->select_param->op_id != UCP_OP_ID_RNDV_RECV) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_ppln_init(init_params);
}

static ucs_status_t
ucp_proto_rndv_recv_ppln_ats_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_ppln_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_rndv_ack_progress(req, &rpriv->ack, UCP_AM_ID_RNDV_ATS,
                                       ucp_proto_rndv_recv_complete);
}

static ucp_proto_t ucp_rndv_recv_ppln_proto = {
    .name       = "rndv/recv/ppln",
    .flags      = 0,
    .init       = ucp_proto_rndv_recv_ppln_init,
    .config_str = ucp_proto_rndv_ppln_config_str,
    .progress   = {
        [UCP_PROTO_RNDV_PPLN_STAGE_SEND] = ucp_proto_rndv_ppln_progress,
        [UCP_PROTO_RNDV_PPLN_STAGE_ACK]  = ucp_proto_rndv_recv_ppln_ats_progress,
    },
};
UCP_PROTO_REGISTER(&ucp_rndv_recv_ppln_proto);
