/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_rndv.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_init.h>


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
    const ucp_proto_select_range_t *frag_range;
    const ucp_proto_select_elem_t *select_elem;
    size_t frag_min_length, frag_max_length;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t sel_param;
    ucp_proto_select_t *proto_select;
    ucs_linear_func_t ppln_overhead;
    ucp_proto_perf_type_t perf_type;
    ucs_linear_func_t *ppln_perf;
    char frag_size_str[32];

    if ((select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        ((select_param->op_id != UCP_OP_ID_RNDV_SEND) &&
         (select_param->op_id != UCP_OP_ID_RNDV_RECV)) ||
        ucp_proto_rndv_init_params_is_ppln_frag(init_params)) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Select a protocol for rndv recv */
    sel_param          = *init_params->select_param;
    sel_param.op_flags = UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG |
                         UCP_PROTO_SELECT_OP_FLAG_INTERNAL |
                         ucp_proto_select_op_attr_to_flags(
                                 UCP_OP_ATTR_FLAG_MULTI_SEND);

    proto_select = ucp_proto_select_get(worker, init_params->ep_cfg_index,
                                        init_params->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return UCS_OK;
    }

    select_elem = ucp_proto_select_lookup_slow(worker, proto_select,
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

    ucs_trace("rndv_ppln frag %s" UCP_PROTO_PERF_FUNC_TYPES_FMT,
              ucs_memunits_to_str(rpriv->frag_size, frag_size_str,
                                  sizeof(frag_size_str)),
              UCP_PROTO_PERF_FUNC_TYPES_ARG(frag_range->super.perf));

    /* Add the single range of the pipeline protocol */
    ucp_proto_common_add_ppln_range(init_params, &frag_range->super, SIZE_MAX);

    /* Add overheads: PPLN overhead */
    ppln_overhead = ucs_linear_func_make(ppln_frag_overhead,
                                         ppln_frag_overhead / rpriv->frag_size);
    for (perf_type = 0; perf_type < UCP_PROTO_PERF_TYPE_LAST; ++perf_type) {
        ppln_perf = &caps->ranges[0].perf[perf_type];
        ucs_linear_func_add_inplace(ppln_perf, ppln_overhead);
    }

    /* Add overheads: ack time */
    return ucp_proto_rndv_ack_init(init_params, &rpriv->ack);
}

static void ucp_proto_rndv_ppln_query(const ucp_proto_query_params_t *params,
                                      ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ppln_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t frag_attr;

    if (params->msg_length <= rpriv->frag_size) {
        /* Message is smaller than fragment size */
        ucp_proto_select_elem_query(params->worker, &rpriv->frag_proto,
                                    params->msg_length, attr);
        attr->max_msg_length = rpriv->frag_size;
    } else {
        /* Message is large and fragmented to frag_size */
        ucp_proto_select_elem_query(params->worker, &rpriv->frag_proto,
                                    rpriv->frag_size, &frag_attr);

        attr->max_msg_length = SIZE_MAX;
        attr->is_estimation  = 0;
        ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "pipeline %s",
                          frag_attr.desc);
        ucs_strncpy_safe(attr->config, frag_attr.config, sizeof(attr->config));
    }
}

static void
ucp_proto_rndv_ppln_frag_complete(ucp_request_t *freq, int send_ack,
                                  ucp_proto_complete_cb_t complete_func,
                                  const char *title)
{
    ucp_request_t *req = ucp_request_get_super(freq);

    if (send_ack) {
        req->send.rndv.ppln.ack_data_size += freq->send.state.dt_iter.length;
    }

    if (!ucp_proto_rndv_frag_complete(req, freq, title)) {
        return;
    }

    if (req->send.rndv.rkey != NULL) {
        ucp_proto_rndv_rkey_destroy(req);
    }

    if (req->send.rndv.ppln.ack_data_size > 0) {
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

    /* Zero-length is not supported */
    ucs_assert(req->send.state.dt_iter.length > 0);

    req->send.state.completed_size    = 0;
    req->send.rndv.ppln.ack_data_size = 0;
    rpriv                             = req->send.proto_config->priv;

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

        /* Empty fragments should not happen */
        ucs_assert(freq->send.state.dt_iter.length > 0);

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

static size_t ucp_proto_rndv_ppln_pack_ack(void *dest, void *arg)
{
    ucp_request_t *req = arg;

    ucs_assert(req->send.rndv.ppln.ack_data_size > 0);
    return ucp_proto_rndv_pack_ack(req, dest,
                                   req->send.rndv.ppln.ack_data_size);
}

static ucs_status_t
ucp_proto_rndv_send_ppln_init(const ucp_proto_init_params_t *init_params)
{
    if (init_params->select_param->op_id != UCP_OP_ID_RNDV_SEND) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_ppln_init(init_params);
}

static ucs_status_t
ucp_proto_rndv_send_ppln_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_ppln_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_rndv_ack_progress(req, &rpriv->ack, UCP_AM_ID_RNDV_ATP,
                                       ucp_proto_rndv_ppln_pack_ack,
                                       ucp_proto_request_zcopy_complete_success);
}

ucp_proto_t ucp_rndv_send_ppln_proto = {
    .name     = "rndv/send/ppln",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_proto_rndv_send_ppln_init,
    .query    = ucp_proto_rndv_ppln_query,
    .progress = {
        [UCP_PROTO_RNDV_PPLN_STAGE_SEND] = ucp_proto_rndv_ppln_progress,
        [UCP_PROTO_RNDV_PPLN_STAGE_ACK]  = ucp_proto_rndv_send_ppln_atp_progress,
    },
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};

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
    const ucp_proto_rndv_ppln_priv_t *rpriv;

    rpriv = req->send.proto_config->priv;
    return ucp_proto_rndv_ack_progress(req, &rpriv->ack, UCP_AM_ID_RNDV_ATS,
                                       ucp_proto_rndv_ppln_pack_ack,
                                       ucp_proto_rndv_recv_complete);
}

ucp_proto_t ucp_rndv_recv_ppln_proto = {
    .name     = "rndv/recv/ppln",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_proto_rndv_recv_ppln_init,
    .query    = ucp_proto_rndv_ppln_query,
    .progress = {
        [UCP_PROTO_RNDV_PPLN_STAGE_SEND] = ucp_proto_rndv_ppln_progress,
        [UCP_PROTO_RNDV_PPLN_STAGE_ACK]  = ucp_proto_rndv_recv_ppln_ats_progress,
    },
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};
