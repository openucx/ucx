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
    ucp_proto_rndv_ack_priv_t ack;                   /* Ack configuration */
    size_t                    frag_size;             /* Fragment size */
    ucp_proto_config_t        frag_proto_cfg;        /* Frag proto config */
    size_t                    frag_proto_min_length; /* Frag proto min length */
} ucp_proto_rndv_ppln_priv_t;

static ucs_status_t
ucp_proto_rndv_ppln_add_overhead(ucp_proto_perf_t *ppln_perf, size_t frag_size)
{
    static const double frag_overhead = 30e-9;
    ucp_proto_perf_factors_t factors  = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    char frag_str[64];

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));
    factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU] =
            ucs_linear_func_make(frag_overhead, frag_overhead / frag_size);
    return ucp_proto_perf_add_funcs(ppln_perf, frag_size + 1, SIZE_MAX, factors,
                                    NULL, "fragment overhead", "frag size: %s",
                                    frag_str);
}

static void
ucp_proto_rndv_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_worker_h worker                          = init_params->worker;
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    ucp_proto_common_init_params_t ack_params;
    ucp_proto_perf_t *ppln_perf, *ack_perf, *result_perf;
    const ucp_proto_perf_segment_t *frag_seg, *first_seg;
    const ucp_proto_select_elem_t *select_elem;
    UCS_STRING_BUFFER_ONSTACK(seg_strb, 128);
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t sel_param;
    ucp_proto_rndv_ppln_priv_t rpriv;
    ucp_proto_select_t *proto_select;
    ucp_proto_init_elem_t *proto;
    char frag_size_str[32];
    void *frag_proto_priv;
    ucs_status_t status;
    uint8_t proto_flags;

    ack_params = ucp_proto_common_init_params(init_params);
    if (worker->context->config.ext.rndv_errh_ppln_enable) {
        ack_params.flags |= UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING;
    }

    if ((select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_init_check_op(init_params, UCP_PROTO_RNDV_OP_ID_MASK) ||
        !ucp_proto_common_init_check_err_handling(&ack_params) ||
        ucp_proto_rndv_init_params_is_ppln_frag(init_params)) {
        return;
    }

    /* Select a protocol for rndv recv */
    sel_param             = *select_param;
    sel_param.op_id_flags = ucp_proto_select_op_id(select_param) |
                            UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG;
    sel_param.op_attr     = ucp_proto_select_op_attr_pack(
            UCP_OP_ATTR_FLAG_MULTI_SEND);

    proto_select = ucp_proto_select_get(worker, init_params->ep_cfg_index,
                                        init_params->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return;
    }

    select_elem = ucp_proto_select_lookup_slow(worker, proto_select, 1,
                                               init_params->ep_cfg_index,
                                               init_params->rkey_cfg_index,
                                               &sel_param);
    if (select_elem == NULL) {
        return;
    }

    ack_params.max_length = SIZE_MAX;
    /* Add each proto as a separate variant */
    ucs_array_for_each(proto, &select_elem->proto_init.protocols) {
        proto_flags = ucp_proto_id_field(proto->proto_id, flags);
        if (proto_flags & UCP_PROTO_FLAG_INVALID) {
            continue;
        }

        ucs_assert(!ucp_proto_perf_is_empty(proto->perf));

        /* Add the single range of the pipeline protocol to ppln_caps */
        status = ucp_proto_perf_create("pipeline", &ppln_perf);
        if (status != UCS_OK) {
            continue;
        }

        frag_seg = ucp_proto_perf_add_ppln(proto->perf, ppln_perf, SIZE_MAX);
        if (frag_seg == NULL) {
            goto out_destroy_ppln_perf;
        }

        /* Initialize private data */
        rpriv.frag_size = ucp_proto_perf_segment_end(frag_seg);
        first_seg       = ucp_proto_perf_find_segment_lb(proto->perf, 0);
        rpriv.frag_proto_min_length = ucp_proto_perf_segment_start(first_seg);
        ucs_assertv(rpriv.frag_size >= rpriv.frag_proto_min_length,
                    "rpriv.frag_size=%zu rpriv.frag_proto_min_length=%zu",
                    rpriv.frag_size, rpriv.frag_proto_min_length);

        frag_proto_priv = &ucs_array_elem(&select_elem->proto_init.priv_buf,
                                          proto->priv_offset);
        ucp_proto_rndv_set_variant_config(init_params, proto,
                                          &sel_param, frag_proto_priv,
                                          &rpriv.frag_proto_cfg);

        ucp_proto_perf_segment_str(frag_seg, &seg_strb);
        ucs_trace("rndv_ppln frag: %s proto: %s segment: %s",
                  ucs_memunits_to_str(rpriv.frag_size, frag_size_str,
                                      sizeof(frag_size_str)),
                  ucp_proto_id_field(proto->proto_id, name),
                  ucs_string_buffer_cstr(&seg_strb));

        /* Add fragment overhead */
        status = ucp_proto_rndv_ppln_add_overhead(ppln_perf, rpriv.frag_size);
        if (status != UCS_OK) {
            goto out_destroy_ppln_perf;
        }

        /* Add ATS overhead */
        status = ucp_proto_rndv_ack_init(&ack_params, UCP_PROTO_RNDV_ATS_NAME,
                                         0, &ack_perf, &rpriv.ack);
        if ((status != UCS_OK) || (rpriv.ack.lane == UCP_NULL_LANE)) {
            goto out_destroy_ppln_perf;
        }

        status = ucp_proto_perf_aggregate2(
                ucp_proto_id_field(init_params->proto_id, name), ppln_perf,
                ack_perf, &result_perf);
        if (status != UCS_OK) {
            goto out_destroy_ack_perf;
        }

        ucp_proto_select_add_proto(init_params, proto->cfg_thresh,
                                   proto->cfg_priority, result_perf, &rpriv,
                                   sizeof(rpriv));

    out_destroy_ack_perf:
        ucp_proto_perf_destroy(ack_perf);
    out_destroy_ppln_perf:
        ucp_proto_perf_destroy(ppln_perf);
    }
}

static void ucp_proto_rndv_ppln_query(const ucp_proto_query_params_t *params,
                                      ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ppln_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t frag_attr;

    if (params->msg_length <= rpriv->frag_size) {
        /* Message is smaller than fragment size */
        ucp_proto_config_query(params->worker, &rpriv->frag_proto_cfg,
                               params->msg_length, attr);
        attr->max_msg_length = rpriv->frag_size;
    } else {
        /* Message is large and fragmented to frag_size */
        ucp_proto_config_query(params->worker, &rpriv->frag_proto_cfg,
                               rpriv->frag_size, &frag_attr);

        attr->max_msg_length = SIZE_MAX;
        attr->is_estimation  = 0;
        attr->lane_map       = frag_attr.lane_map;
        ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "pipeline %s",
                          frag_attr.desc);
        ucs_strncpy_safe(attr->config, frag_attr.config, sizeof(attr->config));
    }

    attr->lane_map |= UCS_BIT(rpriv->ack.lane);
}

static void
ucp_proto_rndv_ppln_frag_complete(ucp_request_t *freq, int send_ack, int abort,
                                  ucp_proto_complete_cb_t complete_func,
                                  const char *title)
{
    ucp_request_t *req = ucp_request_get_super(freq);

    if (send_ack) {
        req->send.rndv.ppln.ack_data_size += freq->send.state.dt_iter.length;
    }

    /* In case of abort we don't destroy super request until all fragments are
     * completed */
    if (!ucp_proto_rndv_frag_complete(req, freq, title)) {
        return;
    }

    if (req->send.rndv.rkey != NULL) {
        ucp_proto_rndv_rkey_destroy(req);
    }

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, UCP_DT_MASK_ALL);

    if (!abort && (req->send.rndv.ppln.ack_data_size > 0)) {
        ucp_proto_request_set_stage(req, UCP_PROTO_RNDV_PPLN_STAGE_ACK);
        ucp_request_send(req);
    } else {
        complete_func(req);
    }
}

void ucp_proto_rndv_ppln_send_frag_complete(ucp_request_t *freq, int send_ack)
{
    ucp_proto_rndv_ppln_frag_complete(freq, send_ack, 0,
                                      ucp_proto_request_complete_success,
                                      "ppln_send");
}

void ucp_proto_rndv_ppln_recv_frag_complete(ucp_request_t *freq, int send_ack,
                                            int abort)
{
    ucp_proto_rndv_ppln_frag_complete(freq, send_ack, abort,
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
    size_t overlap;

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
        overlap = ucp_datatype_iter_next_slice_overlap(
                &req->send.state.dt_iter, rpriv->frag_size,
                rpriv->frag_proto_min_length, &freq->send.state.dt_iter,
                &next_iter);
        req->send.rndv.ppln.ack_data_size -= overlap;
        req->send.state.completed_size    -= overlap;

        /* Empty fragments should not happen */
        ucs_assert(freq->send.state.dt_iter.length > 0);

        /* Initialize rendezvous parameters */
        freq->send.rndv.remote_req_id  = req->send.rndv.remote_req_id;
        freq->send.rndv.remote_address = req->send.rndv.remote_address +
                                         req->send.state.dt_iter.offset;
        freq->send.rndv.rkey           = req->send.rndv.rkey;
        freq->send.rndv.offset         = req->send.rndv.offset +
                                         req->send.state.dt_iter.offset;

        ucp_proto_request_set_proto(freq, &rpriv->frag_proto_cfg,
                                    freq->send.state.dt_iter.length);

        ucp_trace_req(req, "send freq %p offset %zu size %zu", freq,
                      freq->send.rndv.offset, freq->send.state.dt_iter.length);
        UCS_PROFILE_CALL_VOID_ALWAYS(ucp_request_send, freq);

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

static void
ucp_proto_rndv_send_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_RNDV_SEND))) {
        return;
    }

    ucp_proto_rndv_ppln_probe(init_params);
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
    .probe    = ucp_proto_rndv_send_ppln_probe,
    .query    = ucp_proto_rndv_ppln_query,
    .progress = {
        [UCP_PROTO_RNDV_PPLN_STAGE_SEND] = ucp_proto_rndv_ppln_progress,
        [UCP_PROTO_RNDV_PPLN_STAGE_ACK]  = ucp_proto_rndv_send_ppln_atp_progress,
    },
    .abort    = ucp_proto_rndv_stub_abort,
    .reset    = (ucp_request_reset_func_t)ucp_proto_reset_fatal_not_implemented
};

static void
ucp_proto_rndv_recv_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_RNDV_RECV))) {
        return;
    }

    ucp_proto_rndv_ppln_probe(init_params);
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

ucs_status_t ucp_proto_rndv_ppln_reset(ucp_request_t *req)
{
    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        return UCS_OK;
    }

    ucs_assert(req->send.state.completed_size == 0);
    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;

    if ((req->send.proto_stage != UCP_PROTO_RNDV_PPLN_STAGE_SEND) &&
        (req->send.proto_stage != UCP_PROTO_RNDV_PPLN_STAGE_ACK)) {
        ucp_proto_fatal_invalid_stage(req, "reset");
    }

    return UCS_OK;
}

ucp_proto_t ucp_rndv_recv_ppln_proto = {
    .name     = "rndv/recv/ppln",
    .desc     = NULL,
    .flags    = 0,
    .probe    = ucp_proto_rndv_recv_ppln_probe,
    .query    = ucp_proto_rndv_ppln_query,
    .progress = {
        [UCP_PROTO_RNDV_PPLN_STAGE_SEND] = ucp_proto_rndv_ppln_progress,
        [UCP_PROTO_RNDV_PPLN_STAGE_ACK]  = ucp_proto_rndv_recv_ppln_ats_progress,
    },
    .abort    = ucp_proto_rndv_stub_abort,
    .reset    = ucp_proto_rndv_ppln_reset
};
