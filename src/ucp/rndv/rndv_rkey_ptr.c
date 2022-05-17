/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"


enum {
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_FETCH = UCP_PROTO_STAGE_START,
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATS
};


typedef struct {
    ucp_proto_rndv_ack_priv_t ack;
    ucp_proto_single_priv_t   spriv;
} ucp_proto_rndv_rkey_ptr_priv_t;


static double ucp_proto_rndv_rkey_ptr_overhead()
{
    switch (ucs_arch_get_cpu_vendor()) {
    case UCS_CPU_VENDOR_FUJITSU_ARM:
        return 500e-9;
    default:
        return 0;
    }
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_rndv_rkey_ptr_priv_t *rpriv = init_params->priv;
    ucp_context_t *context                = init_params->worker->context;
    uint64_t rndv_modes                   = UCS_BIT(UCP_RNDV_MODE_RKEY_PTR);
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 0,
        .super.overhead      = ucp_proto_rndv_rkey_ptr_overhead(),
        .super.latency       = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_LAST,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_RKEY_PTR,
        .tl_cap_flags        = 0,
    };
    ucs_status_t status;

    if (!ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_RECV, 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_single_init_priv(&params, &rpriv->spriv);
    if (status != UCS_OK) {
        return status;
    }

    *init_params->priv_size = sizeof(*rpriv);
    return ucp_proto_rndv_ack_init(init_params, &rpriv->ack);
}

static void
ucp_proto_rndv_rkey_ptr_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr)
{
    UCS_STRING_BUFFER_FIXED(config_strb, attr->config, sizeof(attr->config));
    const ucp_proto_rndv_rkey_ptr_priv_t *rpriv = params->priv;

    ucp_proto_default_query(params, attr);
    ucp_proto_common_lane_priv_str(params, &rpriv->spriv.super, 1, 0,
                                   &config_strb);
}

static unsigned ucp_proto_rndv_progress_rkey_ptr(void *arg)
{
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucp_request_t *req  = ucs_queue_head_elem_non_empty(&worker->rkey_ptr_reqs,
                                                        ucp_request_t,
                                                        send.rndv.rkey_ptr.queue_elem);
    size_t max_seg_size = worker->context->config.ext.rkey_ptr_seg_size;
    size_t length       = req->send.state.dt_iter.length;
    size_t offset       = req->send.state.completed_size;
    size_t seg_size     = ucs_min(max_seg_size, length - offset);
    ucs_status_t status;
    const void *src;

    src = UCS_PTR_BYTE_OFFSET(req->send.rndv.rkey_ptr_addr, offset);

    ucp_trace_req(req, "rkey_ptr unpack %zd from %p at offset %zd/%zd",
                  seg_size, src, offset, length);

    status = ucp_datatype_iter_unpack(&req->send.state.dt_iter, worker,
                                      seg_size, offset, src);
    if (ucs_unlikely(status != UCS_OK)) {
        ucp_proto_request_abort(req, status);
        return 0;
    }

    if (!ucp_proto_common_frag_complete(req, seg_size, "rkey_ptr")) {
        return 1;
    }

    ucs_queue_pull_non_empty(&worker->rkey_ptr_reqs);
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, UCP_DT_MASK_ALL);

    ucp_proto_rndv_recv_complete_with_ats(req,
                                          UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATS);
    if (ucs_queue_is_empty(&worker->rkey_ptr_reqs)) {
        uct_worker_progress_unregister_safe(worker->uct,
                                            &worker->rkey_ptr_cb_id);
    }

    return 1;
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_fetch_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req                          = ucs_container_of(uct_req,
                                                                   ucp_request_t,
                                                                   send.uct);
    const ucp_proto_rndv_rkey_ptr_priv_t *rpriv = req->send.proto_config->priv;
    ucp_worker_h worker                         = req->send.ep->worker;
    unsigned rkey_index                         = rpriv->spriv.super.rkey_index;
    ucp_rkey_h rkey                             = req->send.rndv.rkey;
    ucs_status_t status;

    ucs_assert(rkey_index != UCP_NULL_RESOURCE);
    status = uct_rkey_ptr(rkey->tl_rkey[rkey_index].cmpt,
                          &rkey->tl_rkey[rkey_index].rkey,
                          req->send.rndv.remote_address,
                          &req->send.rndv.rkey_ptr_addr);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    req->send.state.completed_size = 0;
    UCP_WORKER_STAT_RNDV(worker, RKEY_PTR, 1);
    ucs_queue_push(&worker->rkey_ptr_reqs, &req->send.rndv.rkey_ptr.queue_elem);
    uct_worker_progress_register_safe(worker->uct,
            ucp_proto_rndv_progress_rkey_ptr, worker,
            UCS_CALLBACKQ_FLAG_FAST, &worker->rkey_ptr_cb_id);

    return UCS_OK;
}

ucp_proto_t ucp_rndv_rkey_ptr_proto = {
    .name     = "rndv/rkey_ptr",
    .desc     = "copy from mapped remote memory",
    .flags    = 0,
    .init     = ucp_proto_rndv_rkey_ptr_init,
    .query    = ucp_proto_rndv_rkey_ptr_query,
    .progress = {
         [UCP_PROTO_RNDV_RKEY_PTR_STAGE_FETCH] = ucp_proto_rndv_rkey_ptr_fetch_progress,
         [UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATS]   = ucp_proto_rndv_ats_progress
    },
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};

