/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"
#include "rndv_mtype.inl"

#define UCP_PROTO_RNDV_RKEY_PTR_DESC "copy to attached"


enum {
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_COPY = UCP_PROTO_STAGE_START,
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_ACK
};


typedef struct {
    ucp_proto_rndv_ack_priv_t ack;
    ucp_proto_single_priv_t   spriv;
} ucp_proto_rndv_rkey_ptr_priv_t;


typedef struct {
    ucp_proto_rndv_rkey_ptr_priv_t super;
    ucp_md_index_t                 dst_md_index;
} ucp_proto_rndv_rkey_ptr_mtype_priv_t;


static void ucp_proto_rndv_rkey_ptr_probe_common(
        const ucp_proto_common_init_params_t *init_params,
        ucp_proto_perf_t *rndv_op_perf, ucp_proto_rndv_rkey_ptr_priv_t *rpriv,
        size_t priv_size)
{
    const char *proto_name = ucp_proto_id_field(init_params->super.proto_id,
                                                name);
    ucp_proto_perf_t *perf, *ack_perf;
    ucs_status_t status;

    ucs_assert(!ucp_proto_perf_is_empty(rndv_op_perf));

    status = ucp_proto_rndv_ack_init(init_params, UCP_PROTO_RNDV_ATS_NAME, 0,
                                     &ack_perf, &rpriv->ack);
    if (status != UCS_OK) {
        return;
    }

    if (rpriv->ack.lane == UCP_NULL_LANE) {
        /* Add perf without ACK in case of pipeline */
        perf = rndv_op_perf;
    } else {
        status = ucp_proto_perf_aggregate2(proto_name, rndv_op_perf, ack_perf,
                                           &perf);
        ucp_proto_perf_destroy(ack_perf);
        ucp_proto_perf_destroy(rndv_op_perf);
        if (status != UCS_OK) {
            return;
        }
    }

    ucp_proto_select_add_proto(&init_params->super, init_params->cfg_thresh,
                               init_params->cfg_priority, perf, rpriv,
                               priv_size);
}

static void
ucp_proto_rndv_rkey_ptr_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                = init_params->worker->context;
    uint64_t rndv_modes                   = UCS_BIT(UCP_RNDV_MODE_RKEY_PTR);
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 80,
        .super.overhead      = context->config.ext.proto_overhead_rkey_ptr,
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
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = UCP_LANE_TYPE_RKEY_PTR,
        .tl_cap_flags        = 0,
    };
    ucp_proto_rndv_rkey_ptr_priv_t rpriv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    if (!ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_RECV, 0) ||
        !ucp_proto_common_init_check_err_handling(&params.super)) {
        return;
    }

    status = ucp_proto_single_init(&params, &perf, &rpriv.spriv);
    if (status != UCS_OK) {
        return;
    }

    ucp_proto_rndv_rkey_ptr_probe_common(&params.super, perf, &rpriv,
                                         sizeof(rpriv));
}

static void
ucp_rndv_rkey_ptr_query_common(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_rkey_ptr_priv_t *rpriv = params->priv;

    ucp_proto_default_query(params, attr);

    ucs_assert(rpriv->spriv.super.lane != UCP_NULL_LANE);
    attr->lane_map = UCS_BIT(rpriv->spriv.super.lane);
    if (rpriv->ack.lane != UCP_NULL_LANE) {
        attr->lane_map |= UCS_BIT(rpriv->ack.lane);
    }
}

static void
ucp_proto_rndv_rkey_ptr_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr)
{
    UCS_STRING_BUFFER_FIXED(config_strb, attr->config, sizeof(attr->config));
    const ucp_proto_rndv_rkey_ptr_priv_t *rpriv = params->priv;

    ucp_rndv_rkey_ptr_query_common(params, attr);
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

    ucp_proto_rndv_recv_complete_with_ats(req,
                                          UCP_PROTO_RNDV_RKEY_PTR_STAGE_ACK);
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
                                      0, &worker->rkey_ptr_cb_id);

    return UCS_OK;
}

ucp_proto_t ucp_rndv_rkey_ptr_proto = {
    .name     = "rndv/rkey_ptr",
    .desc     = "copy from mapped remote memory",
    .flags    = 0,
    .probe    = ucp_proto_rndv_rkey_ptr_probe,
    .query    = ucp_proto_rndv_rkey_ptr_query,
    .progress = {
         [UCP_PROTO_RNDV_RKEY_PTR_STAGE_COPY] = ucp_proto_rndv_rkey_ptr_fetch_progress,
         [UCP_PROTO_RNDV_RKEY_PTR_STAGE_ACK]  = ucp_proto_rndv_ats_progress
    },
    .abort    = ucp_proto_abort_fatal_not_implemented,
    .reset    = (ucp_request_reset_func_t)ucp_proto_reset_fatal_not_implemented
};

static void
ucp_proto_rndv_rkey_ptr_mtype_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                = init_params->worker->context;
    uint64_t rndv_modes                   = UCS_BIT(UCP_RNDV_MODE_PUT_PIPELINE);
    ucp_lane_index_t rkey_ptr_lane        = init_params->ep_config_key->rkey_ptr_lane;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.overhead      = 0,
        .super.latency       = 0,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 80,
        .super.min_length    = 0,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_LAST,
        .super.memtype_op    = UCT_EP_OP_GET_ZCOPY,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR   |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS,
        .super.exclude_map   = (rkey_ptr_lane == UCP_NULL_LANE) ?
                               0 : UCS_BIT(rkey_ptr_lane),
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = UCP_LANE_TYPE_LAST,
        .tl_cap_flags        = 0
    };
    ucp_proto_rndv_rkey_ptr_mtype_priv_t rpriv;
    ucp_md_map_t mdesc_md_map;
    ucp_proto_perf_t *perf;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (!context->config.ext.rndv_shm_ppln_enable ||
        !ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_SEND, 1) ||
        !ucp_proto_common_init_check_err_handling(&params.super)) {
        return;
    }

    /* 2-stage ppln protocols work with host staging buffers only */
    status = ucp_proto_rndv_mtype_init(init_params, UCS_MEMORY_TYPE_HOST,
                                       &mdesc_md_map, &params.super.max_length);
    if (status != UCS_OK) {
        return;
    }

    status = ucp_proto_single_init(&params, &perf, &rpriv.super.spriv);
    if (status != UCS_OK) {
        return;
    }

    lane               = rpriv.super.spriv.super.lane;
    rpriv.dst_md_index = init_params->ep_config_key->lanes[lane].dst_md_index;

    ucs_assertv(UCS_BIT(rpriv.dst_md_index) &
                init_params->rkey_config_key->md_map,
                "dst_md_index %u rkey->md_map 0x%lx", rpriv.dst_md_index,
                init_params->rkey_config_key->md_map);

    ucp_proto_rndv_rkey_ptr_probe_common(&params.super, perf, &rpriv.super,
                                         sizeof(rpriv));
}

static ucs_status_t ucp_proto_rndv_rkey_ptr_mtype_completion(ucp_request_t *req)
{
    ucp_trace_req(req, "ucp_proto_rndv_rkey_ptr_mtype_completion");
    ucp_proto_rndv_rkey_destroy(req);
    ucp_proto_request_zcopy_complete(req, UCS_OK);
    return UCS_OK;
}

static void
ucp_proto_rndv_rkey_ptr_mtype_copy_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "rkey_ptr_mtype_copy_completion_send_atp");
    ucp_proto_request_set_stage(req, UCP_PROTO_RNDV_RKEY_PTR_STAGE_ACK);
    ucp_request_send(req);
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_mtype_copy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req    = ucs_container_of(uct_req, ucp_request_t, send.uct);
    ucp_context_h context = req->send.ep->worker->context;
    uint64_t remote_address           = req->send.rndv.remote_address;
    ucs_memory_type_t local_mem_type  = req->send.state.dt_iter.mem_info.type;
    const void *rkey_buffer           = req->send.rndv.rkey_buffer;
    const ucp_proto_rndv_rkey_ptr_mtype_priv_t
                               *rpriv = req->send.proto_config->priv;
    ucp_ep_peer_mem_data_t *ppln_data;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED));
    ucs_assert(rkey_buffer != NULL);

    req->send.rndv.rkey_buffer = NULL;

    ppln_data = ucp_ep_peer_mem_get(context, req->send.ep, remote_address,
                                    req->send.state.dt_iter.length,
                                    rkey_buffer, local_mem_type,
                                    rpriv->dst_md_index);
    if (ppln_data->rkey == NULL) {
        ucs_error("ep %p: failed to get local ptr for address 0x%" PRIx64
                  " length %zu mem_type %s on md_index %u",
                  req->send.ep, remote_address, req->send.state.dt_iter.length,
                  ucs_memory_type_names[local_mem_type], rpriv->dst_md_index);
        ucp_proto_request_abort(req, UCS_ERR_UNREACHABLE);
        return UCS_OK;
    }

    req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    ucp_proto_rndv_mtype_copy(req, ppln_data->local_ptr, ppln_data->uct_memh,
                              UCS_MEMORY_TYPE_HOST, uct_ep_get_zcopy,
                              ucp_proto_rndv_rkey_ptr_mtype_copy_completion,
                              "in from");

    return UCS_OK;
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_mtype_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_rkey_ptr_mtype_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_rndv_ack_progress(req, &rpriv->super.ack,
                                       UCP_AM_ID_RNDV_ATP,
                                       ucp_proto_rndv_common_pack_ack,
                                       ucp_proto_rndv_rkey_ptr_mtype_completion);
}

static void
ucp_proto_rndv_rkey_ptr_mtype_query(const ucp_proto_query_params_t *params,
                                    ucp_proto_query_attr_t *attr)
{
    const char *desc = UCP_PROTO_RNDV_RKEY_PTR_DESC;

    ucp_rndv_rkey_ptr_query_common(params, attr);
    ucp_proto_rndv_mtype_query_desc(params, UCS_MEMORY_TYPE_HOST, attr, desc);
}

ucp_proto_t ucp_rndv_rkey_ptr_mtype_proto = {
    .name     = "rndv/rkey_ptr/mtype",
    .desc     = "copy to mapped remote memory",
    .flags    = 0,
    .probe     = ucp_proto_rndv_rkey_ptr_mtype_probe,
    .query    = ucp_proto_rndv_rkey_ptr_mtype_query,
    .progress = {
        [UCP_PROTO_RNDV_RKEY_PTR_STAGE_COPY] = ucp_proto_rndv_rkey_ptr_mtype_copy_progress,
        [UCP_PROTO_RNDV_RKEY_PTR_STAGE_ACK]  = ucp_proto_rndv_rkey_ptr_mtype_atp_progress,
    },
    .abort    = (ucp_request_abort_func_t)ucp_proto_abort_fatal_not_implemented,
    .reset    = (ucp_request_reset_func_t)ucp_proto_reset_fatal_not_implemented
};
