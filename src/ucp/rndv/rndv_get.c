/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"
#include "rndv_mtype.inl"
#include <ucp/proto/proto_debug.h>


#define UCP_PROTO_RNDV_GET_DESC "read from remote"

static void
ucp_proto_rndv_get_common_probe(const ucp_proto_init_params_t *init_params,
                                uint64_t rndv_modes, size_t max_length,
                                uct_ep_operation_t memtype_op, unsigned flags,
                                ucp_md_map_t initial_reg_md_map,
                                int support_ppln,
                                const ucp_memory_info_t *reg_mem_info)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 80,
        .super.overhead      = 0,
        .super.latency       = 0,
        .super.min_length    = 0,
        .super.max_length    = max_length,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.get.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_GET_ZCOPY,
        .super.memtype_op    = memtype_op,
        .super.flags         = flags | UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_MIN_FRAG,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = *reg_mem_info,
        .max_lanes           = context->config.ext.max_rndv_lanes,
        .initial_reg_md_map  = initial_reg_md_map,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_ZCOPY,
        .opt_align_offs      = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.opt_zcopy_align),
    };
    ucp_proto_rndv_bulk_priv_t rpriv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_RECV,
                                 support_ppln)) {
        return;
    }

    status = ucp_proto_rndv_bulk_init(&params, UCP_PROTO_RNDV_GET_DESC,
                                      UCP_PROTO_RNDV_ATS_NAME, &perf, &rpriv);
    if (status != UCS_OK) {
        return;
    }

    ucp_proto_select_add_proto(&params.super.super, params.super.cfg_thresh,
                               params.super.cfg_priority, perf, &rpriv,
                               UCP_PROTO_MULTI_EXTENDED_PRIV_SIZE(&rpriv,
                                                                  mpriv));
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_get_common_request_init(ucp_request_t *req)
{
    /* coverity[tainted_data_downcast] */
    ucp_proto_rndv_bulk_request_init(req, req->send.proto_config->priv);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_get_common_send(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        const uct_iov_t *iov, size_t offset, uct_completion_t *comp)
{
    uct_rkey_t tl_rkey      = ucp_rkey_get_tl_rkey(req->send.rndv.rkey,
                                                   lpriv->super.rkey_index);
    uint64_t remote_address = req->send.rndv.remote_address + offset;

    return uct_ep_get_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            iov, 1, remote_address, tl_rkey, comp);
}

static void
ucp_proto_rndv_get_zcopy_fetch_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_datatype_iter_mem_dereg(&req->send.state.dt_iter,
                                UCS_BIT(UCP_DATATYPE_CONTIG));
    if (ucs_unlikely(uct_comp->status != UCS_OK)) {
        ucp_proto_rndv_rkey_destroy(req);
        ucp_proto_rndv_recv_complete_status(req, uct_comp->status);
        return;
    }

    UCP_WORKER_STAT_RNDV(req->send.ep->worker, GET_ZCOPY, +1);
    ucp_proto_rndv_recv_complete_with_ats(req, UCP_PROTO_RNDV_GET_STAGE_ATS);
}

static void
ucp_proto_rndv_get_zcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_memory_info_t reg_mem_info = {
        .type    = init_params->select_param->mem_type,
        .sys_dev = init_params->select_param->sys_dev
    };

    ucp_proto_rndv_get_common_probe(
            init_params, UCS_BIT(UCP_RNDV_MODE_GET_ZCOPY), SIZE_MAX,
            UCT_EP_OP_LAST,
            UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
            UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
            0, 0, &reg_mem_info);
}

static void
ucp_proto_rndv_get_zcopy_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    ucp_proto_default_query(params, attr);
    ucp_proto_rndv_bulk_query(params, attr);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_get_zcopy_send_func(ucp_request_t *req,
                                   const ucp_proto_multi_lane_priv_t *lpriv,
                                   ucp_datatype_iter_t *next_iter,
                                   ucp_lane_index_t *lane_shift)
{
    /* coverity[tainted_data_downcast] */
    const ucp_proto_rndv_bulk_priv_t *rpriv = req->send.proto_config->priv;
    size_t offset                           = req->send.state.dt_iter.offset;
    size_t max_payload;
    uct_iov_t iov;

    max_payload = ucp_proto_rndv_bulk_max_payload_align(req, rpriv, lpriv,
                                                        lane_shift);
    ucp_datatype_iter_next_iov(&req->send.state.dt_iter, max_payload,
                               lpriv->super.md_index,
                               UCS_BIT(UCP_DATATYPE_CONTIG), next_iter, &iov,
                               1);

    ucs_assert(iov.count == 1);
    ucp_proto_common_zcopy_adjust_min_frag(req, rpriv->mpriv.min_frag,
                                           iov.length, &iov, 1, &offset);
    return ucp_proto_rndv_get_common_send(req, lpriv, &iov, offset,
                                          &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_rndv_get_zcopy_fetch_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req                      = ucs_container_of(uct_req,
                                                               ucp_request_t,
                                                               send.uct);
    /* coverity[tainted_data_downcast] */
    const ucp_proto_rndv_bulk_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_multi_zcopy_progress(
            req, &rpriv->mpriv, ucp_proto_rndv_get_common_request_init,
            UCT_MD_MEM_ACCESS_LOCAL_WRITE, UCS_BIT(UCP_DATATYPE_CONTIG),
            ucp_proto_rndv_get_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_rndv_get_zcopy_fetch_completion);
}

static void ucp_rndv_get_zcopy_proto_abort(ucp_request_t *request,
                                           ucs_status_t status)
{
    switch (request->send.proto_stage) {
    case UCP_PROTO_RNDV_GET_STAGE_FETCH:
        ucp_proto_request_zcopy_abort(request, status);
        break;
    case UCP_PROTO_RNDV_GET_STAGE_ATS:
        /* The data was received locally and not invalidated, so we can complete
         * with UCS_OK
         */
        ucp_proto_rndv_recv_complete(request);
        break;
    default:
        ucs_fatal("req %p: %s has invalid stage %d", request,
                  request->send.proto_config->proto->name,
                  request->send.proto_stage);
    }
}

static ucs_status_t ucp_rndv_get_zcopy_proto_reset(ucp_request_t *req)
{
    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        return UCS_OK;
    }

    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;

    switch (req->send.proto_stage) {
    case UCP_PROTO_RNDV_GET_STAGE_FETCH:
        ucp_datatype_iter_mem_dereg(&req->send.state.dt_iter, UCP_DT_MASK_ALL);
        /* Fall through */
    case UCP_PROTO_RNDV_GET_STAGE_ATS:
        break;
    default:
        ucp_proto_fatal_invalid_stage(req, "reset");
    }

    return UCS_OK;
}

ucp_proto_t ucp_rndv_get_zcopy_proto = {
    .name     = "rndv/get/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC " " UCP_PROTO_RNDV_GET_DESC,
    .flags    = 0,
    .probe    = ucp_proto_rndv_get_zcopy_probe,
    .query    = ucp_proto_rndv_get_zcopy_query,
    .progress = {
         [UCP_PROTO_RNDV_GET_STAGE_FETCH] = ucp_proto_rndv_get_zcopy_fetch_progress,
         [UCP_PROTO_RNDV_GET_STAGE_ATS]   = ucp_proto_rndv_ats_progress
    },
    .abort    = ucp_rndv_get_zcopy_proto_abort,
    .reset    = ucp_rndv_get_zcopy_proto_reset
};

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_get_mtype_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    /* coverity[tainted_data_downcast] */
    const ucp_proto_rndv_bulk_priv_t *rpriv = req->send.proto_config->priv;
    size_t offset                           = req->send.state.dt_iter.offset;
    uct_iov_t iov;

    ucp_proto_rndv_mtype_next_iov(req, rpriv, lpriv, next_iter, &iov);

    ucs_assert(iov.count == 1);
    ucp_proto_common_zcopy_adjust_min_frag(req, rpriv->mpriv.min_frag,
                                           iov.length, &iov, 1, &offset);
    return ucp_proto_rndv_get_common_send(req, lpriv, &iov, offset,
                                          &req->send.state.uct_comp);
}

static void
ucp_proto_rndv_get_mtype_unpack_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucs_mpool_put_inline(req->send.rndv.mdesc);
    if (ucp_proto_rndv_request_is_ppln_frag(req)) {
        ucp_proto_rndv_ppln_recv_frag_complete(req, 1, 0);
    } else {
        ucp_proto_rndv_recv_complete_with_ats(req,
                                              UCP_PROTO_RNDV_GET_STAGE_ATS);
    }
}

static void
ucp_proto_rndv_get_mtype_fetch_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_proto_rndv_mdesc_mtype_copy(req, uct_ep_put_zcopy,
                                    ucp_proto_rndv_get_mtype_unpack_completion,
                                    "out to");
}

static ucs_status_t
ucp_proto_rndv_get_mtype_fetch_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_bulk_priv_t *rpriv;
    ucs_status_t status;

    /* coverity[tainted_data_downcast] */
    rpriv = req->send.proto_config->priv;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_proto_rndv_mtype_request_init(req, rpriv->frag_mem_type);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        ucp_proto_rndv_get_common_request_init(req);
        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_rndv_get_mtype_fetch_completion);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, &rpriv->mpriv,
                                    ucp_proto_rndv_get_mtype_send_func,
                                    ucp_request_invoke_uct_completion_success,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

static void
ucp_proto_rndv_get_mtype_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context = init_params->worker->context;
    ucp_md_map_t mdesc_md_map;
    ucs_status_t status;
    size_t frag_size;
    ucp_md_index_t UCS_V_UNUSED dummy_md_id;
    ucp_memory_info_t frag_mem_info;

    ucs_for_each_bit(frag_mem_info.type,
                     context->config.ext.rndv_frag_mem_types) {
        status = ucp_proto_rndv_mtype_init(init_params, frag_mem_info.type,
                                           &mdesc_md_map, &frag_size);
        if (status != UCS_OK) {
            continue;
        }

        status = ucp_mm_get_alloc_md_index(context, frag_mem_info.type,
                                           &dummy_md_id,
                                           &frag_mem_info.sys_dev);
        if (status != UCS_OK) {
            continue;
        }


        ucp_proto_rndv_get_common_probe(init_params,
                                        UCS_BIT(UCP_RNDV_MODE_GET_PIPELINE),
                                        frag_size, UCT_EP_OP_PUT_ZCOPY, 0,
                                        mdesc_md_map, 1, &frag_mem_info);
    }
}

static void
ucp_proto_rndv_get_mtype_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_bulk_priv_t *rpriv = params->priv;

    ucp_proto_rndv_bulk_query(params, attr);
    ucp_proto_rndv_mtype_query_desc(params, rpriv->frag_mem_type, attr,
                                    UCP_PROTO_RNDV_GET_DESC);
}

static ucs_status_t ucp_proto_rndv_get_mtype_reset(ucp_request_t *req)
{
    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        return UCS_OK;
    }

    ucs_mpool_put_inline(req->send.rndv.mdesc);
    req->send.rndv.mdesc = NULL;
    req->flags          &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;

    if ((req->send.proto_stage != UCP_PROTO_RNDV_GET_STAGE_FETCH) &&
        (req->send.proto_stage != UCP_PROTO_RNDV_GET_STAGE_ATS)) {
        ucp_proto_fatal_invalid_stage(req, "reset");
    }

    return UCS_OK;
}

ucp_proto_t ucp_rndv_get_mtype_proto = {
    .name     = "rndv/get/mtype",
    .desc     = NULL,
    .flags    = 0,
    .probe    = ucp_proto_rndv_get_mtype_probe,
    .query    = ucp_proto_rndv_get_mtype_query,
    .progress = {
        [UCP_PROTO_RNDV_GET_STAGE_FETCH] = ucp_proto_rndv_get_mtype_fetch_progress,
        [UCP_PROTO_RNDV_GET_STAGE_ATS]   = ucp_proto_rndv_ats_progress
    },
    .abort    = ucp_proto_abort_fatal_not_implemented,
    .reset    = ucp_proto_rndv_get_mtype_reset
};
