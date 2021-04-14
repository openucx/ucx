/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"


static ucs_status_t
ucp_proto_rndv_get_zcopy_init(const ucp_proto_init_params_t *init_params)
{
    static const uint64_t rndv_modes     = UCS_BIT(UCP_RNDV_MODE_GET_ZCOPY);
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .super.overhead      = 0,
        .super.latency       = 0,
        .max_lanes           = context->config.ext.max_rndv_lanes,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_ZCOPY,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_zcopy),
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .super.hdr_size      = 0,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW
    };

    if ((init_params->select_param->op_id != UCP_OP_ID_RNDV_RECV) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_bulk_init(&params);
}

static ucs_status_t ucp_proto_rndv_get_complete(ucp_request_t *req)
{
    ucp_rkey_destroy(req->send.rndv.rkey);
    ucp_proto_request_zcopy_complete(req, req->send.state.uct_comp.status);
    return UCS_OK;
}

static void ucp_proto_rndv_get_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "%s completed", req->send.proto_config->proto->name);
    ucp_request_send(req, 0); /* reschedule to send ATS */
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_get_zcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    ucp_rkey_h rkey    = req->send.rndv.rkey;
    uct_rkey_t tl_rkey = rkey->tl_rkey[lpriv->super.rkey_index].rkey.rkey;
    uct_iov_t iov;

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               lpriv->super.memh_index,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               next_iter, &iov);
    return uct_ep_get_zcopy(req->send.ep->uct_eps[lpriv->super.lane], &iov, 1,
                            req->send.rndv.remote_address +
                                    req->send.state.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_rndv_get_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_bulk_priv_t *rpriv = req->send.proto_config->priv;

    if (ucp_datatype_iter_is_end(&req->send.state.dt_iter)) {
        if (req->send.state.dt_iter.length > 0) {
            ucs_assert(req->send.state.uct_comp.count == 0);
        }
        return ucp_proto_rndv_ack_progress(req, UCP_AM_ID_RNDV_ATS,
                                           ucp_proto_rndv_get_complete);
    } else {
        return ucp_proto_multi_zcopy_progress(req, &rpriv->mpriv, NULL,
                                              UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                                              ucp_proto_rndv_get_zcopy_send_func,
                                              ucp_proto_rndv_get_completion);
    }
}

static ucp_proto_t ucp_rndv_get_zcopy_proto = {
    .name       = "rndv/get/zcopy",
    .flags      = 0,
    .init       = ucp_proto_rndv_get_zcopy_init,
    .config_str = ucp_proto_rndv_bulk_config_str,
    .progress   = ucp_proto_rndv_get_zcopy_progress
};
UCP_PROTO_REGISTER(&ucp_rndv_get_zcopy_proto);


static ucs_status_t
ucp_proto_rndv_ats_init(const ucp_proto_init_params_t *params)
{
    ucs_status_t status;

    if (params->select_param->op_id != UCP_OP_ID_RNDV_RECV) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (params->rkey_config_key != NULL) {
        /* This ATS-only protocol will not take care of releasing the remote, so
           disqualify if remote key is present */
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_rndv_ack_init(params);
    if (status != UCS_OK) {
        return status;
    }

    /* Support only 0-length messages */
    *params->priv_size                 = sizeof(ucp_proto_rndv_ack_priv_t);
    params->caps->cfg_thresh           = 0;
    params->caps->cfg_priority         = 1;
    params->caps->min_length           = 0;
    params->caps->num_ranges           = 1;
    params->caps->ranges[0].max_length = 0;
    params->caps->ranges[0].perf       = ucp_proto_rndv_ack_time(params);
    return UCS_OK;
}

static ucs_status_t ucp_proto_rndv_ats_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_rndv_ack_progress(
            req, UCP_AM_ID_RNDV_ATS, ucp_proto_request_zcopy_complete_success);
}

static ucp_proto_t ucp_rndv_ats_proto = {
    .name       = "rndv/ats",
    .flags      = 0,
    .init       = ucp_proto_rndv_ats_init,
    .config_str = ucp_proto_rndv_ack_config_str,
    .progress   = ucp_proto_rndv_ats_progress
};
UCP_PROTO_REGISTER(&ucp_rndv_ats_proto);
