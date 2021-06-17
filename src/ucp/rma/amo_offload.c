/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "amo.inl"
#include "rma.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_single.inl>


static void ucp_proto_amo_progress_prep(uct_pending_req_t *self,
                                        ucp_request_t **req_p,
                                        uct_rkey_t *rkey_p)
{
    const ucp_proto_single_priv_t *spriv;
    ucp_request_t *req;

    req            = ucs_container_of(self, ucp_request_t, send.uct);
    spriv          = req->send.proto_config->priv;
    req->send.lane = spriv->super.lane;

    *rkey_p        = ucp_rma_request_get_tl_rkey(req, spriv->super.rkey_index);
    *req_p         = req;
}

static ucs_status_t
ucp_proto_amo_init(const ucp_proto_init_params_t *init_params, unsigned flags)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 40e-9,
        .super.cfg_thresh    = 0,
        .super.cfg_priority  = 20,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.flags         = flags,
        .lane_type           = UCP_LANE_TYPE_AMO,
        .tl_cap_flags        = 0
    };

    return ucp_proto_single_init(&params);
}

static ucs_status_t ucp_proto_amo_progress_post(uct_pending_req_t *self)
{
    ucp_request_t *req;
    uct_rkey_t rkey;

    ucp_proto_amo_progress_prep(self, &req, &rkey);
    return ucp_amo_progress_post(req, rkey);
}

static ucs_status_t
ucp_proto_amo_init_post(const ucp_proto_init_params_t *init_params)
{
    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_AMO_POST);

    return ucp_proto_amo_init(init_params,
                              UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                              UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS);
}

static ucp_proto_t ucp_get_amo_post_proto = {
    .name       = "amo/offload/post",
    .flags      = 0,
    .init       = ucp_proto_amo_init_post,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_proto_amo_progress_post
};
UCP_PROTO_REGISTER(&ucp_get_amo_post_proto);

static ucs_status_t ucp_proto_amo_progress_fetch(uct_pending_req_t *self)
{
    ucp_request_t *req;
    uct_rkey_t rkey;

    ucp_proto_amo_progress_prep(self, &req, &rkey);
    return ucp_amo_progress_fetch(req, rkey);
}

static ucs_status_t
ucp_proto_amo_init_fetch(const ucp_proto_init_params_t *init_params)
{
    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_AMO_FETCH);

    return ucp_proto_amo_init(init_params,
                              UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                              UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                              UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                              UCP_PROTO_COMMON_INIT_FLAG_RESPONSE);
}

static ucp_proto_t ucp_get_amo_fetch_proto = {
    .name       = "amo/offload/fetch",
    .flags      = 0,
    .init       = ucp_proto_amo_init_fetch,
    .config_str = ucp_proto_single_config_str,
    .progress   = ucp_proto_amo_progress_fetch
};
UCP_PROTO_REGISTER(&ucp_get_amo_fetch_proto);
