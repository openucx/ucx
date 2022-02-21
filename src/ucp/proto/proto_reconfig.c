/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_select.h"
#include "proto_common.inl"

#include <ucp/core/ucp_worker.inl>


/* Select a new protocol and start progressing it */
static ucs_status_t ucp_proto_reconfig_select_progress(uct_pending_req_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep         = req->send.ep;
    ucp_worker_h worker = ep->worker;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_t *proto_select;
    ucs_status_t status;

    proto_select = ucp_proto_select_get(worker, ep->cfg_index,
                                        req->send.proto_config->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return UCS_OK;
    }

    /* Select from protocol hash according to saved request parameters */
    status = ucp_proto_request_lookup_proto(
            worker, ep, req, proto_select, rkey_cfg_index,
            &req->send.proto_config->select_param,
            req->send.state.dt_iter.length);
    if (status != UCS_OK) {
        /* will try again later */
        return UCS_ERR_NO_RESOURCE;
    }

    return req->send.uct.func(&req->send.uct);
}

static ucs_status_t ucp_proto_reconfig_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep        = req->send.ep;

    /* This protocol should not be selected for valid and connected endpoint */
    ucs_assert(!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED));

    if (ep->cfg_index != req->send.proto_config->ep_cfg_index) {
        ucp_trace_req(req,
                      "ep configuration changed from %d to %d,"
                      " reconfigure protocol",
                      req->send.proto_config->ep_cfg_index, ep->cfg_index);
        return ucp_proto_reconfig_select_progress(self);
    }

    /* TODO select wireup lane when needed */
    req->send.lane = ucp_ep_config(ep)->key.am_lane;
    return UCS_ERR_NO_RESOURCE;
}

static ucs_status_t
ucp_proto_reconfig_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_perf_type_t perf_type;

    /* Default reconfiguration protocol is a fallback for any case protocol
     * selection is unsuccessful. The protocol keeps queuing requests until they
     * can be executed.
     */
    *init_params->priv_size                 = 0;
    init_params->caps->cfg_thresh           = UCS_MEMUNITS_INF;
    init_params->caps->cfg_priority         = 0;
    init_params->caps->min_length           = 0;
    init_params->caps->num_ranges           = 1;
    init_params->caps->ranges[0].max_length = SIZE_MAX;
    init_params->caps->ranges[0].name       = "reconfig";
    for (perf_type = 0; perf_type < UCP_PROTO_PERF_TYPE_LAST; ++perf_type) {
        init_params->caps->ranges[0].perf[perf_type] =
                ucs_linear_func_make(INFINITY, 0);
    }
    return UCS_OK;
}

ucp_proto_t ucp_reconfig_proto = {
    .name     = "reconfig",
    .desc     = "stub protocol",
    .flags    = UCP_PROTO_FLAG_INVALID,
    .init     = ucp_proto_reconfig_init,
    .query    = ucp_proto_default_query,
    .progress = {ucp_proto_reconfig_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};
