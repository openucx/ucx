/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_common.inl"

#include <ucp/core/ucp_worker.inl>


/* Select a new protocol and start progressing it */
static ucs_status_t ucp_proto_reconfig_select_progress(uct_pending_req_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep         = req->send.ep;
    ucp_worker_h worker = ep->worker;
    ucp_worker_cfg_index_t prev_rkey_cfg_index;
    ucp_rkey_config_key_t rkey_config_key;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_t *proto_select;
    ucs_status_t status;

    /*
     * Find the protocol selection hash: could be either on the endpoint or on
     * the remote key
     */
    prev_rkey_cfg_index = req->send.proto_config->rkey_cfg_index;
    if (prev_rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        proto_select   = &worker->ep_config[ep->cfg_index].proto_select;
        rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
    } else {
        rkey_config_key = worker->rkey_config[prev_rkey_cfg_index].key;
        rkey_config_key.ep_cfg_index = ep->cfg_index;

        status = ucp_worker_rkey_config_get(worker, &rkey_config_key, NULL,
                                            &rkey_cfg_index);
        if (status != UCS_OK) {
            ucs_error("failed to switch to new rkey");
            return UCS_OK;
        }

        proto_select = &worker->rkey_config[rkey_cfg_index].proto_select;
    }

    /* Select from protocol hash according to saved request parameters */
    status = ucp_proto_request_set_proto(worker, ep, req, proto_select,
                                         rkey_cfg_index,
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
    init_params->caps->ranges[0].perf       = ucs_linear_func_make(INFINITY, 0);

    return UCS_OK;
}

static ucp_proto_t ucp_reconfig_proto = {
    .name       = "reconfig",
    .flags      = UCP_PROTO_FLAG_INVALID,
    .init       = ucp_proto_reconfig_init,
    .config_str = (ucp_proto_config_str_func_t)ucs_empty_function,
    .progress   = ucp_proto_reconfig_progress
};
UCP_PROTO_REGISTER(&ucp_reconfig_proto);
