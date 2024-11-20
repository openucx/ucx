/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_debug.h"
#include "proto_select.h"
#include "proto_common.inl"

#include <ucp/core/ucp_worker.inl>


/* Select a new protocol and start progressing it */
static ucs_status_t ucp_proto_reconfig_select_progress(uct_pending_req_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    status = ucp_proto_request_init(req, &req->send.proto_config->select_param);
    if (ucs_unlikely(status != UCS_OK)) {
        /* will try again later */
        return UCS_ERR_NO_RESOURCE;
    }

    return req->send.uct.func(&req->send.uct);
}

static ucs_status_t ucp_proto_reconfig_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep        = req->send.ep;
    UCS_STRING_BUFFER_ONSTACK(strb, 256);

    /* This protocol should not be selected for valid and connected endpoint */
    if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
        ucp_ep_config_name(ep->worker, req->send.proto_config->ep_cfg_index,
                           &strb);
        ucs_string_buffer_appendf(&strb, " | ");
        ucp_proto_select_info_str(ep->worker,
                                  req->send.proto_config->rkey_cfg_index,
                                  &req->send.proto_config->select_param,
                                  ucp_operation_names, &strb);
        ucs_error("cannot find remote protocol for: %s",
                  ucs_string_buffer_cstr(&strb));
        ucp_request_complete_send(req, UCS_ERR_CANCELED);
        return UCS_OK;
    }

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

static void ucp_proto_reconfig_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_perf_factors_t perf_factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    /* Default reconfiguration protocol is a fallback for any case protocol
     * selection is unsuccessful. The protocol keeps queuing requests until they
     * can be executed.
     * Its performance estimation is an "infinity" value, that is worse than any
     * other protocol.
     */
    status = ucp_proto_perf_create("reconfig", &perf);
    if (status != UCS_OK) {
        return;
    }

    perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] =
            ucs_linear_func_make(INFINITY, 0);
    ucp_proto_perf_add_funcs(perf, 0, SIZE_MAX, perf_factors, NULL, "dummy",
                             "");

    ucp_proto_select_add_proto(init_params, UCS_MEMUNITS_INF, 0, perf, NULL, 0);
}

ucp_proto_t ucp_reconfig_proto = {
    .name     = "reconfig",
    .desc     = "stub protocol",
    .flags    = UCP_PROTO_FLAG_INVALID,
    .probe    = ucp_proto_reconfig_probe,
    .query    = ucp_proto_default_query,
    .progress = {ucp_proto_reconfig_progress},
    .abort    = ucp_request_complete_send,
    .reset    = (ucp_request_reset_func_t)ucs_empty_function_return_success
};
