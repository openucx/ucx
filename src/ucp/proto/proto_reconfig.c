/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_debug.h"
#include "proto_select.h"
#include "proto_am.inl"
#include "proto_common.inl"

#include <ucp/am/ucp_am.inl>
#include <ucp/core/ucp_worker.inl>
#include <ucs/memory/memory_type.h>
#include <ucs/sys/math.h>


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

    /* coverity[address_free] */
    return req->send.uct.func(&req->send.uct);
}

static void ucp_proto_reconfig_abort(ucp_request_t *req, ucs_status_t status)
{
    if (ucp_proto_config_is_am(req->send.proto_config)) {
        ucp_am_release_user_header(req);
    }

    ucp_request_complete_send(req, status);
}

static int
ucp_proto_reconfig_report_no_rma_emulation_no_proto(ucp_request_t *req,
                                                    ucp_ep_h ep)
{
    ucp_operation_id_t op_id;
    ucs_memory_type_t local_mem_type, remote_mem_type;

    if (ep->worker->context->config.ext.proto_emulation_enable) {
        return 0;
    }

    op_id = ucp_proto_select_op_id(&req->send.proto_config->select_param);
    if (((op_id != UCP_OP_ID_PUT) && (op_id != UCP_OP_ID_GET))) {
        return 0;
    }

    local_mem_type  = req->send.proto_config->select_param.mem_type;
    remote_mem_type = req->send.rma.rkey->mem_type;

    ucs_error("No zero-copy protocol found for %s %s %s %s, %zu bytes. "
              "Please check for proper GPU and/or HCA support, or set "
              "UCX_PROTO_EMULATION_ENABLE=y to proceed by allowing slower "
              "software emulation.",
              (op_id == UCP_OP_ID_PUT) ? "put from" : "get into",
              ucs_memory_type_names[local_mem_type],
              (op_id == UCP_OP_ID_PUT) ? "to" : "from",
              ucs_memory_type_names[remote_mem_type],
              req->send.state.dt_iter.length);
    return 1;
}

static ucs_status_t ucp_proto_reconfig_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep        = req->send.ep;
    UCS_STRING_BUFFER_ONSTACK(strb, 256);
    ucs_status_t status;

    /* This protocol should not be selected for valid and connected endpoint */
    if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
        if (ucp_proto_reconfig_report_no_rma_emulation_no_proto(req, ep)) {
            ucp_proto_request_abort(req, UCS_ERR_CANCELED);
            return UCS_OK;
        }

        ucp_ep_config_name(ep->worker, req->send.proto_config->ep_cfg_index,
                           &strb);
        ucs_string_buffer_appendf(&strb, " | ");
        ucp_proto_select_info_str(ep->worker,
                                  req->send.proto_config->rkey_cfg_index,
                                  &req->send.proto_config->select_param,
                                  ucp_operation_names, &strb);
        ucs_error("cannot find remote protocol for: %s",
                  ucs_string_buffer_cstr(&strb));
        ucp_proto_request_abort(req, UCS_ERR_CANCELED);
        return UCS_OK;
    }

    if (ucp_proto_config_is_am(req->send.proto_config)) {
        status = ucp_proto_am_req_copy_header(req);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }
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
            ucs_linear_func_make(UCS_INFINITY, 0);
    ucp_proto_perf_add_funcs(perf, 0, SIZE_MAX, perf_factors,
                             ucp_proto_perf_node_new_data("dummy", ""), NULL);
    ucp_proto_select_add_proto(init_params, UCS_MEMUNITS_INF, 0, perf, NULL, 0);
}

ucp_proto_t ucp_reconfig_proto = {
    .name     = "reconfig",
    .desc     = "stub protocol",
    .flags    = UCP_PROTO_FLAG_INVALID,
    .probe    = ucp_proto_reconfig_probe,
    .query    = ucp_proto_default_query,
    .progress = {ucp_proto_reconfig_progress},
    .abort    = ucp_proto_reconfig_abort,
    .reset    = (ucp_request_reset_func_t)ucs_empty_function_return_success
};
