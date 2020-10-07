/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_COMMON_INL_
#define UCP_PROTO_COMMON_INL_

#include "proto_common.h"
#include "proto_select.inl"

#include <ucp/dt/datatype_iter.inl>
#include <ucp/core/ucp_request.inl>


static UCS_F_ALWAYS_INLINE void
ucp_proto_request_bcopy_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_datatype_iter_cleanup(&req->send.dt_iter, UINT_MAX);
    ucp_request_complete_send(req, status);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_completion_init(ucp_request_t *req,
                                  uct_completion_callback_t comp_func)
{
    req->send.state.uct_comp.func   = comp_func;
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.status = UCS_OK;
    /* extra ref to be decremented when all sent */
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_request_zcopy_init(ucp_request_t *req, ucp_md_map_t md_map,
                             uct_completion_callback_t comp_func)
{
    ucp_ep_h ep = req->send.ep;
    ucs_status_t status;

    ucp_trace_req(req, "ucp_proto_zcopy_request_init for %s",
                  req->send.proto_config->proto->name);

    ucp_proto_request_completion_init(req, comp_func);

    status = ucp_datatype_iter_mem_reg(ep->worker->context, &req->send.dt_iter,
                                       md_map);
    if (status != UCS_OK) {
        return status;
    }

    ucp_trace_req(req, "registered md_map 0x%"PRIx64"/0x%"PRIx64,
                  req->send.dt_iter.type.contig.reg.md_map, md_map);

    /* We expect the registration to happen on all desired memory domains, since
     * the protocol initialization code would already disqualify any memory
     * domain which does not support registration, or does not require a local
     * memory key for zero-copy operations. This assumption simplifies memory
     * key lookups during protocol progress.
     */
    ucs_assert(req->send.dt_iter.type.contig.reg.md_map == md_map);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_zcopy_cleanup(ucp_request_t *req)
{
    ucp_datatype_iter_mem_dereg(req->send.ep->worker->context,
                                &req->send.dt_iter);
    ucp_datatype_iter_cleanup(&req->send.dt_iter, UCS_BIT(UCP_DATATYPE_CONTIG));
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_zcopy_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_proto_request_zcopy_cleanup(req);
    ucp_request_complete_send(req, status);
}

/* Select protocol for the request and initialize protocol-related fields */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_request_set_proto(ucp_worker_h worker, ucp_ep_h ep,
                            ucp_request_t *req, ucp_proto_select_t *proto_select,
                            ucp_worker_cfg_index_t rkey_cfg_index,
                            const ucp_proto_select_param_t *sel_param,
                            size_t msg_length)
{
    const ucp_proto_threshold_elem_t *thresh_elem;
    const ucp_proto_t *proto;
    ucs_string_buffer_t strb;

    thresh_elem = ucp_proto_select_lookup(worker, proto_select, ep->cfg_index,
                                          rkey_cfg_index, sel_param, msg_length);
    if (ucs_unlikely(thresh_elem == NULL)) {
        ucp_proto_request_select_error(req, proto_select, rkey_cfg_index,
                                       sel_param, msg_length);
        return UCS_ERR_UNREACHABLE;
    }

    proto                  = thresh_elem->proto_config.proto;
    req->send.proto_config = &thresh_elem->proto_config;
    req->send.uct.func     = proto->progress;

    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        ucp_proto_select_param_str(sel_param, &strb);
        ucp_trace_req(req, "selected protocol %s for %s length %zu",
                      proto->name, ucs_string_buffer_cstr(&strb), msg_length);
        ucs_string_buffer_cleanup(&strb);
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_proto_request_send_op(ucp_ep_h ep, ucp_proto_select_t *proto_select,
                          ucp_worker_cfg_index_t rkey_cfg_index,
                          ucp_request_t *req, ucp_operation_id_t op_id,
                          const void *buffer, size_t count, ucp_datatype_t datatype,
                          size_t contig_length, const ucp_request_param_t *param)
{
    ucp_worker_h worker     = ep->worker;
    ucp_proto_select_param_t sel_param;
    ucs_status_t status;
    uint8_t sg_count;

    req->flags   = 0;
    req->send.ep = ep;

    ucp_datatype_iter_init(worker->context, (void*)buffer, count, datatype,
                           contig_length, &req->send.dt_iter, &sg_count);

    ucp_proto_select_param_init(&sel_param, op_id, param->op_attr_mask,
                                req->send.dt_iter.dt_class,
                                req->send.dt_iter.mem_type,
                                sg_count);

    status = ucp_proto_request_set_proto(worker, ep, req, proto_select, rkey_cfg_index,
                                      &sel_param, contig_length);
    if (status != UCS_OK) {
        goto out_put_request;
    }

    ucp_request_send(req, 0);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        goto out_put_request;
    }

    /* set callback flag to allow calling it. we didn't set it before to prevent
     * it from being called if the send is completed immediately.
     */
    ucp_request_set_send_callback_param(param, req, send);

    ucs_trace_req("returning send request %p", req);
    return req + 1;

out_put_request:
    ucs_trace_req("releasing send request %p, returning status %s", req,
                  ucs_status_string(status));
    status = req->status;
    ucp_request_put_param(param, req);
    return UCS_STATUS_PTR(status);
}

#endif
