/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_COMMON_INL_
#define UCP_PROTO_COMMON_INL_

#include "proto_common.h"
#include "proto_select.inl"

#include <ucp/dt/datatype_iter.inl>
#include <ucp/core/ucp_request.inl>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_request_complete_success(ucp_request_t *req)
{
    ucp_request_complete_send(req, UCS_OK);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_request_bcopy_complete_success(ucp_request_t *req)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0, UCP_DT_MASK_ALL);
    if (ucp_proto_select_op_id(&req->send.proto_config->select_param) ==
        UCP_OP_ID_TAG_SEND) {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER)
    }

    return ucp_proto_request_complete_success(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_bcopy_send_func_status(ssize_t packed_size)
{
    if (ucs_unlikely(packed_size < 0)) {
        return (ucs_status_t)packed_size;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_msg_multi_request_init(ucp_request_t *req)
{
    if (!ucp_datatype_iter_is_begin(&req->send.state.dt_iter)) {
        return;
    }

    req->send.msg_proto.message_id = req->send.ep->worker->am_message_id++;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_completion_init(uct_completion_t *comp,
                          uct_completion_callback_t comp_func)
{
    comp->func   = comp_func;
    comp->count  = 1;
    comp->status = UCS_OK;
    /* extra ref to be decremented when all sent */
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_request_zcopy_init(ucp_request_t *req, ucp_md_map_t md_map,
                             uct_completion_callback_t comp_func,
                             unsigned uct_reg_flags, unsigned dt_mask)
{
    ucp_ep_h ep = req->send.ep;

    ucp_trace_req(req, "ucp_proto_request_zcopy_init for %s",
                  req->send.proto_config->proto->name);

    ucp_proto_completion_init(&req->send.state.uct_comp, comp_func);

    return UCS_PROFILE_CALL(ucp_datatype_iter_mem_reg, ep->worker->context,
                            &req->send.state.dt_iter, md_map, uct_reg_flags,
                            dt_mask);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_zcopy_clean(ucp_request_t *req, unsigned dt_mask)
{
    ucp_datatype_iter_mem_dereg(&req->send.state.dt_iter, dt_mask);
    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_zcopy_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1,
                              UCP_DT_MASK_CONTIG_IOV);
    if (ucp_proto_select_op_id(&req->send.proto_config->select_param) ==
        UCP_OP_ID_TAG_SEND) {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER)
    }

    ucp_request_complete_send(req, status);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_request_zcopy_complete_success(ucp_request_t *req)
{
    ucp_proto_request_zcopy_complete(req, UCS_OK);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_common_multi_frag_is_completed(ucp_request_t *req)
{
    return req->send.state.completed_size == req->send.state.dt_iter.length;
}

/**
 * Add 'frag_size' to 'req->send.state.completed_size' and return nonzero if
 * completed_size reached dt_iter.length
 */
static UCS_F_ALWAYS_INLINE int
ucp_proto_common_frag_complete(ucp_request_t *req, size_t frag_size,
                               const char *title)
{
    req->send.state.completed_size += frag_size;

    ucp_trace_req(req, "%s completed %zu, overall %zu/%zu", title, frag_size,
                  req->send.state.completed_size,
                  req->send.state.dt_iter.length);

    ucs_assertv(req->send.state.completed_size <=
                req->send.state.dt_iter.length,
                "req %p: proto %s completed_size=%zu dt_iter.length=%zu", req,
                req->send.proto_config->proto->name,
                req->send.state.completed_size, req->send.state.dt_iter.length);

    return ucp_proto_common_multi_frag_is_completed(req);
}

/* Adjust a zero-copy operation to a minimal fragment size, by overlapping with
 * the previous or subsequent fragment, assuming it will not impact data
 * delivery. For example, it's safe to do with one-sided operations.
 */
static UCS_F_ALWAYS_INLINE void
ucp_proto_common_zcopy_adjust_min_frag(ucp_request_t *req, size_t min_frag,
                                       size_t length, uct_iov_t *iov,
                                       size_t iovcnt, size_t *offset_p)
{
    ssize_t min_frag_diff = min_frag - length;
    if (ucs_likely(min_frag_diff < 0)) {
        /* if length is already larger than min_frag - no need to adjust */
        return;
    }

    /* Make sure increasing payload size would not exceed request data size */
    ucs_assertv((length + min_frag_diff) <= req->send.state.dt_iter.length,
                "req=%p length=%zu min_frag_diff=%zd dt_iter.length=%zu", req,
                length, min_frag_diff, req->send.state.dt_iter.length);

    ucp_proto_common_zcopy_adjust_min_frag_always(req, min_frag_diff, iov,
                                                  iovcnt, offset_p);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_set_stage(ucp_request_t *req, uint8_t proto_stage)
{
    const ucp_proto_t *proto = req->send.proto_config->proto;

    ucs_assertv(proto_stage < UCP_PROTO_STAGE_LAST, "stage=%"PRIu8,
                proto_stage);
    ucs_assert(proto->progress[proto_stage] != NULL);

    ucp_trace_req(req, "set to stage %u, progress function '%s'", proto_stage,
                  ucs_debug_get_symbol_name(proto->progress[proto_stage]));
    req->send.proto_stage = proto_stage;

    /* Set pointer to progress function */
    if (req->send.ep->worker->context->config.progress_wrapper_enabled) {
        req->send.uct.func = ucp_request_progress_wrapper;
    } else {
        req->send.uct.func = proto->progress[proto_stage];
    }
}

/* Select protocol for the request and initialize protocol-related fields */
static void ucp_proto_request_set_proto(ucp_request_t *req,
                                        const ucp_proto_config_t *proto_config,
                                        size_t msg_length)
{
    ucs_assertv(req->flags & UCP_REQUEST_FLAG_PROTO_SEND, "flags=0x%"PRIx32,
                req->flags);

    req->send.proto_config = proto_config;
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        ucp_proto_trace_selected(req, msg_length);
    }

    ucp_proto_request_set_stage(req, UCP_PROTO_STAGE_START);
}

/* Select protocol for the request and initialize protocol-related fields */
static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_request_lookup_proto(
        ucp_worker_h worker, ucp_ep_h ep, ucp_request_t *req,
        ucp_proto_select_t *proto_select, ucp_worker_cfg_index_t rkey_cfg_index,
        const ucp_proto_select_param_t *sel_param, size_t msg_length)
{
    const ucp_proto_threshold_elem_t *thresh_elem;

    thresh_elem = ucp_proto_select_lookup(worker, proto_select, ep->cfg_index,
                                          rkey_cfg_index, sel_param, msg_length);
    if (UCS_ENABLE_ASSERT && (thresh_elem == NULL)) {
        /* We expect that a protocol will always be found, or we will fallback
           to 'reconfig' placeholder */
        ucp_proto_request_select_error(req, proto_select, rkey_cfg_index,
                                       sel_param, msg_length);
        return UCS_ERR_UNREACHABLE;
    }

    /* Set pointer to request's protocol configuration */
    ucs_assert(thresh_elem->proto_config.ep_cfg_index == ep->cfg_index);
    ucs_assert(thresh_elem->proto_config.rkey_cfg_index == rkey_cfg_index);
    ucp_proto_request_set_proto(req, &thresh_elem->proto_config, msg_length);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_request_send_init(ucp_request_t *req, ucp_ep_h ep, uint32_t flags)
{
    req->flags   = UCP_REQUEST_FLAG_PROTO_SEND | flags;
    req->send.ep = ep;
}


static UCS_F_ALWAYS_INLINE ucs_status_ptr_t ucp_proto_request_send_op_common(
        ucp_worker_h worker, ucp_ep_h ep, ucp_proto_select_t *proto_select,
        ucp_worker_cfg_index_t rkey_cfg_index, ucp_request_t *req,
        const ucp_request_param_t *param,
        const ucp_proto_select_param_t *select_param, size_t msg_length)
{
    ucs_string_buffer_t strb;
    ucs_status_t status;

    status = UCS_PROFILE_CALL(ucp_proto_request_lookup_proto, worker, ep, req,
                              proto_select, rkey_cfg_index, select_param,
                              msg_length);
    if (status != UCS_OK) {
        ucp_request_put_param(param, req);
        return UCS_STATUS_PTR(status);
    }

    UCS_PROFILE_CALL_VOID(ucp_request_send, req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        /* coverity[offset_free] */
        ucp_request_imm_cmpl_param(param, req, send);
    }

    ucp_request_set_send_callback_param(param, req, send);

    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        ucs_string_buffer_init(&strb);
        ucp_datatype_iter_str(&req->send.state.dt_iter, &strb);
        ucs_trace_req("returning send request %p: %s %s", req,
                      ucp_operation_names[ucp_proto_select_op_id(select_param)],
                      ucs_string_buffer_cstr(&strb));
        ucs_string_buffer_cleanup(&strb);
    }

    return req + 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_proto_request_send_op(ucp_ep_h ep, ucp_proto_select_t *proto_select,
                          ucp_worker_cfg_index_t rkey_cfg_index,
                          ucp_request_t *req, ucp_operation_id_t op_id,
                          const void *buffer, size_t count,
                          ucp_datatype_t datatype, size_t contig_length,
                          const ucp_request_param_t *param,
                          size_t header_length, uint8_t op_flags)
{
    ucp_worker_h worker = ep->worker;
    ucp_proto_select_param_t sel_param;
    ucs_status_t status;
    size_t msg_length;
    uint8_t sg_count;

    ucp_proto_request_send_init(req, ep, 0);

    status = UCS_PROFILE_CALL(ucp_datatype_iter_init, worker->context,
                              (void*)buffer, count, datatype, contig_length, 1,
                              &req->send.state.dt_iter, &sg_count, param);
    if (status != UCS_OK) {
        ucp_request_put_param(param, req);
        return UCS_STATUS_PTR(status);
    }

    ucp_proto_select_param_init(&sel_param, op_id, param->op_attr_mask,
                                op_flags, req->send.state.dt_iter.dt_class,
                                &req->send.state.dt_iter.mem_info, sg_count);

    msg_length = req->send.state.dt_iter.length + header_length;
    return ucp_proto_request_send_op_common(worker, ep, proto_select,
                                            rkey_cfg_index, req, param,
                                            &sel_param, msg_length);
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t ucp_proto_request_send_op_reply(
        ucp_ep_h ep, ucp_proto_select_t *proto_select,
        ucp_worker_cfg_index_t rkey_cfg_index, ucp_request_t *req,
        ucp_operation_id_t op_id, const void *buffer, size_t count,
        ucp_datatype_t datatype, size_t contig_length,
        const ucp_request_param_t *param)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    ucp_proto_select_param_t sel_param;
    ucp_memory_info_t reply_mem_info;
    ucs_status_t status;
    uint8_t sg_count;

    ucp_proto_request_send_init(req, ep, 0);

    status = UCS_PROFILE_CALL(ucp_datatype_iter_init, context, (void*)buffer,
                              count, datatype, contig_length, 1,
                              &req->send.state.dt_iter, &sg_count, param);
    if (status != UCS_OK) {
        ucp_request_put_param(param, req);
        return UCS_STATUS_PTR(status);
    }

    UCS_PROFILE_CALL_VOID(ucp_memory_detect, context, param->reply_buffer,
                          contig_length, &reply_mem_info);

    ucp_proto_select_param_init_reply(&sel_param, op_id, param->op_attr_mask, 0,
                                      req->send.state.dt_iter.dt_class,
                                      &req->send.state.dt_iter.mem_info,
                                      sg_count, &reply_mem_info);

    return ucp_proto_request_send_op_common(worker, ep, proto_select,
                                            rkey_cfg_index, req, param,
                                            &sel_param,
                                            req->send.state.dt_iter.length);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_request_pack_rkey(ucp_request_t *req, ucp_md_map_t md_map,
                            uint64_t distance_dev_map,
                            const ucs_sys_dev_distance_t *dev_distance,
                            void *rkey_buffer)
{
    const ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_mem_h memh;
    ssize_t packed_rkey_size;

    /* For contiguous buffer, pack one rkey
     * TODO to support IOV datatype write N [address+length] records,
     */
    ucs_assertv(dt_iter->dt_class == UCP_DATATYPE_CONTIG, "dt_class=%s",
                ucp_datatype_class_names[dt_iter->dt_class]);

    memh = dt_iter->type.contig.memh;

    /* Since global VA registration doesn't support invalidation yet, and error
     * handling is enabled on this EP, we replace GVA registrations with
     * regular ones */
    if (ucp_ep_config_err_mode_eq(req->send.ep,
                                  UCP_ERR_HANDLING_MODE_PEER) &&
        ucs_unlikely(memh->flags & UCP_MEMH_FLAG_HAS_AUTO_GVA)) {
        ucp_memh_disable_gva(memh, md_map);
    }

    if (!ucs_test_all_flags(memh->md_map, md_map)) {
        ucs_trace("dt_iter_md_map=0x%"PRIx64" md_map=0x%"PRIx64, memh->md_map,
                  md_map);
    }

    packed_rkey_size = ucp_rkey_pack_memh(
            req->send.ep->worker->context, md_map & memh->md_map, memh,
            dt_iter->type.contig.buffer, dt_iter->length, &dt_iter->mem_info,
            distance_dev_map, dev_distance,
            ucp_ep_config(req->send.ep)->uct_rkey_pack_flags, rkey_buffer);

    if (packed_rkey_size < 0) {
        ucs_error("failed to pack remote key: %s",
                  ucs_status_string((ucs_status_t)packed_rkey_size));
        return 0;
    }

    req->flags |= UCP_REQUEST_FLAG_RKEY_INUSE;

    return packed_rkey_size;
}

#endif
