/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tag_match.inl"
#include "eager.h"
#include "tag_rndv.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_common.inl>
#include <ucs/datastruct/mpool.inl>
#include <string.h>


static UCS_F_ALWAYS_INLINE size_t
ucp_tag_get_rndv_threshold(const ucp_request_t *req, size_t count,
                           size_t max_iov, size_t rndv_rma_thresh,
                           size_t rndv_am_thresh)
{
    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_IOV:
        if ((count > max_iov) &&
            ucp_ep_is_tag_offload_enabled(ucp_ep_config(req->send.ep))) {
            /* Make sure SW RNDV will be used, because tag offload does
             * not support multi-packet eager protocols. */
            return 1;
        }
        /* Fall through */
    case UCP_DATATYPE_CONTIG:
        return ucs_min(rndv_rma_thresh, rndv_am_thresh);
    case UCP_DATATYPE_GENERIC:
        return rndv_am_thresh;
    default:
        ucs_error("Invalid data type 0x%"PRIx64, req->send.datatype);
    }

    return SIZE_MAX;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_tag_send_req(ucp_request_t *req, size_t dt_count,
                 const ucp_ep_msg_config_t* msg_config,
                 const ucp_request_param_t *param,
                 const ucp_request_send_proto_t *proto)
{
    ssize_t max_short          = ucp_proto_get_short_max(req, msg_config);
    ucp_ep_config_t *ep_config = ucp_ep_config(req->send.ep);
    ucs_status_t status;
    size_t zcopy_thresh;
    size_t rndv_thresh;
    size_t rndv_rma_thresh;
    size_t rndv_am_thresh;

    ucp_request_param_rndv_thresh(req, param, &ep_config->tag.rndv.rma_thresh,
                                  &ep_config->tag.rndv.am_thresh,
                                  &rndv_rma_thresh, &rndv_am_thresh);

    rndv_thresh = ucp_tag_get_rndv_threshold(req, dt_count, msg_config->max_iov,
                                             rndv_rma_thresh, rndv_am_thresh);

    if (!(param->op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) ||
        ucs_unlikely(!UCP_MEM_IS_HOST(req->send.mem_type))) {
        zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config, dt_count,
                                                     rndv_thresh);
    } else {
        zcopy_thresh = rndv_thresh;
    }

    ucs_trace_req("select tag request(%p) progress algorithm datatype=0x%"PRIx64
                  " buffer=%p length=%zu mem_type:%s max_short=%zd rndv_thresh=%zu "
                  "zcopy_thresh=%zu zcopy_enabled=%d",
                  req, req->send.datatype, req->send.buffer, req->send.length,
                  ucs_memory_type_names[req->send.mem_type],
                  max_short, rndv_thresh, zcopy_thresh,
                  !(param->op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL));

    status = ucp_request_send_start(req, max_short, zcopy_thresh, rndv_thresh,
                                    dt_count, 0, req->send.length, msg_config,
                                    proto);
    if (ucs_unlikely(status != UCS_OK)) {
        if (status == UCS_ERR_NO_PROGRESS) {
            /* RMA/AM rendezvous */
            ucs_assert(req->send.length >= rndv_thresh);
            status = ucp_tag_send_start_rndv(req);
            if (status != UCS_OK) {
                return UCS_STATUS_PTR(status);
            }

            UCP_EP_STAT_TAG_OP(req->send.ep, RNDV);
        } else {
            return UCS_STATUS_PTR(status);
        }
    }

    if (req->flags & UCP_REQUEST_FLAG_SYNC) {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER_SYNC);
    } else {
        UCP_EP_STAT_TAG_OP(req->send.ep, EAGER);
    }

    /*
     * Start the request.
     * If it is completed immediately and this completion is allowed,
     * release the request and return the status.
     * Otherwise, return the request.
     */
    ucp_request_send(req, 0);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucp_request_imm_cmpl_param(param, req, send);
    }

    ucp_request_set_send_callback_param(param, req, send);
    ucs_trace_req("returning send request %p", req);
    return req + 1;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_send_req_init(ucp_request_t* req, ucp_ep_h ep, const void* buffer,
                      uintptr_t datatype, ucs_memory_type_t memory_type,
                      size_t count, ucp_tag_t tag, uint32_t flags)
{
    req->flags                  = flags | UCP_REQUEST_FLAG_SEND_TAG;
    req->send.ep                = ep;
    req->send.buffer            = (void*)buffer;
    req->send.datatype          = datatype;
    req->send.msg_proto.tag.tag = tag;
    ucp_request_send_state_init(req, datatype, count);
    req->send.length       = ucp_dt_length(req->send.datatype, count,
                                           req->send.buffer,
                                           &req->send.state.dt);
    req->send.mem_type     = ucp_get_memory_type(ep->worker->context, (void*)buffer,
                                                 req->send.length, memory_type);
    req->send.lane         = ucp_ep_config(ep)->tag.lane;
    req->send.pending_lane = UCP_NULL_LANE;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_send_inline(ucp_ep_h ep, const void *buffer, size_t length, ucp_tag_t tag)
{
    ucs_status_t status;

    if (ucp_proto_is_inline(ep, &ucp_ep_config(ep)->tag.max_eager_short, length)) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(ucp_eager_hdr_t));
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uint64_t));
        status = uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_EAGER_ONLY,
                                 tag, buffer, length);
    } else if (ucp_proto_is_inline(ep,
                                   &ucp_ep_config(ep)->tag.offload.max_eager_short,
                                   length)) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uct_tag_t));
        status = uct_ep_tag_eager_short(ucp_ep_get_tag_uct_ep(ep), tag, buffer,
                                        length);
    } else {
        return UCS_ERR_NO_RESOURCE;
    }

    if (status != UCS_ERR_NO_RESOURCE) {
        UCP_EP_STAT_TAG_OP(ep, EAGER);
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_nb,
                 (ep, buffer, count, datatype, tag, cb),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb,
        .datatype     = datatype
    };

    return ucp_tag_send_nbx(ep, buffer, count, tag, &param);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_send_nbr,
                 (ep, buffer, count, datatype, tag, request),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_tag_t tag, void *request)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_REQUEST |
                        UCP_OP_ATTR_FLAG_FAST_CMPL,
        .datatype     = datatype,
        .request      = request
    };
    ucs_status_ptr_t status;

    status = ucp_tag_send_nbx(ep, buffer, count, tag, &param);
    if (ucs_likely(status == UCS_OK)) {
        return UCS_OK;
    }

    if (ucs_unlikely(UCS_PTR_IS_ERR(status))) {
        return UCS_PTR_STATUS(status);
    }
    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_sync_nb,
                 (ep, buffer, count, datatype, tag, cb),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb,
        .datatype     = datatype
    };

    return ucp_tag_send_sync_nbx(ep, buffer, count, tag, &param);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_nbx,
                 (ep, buffer, count, tag, param),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_tag_t tag, const ucp_request_param_t *param)
{
    size_t contig_length = 0;
    ucs_status_t status;
    ucp_request_t *req;
    ucs_status_ptr_t ret;
    uintptr_t datatype;
    ucs_memory_type_t memory_type;
    uint32_t attr_mask;
    ucp_worker_h worker;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(ep->worker->context, UCP_FEATURE_TAG,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("send_nbx buffer %p count %zu tag %"PRIx64" to %s",
                  buffer, count, tag, ucp_ep_peer_name(ep));

    attr_mask = param->op_attr_mask &
                (UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FLAG_NO_IMM_CMPL);

    if (ucs_likely(attr_mask == 0)) {
        status = UCS_PROFILE_CALL(ucp_tag_send_inline, ep, buffer, count, tag);
        ucp_request_send_check_status(status, ret, goto out);
        datatype      = ucp_dt_make_contig(1);
        contig_length = count;
    } else if (attr_mask == UCP_OP_ATTR_FIELD_DATATYPE) {
        datatype = param->datatype;
        if (ucs_likely(UCP_DT_IS_CONTIG(datatype))) {
            contig_length = ucp_contig_dt_length(datatype, count);
            status        = UCS_PROFILE_CALL(ucp_tag_send_inline, ep, buffer,
                                             contig_length, tag);
            ucp_request_send_check_status(status, ret, goto out);
        }
    } else {
        datatype = ucp_dt_make_contig(1);
    }

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
        goto out;
    }

    worker      = ep->worker;
    memory_type = ucp_request_param_mem_type(param);
    req         = ucp_request_get_param(worker, param,
                                        {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                        goto out;});

    if (worker->context->config.ext.proto_enable) {
        req->send.msg_proto.tag.tag = tag;

        ret = ucp_proto_request_send_op(ep, &ucp_ep_config(ep)->proto_select,
                                        UCP_WORKER_CFG_INDEX_NULL, req,
                                        UCP_OP_ID_TAG_SEND, buffer, count,
                                        datatype, contig_length, param);
    } else {
        ucp_tag_send_req_init(req, ep, buffer, datatype, memory_type, count,
                              tag, 0);
        ret = ucp_tag_send_req(req, count, &ucp_ep_config(ep)->tag.eager,
                               param, ucp_ep_config(ep)->tag.proto);
    }
out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_send_sync_nbx,
                 (ep, buffer, count, tag, param),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_tag_t tag, const ucp_request_param_t *param)
{
    ucs_status_t status;
    ucp_request_t *req;
    ucs_status_ptr_t ret;
    uintptr_t datatype;
    ucs_memory_type_t memory_type;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(ep->worker->context, UCP_FEATURE_TAG,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("send_sync_nbx buffer %p count %zu tag %"PRIx64" to %s",
                  buffer, count, tag, ucp_ep_peer_name(ep));

    datatype    = ucp_request_param_datatype(param);
    memory_type = ucp_request_param_mem_type(param);

    if (!ucp_ep_config_test_rndv_support(ucp_ep_config(ep))) {
        ret = UCS_STATUS_PTR(UCS_ERR_UNSUPPORTED);
        goto out;
    }

    status = ucp_ep_resolve_remote_id(ep, ucp_ep_config(ep)->tag.lane);
    if (status != UCS_OK) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }

    req = ucp_request_get_param(ep->worker, param,
                                {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                 goto out;});

    ucp_tag_send_req_init(req, ep, buffer, datatype, memory_type, count, tag,
                          UCP_REQUEST_FLAG_SYNC);
    ret = ucp_tag_send_req(req, count, &ucp_ep_config(ep)->tag.eager,
                           param, ucp_ep_config(ep)->tag.sync_proto);
out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ret;
}
