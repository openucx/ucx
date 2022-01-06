/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"
#include "tag_rndv.h"
#include "tag_match.inl"
#include "offload.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>


static void ucp_tag_recv_eager_multi(ucp_worker_h worker, ucp_request_t *req,
                                     ucp_recv_desc_t *rdesc)
{
    ucp_eager_first_hdr_t *first_hdr;
    uint64_t msg_id;
    ucs_status_t status;

    UCP_WORKER_STAT_EAGER_MSG(worker, rdesc->flags);
    UCP_WORKER_STAT_EAGER_CHUNK(worker, UNEXP);
    ucs_assert(rdesc->flags & UCP_RECV_DESC_FLAG_EAGER);

    req->recv.tag.info.sender_tag = ucp_rdesc_get_tag(rdesc);

    if (rdesc->flags & UCP_RECV_DESC_FLAG_EAGER_OFFLOAD) {
        ucp_request_recv_offload_first(
               req, (ucp_offload_first_desc_t*)(rdesc + 1), rdesc->length,
               rdesc->flags
               UCS_STATS_ARG(UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP));
    } else {
        first_hdr           = (ucp_eager_first_hdr_t*)(rdesc + 1);
        req->recv.remaining = req->recv.tag.info.length = first_hdr->total_len;
        msg_id              = first_hdr->msg_id;

        if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_EAGER_SYNC)) {
            ucp_tag_eager_sync_send_ack(worker, rdesc + 1, rdesc->flags);
        }

        status = ucp_tag_recv_request_process_rdesc(req, rdesc, 0, 0);
        if (status == UCS_INPROGRESS) {
            ucp_tag_frag_list_process_queue(
                    &worker->tm, req, msg_id
                    UCS_STATS_ARG(UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP));
        }
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_tag_recv_common(ucp_worker_h worker, void *buffer, size_t count,
                    uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                    ucp_request_t *req, ucp_recv_desc_t *rdesc,
                    const ucp_request_param_t *param, const char *debug_name)
{
    uint32_t req_flags = (param->op_attr_mask & UCP_OP_ATTR_FIELD_CALLBACK) ?
                         UCP_REQUEST_FLAG_CALLBACK : 0;
    ucp_request_queue_t *req_queue;
    ucs_memory_type_t memory_type;
    size_t hdr_len, recv_len;
    ucs_status_t status;

    ucp_trace_req(req, "%s buffer %p dt 0x%lx count %zu tag %"PRIx64"/%"PRIx64,
                  debug_name, buffer, datatype, count, tag, tag_mask);

#if ENABLE_DEBUG_DATA
    req->recv.proto_rndv_config = NULL;
#endif

    /* First, check the fast path case - single fragment
     * in this case avoid initializing most of request fields
     * */
    if (ucs_likely((rdesc != NULL) && (rdesc->flags & UCP_RECV_DESC_FLAG_EAGER_ONLY))) {
        UCS_PROFILE_REQUEST_EVENT(req, "eager_only_match", 0);
        UCP_WORKER_STAT_EAGER_MSG(worker, rdesc->flags);
        UCP_WORKER_STAT_EAGER_CHUNK(worker, UNEXP);

        if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_EAGER_SYNC)) {
            ucp_tag_eager_sync_send_ack(worker, rdesc + 1, rdesc->flags);
        }

        req->flags                    = UCP_REQUEST_FLAG_COMPLETED |
                                        UCP_REQUEST_FLAG_RECV_TAG;
        hdr_len                       = rdesc->payload_offset;
        recv_len                      = rdesc->length - hdr_len;
        req->recv.tag.info.sender_tag = ucp_rdesc_get_tag(rdesc);
        req->recv.tag.info.length     = recv_len;

        memory_type = ucp_request_get_memory_type(worker->context, buffer,
                                                  recv_len, param);
        status      = ucp_dt_unpack_only(worker, buffer, count, datatype,
                                         memory_type,
                                         UCS_PTR_BYTE_OFFSET(rdesc + 1, hdr_len),
                                         recv_len, 1);
        ucp_recv_desc_release(rdesc);

        req->status = status;
        UCS_PROFILE_REQUEST_EVENT(req, "complete_imm_tag_recv", 0);

        ucp_request_imm_cmpl_param(param, req, recv, &req->recv.tag.info);
    }

    /* TODO: allocate request only in case if flag
     * UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL is not set */
    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        status = UCS_ERR_NO_RESOURCE;
        goto err;
    }

    /* Initialize receive request */
    req->status             = UCS_OK;
    req->recv.worker        = worker;
    req->recv.buffer        = buffer;
    req->recv.datatype      = datatype;
    req->flags              = UCP_REQUEST_FLAG_RECV_TAG | req_flags;

    ucp_dt_recv_state_init(&req->recv.state, buffer, datatype, count);

    if (!UCP_DT_IS_CONTIG(datatype)) {
        req->flags         |= UCP_REQUEST_FLAG_BLOCK_OFFLOAD;
    }

    req->recv.length        = ucp_dt_length(datatype, count, buffer,
                                            &req->recv.state);
    req->recv.mem_type      = ucp_request_get_memory_type(worker->context, buffer,
                                                         req->recv.length, param);

    req->recv.tag.tag       = tag;
    req->recv.tag.tag_mask  = tag_mask;
    if (param->op_attr_mask & UCP_OP_ATTR_FIELD_CALLBACK) {
        req->recv.tag.cb    = param->cb.recv;

        if (param->op_attr_mask & UCP_OP_ATTR_FIELD_USER_DATA) {
            req->user_data = param->user_data;
        } else {
            req->user_data = NULL;
        }
    }

    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        req->recv.tag.info.sender_tag = 0;
    }

    status = ucp_recv_request_set_user_memh(req, param);
    if (status != UCS_OK) {
        goto err;
    }

    if (ucs_unlikely(rdesc == NULL)) {
        /* If not found on unexpected, wait until it arrives.
         * If was found but need this receive request for later completion, save it */
        req_queue = ucp_tag_exp_get_queue(&worker->tm, tag, tag_mask);

        /* If offload supported, post this tag to transport as well.
         * TODO: need to distinguish the cases when posting is not needed. */
        ucp_tag_offload_try_post(worker, req, req_queue);

        ucp_tag_exp_push(&worker->tm, req_queue, req);

        ucs_trace_req("%s returning expected request %p (%p)", debug_name, req,
                      req + 1);
        return req + 1;
    }

    /* Check rendezvous case */
    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_RNDV)) {
        ucp_tag_rndv_matched(worker, req, ucp_tag_rndv_rts_from_rdesc(rdesc),
                             rdesc->length);
        UCP_WORKER_STAT_RNDV(worker, UNEXP, 1);
        ucp_recv_desc_release(rdesc);
    } else {
        ucp_tag_recv_eager_multi(worker, req, rdesc);
    }

    return req + 1;

err:
    ucp_request_put_param(param, req);
    return UCS_STATUS_PTR(status);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_recv_nbr,
                 (worker, buffer, count, datatype, tag, tag_mask, request),
                 ucp_worker_h worker, void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                 void *request)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FIELD_REQUEST  |
                        UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
        .request      = request,
        .datatype     = datatype
    };
    ucs_status_ptr_t status;

    status = ucp_tag_recv_nbx(worker, buffer, count, tag, tag_mask, &param);
    return UCS_PTR_IS_ERR(status) ? UCS_PTR_STATUS(status) : UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_recv_nb,
                 (worker, buffer, count, datatype, tag, tag_mask, cb),
                 ucp_worker_h worker, void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                 ucp_tag_recv_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
        .cb.recv      = (ucp_tag_recv_nbx_callback_t)cb,
        .datatype     = datatype
    };

    return ucp_tag_recv_nbx(worker, buffer, count, tag, tag_mask, &param);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_recv_nbx,
                 (worker, buffer, count, tag, tag_mask, param),
                 ucp_worker_h worker, void *buffer, size_t count,
                 ucp_tag_t tag, ucp_tag_t tag_mask,
                 const ucp_request_param_t *param)
{
    ucp_recv_desc_t *rdesc;
    ucs_status_ptr_t ret;
    ucp_request_t *req;
    ucp_datatype_t datatype;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_TAG,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_REQUEST_CHECK_PARAM(param);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    datatype = ucp_request_param_datatype(param);
    req      = ucp_request_get_param(worker, param,
                                     {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                      goto out;});

    rdesc    = ucp_tag_unexp_search(&worker->tm, tag, tag_mask, 1, "recv_nbx");
    ret      = ucp_tag_recv_common(worker, buffer, count, datatype, tag,
                                   tag_mask, req, rdesc, param, "recv_nbx");

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

ucs_status_ptr_t ucp_tag_msg_recv_nb(ucp_worker_h worker, void *buffer, size_t count,
                                     ucp_datatype_t datatype, ucp_tag_message_h message,
                                     ucp_tag_recv_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FLAG_NO_IMM_CMPL,
        .datatype     = datatype,
        .cb.recv      = (ucp_tag_recv_nbx_callback_t)cb
    };

    return ucp_tag_msg_recv_nbx(worker, buffer, count, message, &param);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_msg_recv_nbx,
                 (worker, buffer, count, message, param),
                 ucp_worker_h worker, void *buffer, size_t count,
                 ucp_tag_message_h message, const ucp_request_param_t *param)
{
    ucp_recv_desc_t *rdesc = message;
    ucs_status_ptr_t ret;
    ucp_request_t *req;
    ucp_datatype_t datatype;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_TAG,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_REQUEST_CHECK_PARAM(param);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    req      = ucp_request_get_param(worker, param,
                                     {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                     goto out;});
    datatype = ucp_request_param_datatype(param);
    ret      =  ucp_tag_recv_common(worker, buffer, count, datatype,
                                    ucp_rdesc_get_tag(rdesc), UCP_TAG_MASK_FULL,
                                    req, rdesc, param, "msg_recv_nbx");

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}
