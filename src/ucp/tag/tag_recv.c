/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"
#include "rndv.h"
#include "tag_match.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_search_unexp(ucp_worker_h worker, void *buffer, size_t buffer_size,
                     ucp_datatype_t datatype, ucp_tag_t tag, uint64_t tag_mask,
                     ucp_request_t *req, ucp_tag_recv_info_t *info,
                     ucp_tag_recv_callback_t cb, ucs_queue_iter_t first_iter,
                     unsigned *save_rreq)
{
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc;
    ucs_queue_iter_t iter;
    ucs_status_t status;
    ucp_tag_t recv_tag;
    unsigned flags;

    if (first_iter == NULL) {
        iter = ucs_queue_iter_begin(&context->tm.unexpected);
    } else {
        iter = first_iter;
    }

    while (!ucs_queue_iter_end(&context->tm.unexpected, iter)) {
        rdesc    = ucs_queue_iter_elem(rdesc, iter, queue);
        recv_tag = ucp_rdesc_get_tag(rdesc);
        flags    = rdesc->flags;
        ucs_trace_req("searching for %"PRIx64"/%"PRIx64"/%"PRIx64" offset %zu, "
                      "checking desc %p %"PRIx64" %c%c%c%c%c",
                      tag, tag_mask, info->sender_tag, req->recv.state.offset,
                      rdesc, recv_tag,
                      (flags & UCP_RECV_DESC_FLAG_FIRST) ? 'f' : '-',
                      (flags & UCP_RECV_DESC_FLAG_LAST)  ? 'l' : '-',
                      (flags & UCP_RECV_DESC_FLAG_EAGER) ? 'e' : '-',
                      (flags & UCP_RECV_DESC_FLAG_SYNC)  ? 's' : '-',
                      (flags & UCP_RECV_DESC_FLAG_RNDV)  ? 'r' : '-');
        if (ucp_tag_recv_is_match(recv_tag, flags, tag, tag_mask,
                                  req->recv.state.offset, info->sender_tag))
        {
            ucp_tag_log_match(recv_tag, rdesc->length - rdesc->hdr_len, req, tag,
                              tag_mask, req->recv.state.offset, "unexpected");
            ucs_queue_del_iter(&context->tm.unexpected, iter);
            if (rdesc->flags & UCP_RECV_DESC_FLAG_EAGER) {
                UCS_PROFILE_REQUEST_EVENT(req, "eager_match", 0);
                status = ucp_eager_unexp_match(worker, rdesc, recv_tag, flags,
                                               buffer, buffer_size, datatype,
                                               &req->recv.state, info);
                ucs_trace_req("release receive descriptor %p", rdesc);
                ucp_tag_unexp_desc_release(rdesc);
                if (status != UCS_INPROGRESS) {
                    return status;
                }
            } else if (rdesc->flags & UCP_RECV_DESC_FLAG_RNDV) {
                *save_rreq         = 0;
                req->recv.buffer   = buffer;
                req->recv.length   = buffer_size;
                req->recv.datatype = datatype;
                req->recv.cb       = cb;
                ucp_rndv_matched(worker, req, (void*)(rdesc + 1));
                ucp_tag_unexp_desc_release(rdesc);
                UCP_WORKER_STAT_RNDV(worker, UNEXP);
                return UCS_INPROGRESS;
            }
        } else {
            iter = ucs_queue_iter_next(iter);
        }
    }

    return UCS_INPROGRESS;
}


static UCS_F_ALWAYS_INLINE void
ucp_tag_recv_request_init(ucp_request_t *req, ucp_worker_h worker, void* buffer,
                          size_t count, ucp_datatype_t datatype,
                          uint16_t req_flags)
{
    ucp_dt_generic_t *dt_gen;
    req->flags = UCP_REQUEST_FLAG_EXPECTED | UCP_REQUEST_FLAG_RECV | req_flags;
    req->recv.state.offset = 0;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_IOV:
        req->recv.state.dt.iov.iov_offset    = 0;
        req->recv.state.dt.iov.iovcnt_offset = 0;
        req->recv.state.dt.iov.iovcnt        = count;
        req->recv.state.dt.iov.memh          = UCT_MEM_HANDLE_NULL;
        break;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        req->recv.state.dt.generic.state =
                        UCS_PROFILE_NAMED_CALL("dt_start", dt_gen->ops.start_unpack,
                                               dt_gen->context, buffer, count);
        ucs_debug("req %p buffer %p count %zu dt_gen state=%p", req, buffer, count,
                  req->recv.state.dt.generic.state);
        break;

    default:
        break;
    }

    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        req->recv.info.sender_tag = 0;
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_recv_request_completed(ucp_request_t *req, ucs_status_t status,
                               ucp_tag_recv_info_t *info, const char *function)
{
    ucs_trace_req("%s returning completed request %p (%p) stag 0x%"PRIx64" len %zu, %s",
                  function, req, req + 1, info->sender_tag, info->length,
                  ucs_status_string(status));

    req->status = status;
    req->flags |= UCP_REQUEST_FLAG_COMPLETED;
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", 0);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_recv_common(ucp_worker_h worker, void *buffer, size_t count,
                    uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                    ucp_request_t *req, uint16_t req_flags, ucp_tag_recv_callback_t cb,
                    ucs_queue_iter_t iter, const char *debug_name)
{
    ucs_status_t status;
    unsigned save_rreq = 1;
    size_t buffer_size;

    ucp_tag_recv_request_init(req, worker, buffer, count, datatype, req_flags);
    buffer_size = ucp_dt_length(datatype, count, buffer, &req->recv.state);

    ucs_trace_req("%s buffer %p buffer_size %zu tag %"PRIx64"/%"PRIx64, debug_name,
                  buffer, buffer_size, tag, tag_mask);

    /* First, search in unexpected list */
    status = ucp_tag_search_unexp(worker, buffer, buffer_size, datatype, tag,
                                  tag_mask, req, &req->recv.info, cb, iter,
                                  &save_rreq);
    if (status != UCS_INPROGRESS) {
        if (req_flags & UCP_REQUEST_FLAG_CALLBACK) {
            cb(req + 1, status, &req->recv.info);
        }
        ucp_tag_recv_request_completed(req, status, &req->recv.info, debug_name);
    } else if (save_rreq) {
        /* If not found on unexpected, wait until it arrives.
         * If was found but need this receive request for later completion, save it */
        req->recv.buffer   = buffer;
        req->recv.length   = buffer_size;
        req->recv.datatype = datatype;
        req->recv.tag      = tag;
        req->recv.tag_mask = tag_mask;
        req->recv.cb       = cb;
        ucp_tag_exp_add(&worker->context->tm, req);
        ucs_trace_req("%s returning expected request %p (%p)", debug_name,
                      req, req + 1);
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_recv_nbr,
                 (worker, buffer, count, datatype, tag, tag_mask, request),
                 ucp_worker_h worker, void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                 void *request)
{
    ucp_request_t *req = (ucp_request_t *)request - 1;
    ucs_status_t status;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->context->mt_lock);

    status = ucp_tag_recv_common(worker, buffer, count, datatype, tag, tag_mask,
                                 req, UCP_REQUEST_DEBUG_FLAG_EXTERNAL, NULL, NULL,
                                 "recv_nbr");

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->context->mt_lock);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_recv_nb,
                 (worker, buffer, count, datatype, tag, tag_mask, cb),
                 ucp_worker_h worker, void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                 ucp_tag_recv_callback_t cb)
{
    ucs_status_ptr_t ret;
    ucp_request_t *req;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->context->mt_lock);

    req = ucp_request_get(worker);
    if (ucs_likely(req != NULL)) {
        ucp_tag_recv_common(worker, buffer, count, datatype, tag, tag_mask, req,
                            UCP_REQUEST_FLAG_CALLBACK, cb, NULL, "recv_nb");
        ret = req + 1;
    } else {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->context->mt_lock);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_msg_recv_nb,
                 (worker, buffer, count, datatype, message, cb),
                 ucp_worker_h worker, void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_message_h message,
                 ucp_tag_recv_callback_t cb)
{
    ucs_queue_iter_t iter = message;
    ucp_recv_desc_t *rdesc;
    ucs_status_ptr_t ret;
    ucp_request_t *req;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->context->mt_lock);

    req = ucp_request_get(worker);
    if (ucs_likely(req != NULL)) {
        rdesc         = ucs_queue_iter_elem(rdesc, iter, queue);
        rdesc->flags |= UCP_RECV_DESC_FLAG_FIRST;
        ucp_tag_recv_common(worker, buffer, count, datatype,
                            ucp_rdesc_get_tag(rdesc), UCP_TAG_MASK_FULL, req,
                            UCP_REQUEST_FLAG_CALLBACK, cb, iter, "msg_recv_nb");
        ret = req + 1;
    } else {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->context->mt_lock);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return ret;
}
