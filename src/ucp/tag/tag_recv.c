/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"
#include "rndv.h"
#include "tag_match.inl"
#include "offload.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>


static UCS_F_ALWAYS_INLINE ucp_recv_desc_t*
ucp_tag_unexp_list_next(ucp_recv_desc_t *rdesc, int i_list)
{
    return ucs_list_next(&rdesc->list[i_list], ucp_recv_desc_t, list[i_list]);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_search_unexp(ucp_worker_h worker, void *buffer, size_t buffer_size,
                     ucp_datatype_t datatype, ucp_tag_t tag, uint64_t tag_mask,
                     ucp_request_t *req, ucp_tag_recv_info_t *info,
                     ucp_tag_recv_callback_t cb, ucp_recv_desc_t *first_rdesc,
                     unsigned *save_rreq)
{
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc, *next;
    ucs_list_link_t *list;
    ucs_status_t status;
    ucp_tag_t recv_tag;
    unsigned flags;
    int i_list;

    /* fast check of global unexpected queue */
    if (ucs_list_is_empty(&context->tm.unexpected.all)) {
        return UCS_INPROGRESS;
    }

    if (first_rdesc == NULL) {
        if (tag_mask == UCP_TAG_MASK_FULL) {
            list   = ucp_tag_unexp_get_list_for_tag(&context->tm, tag);
            if (ucs_list_is_empty(list)) {
                return UCS_INPROGRESS;
            }

            i_list = UCP_RDESC_HASH_LIST;
        } else {
            list   = &context->tm.unexpected.all;
            i_list = UCP_RDESC_ALL_LIST;
        }
        rdesc = ucs_list_head(list, ucp_recv_desc_t, list[i_list]);
    } else {
        ucs_assert(tag_mask == UCP_TAG_MASK_FULL);
        list   = ucp_tag_unexp_get_list_for_tag(&context->tm, tag);
        i_list = UCP_RDESC_HASH_LIST;
        rdesc  = first_rdesc;
    }

    do {
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
            ucp_tag_unexp_remove(rdesc);

            if (rdesc->flags & UCP_RECV_DESC_FLAG_EAGER) {
                UCS_PROFILE_REQUEST_EVENT(req, "eager_match", 0);
                status = ucp_eager_unexp_match(worker, rdesc, recv_tag, flags,
                                               buffer, buffer_size, datatype,
                                               &req->recv.state, info);
                ucs_trace_req("release receive descriptor %p", rdesc);
                if (status != UCS_INPROGRESS) {
                    goto out_release_desc;
                }

                next = ucp_tag_unexp_list_next(rdesc, i_list);
                ucp_tag_unexp_desc_release(rdesc);
                rdesc = next;
            } else {
                ucs_assert_always(rdesc->flags & UCP_RECV_DESC_FLAG_RNDV);
                *save_rreq         = 0;
                req->recv.buffer   = buffer;
                req->recv.length   = buffer_size;
                req->recv.datatype = datatype;
                req->recv.cb       = cb;
                ucp_rndv_matched(worker, req, (void*)(rdesc + 1));
                UCP_WORKER_STAT_RNDV(worker, UNEXP);
                status = UCS_INPROGRESS;
                goto out_release_desc;
            }
        } else {
            rdesc = ucp_tag_unexp_list_next(rdesc, i_list);
        }
    } while (&rdesc->list[i_list] != list);
    return UCS_INPROGRESS;

out_release_desc:
    ucp_tag_unexp_desc_release(rdesc);
    return status;
}


static UCS_F_ALWAYS_INLINE void
ucp_tag_recv_request_init(ucp_request_t *req, ucp_worker_h worker, void* buffer,
                          size_t count, ucp_datatype_t datatype,
                          uint16_t req_flags)
{
    ucp_dt_extended_t *dt_ex;
    req->flags = UCP_REQUEST_FLAG_EXPECTED | UCP_REQUEST_FLAG_RECV | req_flags;
    req->recv.state.offset = 0;
    req->recv.worker       = worker;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_STRIDE_R:
        dt_ex = ucp_dt_ptr(datatype);
        memset(req->recv.state.dt.stride.dim_index, 0,
               UCP_DT_STRIDE_MAX_DIMS * sizeof(size_t));
        req->recv.state.dt.stride.contig_memh = dt_ex->reusable.nc_memh;
        req->recv.state.dt.stride.memh        = dt_ex->reusable.stride_memh;
        req->recv.state.dt.stride.item_offset = 0;
        req->recv.state.dt.stride.count       = 0;
        break;

    case UCP_DATATYPE_STRIDE:
        memset(req->recv.state.dt.stride.dim_index, 0,
               UCP_DT_STRIDE_MAX_DIMS * sizeof(size_t));
        req->recv.state.dt.stride.contig_memh = UCT_MEM_HANDLE_NULL;
        req->recv.state.dt.stride.memh        = UCT_MEM_HANDLE_NULL;
        req->recv.state.dt.stride.item_offset = 0;
        req->recv.state.dt.stride.count       = 0;
        break;

    case UCP_DATATYPE_IOV_R:
        dt_ex = ucp_dt_ptr(datatype);
        req->recv.state.dt.iov.iov_offset    = 0;
        req->recv.state.dt.iov.iovcnt_offset = 0;
        req->recv.state.dt.iov.iovcnt        = count;
        req->recv.state.dt.iov.contig_memh   = dt_ex->reusable.nc_memh;
        req->recv.state.dt.iov.memh          = dt_ex->reusable.iov_memh;
        break;

    case UCP_DATATYPE_IOV:
        req->recv.state.dt.iov.iov_offset    = 0;
        req->recv.state.dt.iov.iovcnt_offset = 0;
        req->recv.state.dt.iov.iovcnt        = count;
        req->recv.state.dt.iov.contig_memh   = UCT_MEM_HANDLE_NULL;
        req->recv.state.dt.iov.memh          = UCT_MEM_HANDLE_NULL;
        break;

    case UCP_DATATYPE_GENERIC:
        dt_ex = ucp_dt_ptr(datatype);
        req->recv.state.dt.generic.state =
                        UCS_PROFILE_NAMED_CALL("dt_start", dt_ex->generic.ops.start_unpack,
                                               dt_ex->generic.context, buffer, count);
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
    if (req->flags & UCP_REQUEST_FLAG_BLOCK_OFFLOAD) {
        --req->recv.worker->context->tm.sw_req_count;
    }
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", 0);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_recv_common(ucp_worker_h worker, void *buffer, size_t count,
                    uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                    ucp_request_t *req, uint16_t req_flags, ucp_tag_recv_callback_t cb,
                    ucp_recv_desc_t *rdesc, const char *debug_name)
{
    unsigned save_rreq = 1;
    ucs_queue_head_t *queue;
    ucp_context_h context;
    ucs_status_t status;
    size_t buffer_size;

    ucp_tag_recv_request_init(req, worker, buffer, count, datatype, req_flags);
    buffer_size = ucp_dt_length(datatype, count, buffer, &req->recv.state);

    ucs_trace_req("%s buffer %p buffer_size %zu tag %"PRIx64"/%"PRIx64, debug_name,
                  buffer, buffer_size, tag, tag_mask);

    /* First, search in unexpected list */
    status = ucp_tag_search_unexp(worker, buffer, buffer_size, datatype, tag,
                                  tag_mask, req, &req->recv.info, cb, rdesc,
                                  &save_rreq);
    if (status != UCS_INPROGRESS) {
        if (req_flags & UCP_REQUEST_FLAG_CALLBACK) {
            cb(req + 1, status, &req->recv.info);
        }
        ucp_tag_recv_request_completed(req, status, &req->recv.info, debug_name);
    } else if (save_rreq) {
        /* If not found on unexpected, wait until it arrives.
         * If was found but need this receive request for later completion, save it */
        context            = worker->context;
        queue              = ucp_tag_exp_get_queue(&context->tm, tag, tag_mask);
        req->recv.buffer   = buffer;
        req->recv.length   = buffer_size;
        req->recv.datatype = datatype;
        req->recv.tag      = tag;
        req->recv.tag_mask = tag_mask;
        req->recv.cb       = cb;
        ucp_tag_exp_push(&context->tm, queue, req);

        /* If offload supported, post this tag to transport as well.
         * TODO: need to distinguish the cases when posting is not needed. */
        ucp_tag_offload_try_post(worker->context, req);
        ucs_trace_req("%s returning expected request %p (%p)", debug_name, req,
                      req + 1);
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
    ucp_recv_desc_t *rdesc = message;
    ucs_status_ptr_t ret;
    ucp_request_t *req;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->context->mt_lock);

    req = ucp_request_get(worker);
    if (ucs_likely(req != NULL)) {
        rdesc->flags |= UCP_RECV_DESC_FLAG_FIRST;
        ucp_tag_recv_common(worker, buffer, count, datatype,
                            ucp_rdesc_get_tag(rdesc), UCP_TAG_MASK_FULL, req,
                            UCP_REQUEST_FLAG_CALLBACK, cb, rdesc, "msg_recv_nb");
        ret = req + 1;
    } else {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->context->mt_lock);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return ret;
}
