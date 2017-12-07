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
    return ucs_list_next(&rdesc->tag_list[i_list], ucp_recv_desc_t,
                         tag_list[i_list]);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_search_unexp(ucp_worker_h worker, void *buffer, size_t buffer_size,
                     ucp_datatype_t datatype, ucp_tag_t tag, uint64_t tag_mask,
                     ucp_request_t *req, ucp_tag_recv_info_t *info,
                     ucp_tag_recv_callback_t cb, ucp_recv_desc_t *first_rdesc,
                     unsigned *save_rreq)
{
    ucp_recv_desc_t *rdesc, *next;
    ucs_list_link_t *list;
    ucs_status_t status;
    ucp_tag_t recv_tag;
    unsigned flags;
    int i_list;

    /* fast check of global unexpected queue */
    if (ucs_list_is_empty(&worker->tm.unexpected.all)) {
        return UCS_INPROGRESS;
    }

    if (first_rdesc == NULL) {
        if (tag_mask == UCP_TAG_MASK_FULL) {
            list = ucp_tag_unexp_get_list_for_tag(&worker->tm, tag);
            if (ucs_list_is_empty(list)) {
                return UCS_INPROGRESS;
            }

            i_list = UCP_RDESC_HASH_LIST;
        } else {
            list   = &worker->tm.unexpected.all;
            i_list = UCP_RDESC_ALL_LIST;
        }
        rdesc = ucs_list_head(list, ucp_recv_desc_t, tag_list[i_list]);
    } else {
        ucs_assert(tag_mask == UCP_TAG_MASK_FULL);
        list   = ucp_tag_unexp_get_list_for_tag(&worker->tm, tag);
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
                req->recv.tag.cb   = cb;
                ucp_rndv_matched(worker, req, (void*)(rdesc + 1));
                UCP_WORKER_STAT_RNDV(worker, UNEXP);
                status = UCS_INPROGRESS;
                goto out_release_desc;
            }
        } else {
            rdesc = ucp_tag_unexp_list_next(rdesc, i_list);
        }
    } while (&rdesc->tag_list[i_list] != list);
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
    req->flags          = UCP_REQUEST_FLAG_EXPECTED | UCP_REQUEST_FLAG_RECV |
                          req_flags;
    req->recv.worker    = worker;
    req->recv.reg_rsc   = UCP_NULL_RESOURCE;

    ucp_request_recv_state_init(req, buffer, datatype, count);

    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        req->recv.tag.info.sender_tag = 0;
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
    if (req->flags & UCP_REQUEST_FLAG_BLOCK_OFFLOAD) {
        --req->recv.worker->tm.offload.sw_req_count;
    }
    if ((req->flags |= UCP_REQUEST_FLAG_COMPLETED) & UCP_REQUEST_FLAG_RELEASED) {
        ucp_request_put(req);
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
    ucs_status_t status;
    size_t buffer_size;

    ucp_tag_recv_request_init(req, worker, buffer, count, datatype, req_flags);
    buffer_size = ucp_dt_length(datatype, count, buffer, &req->recv.state);

    ucs_trace_req("%s buffer %p buffer_size %zu tag %"PRIx64"/%"PRIx64, debug_name,
                  buffer, buffer_size, tag, tag_mask);

    /* First, search in unexpected list */
    status = ucp_tag_search_unexp(worker, buffer, buffer_size, datatype, tag,
                                  tag_mask, req, &req->recv.tag.info, cb, rdesc,
                                  &save_rreq);
    if (status != UCS_INPROGRESS) {
        if (req_flags & UCP_REQUEST_FLAG_CALLBACK) {
            cb(req + 1, status, &req->recv.tag.info);
        }
        ucp_tag_recv_request_completed(req, status, &req->recv.tag.info,
                                       debug_name);
    } else if (save_rreq) {
        /* If not found on unexpected, wait until it arrives.
         * If was found but need this receive request for later completion, save it */
        queue = ucp_tag_exp_get_queue(&worker->tm, tag, tag_mask);

        req->recv.buffer        = buffer;
        req->recv.length        = buffer_size;
        req->recv.datatype      = datatype;
        req->recv.tag.tag       = tag;
        req->recv.tag.tag_mask  = tag_mask;
        req->recv.tag.cb        = cb;

        ucp_tag_exp_push(&worker->tm, queue, req);

        /* If offload supported, post this tag to transport as well.
         * TODO: need to distinguish the cases when posting is not needed. */
        ucp_tag_offload_try_post(worker, req);
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

    status = ucp_tag_recv_common(worker, buffer, count, datatype, tag, tag_mask,
                                 req, UCP_REQUEST_DEBUG_FLAG_EXTERNAL, NULL, NULL,
                                 "recv_nbr");

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

    req = ucp_request_get(worker);
    if (ucs_likely(req != NULL)) {
        ucp_tag_recv_common(worker, buffer, count, datatype, tag, tag_mask, req,
                            UCP_REQUEST_FLAG_CALLBACK, cb, NULL, "recv_nb");
        ret = req + 1;
    } else {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

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

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return ret;
}
