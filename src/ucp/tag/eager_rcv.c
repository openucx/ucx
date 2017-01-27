/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"
#include "match.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/datastruct/queue.h>
#include <ucp/core/ucp_request.inl>


static UCS_F_ALWAYS_INLINE void
ucp_eager_sync_send_handler(void *arg, void *data, uint16_t flags)
{
    ucp_eager_sync_hdr_t        *eagers_hdr;
    ucp_eager_sync_first_hdr_t  *eagers_first_hdr;

    if (ucs_test_all_flags(flags, UCP_RECV_DESC_FLAG_EAGER|
                                  UCP_RECV_DESC_FLAG_FIRST|
                                  UCP_RECV_DESC_FLAG_LAST|
                                  UCP_RECV_DESC_FLAG_SYNC)) {
        eagers_hdr = data;
        ucp_tag_eager_sync_send_ack(arg, eagers_hdr->req.sender_uuid,
                                    eagers_hdr->req.reqptr);
    } else if (ucs_test_all_flags(flags, UCP_RECV_DESC_FLAG_EAGER|
                                         UCP_RECV_DESC_FLAG_FIRST|
                                         UCP_RECV_DESC_FLAG_SYNC)) {
        eagers_first_hdr = data;
        ucp_tag_eager_sync_send_ack(arg, eagers_first_hdr->req.sender_uuid,
                                    eagers_first_hdr->req.reqptr);
    } else {
        ucs_fatal("wrong UCP_RECV_DESC_FLAG bit mask");
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_handler(void *arg, void *data, size_t length, unsigned am_flags,
                  uint16_t flags, uint16_t hdr_len)
{
    ucp_worker_h worker = arg;
    ucp_eager_hdr_t *eager_hdr = data;
    ucp_eager_first_hdr_t *eager_first_hdr = data;
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc;
    ucp_request_t *req;
    ucs_queue_iter_t iter;
    ucs_status_t status;
    size_t recv_len;
    ucp_tag_t recv_tag;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&context->mt_lock);

    ucs_assert(length >= hdr_len);
    recv_tag = eager_hdr->super.tag;

    /* Search in expected queue */
    ucs_queue_for_each_safe(req, iter, &context->tag.expected, recv.queue) {
        req = ucs_container_of(*iter, ucp_request_t, recv.queue);
        if (ucp_tag_recv_is_match(recv_tag, flags, req->recv.tag, req->recv.tag_mask,
                                  req->recv.state.offset, req->recv.info.sender_tag))
        {
            recv_len = length - hdr_len;
            ucp_tag_log_match(recv_tag, recv_len, req, req->recv.tag,
                              req->recv.tag_mask, req->recv.state.offset, "expected");
            UCS_PROFILE_REQUEST_EVENT(req, "eager_recv", recv_len);
            status = ucp_dt_unpack(req->recv.datatype, req->recv.buffer,
                                   req->recv.length, &req->recv.state,
                                   data + hdr_len, recv_len,
                                   flags & UCP_RECV_DESC_FLAG_LAST);

            /* First fragment fills the receive information */
            if (flags & UCP_RECV_DESC_FLAG_FIRST) {
                UCP_WORKER_STAT_EAGER_MSG(worker, flags);
                req->recv.info.sender_tag = recv_tag;
                if (flags & UCP_RECV_DESC_FLAG_LAST) {
                    req->recv.info.length = recv_len;
                } else {
                    req->recv.info.length = eager_first_hdr->total_len;
                }
            }

            /* Last fragment completes the request */
            if (flags & UCP_RECV_DESC_FLAG_LAST) {
                ucs_queue_del_iter(&context->tag.expected, iter);
                ucp_request_complete_recv(req, status);
            } else {
                req->recv.state.offset += recv_len;
            }
            UCP_WORKER_STAT_EAGER_CHUNK(worker, EXP);
            /* TODO In case an error status is returned from ucp_tag_process_recv,
             * need to discard the rest of the messages */
            status = UCS_OK;

            if (flags & UCP_RECV_DESC_FLAG_SYNC) {
                ucp_eager_sync_send_handler(arg, data, flags);
            }

            goto out;
        }
    }

    rdesc = ucp_recv_desc_get(worker, data, length, hdr_len, am_flags, flags);
    if (!rdesc) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    ucs_trace_req("unexp recv %c%c%c tag %"PRIx64" length %zu desc %p",
                  (flags & UCP_RECV_DESC_FLAG_FIRST) ? 'f' : '-',
                  (flags & UCP_RECV_DESC_FLAG_LAST)  ? 'l' : '-',
                  (flags & UCP_RECV_DESC_FLAG_EAGER) ? 'e' : '-',
                  recv_tag, length, rdesc);

    ucs_queue_push(&context->tag.unexpected, &rdesc->queue);

    status = (am_flags & UCT_AM_FLAG_DESC) ? UCS_INPROGRESS : UCS_OK;

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&context->mt_lock);
    return status;
}

static ucs_status_t ucp_eager_only_handler(void *arg, void *data, size_t length,
                                           unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_FIRST|
                             UCP_RECV_DESC_FLAG_LAST,
                             sizeof(ucp_eager_hdr_t));
}

static ucs_status_t ucp_eager_first_handler(void *arg, void *data, size_t length,
                                            unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_FIRST,
                             sizeof(ucp_eager_first_hdr_t));
}

static ucs_status_t ucp_eager_middle_handler(void *arg, void *data, size_t length,
                                             unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_EAGER,
                             sizeof(ucp_eager_hdr_t));
}

static ucs_status_t ucp_eager_last_handler(void *arg, void *data, size_t length,
                                           unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_LAST,
                             sizeof(ucp_eager_hdr_t));
}

static ucs_status_t ucp_eager_sync_only_handler(void *arg, void *data,
                                                size_t length, unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_FIRST|
                             UCP_RECV_DESC_FLAG_LAST|
                             UCP_RECV_DESC_FLAG_SYNC,
                             sizeof(ucp_eager_sync_hdr_t));
}

static ucs_status_t ucp_eager_sync_first_handler(void *arg, void *data,
                                                 size_t length, unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_FIRST|
                             UCP_RECV_DESC_FLAG_SYNC,
                             sizeof(ucp_eager_sync_first_hdr_t));
}

static ucs_status_t ucp_eager_sync_ack_handler(void *arg, void *data,
                                               size_t length, unsigned am_flags)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *req;

    req = (ucp_request_t*)rep_hdr->reqptr;
    ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_REMOTE_COMPLETED);
    return UCS_OK;
}

static void ucp_eager_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                           uint8_t id, const void *data, size_t length,
                           char *buffer, size_t max)
{
    const ucp_eager_first_hdr_t *eager_first_hdr = data;
    const ucp_eager_hdr_t *eager_hdr             = data;
    const ucp_eager_sync_first_hdr_t *eagers_first_hdr = data;
    const ucp_eager_sync_hdr_t *eagers_hdr       = data;
    const ucp_reply_hdr_t *rep_hdr               = data;
    size_t header_len;
    char *p;

    switch (id) {
    case UCP_AM_ID_EAGER_ONLY:
        snprintf(buffer, max, "EGR tag %"PRIx64, eager_hdr->super.tag);
        header_len = sizeof(*eager_hdr);
        break;
    case UCP_AM_ID_EAGER_FIRST:
        snprintf(buffer, max, "EGR_F tag %"PRIx64" len %zu",
                 eager_first_hdr->super.super.tag, eager_first_hdr->total_len);
        header_len = sizeof(*eager_first_hdr);
        break;
    case UCP_AM_ID_EAGER_MIDDLE:
        snprintf(buffer, max, "EGR_M tag %"PRIx64, eager_hdr->super.tag);
        header_len = sizeof(*eager_hdr);
        break;
    case UCP_AM_ID_EAGER_LAST:
        snprintf(buffer, max, "EGR_L tag %"PRIx64, eager_hdr->super.tag);
        header_len = sizeof(*eager_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_ONLY:
        snprintf(buffer, max, "EGRS tag %"PRIx64" uuid %"PRIx64" request 0x%lx",
                 eagers_hdr->super.super.tag, eagers_hdr->req.sender_uuid,
                 eagers_hdr->req.reqptr);
        header_len = sizeof(*eagers_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_FIRST:
        snprintf(buffer, max, "EGRS_F tag %"PRIx64" len %zu uuid %"PRIx64" request 0x%lx",
                 eagers_first_hdr->super.super.super.tag,
                 eagers_first_hdr->super.total_len,
                 eagers_first_hdr->req.sender_uuid,
                 eagers_first_hdr->req.reqptr);
        header_len = sizeof(*eagers_first_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_ACK:
        snprintf(buffer, max, "EGRS_A request 0x%lx status '%s'", rep_hdr->reqptr,
                 ucs_status_string(rep_hdr->status));
        header_len = sizeof(*rep_hdr);
        break;
    default:
        return;
    }

    p = buffer + strlen(buffer);
    ucp_dump_payload(worker->context, p, buffer + max - p, data + header_len,
                     length - header_len);
}

UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_ONLY, ucp_eager_only_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_FIRST, ucp_eager_first_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_MIDDLE, ucp_eager_middle_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_LAST, ucp_eager_last_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_ONLY, ucp_eager_sync_only_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_FIRST, ucp_eager_sync_first_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_ACK, ucp_eager_sync_ack_handler,
              ucp_eager_dump, UCT_AM_CB_FLAG_SYNC);
