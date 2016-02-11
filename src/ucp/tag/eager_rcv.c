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


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_handler(void *arg, void *data, size_t length, void *desc,
                  unsigned flags)
{
    ucp_worker_h worker = arg;
    ucp_eager_hdr_t *eager_hdr = data;
    ucp_eager_first_hdr_t *eager_first_hdr = data;
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc = desc;
    ucp_request_t *req;
    ucs_queue_iter_t iter;
    ucs_status_t status;
    size_t recv_len, hdr_len;
    ucp_tag_t recv_tag;

    hdr_len = ucp_eager_hdr_len(flags);

    ucs_assert(length >= hdr_len);
    recv_tag = eager_hdr->super.tag;

    /* Search in expected queue */
    ucs_queue_for_each_safe(req, iter, &context->tag.expected, recv.queue) {
        req = ucs_container_of(*iter, ucp_request_t, recv.queue);
        if (ucp_tag_recv_is_match(recv_tag, flags, req->recv.tag, req->recv.tag_mask,
                                  req->recv.state.offset, req->recv.info.sender_tag))
        {
            ucp_tag_log_match(recv_tag, req, req->recv.tag, req->recv.tag_mask,
                              req->recv.state.offset, "expected");
            recv_len = length - hdr_len;
            status = ucp_tag_process_recv(req->recv.buffer, req->recv.count,
                                          req->recv.datatype, &req->recv.state,
                                          data + hdr_len, recv_len,
                                          flags & UCP_RECV_DESC_FLAG_LAST);

            /* First fragment fills the receive information */
            if (flags & UCP_RECV_DESC_FLAG_FIRST) {
                req->recv.info.sender_tag = recv_tag;
                if (flags & UCP_RECV_DESC_FLAG_LAST) {
                    req->recv.info.length = recv_len;
                } else {
                    ucs_assert(hdr_len == sizeof(*eager_first_hdr));
                    req->recv.info.length = eager_first_hdr->total_len;
                }
            }

            /* Last fragment completes the request */
            if (flags & UCP_RECV_DESC_FLAG_LAST) {
                ucs_queue_del_iter(&context->tag.expected, iter);
                ucp_request_complete(req, req->cb.tag_recv, status, &req->recv.info);
            } else {
                req->recv.state.offset += recv_len;
            }
            return UCS_OK;
        }
    }

    ucs_trace_req("unexp recv %c%c%c tag %"PRIx64" length %zu desc %p",
                  (flags & UCP_RECV_DESC_FLAG_FIRST) ? 'f' : '-',
                  (flags & UCP_RECV_DESC_FLAG_LAST)  ? 'l' : '-',
                  (flags & UCP_RECV_DESC_FLAG_EAGER) ? 'e' : '-',
                  recv_tag, length, rdesc);

    if (data != rdesc + 1) {
        memcpy(rdesc + 1, data, length);
    }

    rdesc->length = length;
    rdesc->flags  = flags;
    ucs_queue_push(&context->tag.unexpected, &rdesc->queue);
    return UCS_INPROGRESS;
}

static ucs_status_t ucp_eager_only_handler(void *arg, void *data, size_t length,
                                           void *desc)
{
    return ucp_eager_handler(arg, data, length, desc,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_FIRST|
                             UCP_RECV_DESC_FLAG_LAST);
}

static ucs_status_t ucp_eager_first_handler(void *arg, void *data, size_t length,
                                            void *desc)
{
    return ucp_eager_handler(arg, data, length, desc,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_FIRST);
}

static ucs_status_t ucp_eager_middle_handler(void *arg, void *data, size_t length,
                                             void *desc)
{
    return ucp_eager_handler(arg, data, length, desc,
                             UCP_RECV_DESC_FLAG_EAGER);
}

static ucs_status_t ucp_eager_last_handler(void *arg, void *data, size_t length,
                                           void *desc)
{
    return ucp_eager_handler(arg, data, length, desc,
                             UCP_RECV_DESC_FLAG_EAGER|
                             UCP_RECV_DESC_FLAG_LAST);
}

static void ucp_eager_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                           uint8_t id, const void *data, size_t length,
                           char *buffer, size_t max)
{
    const ucp_eager_first_hdr_t *eager_first_hdr = data;
    const ucp_eager_hdr_t *eager_hdr             = data;
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

