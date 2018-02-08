/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"
#include "tag_match.inl"
#include "offload.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/datastruct/queue.h>
#include <ucp/core/ucp_request.inl>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_tagged_handler(void *arg, void *data, size_t length, unsigned am_flags,
                         uint16_t flags, uint16_t hdr_len)
{
    ucp_worker_h worker        = arg;
    ucp_eager_hdr_t *eager_hdr = data;
    ucp_eager_first_hdr_t *eagerf_hdr;
    ucp_recv_desc_t *rdesc;
    ucp_request_t *req;
    ucs_status_t status;
    ucp_tag_t recv_tag;
    size_t recv_len;

    ucs_assert(length >= hdr_len);
    ucs_assert(flags & UCP_RECV_DESC_FLAG_EAGER);

    recv_tag = eager_hdr->super.tag;
    recv_len = length - hdr_len;

    req = ucp_tag_exp_search(&worker->tm, recv_tag);
    if (req != NULL) {
        ucs_trace_req("found req %p", req);
        UCS_PROFILE_REQUEST_EVENT(req, "eager_recv", recv_len);

        /* First fragment fills the receive information */
        UCP_WORKER_STAT_EAGER_MSG(worker, flags);
        UCP_WORKER_STAT_EAGER_CHUNK(worker, EXP);

        req->recv.tag.info.sender_tag = recv_tag;

        /* Cancel req in transport if it was offloaded,
         * because it arrived either:
         * 1) via SW TM (e. g. peer doesn't support offload)
         * 2) as unexpected via HW TM */
        ucp_tag_offload_try_cancel(worker, req,
                                   UCP_TAG_OFFLOAD_CANCEL_FORCE |
                                   UCP_TAG_OFFLOAD_CANCEL_DEREG);

        if (flags & UCP_RECV_DESC_FLAG_EAGER_SYNC) {
            ucp_tag_eager_sync_send_ack(worker, data, flags);
        }

        if (flags & UCP_RECV_DESC_FLAG_EAGER_ONLY) {
            req->recv.tag.info.length = recv_len;
            status = ucp_request_recv_data_unpack(req, data + hdr_len, recv_len,
                                                  0, 1);
            ucp_request_complete_tag_recv(req, status);
        } else {
            eagerf_hdr                = data;
            req->recv.tag.info.length =
            req->recv.tag.remaining   = eagerf_hdr->total_len;

            status = ucp_tag_request_process_recv_data(req, data + hdr_len,
                                                       recv_len, 0, 0);
            ucs_assert(status == UCS_INPROGRESS);

            ucp_tag_frag_list_process_queue(&worker->tm, req, eagerf_hdr->msg_id
                                            UCS_STATS_ARG(UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP));
        }

        status = UCS_OK;
    } else {
        status = ucp_recv_desc_init(worker, data, length, am_flags, hdr_len,
                                    flags, &rdesc);
        if (!UCS_STATUS_IS_ERR(status)) {
            ucp_tag_unexp_recv(&worker->tm, rdesc, recv_tag);
        }
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_only_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    return ucp_eager_tagged_handler(arg, data, length, am_flags,
                                    UCP_RECV_DESC_FLAG_EAGER |
                                    UCP_RECV_DESC_FLAG_EAGER_ONLY,
                                    sizeof(ucp_eager_hdr_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_first_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    return ucp_eager_tagged_handler(arg, data, length, am_flags,
                                    UCP_RECV_DESC_FLAG_EAGER,
                                    sizeof(ucp_eager_first_hdr_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_middle_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_worker_h worker         = arg;
    ucp_eager_middle_hdr_t *hdr = data;
    ucp_tag_frag_match_t *matchq;
    ucp_recv_desc_t *rdesc;
    ucp_request_t *req;
    ucs_status_t status;
    size_t recv_len;
    khiter_t iter;
    int ret;

    iter   = kh_put(ucp_tag_frag_hash, &worker->tm.frag_hash, hdr->msg_id, &ret);
    matchq = &kh_value(&worker->tm.frag_hash, iter);
    if (ret != 0) {
        /* initialize a previously empty hash entry */
        ucp_tag_frag_match_init_unexp(matchq);
    }

    if (ucp_tag_frag_match_is_unexp(matchq)) {
        /* add new received descriptor to the queue */
        status = ucp_recv_desc_init(worker, data, length, am_flags, sizeof(*hdr),
                                    UCP_RECV_DESC_FLAG_EAGER, &rdesc);
        if (!UCS_STATUS_IS_ERR(status)) {
            ucp_tag_frag_match_add_unexp(matchq, rdesc, hdr->offset);
        }
    } else {
        /* hash entry contains a request, copy data to user buffer */
        req      = matchq->exp_req;
        recv_len = length - sizeof(*hdr);

        UCP_WORKER_STAT_EAGER_CHUNK(worker, EXP);
        status = ucp_tag_request_process_recv_data(req, data + sizeof(*hdr),
                                                   recv_len, hdr->offset, 0);
        if (status != UCS_INPROGRESS) {
            /* request completed, delete hash entry */
            kh_del(ucp_tag_frag_hash, &worker->tm.frag_hash, iter);
        }

        status = UCS_OK;
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_sync_only_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    return ucp_eager_tagged_handler(arg, data, length, am_flags,
                                    UCP_RECV_DESC_FLAG_EAGER|
                                    UCP_RECV_DESC_FLAG_EAGER_ONLY|
                                    UCP_RECV_DESC_FLAG_EAGER_SYNC,
                                    sizeof(ucp_eager_sync_hdr_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_sync_first_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    return ucp_eager_tagged_handler(arg, data, length, am_flags,
                                    UCP_RECV_DESC_FLAG_EAGER|
                                    UCP_RECV_DESC_FLAG_EAGER_SYNC,
                                    sizeof(ucp_eager_sync_first_hdr_t));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_offload_sync_ack_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_offload_ssend_hdr_t *rep_hdr = data;
    ucp_worker_t *worker             = arg;
    ucs_queue_head_t *queue          = &worker->tm.offload.sync_reqs;
    ucp_request_t *sreq;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(sreq, iter, queue, send.tag_offload.queue) {
        if ((sreq->send.tag_offload.ssend_tag == rep_hdr->sender_tag) &&
            (worker->uuid == rep_hdr->sender_uuid)) {
            ucp_tag_eager_sync_completion(sreq, UCP_REQUEST_FLAG_REMOTE_COMPLETED,
                                          UCS_OK);
            ucs_queue_del_iter(queue, iter);
            return UCS_OK;
        }
    }
    ucs_error("Unexpected sync ack received: tag %"PRIx64" uuid %"PRIx64"",
              rep_hdr->sender_tag, rep_hdr->sender_uuid);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_sync_ack_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *req;

    req = (ucp_request_t*)rep_hdr->reqptr;
    ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_REMOTE_COMPLETED, UCS_OK);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_offload_unexp_eager,
                 (arg, data, length, tl_flags, stag, imm),
                 void *arg, void *data, size_t length, unsigned tl_flags,
                 uct_tag_t stag, uint64_t imm)
{
    /* Align data with AM protocol. We should add tag before the data. */
    ucp_worker_iface_t *wiface = arg;
    uint16_t flags             = UCP_RECV_DESC_FLAG_EAGER |
                                 UCP_RECV_DESC_FLAG_EAGER_ONLY |
                                 UCP_RECV_DESC_FLAG_EAGER_OFFLOAD;
    ucp_eager_hdr_t *hdr;
    ucp_eager_sync_hdr_t *sync_hdr;
    unsigned hdr_len;

    hdr_len = ucs_unlikely(imm) ? sizeof(ucp_eager_sync_hdr_t) :
                                  sizeof(ucp_eager_hdr_t);

    if (ucs_unlikely(tl_flags & UCT_CB_PARAM_FLAG_DESC)) {
        hdr = (ucp_eager_hdr_t*)((char*)data - hdr_len);
    } else {
        /* Can not shift back, no headroom */
        hdr = ucs_alloca(length + hdr_len);
        memcpy((char*)hdr + hdr_len, data, length);
    }

    hdr->super.tag = stag;

    if (ucs_unlikely(imm)) {
        /* It is a sync send, imm data contains sender uuid */
        sync_hdr = ucs_derived_of(hdr, ucp_eager_sync_hdr_t);
        flags   |= UCP_RECV_DESC_FLAG_EAGER_SYNC;
        sync_hdr->req.reqptr      = 0ul;
        sync_hdr->req.sender_uuid = imm;
    }

    UCP_WORKER_STAT_TAG_OFFLOAD(wiface->worker, RX_UNEXP_EGR);

    ucp_tag_offload_unexp(wiface, stag);

    return ucp_eager_tagged_handler(wiface->worker, hdr, length + hdr_len,
                                    tl_flags, flags, hdr_len);
}

static void ucp_eager_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                           uint8_t id, const void *data, size_t length,
                           char *buffer, size_t max)
{
    const ucp_eager_first_hdr_t *eager_first_hdr = data;
    const ucp_eager_hdr_t *eager_hdr             = data;
    const ucp_eager_middle_hdr_t *eager_mid_hdr  = data;
    const ucp_eager_sync_first_hdr_t *eagers_first_hdr = data;
    const ucp_eager_sync_hdr_t *eagers_hdr       = data;
    const ucp_reply_hdr_t *rep_hdr               = data;
    const ucp_offload_ssend_hdr_t *off_rep_hdr   = data;
    size_t header_len;
    char *p;

    switch (id) {
    case UCP_AM_ID_EAGER_ONLY:
        snprintf(buffer, max, "EGR_O tag %"PRIx64, eager_hdr->super.tag);
        header_len = sizeof(*eager_hdr);
        break;
    case UCP_AM_ID_EAGER_FIRST:
        snprintf(buffer, max, "EGR_F tag %"PRIx64" msgid %"PRIx64" len %zu",
                 eager_first_hdr->super.super.tag, eager_first_hdr->msg_id,
                 eager_first_hdr->total_len);
        header_len = sizeof(*eager_first_hdr);
        break;
    case UCP_AM_ID_EAGER_MIDDLE:
        snprintf(buffer, max, "EGR_M msgid %"PRIx64" offset %zu",
                 eager_mid_hdr->msg_id, eager_mid_hdr->offset);
        header_len = sizeof(*eager_mid_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_ONLY:
        snprintf(buffer, max, "EGRS tag %"PRIx64" uuid %"PRIx64" request 0x%lx",
                 eagers_hdr->super.super.tag, eagers_hdr->req.sender_uuid,
                 eagers_hdr->req.reqptr);
        header_len = sizeof(*eagers_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_FIRST:
        snprintf(buffer, max, "EGRS_F tag %"PRIx64" msgid %"PRIx64" len %zu "
                 "uuid %"PRIx64" request 0x%lx",
                 eagers_first_hdr->super.super.super.tag,
                 eagers_first_hdr->super.msg_id,
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
    case UCP_AM_ID_OFFLOAD_SYNC_ACK:
        snprintf(buffer, max, "EGRS_A_O tag %"PRIx64" uuid %"PRIx64"",
                 off_rep_hdr->sender_tag, off_rep_hdr->sender_uuid);
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
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_FIRST, ucp_eager_first_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_MIDDLE, ucp_eager_middle_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_ONLY, ucp_eager_sync_only_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_FIRST, ucp_eager_sync_first_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_ACK, ucp_eager_sync_ack_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_TAG, UCP_AM_ID_OFFLOAD_SYNC_ACK,
              ucp_eager_offload_sync_ack_handler, ucp_eager_dump, UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_EAGER_ONLY);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_EAGER_FIRST);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_EAGER_MIDDLE);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_EAGER_SYNC_ONLY);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_EAGER_SYNC_FIRST);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_EAGER_SYNC_ACK);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_OFFLOAD_SYNC_ACK);
