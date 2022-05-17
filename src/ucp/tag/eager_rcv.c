/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"
#include "tag_match.inl"
#include "offload.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/datastruct/queue.h>
#include <ucp/core/ucp_request.inl>


/* Common handler for HW unexpected and SW tag flows when the message is
 * matched. */
static UCS_F_ALWAYS_INLINE void
ucp_eager_common_matched(ucp_worker_t *worker, ucp_request_t *req, void *data,
                         size_t recv_len, ucp_tag_t recv_tag, uint16_t flags)
{
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
    ucp_tag_offload_try_cancel(worker, req, UCP_TAG_OFFLOAD_CANCEL_FORCE);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_offload_handler(void *arg, void *data, size_t length,
                          unsigned tl_flags, uint16_t flags, ucp_tag_t recv_tag,
                          const char *name)
{
    ucp_worker_t *worker = arg;
    ucp_request_t *req;
    ucp_recv_desc_t *rdesc;
    ucp_tag_t *rdesc_hdr;
    ucs_status_t status;

    req = ucp_tag_exp_search(&worker->tm, recv_tag);
    if (req != NULL) {
        ucp_eager_common_matched(worker, req, data, length, recv_tag, flags);
        req->recv.tag.info.length = length;
        status = ucp_request_recv_data_unpack(req, data, length, 0, 1);
        ucp_request_complete_tag_recv(req, status);
        status = UCS_OK;
    } else {
        status = ucp_recv_desc_init(worker, data, length, sizeof(ucp_tag_t),
                                    tl_flags, sizeof(ucp_tag_t), flags,
                                    sizeof(ucp_tag_t), 1, name, &rdesc);
        if (!UCS_STATUS_IS_ERR(status)) {
            rdesc_hdr  = (ucp_tag_t*)(rdesc + 1);
            *rdesc_hdr = recv_tag;
            ucp_tag_unexp_recv(&worker->tm, rdesc, recv_tag);
        }
    }

    return status;
}

/* Common handler for eager only, eager sync only, eager first, eager sync
 * first, eager offload only and eager sync offload only messages
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_tagged_handler(void *arg, void *data, size_t length, unsigned am_flags,
                         uint16_t flags, uint16_t hdr_len, uint16_t priv_length,
                         const char *name)
{
    ucp_worker_h worker        = arg;
    ucp_eager_hdr_t *eager_hdr = data;
    ucp_tag_t recv_tag         = eager_hdr->super.tag;
    ucp_eager_first_hdr_t *eagerf_hdr;
    size_t recv_len;
    void *payload;
    ucp_recv_desc_t *rdesc;
    ucp_request_t *req;
    ucs_status_t status;

    req = ucp_tag_exp_search(&worker->tm, recv_tag);
    if (req != NULL) {
        recv_len = length - hdr_len;
        payload  = UCS_PTR_BYTE_OFFSET(data, hdr_len);

        ucp_eager_common_matched(worker, req, data, recv_len, recv_tag, flags);

        if (flags & UCP_RECV_DESC_FLAG_EAGER_SYNC) {
            ucp_tag_eager_sync_send_ack(worker, data, flags);
        }

        if (flags & UCP_RECV_DESC_FLAG_EAGER_ONLY) {
            req->recv.tag.info.length = recv_len;
            status = ucp_request_recv_data_unpack(req, payload, recv_len, 0, 1);
            ucp_request_complete_tag_recv(req, status);
        } else {
            /* Multi fragment tag offload flow does not use this handler */
            ucs_assert(!(flags & UCP_RECV_DESC_FLAG_EAGER_OFFLOAD));

            eagerf_hdr                = data;
            req->recv.tag.info.length = eagerf_hdr->total_len;
            req->recv.remaining       = eagerf_hdr->total_len;

            status = ucp_request_process_recv_data(req, payload, recv_len, 0, 0,
                                                   0);
            if (status == UCS_INPROGRESS) {
                ucp_tag_frag_list_process_queue(
                        &worker->tm, req, eagerf_hdr->msg_id
                        UCS_STATS_ARG(UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP));
            }
        }

        status = UCS_OK;
    } else {
        status = ucp_recv_desc_init(worker, data, length, 0, am_flags, hdr_len,
                                    flags, priv_length, 1, name, &rdesc);
        if (!UCS_STATUS_IS_ERR(status)) {
            ucp_tag_unexp_recv(&worker->tm, rdesc, eager_hdr->super.tag);
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
                                    sizeof(ucp_eager_hdr_t), 0,
                                    "eager_only_handler");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_first_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    return ucp_eager_tagged_handler(arg, data, length, am_flags,
                                    UCP_RECV_DESC_FLAG_EAGER,
                                    sizeof(ucp_eager_first_hdr_t), 0,
                                    "eager_first_handler");
}

/* Handler for middle fragments of SW eager messages */
UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_middle_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_worker_h worker         = arg;
    ucp_eager_middle_hdr_t *hdr = data;
    ucp_recv_desc_t *rdesc      = NULL;
    ucp_tag_frag_match_t *matchq;
    ucp_request_t *req;
    ucs_status_t status;
    size_t recv_len;
    khiter_t iter;
    int ret;

    iter   = kh_put(ucp_tag_frag_hash, &worker->tm.frag_hash, hdr->msg_id, &ret);
    ucs_assert(ret >= 0);
    matchq = &kh_value(&worker->tm.frag_hash, iter);
    if (ret != 0) {
        /* initialize a previously empty hash entry */
        ucp_tag_frag_match_init_unexp(matchq);
    }

    if (ucp_tag_frag_match_is_unexp(matchq)) {
        /* add new received descriptor to the queue */
        status = ucp_recv_desc_init(worker, data, length, 0, am_flags,
                                    sizeof(*hdr), UCP_RECV_DESC_FLAG_EAGER, 0,
                                    1, "eager_middle_handler", &rdesc);
        if (ucs_likely(!UCS_STATUS_IS_ERR(status))) {
            ucp_tag_frag_match_add_unexp(matchq, rdesc, hdr->offset);
        } else if (ucs_queue_is_empty(&matchq->unexp_q)) {
            /* If adding the first fragment to the unexpected queue fails,
             * remove the element from the hash. Otherwise hash would contain an
             * empty queue, which is not allowed, because queue implementation
             * relies on the address of its head for certain operations (e.g.
             * ucs_queue_is_empty). And khash may change address of its elements
             * during resize (provoked by kh_put). */
            kh_del(ucp_tag_frag_hash, &worker->tm.frag_hash, iter);
        }
    } else {
        /* If fragment is expected, the corresponding element must be present
         * in the hash (added in ucp_tag_frag_list_process_queue). */
        ucs_assert(ret == 0);

        /* hash entry contains a request, copy data to user buffer */
        req      = matchq->exp_req;
        recv_len = length - sizeof(*hdr);

        UCP_WORKER_STAT_EAGER_CHUNK(worker, EXP);

        status = ucp_request_process_recv_data(req, hdr + 1, recv_len,
                                               hdr->offset, 0, 0);
        if (status != UCS_INPROGRESS) {
            /* request completed, delete hash entry */
            kh_del(ucp_tag_frag_hash, &worker->tm.frag_hash, iter);
        }

        status = UCS_OK;
    }

    /* If hash contains queue of unexpected fragments, it should not be empty */
    ucs_assert(!ucp_tag_frag_match_is_unexp(matchq) ||
               !ucs_queue_is_empty(&matchq->unexp_q));

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
                                    sizeof(ucp_eager_sync_hdr_t), 0,
                                    "eager_sync_only_handler");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_sync_first_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    return ucp_eager_tagged_handler(arg, data, length, am_flags,
                                    UCP_RECV_DESC_FLAG_EAGER|
                                    UCP_RECV_DESC_FLAG_EAGER_SYNC,
                                    sizeof(ucp_eager_sync_first_hdr_t), 0,
                                    "eager_sync_first_handler");
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
            !(sreq->send.ep->flags & UCP_EP_FLAG_FAILED) &&
            (ucp_ep_local_id(sreq->send.ep) == rep_hdr->ep_id)) {
            ucp_send_request_id_release(sreq);
            ucp_tag_eager_sync_completion(
                    sreq, UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED, UCS_OK);
            ucs_queue_del_iter(queue, iter);
            return UCS_OK;
        }
    }

    ucs_error("unexpected sync ack received: tag %"PRIx64" ep_id 0x%"PRIx64,
              rep_hdr->sender_tag, rep_hdr->ep_id);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_eager_sync_ack_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_worker_h    worker   = arg;
    ucp_reply_hdr_t *rep_hdr = data;
    ucp_request_t *req;

    if (worker->context->config.ext.proto_enable) {
        ucp_proto_eager_sync_ack_handler(worker, rep_hdr);
    } else {
        UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rep_hdr->req_id, 1,
                                   return UCS_OK, "EAGER_S ACK %p", rep_hdr);
        ucp_tag_eager_sync_completion(req,
                                      UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED,
                                      UCS_OK);
    }

    return UCS_OK;
}

#define ucp_tag_eager_offload_priv(_flags, _data, _length, _priv_type) \
    ({ \
         size_t _priv_len = sizeof(_priv_type); \
         typeof(_priv_type) *priv_data; \
         if (ucs_unlikely((_flags) & UCT_CB_PARAM_FLAG_DESC)) { \
             priv_data = UCS_PTR_BYTE_OFFSET(_data, -_priv_len); \
         } else { /* Can not shift back, no headroom */ \
             priv_data = ucs_alloca((_length) + _priv_len); \
             memcpy(UCS_PTR_BYTE_OFFSET(priv_data, _priv_len), _data, (_length)); \
         } \
         priv_data; \
    })

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_offload_eager_first_handler(ucp_worker_h worker, void *data,
                                    size_t length, unsigned tl_flags,
                                    uct_tag_t stag, uint16_t flags,
                                    void **context)
{
    const size_t priv_len = sizeof(ucp_offload_first_desc_t);
    ucp_tag_frag_match_t *matchq;
    ucp_offload_first_desc_t *priv_hdr;
    ucp_recv_desc_t *rdesc;
    ucp_request_t *req;
    ucs_status_t status;

    /* We have always keep the first fragment until the messages is processed,
     * because matchq is stored in its private data.
     */
    status = ucp_recv_desc_init(worker, data, length, priv_len,
                                tl_flags, priv_len, flags, priv_len, 1,
                                "eager_offload_first_handler", &rdesc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        return UCS_OK;
    }

    priv_hdr                  = (ucp_offload_first_desc_t*)(rdesc + 1);
    priv_hdr->super.super.tag = stag;
    priv_hdr->total_length    = length; /* total length is not final at this point */
    matchq                    = ucs_unaligned_ptr(&priv_hdr->matchq);

    /* Set msg context to the first fragment address, so that the other
     * fragments could take matchq from it.
     */
    *(ucp_offload_first_desc_t**)context = priv_hdr;

    req = ucp_tag_exp_search(&worker->tm, stag);
    if (req != NULL) {
        req->recv.offset = 0ul;
        ucp_tag_frag_hash_init_exp(matchq, req);
        ucp_eager_common_matched(worker, req, data, length, stag, flags);
        ucp_request_recv_offload_data(req, data, length, flags);
    } else {
        ucp_tag_frag_match_init_unexp(matchq);
        ucp_tag_unexp_recv(&worker->tm, rdesc, stag);
    }

    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_offload_eager_middle_handler(ucp_worker_h worker, void *data,
                                     size_t length, unsigned tl_flags,
                                     uct_tag_t stag, uint64_t imm,
                                     uint16_t flags, void **context)
{
    ucp_offload_first_desc_t *priv_hdr = *(ucp_offload_first_desc_t**)context;
    ucp_tag_frag_match_t *matchq       = ucs_unaligned_ptr(&priv_hdr->matchq);
    ucp_recv_desc_t *rdesc             = NULL;
    ucp_offload_ssend_hdr_t *sync_hdr;
    ucp_recv_desc_t *first_rdesc;
    void *hdr;
    size_t hdr_length;
    ucs_status_t status;

    /* With tag offload, the total message length is not sent with the first
     * fragment due to lack of space in the header, every incoming fragment
     * adds its length to the first_fragment->total_length.
     */
    priv_hdr->total_length += length;

    if (!(tl_flags & UCT_CB_PARAM_FLAG_MORE)) {
        first_rdesc         = (ucp_recv_desc_t*)priv_hdr - 1;
        flags              |= UCP_RECV_DESC_FLAG_EAGER_LAST;
        first_rdesc->flags &= ~UCP_RECV_DESC_FLAG_RECV_STARTED;
    }

    /* Last fragment may contain immediate data, indicating that it is
     * synchronous send
     */
    if (imm) {
        ucs_assert(!(tl_flags & UCT_CB_PARAM_FLAG_MORE));

        sync_hdr             = ucp_tag_eager_offload_priv(
                                   tl_flags, data, length,
                                   ucp_offload_ssend_hdr_t);
        sync_hdr->sender_tag = stag;
        sync_hdr->ep_id      = imm;
        flags               |= UCP_RECV_DESC_FLAG_EAGER_SYNC;
        hdr                  = sync_hdr;
        hdr_length           = sizeof(*sync_hdr);
    } else {
        hdr                  = data;
        hdr_length           = 0;
    }

    if (ucp_tag_frag_match_is_unexp(matchq)) {
        status = ucp_recv_desc_init(worker, hdr, length + hdr_length, 0,
                                    tl_flags, hdr_length, flags, hdr_length, 1,
                                    "tag_offload_eager_middle_handler", &rdesc);
        if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
            return UCS_OK;
        }

        /* Offset is not known at this point, pass 0 */
        ucp_tag_frag_match_add_unexp(matchq, rdesc, 0ul);
    } else {
        status = ucp_request_recv_offload_data(matchq->exp_req, data, length,
                                               flags);
        if (status != UCS_INPROGRESS) {
            ucp_tag_offload_release_first(priv_hdr);
        }
        status = UCS_OK;
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_offload_unexp_eager,
                 (arg, data, length, tl_flags, stag, imm, context),
                 void *arg, void *data, size_t length, unsigned tl_flags,
                 uct_tag_t stag, uint64_t imm, void **context)
{
    /* Align data with AM protocol. We should add tag before the data. */
    ucp_worker_iface_t *wiface = arg;
    ucp_worker_t *worker       = wiface->worker;
    uint16_t flags             = UCP_RECV_DESC_FLAG_EAGER |
                                 UCP_RECV_DESC_FLAG_EAGER_OFFLOAD;
    ucp_eager_sync_hdr_t *priv;
    int priv_len;

    UCP_WORKER_STAT_TAG_OFFLOAD(wiface->worker, RX_UNEXP_EGR);

    /* Fast path - single-fragment, non-sync eager message */
    if (ucs_likely((tl_flags & UCT_CB_PARAM_FLAG_FIRST) &&
                   !(tl_flags & UCT_CB_PARAM_FLAG_MORE) &&
                   !imm)) {
        ucp_tag_offload_unexp(wiface, stag, length);

        return ucp_eager_offload_handler(wiface->worker, data, length, tl_flags,
                                         flags | UCP_RECV_DESC_FLAG_EAGER_ONLY,
                                         stag, "tag_offload_unexp_eager");
    }

    if (!(tl_flags & UCT_CB_PARAM_FLAG_FIRST)) {
        /* Either middle or last fragment */
        return ucp_tag_offload_eager_middle_handler(worker, data, length,
                                                    tl_flags, stag, imm, flags,
                                                    context);
    }

    /* Either first eager fragment or entire sync eager message */
    ucp_tag_offload_unexp(wiface, stag, length);

    if (tl_flags & UCT_CB_PARAM_FLAG_MORE) {
        /* First part of the fragmented message */
        return ucp_tag_offload_eager_first_handler(
                   worker, data, length, tl_flags, stag,
                   flags | UCP_RECV_DESC_FLAG_RECV_STARTED, context);
    }

    /* Sync eager only packet */
    ucs_assert(!(tl_flags & UCT_CB_PARAM_FLAG_MORE));
    ucs_assert(imm);

    flags                |= UCP_RECV_DESC_FLAG_EAGER_ONLY |
                            UCP_RECV_DESC_FLAG_EAGER_SYNC;
    priv_len              = sizeof(*priv);
    priv                  = ucp_tag_eager_offload_priv(tl_flags, data, length,
                                                       ucp_eager_sync_hdr_t);
    priv->req.req_id      = UCS_PTR_MAP_KEY_INVALID;
    priv->req.ep_id       = imm;
    priv->super.super.tag = stag;
    return ucp_eager_tagged_handler(worker, priv, length + priv_len,
                                    tl_flags, flags, priv_len, priv_len,
                                    "tag_offload_unexp_eager_sync");
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
        ucs_assert(eagers_hdr->req.ep_id != UCS_PTR_MAP_KEY_INVALID);
        snprintf(buffer, max,
                 "EGRS tag %" PRIx64 " ep_id 0x%" PRIx64 " req_id 0x%" PRIx64
                 " len %zu",
                 eagers_hdr->super.super.tag, eagers_hdr->req.ep_id,
                 eagers_hdr->req.req_id, length - sizeof(*eagers_hdr));
        header_len = sizeof(*eagers_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_FIRST:
        snprintf(buffer, max, "EGRS_F tag %"PRIx64" msgid %"PRIx64" len %zu "
                 "ep_id 0x%"PRIx64" req_id 0x%"PRIx64,
                 eagers_first_hdr->super.super.super.tag,
                 eagers_first_hdr->super.msg_id,
                 eagers_first_hdr->super.total_len,
                 eagers_first_hdr->req.ep_id,
                 eagers_first_hdr->req.req_id);
        header_len = sizeof(*eagers_first_hdr);
        break;
    case UCP_AM_ID_EAGER_SYNC_ACK:
        snprintf(buffer, max, "EGRS_A req_id %"PRIx64" status '%s'",
                 rep_hdr->req_id, ucs_status_string(rep_hdr->status));
        header_len = sizeof(*rep_hdr);
        break;
    case UCP_AM_ID_OFFLOAD_SYNC_ACK:
        snprintf(buffer, max, "EGRS_A_O tag %"PRIx64" ep_id 0x%"PRIx64,
                 off_rep_hdr->sender_tag, off_rep_hdr->ep_id);
        header_len = sizeof(*rep_hdr);
        break;
    default:
        return;
    }

    p = buffer + strlen(buffer);
    ucp_dump_payload(worker->context, p, buffer + max - p,
                     UCS_PTR_BYTE_OFFSET(data, header_len), length - header_len);
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_ONLY,
                         ucp_eager_only_handler, ucp_eager_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_FIRST,
                         ucp_eager_first_handler, ucp_eager_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_MIDDLE,
                         ucp_eager_middle_handler, ucp_eager_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_ONLY,
                        ucp_eager_sync_only_handler, ucp_eager_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_FIRST,
                         ucp_eager_sync_first_handler, ucp_eager_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_EAGER_SYNC_ACK,
                         ucp_eager_sync_ack_handler, ucp_eager_dump, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_TAG, UCP_AM_ID_OFFLOAD_SYNC_ACK,
                         ucp_eager_offload_sync_ack_handler, ucp_eager_dump, 0);
