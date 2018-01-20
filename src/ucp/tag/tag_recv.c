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


static UCS_F_ALWAYS_INLINE void
ucp_tag_recv_request_completed(ucp_request_t *req, ucs_status_t status,
                               ucp_tag_recv_info_t *info, const char *function)
{
    ucs_trace_req("%s returning completed request %p (%p) stag 0x%"PRIx64" len %zu, %s",
                  function, req, req + 1, info->sender_tag, info->length,
                  ucs_status_string(status));

    req->status = status;
    if ((req->flags |= UCP_REQUEST_FLAG_COMPLETED) & UCP_REQUEST_FLAG_RELEASED) {
        ucp_request_put(req);
    }
    UCS_PROFILE_REQUEST_EVENT(req, "complete_recv", 0);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_recv_common(ucp_worker_h worker, void *buffer, size_t count,
                    uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                    ucp_request_t *req, uint16_t req_flags, ucp_tag_recv_callback_t cb,
                    ucp_recv_desc_t *rdesc, const char *debug_name)
{
    unsigned common_flags = UCP_REQUEST_FLAG_RECV | UCP_REQUEST_FLAG_EXPECTED;
    ucp_eager_first_hdr_t *eagerf_hdr;
    ucp_request_queue_t *req_queue;
    uct_memory_type_t mem_type;
    size_t hdr_len, recv_len;
    ucs_status_t status;
    uint64_t msg_id;

    ucp_trace_req(req, "%s buffer %p dt 0x%lx count %zu tag %"PRIx64"/%"PRIx64,
                  debug_name, buffer, datatype, count, tag, tag_mask);

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

        req->flags                    = UCP_REQUEST_FLAG_RECV | req_flags;
        hdr_len                       = rdesc->payload_offset;
        recv_len                      = rdesc->length - hdr_len;
        req->recv.tag.info.sender_tag = ucp_rdesc_get_tag(rdesc);
        req->recv.tag.info.length     = recv_len;

        ucp_memory_type_detect_mds(worker->context, buffer, recv_len, &mem_type);

        status = ucp_dt_unpack_only(worker, buffer, count, datatype, mem_type,
                                    (void*)(rdesc + 1) + hdr_len, recv_len, 1);
        ucp_tag_unexp_desc_release(rdesc);

        if (req_flags & UCP_REQUEST_FLAG_CALLBACK) {
            cb(req + 1, status, &req->recv.tag.info);
        }
        ucp_tag_recv_request_completed(req, status, &req->recv.tag.info,
                                       debug_name);
        return;
    }

    /* Initialize receive request */
    req->status             = UCS_OK;
    req->recv.worker        = worker;
    req->recv.buffer        = buffer;
    req->recv.datatype      = datatype;

    ucp_dt_recv_state_init(&req->recv.state, buffer, datatype, count);

    if (!UCP_DT_IS_CONTIG(datatype)) {
        common_flags       |= UCP_REQUEST_FLAG_BLOCK_OFFLOAD;
    }

    req->flags              = common_flags | req_flags;
    req->recv.length        = ucp_dt_length(datatype, count, buffer,
                                            &req->recv.state);

    ucp_memory_type_detect_mds(worker->context, buffer, req->recv.length, &mem_type);

    req->recv.mem_type      = mem_type;
    req->recv.tag.tag       = tag;
    req->recv.tag.tag_mask  = tag_mask;
    req->recv.tag.cb        = cb;
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_REQ)) {
        req->recv.tag.info.sender_tag = 0;
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
        return;
    }

    /* Check rendezvous case */
    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_RNDV)) {
        ucp_rndv_matched(worker, req, (void*)(rdesc + 1));
        UCP_WORKER_STAT_RNDV(worker, UNEXP);
        ucp_tag_unexp_desc_release(rdesc);
        return;
    }

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_EAGER_SYNC)) {
        ucp_tag_eager_sync_send_ack(worker, rdesc + 1, rdesc->flags);
    }

    UCP_WORKER_STAT_EAGER_MSG(worker, rdesc->flags);
    ucs_assert(rdesc->flags & UCP_RECV_DESC_FLAG_EAGER);
    eagerf_hdr                    = (void*)(rdesc + 1);
    req->recv.tag.info.sender_tag = ucp_rdesc_get_tag(rdesc);
    req->recv.tag.info.length     =
    req->recv.tag.remaining       = eagerf_hdr->total_len;

    /* process first fragment */
    UCP_WORKER_STAT_EAGER_CHUNK(worker, UNEXP);
    msg_id = eagerf_hdr->msg_id;
    status = ucp_tag_recv_request_process_rdesc(req, rdesc, 0);
    ucs_assert(status == UCS_INPROGRESS);

    /* process additional fragments */
    ucp_tag_frag_list_process_queue(&worker->tm, req, msg_id
                                    UCS_STATS_ARG(UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP));
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_recv_nbr,
                 (worker, buffer, count, datatype, tag, tag_mask, request),
                 ucp_worker_h worker, void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                 void *request)
{
    ucp_request_t *req = (ucp_request_t *)request - 1;
    ucp_recv_desc_t *rdesc;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    rdesc = ucp_tag_unexp_search(&worker->tm, tag, tag_mask, 1, "recv_nbr");
    ucp_tag_recv_common(worker, buffer, count, datatype, tag, tag_mask,
                        req, UCP_REQUEST_DEBUG_FLAG_EXTERNAL, NULL, rdesc,
                        "recv_nbr");

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_tag_recv_nb,
                 (worker, buffer, count, datatype, tag, tag_mask, cb),
                 ucp_worker_h worker, void *buffer, size_t count,
                 uintptr_t datatype, ucp_tag_t tag, ucp_tag_t tag_mask,
                 ucp_tag_recv_callback_t cb)
{
    ucp_recv_desc_t *rdesc;
    ucs_status_ptr_t ret;
    ucp_request_t *req;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    req = ucp_request_get(worker);
    if (ucs_likely(req != NULL)) {
        rdesc = ucp_tag_unexp_search(&worker->tm, tag, tag_mask, 1, "recv_nb");
        ucp_tag_recv_common(worker, buffer, count, datatype, tag, tag_mask, req,
                            UCP_REQUEST_FLAG_CALLBACK, cb, rdesc,"recv_nb");
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
