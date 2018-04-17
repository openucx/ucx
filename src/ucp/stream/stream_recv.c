/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/stream/stream.h>

#include <ucs/datastruct/mpool.inl>
#include <ucs/profile/profile.h>


/* @verbatim
 * Data layout within Stream AM
 * |---------------------------------------------------------------------------------------------------------------------------|
 * | ucp_recv_desc_t                                                 | \    / | ucp_stream_am_data_t | payload                 |
 * |-----------------------------------------------------------------|  \  /  |----------------------|-------------------------|
 * | stream_queue        | length         | payload_offset | flags   |   \/   | am_header            |                         |
 * | tag_list (not used) |                |                |         |   /\   | rdesc                |                         |
 * |---------------------|----------------|----------------|---------|  /  \  |----------------------|-------------------------|
 * | 4 * sizeof(ptr)     | 32 bits        | 32 bits        | 16 bits | /    \ | 64 bits              | up to TL AM buffer size |
 * |---------------------------------------------------------------------------------------------------------------------------|
 * @endverbatim
 *
 * stream_queue   is an entry link in the "unexpected" queue per endpoint
 * length         is an actual size of 'payload'
 * payload_offset is a distance between 'ucp_recv_desc_t *' and 'payload *'
 * X              is an optional empty space which is a result of partial
 *                handled payload in case when 'length' greater than user's
 *                buffer size passed to @ref ucp_stream_recv_nb
 * am_header      is an active message header, not actual after ucp_recv_desc_t
 *                initialization and setup of offsets
 * rdesc          pointer to 'ucp_recv_desc_t *', it's needed to get access to
 *                'ucp_recv_desc_t *' inside @ref ucp_stream_release_data after
 *                the buffer was returned to user by
 *                @ref ucp_stream_recv_data_nb as a pointer to 'paylod'
 */


#define ucp_stream_rdesc_payload(_rdesc)                                      \
    (UCS_PTR_BYTE_OFFSET((_rdesc), (_rdesc)->payload_offset))


#define ucp_stream_rdesc_am_data(_rdesc)                                      \
    ((ucp_stream_am_data_t *)                                                 \
     UCS_PTR_BYTE_OFFSET(ucp_stream_rdesc_payload(_rdesc),                    \
                         -sizeof(ucp_stream_am_data_t)))


#define ucp_stream_rdesc_from_data(_data)                                     \
    ((ucp_stream_am_data_t *)_data - 1)->rdesc


static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_stream_rdesc_dequeue(ucp_ep_ext_proto_t *ep_ext)
{
    ucp_recv_desc_t *rdesc = ucs_queue_pull_elem_non_empty(&ep_ext->stream.match_q,
                                                           ucp_recv_desc_t,
                                                           stream_queue);
    ucs_assert(ucp_stream_ep_has_data(ep_ext));
    if (ucs_unlikely(ucs_queue_is_empty(&ep_ext->stream.match_q))) {
        ucp_ep_from_ext_proto(ep_ext)->flags &= ~UCP_EP_FLAG_STREAM_HAS_DATA;
        if (ucp_stream_ep_is_queued(ep_ext)) {
            ucp_stream_ep_dequeue(ep_ext);
        }
    }

    return rdesc;
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_stream_rdesc_get(ucp_ep_ext_proto_t *ep_ext)
{
    ucp_recv_desc_t *rdesc = ucs_queue_head_elem_non_empty(&ep_ext->stream.match_q,
                                                           ucp_recv_desc_t,
                                                           stream_queue);

    ucs_assert(ucp_stream_ep_has_data(ep_ext));
    ucs_trace_data("ep %p, rdesc %p with %u stream bytes",
                   ucp_ep_from_ext_proto(ep_ext), rdesc, rdesc->length);

    return rdesc;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_data_nb, (ep, length),
                 ucp_ep_h ep, size_t *length)
{
    ucp_ep_ext_proto_t   *ep_ext = ucp_ep_ext_proto(ep);
    ucp_recv_desc_t      *rdesc;
    ucp_stream_am_data_t *am_data;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    if (ucs_unlikely(!ucp_stream_ep_has_data(ep_ext))) {
        UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
        return UCS_STATUS_PTR(UCS_OK);
    }

    rdesc = ucp_stream_rdesc_dequeue(ep_ext);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);

    *length         = rdesc->length;
    am_data         = ucp_stream_rdesc_am_data(rdesc);
    am_data->rdesc  = rdesc;
    return am_data + 1;
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_rdesc_dequeue_and_release(ucp_recv_desc_t *rdesc,
                                     ucp_ep_ext_proto_t *ep_ext)
{
    ucs_assert(ucp_stream_ep_has_data(ep_ext));
    ucs_assert(rdesc == ucs_queue_head_elem_non_empty(&ep_ext->stream.match_q,
                                                      ucp_recv_desc_t,
                                                      stream_queue));
    ucp_stream_rdesc_dequeue(ep_ext);
    ucp_recv_desc_release(rdesc);
}

UCS_PROFILE_FUNC_VOID(ucp_stream_data_release, (ep, data),
                      ucp_ep_h ep, void *data)
{
    ucp_recv_desc_t *rdesc = ucp_stream_rdesc_from_data(data);

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    ucp_recv_desc_release(rdesc);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_stream_rdata_unpack(const void *rdata, size_t length, ucp_request_t *dst_req)
{
    size_t       valid_len;
    int          last;
    ucs_status_t status;

    /* Truncated error is not actual for stream, need to adjust */
    valid_len = dst_req->recv.length - dst_req->recv.stream.offset;
    if (valid_len <= length) {
        last = (valid_len == length);
    } else {
        valid_len = length;
        last      = !(dst_req->flags & UCP_REQUEST_FLAG_STREAM_RECV_WAITALL);
    }

    status = ucp_request_recv_data_unpack(dst_req, rdata, valid_len,
                                          dst_req->recv.stream.offset, last);
    if (ucs_likely(status == UCS_OK)) {
        dst_req->recv.stream.offset += valid_len;
        ucs_trace_data("unpacked %zd bytes of stream data %p",
                       valid_len, rdata);
        return valid_len;
    }

    ucs_assert(status != UCS_ERR_MESSAGE_TRUNCATED);
    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_rdesc_advance(ucp_recv_desc_t *rdesc, ssize_t offset,
                         ucp_ep_ext_proto_t *ep_ext)
{
    ucs_assert(offset <= rdesc->length);

    if (ucs_unlikely(offset < 0)) {
        return offset;
    } else if (ucs_likely(offset == rdesc->length)) {
        ucp_stream_rdesc_dequeue_and_release(rdesc, ep_ext);
    } else {
        rdesc->length         -= offset;
        rdesc->payload_offset += offset;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_process_rdesc_inplace(ucp_recv_desc_t *rdesc, ucp_datatype_t dt,
                                 void *buffer, size_t count, size_t length,
                                 ucp_ep_ext_proto_t *ep_ext)
{
    ucs_status_t status;
    ssize_t unpacked;

    status   = ucp_dt_unpack_only(ucp_ep_from_ext_proto(ep_ext)->worker, buffer, count, dt,
                                  UCT_MD_MEM_TYPE_HOST, ucp_stream_rdesc_payload(rdesc),
                                  length, 0);

    unpacked = ucs_likely(status == UCS_OK) ? length : status;

    return ucp_stream_rdesc_advance(rdesc, unpacked, ep_ext);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_process_rdesc(ucp_recv_desc_t *rdesc, ucp_ep_ext_proto_t *ep_ext,
                         ucp_request_t *req)
{
    ssize_t unpacked;

    unpacked = ucp_stream_rdata_unpack(ucp_stream_rdesc_payload(rdesc),
                                       rdesc->length, req);
    ucs_assert(req->recv.stream.offset <= req->recv.length);

    return ucp_stream_rdesc_advance(rdesc, unpacked, ep_ext);
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_recv_request_init(ucp_request_t *req, void *buffer, size_t count,
                             size_t length, ucp_datatype_t datatype,
                             ucp_stream_recv_callback_t cb,
                             uint16_t request_flags)
{
    req->flags              = UCP_REQUEST_FLAG_CALLBACK | request_flags;
#if ENABLE_ASSERT
    req->flags             |= UCP_REQUEST_FLAG_STREAM_RECV;
    req->status             = UCS_OK; /* for ucp_request_recv_data_unpack() */
#endif
    req->recv.stream.cb     = cb;
    req->recv.stream.length = 0;
    req->recv.stream.offset = 0;

    ucp_dt_recv_state_init(&req->recv.state, buffer, datatype, count);

    req->recv.buffer   = buffer;
    req->recv.datatype = datatype;
    req->recv.length   = ucs_likely(!UCP_DT_IS_GENERIC(datatype)) ? length :
                         ucp_dt_length(datatype, count, NULL, &req->recv.state);
    req->recv.mem_type = UCT_MD_MEM_TYPE_HOST;
}

static UCS_F_ALWAYS_INLINE int
ucp_stream_recv_nb_is_inplace(ucp_ep_ext_proto_t *ep_ext, size_t dt_length)
{
    return ucp_stream_ep_has_data(ep_ext) &&
           (ucp_stream_rdesc_get(ep_ext)->length >= dt_length);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_nb,
                 (ep, buffer, count, datatype, cb, length, flags),
                 ucp_ep_h ep, void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_stream_recv_callback_t cb,
                 size_t *length, unsigned flags)
{
    ucs_status_t        status     = UCS_OK;
    ucp_ep_ext_proto_t  *ep_ext    = ucp_ep_ext_proto(ep);
    size_t              dt_length;
    ucp_request_t       *req;
    ucp_recv_desc_t     *rdesc;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    if (ucs_likely(!UCP_DT_IS_GENERIC(datatype))) {
        dt_length = ucp_dt_length(datatype, count, buffer, NULL);
        if (ucs_likely(ucp_stream_recv_nb_is_inplace(ep_ext, dt_length))) {
            status = ucp_stream_process_rdesc_inplace(ucp_stream_rdesc_get(ep_ext),
                                                      datatype, buffer, count,
                                                      dt_length, ep_ext);
            *length = dt_length;
            goto out_status;
        }
    } else {
        dt_length = 0; /* Suppress warnings of paranoid compilers */
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(req == NULL)) {
        status = UCS_ERR_NO_MEMORY;
        goto out_status;
    }

    ucp_stream_recv_request_init(req, buffer, count, dt_length, datatype, cb,
                                 (flags & UCP_STREAM_RECV_FLAG_WAITALL) ?
                                 UCP_REQUEST_FLAG_STREAM_RECV_WAITALL : 0);

    /* OK, lets obtain all arrived data which matches the recv size */
    while ((req->recv.stream.offset < req->recv.length) &&
           ucp_stream_ep_has_data(ep_ext)) {

        rdesc  = ucp_stream_rdesc_get(ep_ext);
        status = ucp_stream_process_rdesc(rdesc, ep_ext, req);
        if (ucs_unlikely(status != UCS_OK)) {
            goto out_put_request;
        }

        /*
         * NOTE: generic datatype can be completed with any amount of data to
         *       avoid extra logic in ucp_stream_process_rdesc, exception is
         *       WAITALL flag
         */
        if (ucs_unlikely(UCP_DT_IS_GENERIC(req->recv.datatype)) &&
            !(req->flags & UCP_REQUEST_FLAG_STREAM_RECV_WAITALL)) {
            break;
        }
    }

    ucs_assert(req->recv.stream.offset <= req->recv.length);

    if (ucp_request_can_complete_stream_recv(req)) {
        *length = req->recv.stream.offset;
    } else {
        ucs_assert(!ucp_stream_ep_has_data(ep_ext));
        ucs_queue_push(&ep_ext->stream.match_q, &req->recv.queue);
        req += 1;
        goto out;
    }

out_put_request:
    ucp_request_put(req);

out_status:
    req = UCS_STATUS_PTR(status);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return req;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_am_data_process(ucp_worker_t *worker, ucp_ep_ext_proto_t *ep_ext,
                           ucp_stream_am_data_t *am_data, size_t length,
                           unsigned am_flags)
{
    ucp_recv_desc_t  rdesc_tmp;
    void            *payload;
    ucp_recv_desc_t *rdesc;
    ucp_request_t   *req;
    ssize_t          unpacked;

    rdesc_tmp.length         = length;
    rdesc_tmp.payload_offset = sizeof(*am_data); /* add sizeof(*rdesc) only if
                                                    am_data wont be handled in
                                                    place */

    /* First, process expected requests */
    if (!ucp_stream_ep_has_data(ep_ext)) {
        while (!ucs_queue_is_empty(&ep_ext->stream.match_q)) {
            req      = ucs_queue_head_elem_non_empty(&ep_ext->stream.match_q,
                                                     ucp_request_t, recv.queue);
            payload  = UCS_PTR_BYTE_OFFSET(am_data, rdesc_tmp.payload_offset);
            unpacked = ucp_stream_rdata_unpack(payload, rdesc_tmp.length, req);
            if (ucs_unlikely(unpacked < 0)) {
                ucs_fatal("failed to unpack from am_data %p with offset %u to request %p",
                          am_data, rdesc_tmp.payload_offset, req);
            } else if (unpacked == rdesc_tmp.length) {
                if (ucp_request_can_complete_stream_recv(req)) {
                    ucp_request_complete_stream_recv(req, ep_ext, UCS_OK);
                }
                return UCS_OK;
            }
            ucp_stream_rdesc_advance(&rdesc_tmp, unpacked, ep_ext);
            /* This request is full, try next one */
            ucs_assert(ucp_request_can_complete_stream_recv(req));
            ucp_request_complete_stream_recv(req, ep_ext, UCS_OK);
        }
    }

    ucs_assert(rdesc_tmp.length > 0);

    /* Now, enqueue the rest of data */
    if (ucs_likely(!(am_flags & UCT_CB_PARAM_FLAG_DESC))) {
        rdesc = (ucp_recv_desc_t*)ucs_mpool_get_inline(&worker->am_mp);
        ucs_assertv_always(rdesc != NULL,
                           "ucp recv descriptor is not allocated");
        rdesc->length         = rdesc_tmp.length;
        /* reset offset to improve locality */
        rdesc->payload_offset = sizeof(*rdesc) + sizeof(*am_data);
        rdesc->flags          = 0;
        memcpy(ucp_stream_rdesc_payload(rdesc),
               UCS_PTR_BYTE_OFFSET(am_data, rdesc_tmp.payload_offset),
               rdesc_tmp.length);
    } else {
        /* slowpath */
        rdesc                 = (ucp_recv_desc_t *)am_data - 1;
        rdesc->length         = rdesc_tmp.length;
        rdesc->payload_offset = rdesc_tmp.payload_offset + sizeof(*rdesc);
        rdesc->priv_length    = 0;
        rdesc->flags          = UCP_RECV_DESC_FLAG_UCT_DESC;
    }

    ucp_ep_from_ext_proto(ep_ext)->flags |= UCP_EP_FLAG_STREAM_HAS_DATA;
    ucs_queue_push(&ep_ext->stream.match_q, &rdesc->stream_queue);

    return UCS_INPROGRESS;
}

void ucp_stream_ep_init(ucp_ep_h ep)
{
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);

    if (ep->worker->context->config.features & UCP_FEATURE_STREAM) {
        ep_ext->stream.ready_list.prev = NULL;
        ep_ext->stream.ready_list.next = NULL;
        ucs_queue_head_init(&ep_ext->stream.match_q);
    }
}

void ucp_stream_ep_cleanup(ucp_ep_h ep)
{
    size_t length;
    void *data;

    if (ep->worker->context->config.features & UCP_FEATURE_STREAM) {
        while ((data = ucp_stream_recv_data_nb(ep, &length)) != NULL) {
            ucs_assert_always(!UCS_PTR_IS_ERR(data));
            ucp_stream_data_release(ep, data);
        }
    }
}

void ucp_stream_ep_activate(ucp_ep_h ep)
{
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);

    if ((ep->worker->context->config.features & UCP_FEATURE_STREAM) &&
        ucp_stream_ep_has_data(ep_ext) && !ucp_stream_ep_is_queued(ep_ext)) {
        ucp_stream_ep_enqueue(ep_ext, ep->worker);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_am_handler(void *am_arg, void *am_data, size_t am_length,
                      unsigned am_flags)
{
    ucp_worker_h          worker    = am_arg;
    ucp_stream_am_data_t *data      = am_data;
    ucp_ep_h              ep;
    ucp_ep_ext_proto_t    *ep_ext;
    ucs_status_t          status;

    ucs_assert(am_length >= sizeof(ucp_stream_am_hdr_t));

    ep     = ucp_worker_get_ep_by_ptr(worker, data->hdr.ep_ptr);
    ep_ext = ucp_ep_ext_proto(ep);

    if (ucs_unlikely(ep->flags & UCP_EP_FLAG_HIDDEN)) {
        ucs_trace_data("ep %p: stream is invalid", ep);
        /* drop the data */
        return UCS_OK;
    }

    status = ucp_stream_am_data_process(worker, ep_ext, data,
                                        am_length - sizeof(data->hdr),
                                        am_flags);
    if (status == UCS_OK) {
        /* rdesc was processed in place */
        return UCS_OK;
    }

    ucs_assert(status == UCS_INPROGRESS);

    if (!ucp_stream_ep_is_queued(ep_ext) && (ep->flags & UCP_EP_FLAG_USED)) {
        ucp_stream_ep_enqueue(ep_ext, worker);
    }

    return (am_flags & UCT_CB_PARAM_FLAG_DESC) ? UCS_INPROGRESS : UCS_OK;
}

static void ucp_stream_am_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                               uint8_t id, const void *data, size_t length,
                               char *buffer, size_t max)
{
    const ucp_stream_am_hdr_t *hdr    = data;
    size_t                    hdr_len = sizeof(*hdr);
    char                      *p;

    snprintf(buffer, max, "STREAM ep_ptr 0x%lx", hdr->ep_ptr);
    p = buffer + strlen(buffer);

    ucs_assert(hdr->ep_ptr != 0);
    ucp_dump_payload(worker->context, p, buffer + max - p, data + hdr_len,
                     length - hdr_len);
}

UCP_DEFINE_AM(UCP_FEATURE_STREAM, UCP_AM_ID_STREAM_DATA,
              ucp_stream_am_handler, ucp_stream_am_dump,
              UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_STREAM_DATA);
