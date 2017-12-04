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
#include <ucs/debug/profile.h>

#include <ucp/tag/eager.h> /* TODO: remove ucp_eager_sync_hdr_t usage */


/* @verbatim
 * Data layout within Stream AM
 * |----------------------------------------------------------------------------------------------------------------------|
 * | ucp_recv_desc_t                                            | \    / | ucp_stream_am_data_t | payload                 |
 * |------------------------------------------------------------|  \  /  |----------------------|-------------------------|
 * | stream_queue    | length         | stream_offset | flags   |   \/   | am_header            |                         |
 * |                 |                |               |         |   /\   | offset               |                         |
 * |-----------------|----------------|---------------|---------|  /  \  |----------------------|-------------------------|
 * | sizeof(ptr)     | sizeof(size_t) | 16 bits       | 16 bits | /    \ | 64 bits              | up to TL AM buffer size |
 * |----------------------------------------------------------------------------------------------------------------------|
 * @endverbatim
 *
 * stream_queue  is an entry link in the "unexpected" queue per endpoint
 * length        is an actual size of 'payload'
 * stream_offset is a distance between 'ucp_recv_desc_t *' and
 *               'ucp_stream_am_data_t *'
 * X             is an optional empty space which is a result of partial
 *               handled payload in case when 'length' greater than user's
 *               buffer size passed to @ref ucp_stream_recv_nb
 * am_header     is an active message header, not actual after ucp_recv_desc_t
 *               initialization and setup of offsets
 * offset        is a distance between 'ucp_recv_desc_t *' and 'payload', it's
 *               needed to get access to ucp_recv_desc_t inside
 *               @ref ucp_stream_release_data after the buffer was returned to
 *               user by @ref ucp_stream_recv_data_nb as a pointer to 'paylod'
 */


#define ucp_stream_rdesc_am_data(_rdesc)                                      \
    ((ucp_stream_am_data_t *)                                                 \
     UCS_PTR_BYTE_OFFSET((_rdesc), (_rdesc)->stream_offset))


#define ucp_stream_rdesc_from_data(_data)                                     \
    ((ucp_recv_desc_t *)                                                      \
     UCS_PTR_BYTE_OFFSET(_data,                                               \
                         -(ucs_container_of(_data, ucp_stream_am_data_t,      \
                                            payload)->offset)))


static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_stream_rdesc_dequeue(ucp_ep_ext_stream_t *ep_stream)
{
    ucp_recv_desc_t *rdesc = ucs_queue_pull_elem_non_empty(&ep_stream->data,
                                                           ucp_recv_desc_t,
                                                           stream_queue);

    rdesc->flags &= ~UCP_RECV_DESC_FLAG_STREAM_Q;
    if (ucs_unlikely(ucs_queue_is_empty(&ep_stream->data))) {
        if (ucp_stream_ep_is_queued(ep_stream->ucp_ep)) {
            ucp_stream_ep_dequeue(ep_stream);
        }
    }

    return rdesc;
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_stream_recv_data_nb_internal(ucp_ep_ext_stream_t *ep_stream, int dequeue)
{
    ucp_recv_desc_t *rdesc;

    if (ucs_queue_is_empty(&ep_stream->data)) {
        return NULL;
    }

    rdesc = dequeue ? ucp_stream_rdesc_dequeue(ep_stream) :
            ucs_queue_head_elem_non_empty(&ep_stream->data, ucp_recv_desc_t,
                                          stream_queue);
    ucs_trace_data("ep %p, rdesc %p with %zu stream bytes",
                   ep_stream->ucp_ep, rdesc, rdesc->length);

    return rdesc;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_data_nb,
                 (ep, length), ucp_ep_h ep, size_t *length)
{
    ucp_recv_desc_t  *rdesc;
    ucs_status_ptr_t ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    rdesc = ucp_stream_recv_data_nb_internal(ep->ext.stream, 1);
    if (ucs_likely(rdesc != NULL)) {
        *length = rdesc->length;
        ret = ucp_stream_rdesc_am_data(rdesc)->payload;
    } else {
        ret = UCS_STATUS_PTR(UCS_OK);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);

    return ret;
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_rdesc_release(ucp_recv_desc_t *rdesc, ucp_ep_ext_stream_t *ep)
{
    if (rdesc->flags & UCP_RECV_DESC_FLAG_STREAM_Q) {
        ucs_assert(!ucs_queue_is_empty(&ep->data));
        ucs_assert(rdesc == ucs_queue_head_elem_non_empty(&ep->data,
                                                          ucp_recv_desc_t,
                                                          stream_queue));
        ucp_stream_rdesc_dequeue(ep);
    }

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        uct_iface_release_desc(UCS_PTR_BYTE_OFFSET(rdesc,
                                                   -sizeof(ucp_eager_sync_hdr_t)));
    } else {
        ucs_mpool_put_inline(rdesc);
    }
}

UCS_PROFILE_FUNC_VOID(ucp_stream_data_release, (ep, data),
                      ucp_ep_h ep, void *data)
{
    ucp_recv_desc_t *rdesc = ucp_stream_rdesc_from_data(data);

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    ucp_stream_rdesc_release(rdesc, ep->ext.stream);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_stream_rdata_unpack(void *rdata, size_t length, ucp_request_t *dst_req)
{
    /* Truncated error is not actual for stream, need to adjust */
    size_t       valid_len = ucs_min((dst_req->recv.length -
                                      dst_req->recv.state.offset),
                                     length);
    ucs_status_t status;

    status = ucp_dt_unpack(dst_req->recv.datatype, dst_req->recv.buffer,
                           dst_req->recv.length, &dst_req->recv.state, rdata,
                           valid_len, UCP_RECV_DESC_FLAG_LAST);

    if (ucs_likely(status == UCS_OK)) {
        dst_req->recv.state.offset += valid_len;
        ucs_trace_data("unpacked %zd bytes of stream data from rdesc %p\n",
                       valid_len, ucp_stream_rdesc_from_data(rdata));
        return valid_len;
    }

    ucs_assert(status != UCS_ERR_MESSAGE_TRUNCATED);
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_rdesc_advance(ucp_recv_desc_t *rdesc, size_t offset)
{
    ucp_stream_am_data_t *am_data;

    ucs_assert(offset < rdesc->length);

    rdesc->length        -= offset;
    rdesc->stream_offset += offset;

    am_data         = ucp_stream_rdesc_am_data(rdesc);
    am_data->offset = rdesc->stream_offset + ucs_offsetof(ucp_stream_am_data_t,
                                                          payload);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_process_rdesc(ucp_recv_desc_t *rdesc, ucp_ep_ext_stream_t *ep_stream,
                         ucp_request_t *req)
{
    void    *rdata = ucp_stream_rdesc_am_data(rdesc)->payload;
    ssize_t unpacked;

    unpacked = ucp_stream_rdata_unpack(rdata, rdesc->length, req);
    if (ucs_unlikely(unpacked < 0)) {
        return unpacked;
    } else if (ucs_likely(unpacked == rdesc->length)) {
        ucp_stream_rdesc_release(rdesc, ep_stream);
    } else {
        ucp_stream_rdesc_advance(rdesc, unpacked);
    }

    ucs_assert(req->recv.state.offset <= req->recv.length);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucp_request_t *
ucp_stream_recv_request_get(ucp_worker_t *worker, void *buffer, size_t count,
                            ucp_datatype_t datatype,
                            ucp_stream_recv_callback_t cb)
{
    ucp_request_t *req = ucp_request_get(worker);

    if (ucs_unlikely(req == NULL)) {
        return NULL;
    }

    req->flags = UCP_REQUEST_FLAG_CALLBACK | UCP_REQUEST_FLAG_STREAM_RECV;
    req->recv.stream.cb     = cb;
    req->recv.stream.length = 0;

    req->recv.buffer   = buffer;
    req->recv.datatype = datatype;
    req->recv.length   = ucp_dt_length(datatype, count, buffer,
                                       &req->recv.state);

    ucp_request_recv_state_init(req, buffer, datatype, count);

    return req;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_recv_request_put(ucp_request_t *req, size_t *length,
                            ucs_status_t status)
{
    *length = req->recv.state.offset;
    ucp_request_put(req);
    return UCS_STATUS_PTR(status);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_nb,
                 (ep, buffer, count, datatype, cb, length, flags),
                 ucp_ep_h ep, void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_stream_recv_callback_t cb,
                 size_t *length, unsigned flags)
{
    ucp_ep_ext_stream_t *ep_stream;
    ucp_request_t       *req;
    ucs_status_ptr_t    rdesc;
    ucs_status_t        status;

    if (UCP_DT_IS_GENERIC(datatype)) {
        ucs_error("ucp_stream_recv_nb doesn't support generic datatype");
        return UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
    }

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    req = ucp_stream_recv_request_get(ep->worker, buffer, count, datatype, cb);
    if (ucs_unlikely(req == NULL)) {
        req = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ep_stream = ep->ext.stream;

    /* if there are already posted requests, need to queue this one */
    if (!ucs_queue_is_empty(&ep_stream->reqs)) {
        ucs_queue_push(&ep_stream->reqs, &req->recv.queue);
        goto ptr_out;
    }

    /* OK, lets obtain all arrived data which matches the recv size */
    while (req->recv.state.offset < req->recv.length) {
        rdesc = ucp_stream_recv_data_nb_internal(ep_stream, 0);
        if (rdesc == NULL) {
            /* No data any more. Enqueue empty request or complete if some
             * data is filled */
            if (req->recv.state.offset == 0) {
                ucs_queue_push(&ep_stream->reqs, &req->recv.queue);
                goto ptr_out;
            } else {
                req = ucp_stream_recv_request_put(req, length, UCS_OK);
                goto out;
            }
        }

        status = ucp_stream_process_rdesc(rdesc, ep_stream, req);
        if (ucs_unlikely(status != UCS_OK)) {
            req = ucp_stream_recv_request_put(req, length, status);
            goto out;
        }
    }

    ucs_assert(req->recv.state.offset == req->recv.length);
    req = ucp_stream_recv_request_put(req, length, UCS_OK);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return req;

ptr_out:
    ucs_assert(UCS_PTR_IS_PTR(req));
    req += 1;
    goto out;
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_stream_am_rdesc_get(ucp_worker_t *worker, ucp_stream_am_data_t *data,
                        size_t length, unsigned am_flags)
{
    ucp_recv_desc_t *rdesc;

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        /* slowpath */
        rdesc        = (ucp_recv_desc_t *)data - 1;
        rdesc->flags = UCP_RECV_DESC_FLAG_UCT_DESC;
    } else {
        /* TODO: optimize if there are posted recvs */
        rdesc = (ucp_recv_desc_t*)ucs_mpool_get_inline(&worker->am_mp);
        if (ucs_likely(rdesc != NULL)) {
            rdesc->flags = 0;
            memcpy(((ucp_stream_am_data_t *)(rdesc + 1))->payload,
                   data->payload, length);
        } else {
            ucs_fatal("ucp recv descriptor is not allocated");
        }
    }

    rdesc->length        = length;
    rdesc->stream_offset = sizeof(*rdesc);
    ucp_stream_rdesc_am_data(rdesc)->offset = sizeof(*rdesc) +
                                              ucs_offsetof(ucp_stream_am_data_t,
                                                           payload);

    return rdesc;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_am_handler(void *am_arg, void *am_data, size_t am_length,
                      unsigned am_flags)
{
    ucp_ep_ext_stream_t  *ep_stream = NULL;
    ucp_worker_h          worker    = am_arg;
    ucp_stream_am_data_t *data      = am_data;
    ucp_stream_am_data_t *data_iter;
    ucp_recv_desc_t      *rdesc;
    ucp_recv_desc_t      *rdesc_iter;
    ucp_ep_h              ep;
    ucp_request_t        *req;
    ssize_t               unpacked;

    ucs_assert(am_length >= sizeof(ucp_stream_am_hdr_t));

    ep = ucp_worker_ep_find(worker, data->hdr.sender_uuid);
    if (ucs_unlikely(ep == NULL)) {
        ucs_error("ep is not found by uuid: %lu", data->hdr.sender_uuid);
        goto out;
    }

    ep_stream = ep->ext.stream;

    rdesc = ucp_stream_am_rdesc_get(worker, data, am_length - sizeof(data->hdr),
                                    am_flags);
    ucs_assert(rdesc != NULL);

    rdesc->flags |= UCP_RECV_DESC_FLAG_STREAM_Q;
    ucs_queue_push(&ep_stream->data, &rdesc->stream_queue);
    if (ucp_stream_ep_is_queued(ep)) {
        goto out;
    } else {
        ucp_stream_ep_enqueue(ep_stream, worker);
        if (ucs_queue_is_empty(&ep_stream->reqs)) {
            goto out;
        }
    }

    do {
        req = ucs_queue_head_elem_non_empty(&ep_stream->reqs, ucp_request_t,
                                            recv.queue);
        while ((rdesc_iter = ucp_stream_recv_data_nb_internal(ep_stream, 0)) !=
                NULL) {
            data_iter = ucp_stream_rdesc_am_data(rdesc_iter);
            unpacked  = ucp_stream_rdata_unpack(data_iter->payload,
                                                rdesc_iter->length, req);
            if (ucs_unlikely(unpacked < 0)) {
                goto out;
            } else if (unpacked < rdesc_iter->length) {
                ucp_stream_rdesc_advance(rdesc_iter, unpacked);
                /* This request is full, try next one */
                ucp_request_complete_stream_recv(req, ep_stream, UCS_OK);
                break;
            } else {
                /* The descriptor is completely processed, go to next */
                ucs_assert(rdesc_iter->length == unpacked);
                /* Do not release currently arrived rdesc directly from
                 * the callback */
                if (rdesc_iter != rdesc) {
                    ucp_stream_rdesc_release(rdesc_iter, ep_stream);
                } else {
                    rdesc_iter = ucp_stream_rdesc_dequeue(ep_stream);
                    ucs_assert(rdesc_iter == rdesc);
                    /* return into the pool buffered rdesc */
                    if (!(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
                        ucs_mpool_put_inline(rdesc_iter);
                    }
                }
                rdesc_iter = NULL;
            }
        }
    } while ((rdesc_iter != NULL ) &&
             !ucs_queue_is_empty(&ep_stream->reqs));

    if (rdesc_iter == NULL) {
        /* Complete partially filled request */
        if (req->recv.state.offset > 0) {
            ucp_request_complete_stream_recv(req, ep_stream, UCS_OK);
        }

        return UCS_OK;
    } else if (!ucp_stream_ep_is_queued(ep)) {
        /* Last fragment is partially processed, enqueue EP back to worker
         * in order to return it by ucp_stream_worker_poll */
        ucp_stream_ep_enqueue(ep_stream, worker);
    }

out:
    return (am_flags & UCT_CB_PARAM_FLAG_DESC) ? UCS_INPROGRESS : UCS_OK;
}

static void ucp_stream_am_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                               uint8_t id, const void *data, size_t length,
                               char *buffer, size_t max)
{
    const ucp_stream_am_hdr_t *hdr    = data;
    size_t                    hdr_len = sizeof(*hdr);
    char                      *p;

    snprintf(buffer, max, "STREAM ep uuid %"PRIx64, hdr->sender_uuid);
    p = buffer + strlen(buffer);

    ucp_dump_payload(worker->context, p, buffer + max - p, data + hdr_len,
                     length - hdr_len);
}

UCP_DEFINE_AM(UCP_FEATURE_STREAM, UCP_AM_ID_STREAM_DATA,
              ucp_stream_am_handler, ucp_stream_am_dump,
              UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_STREAM_DATA);
