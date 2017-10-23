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


#define ucp_stream_rdesc_data(_rdesc)                           \
    UCS_PTR_BYTE_OFFSET((ucp_recv_desc_t *)(_rdesc) + 1,        \
                        ((ucp_recv_desc_t *)(_rdesc))->hdr_len)


#define ucp_stream_rdesc_data_offset(_rdesc, _offset)           \
    UCS_PTR_BYTE_OFFSET(ucp_stream_rdesc_data(_rdesc), _offset)


#define ucp_stream_rdesc_from_data(data)            \
    ((ucp_recv_desc_t *)UCS_PTR_BYTE_OFFSET(data,   \
                                            -sizeof(ucp_stream_am_hdr_t)) - 1)


static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_recv_data_nb_internal(ucp_ep_ext_stream_t *ep_stream, size_t *length)
{
    ucp_recv_desc_t     *rdesc;

    if (ucs_unlikely(ucs_queue_is_empty(&ep_stream->data))) {
        return UCS_STATUS_PTR(UCS_OK);
    }

    rdesc = ucs_queue_pull_elem_non_empty(&ep_stream->data, ucp_recv_desc_t,
                                          stream_queue);
    ucs_trace_data("dequeued %zu stream bytes", rdesc->length);
    *length = rdesc->length;

    return ucp_stream_rdesc_data(rdesc);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_data_nb,
                 (ep, length), ucp_ep_h ep, size_t *length)
{
    ucs_status_ptr_t ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);
    ret = ucp_stream_recv_data_nb_internal(ep->ext.stream, length);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);

    return ret;
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_rdesc_release(ucp_recv_desc_t *rdesc)
{
    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        uct_iface_release_desc((char*)rdesc - sizeof(ucp_eager_sync_hdr_t));
    } else {
        ucs_mpool_put_inline(rdesc);
    }
}

UCS_PROFILE_FUNC_VOID(ucp_stream_data_release, (ep, data),
                      ucp_ep_h ep, void *data)
{
    ucp_recv_desc_t *rdesc = ucp_stream_rdesc_from_data(data);

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    ucp_stream_rdesc_release(rdesc);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_stream_data_unpack(void *data, size_t length, ucp_request_t *req)
{
    /* Truncated error is not actual for stream, need to adjust */
    size_t       valid_len = ucs_min(req->recv.length - req->recv.state.offset,
                                     length);
    ucs_status_t status    = ucp_dt_unpack(req->recv.datatype, req->recv.buffer,
                                           req->recv.length, &req->recv.state,
                                           data, valid_len,
                                           UCP_RECV_DESC_FLAG_LAST);
    if (ucs_likely(status == UCS_OK)) {
        req->recv.state.offset += valid_len;
        ucs_trace_data("unpacked %zd\n", valid_len);

        return valid_len;
    }

    ucs_assert(status != UCS_ERR_MESSAGE_TRUNCATED);

    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_request_recv_get(ucp_worker_t *worker, void *buffer, size_t *count,
                            ucp_datatype_t datatype,
                            ucp_stream_recv_callback_t cb)
{
    ucp_request_t *req = ucp_request_get(worker);
    if (ucs_unlikely(req == NULL)) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    req->flags = UCP_REQUEST_FLAG_CALLBACK;
    req->recv.stream.cb    = cb;
    req->recv.state.offset = 0;
    req->recv.stream.count = 0;

    if (ucs_unlikely(UCP_DT_IS_IOV(datatype))) {
        req->recv.state.dt.iov.iov_offset    = 0;
        req->recv.state.dt.iov.iovcnt_offset = 0;
        req->recv.state.dt.iov.iovcnt        = *count;
        req->recv.state.dt.iov.memh          = UCT_MEM_HANDLE_NULL;
    }

    req->recv.buffer   = buffer;
    req->recv.datatype = datatype;
    req->recv.length   = ucp_dt_length(datatype, *count, buffer,
                                       &req->recv.state);

    return req;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_nb,
                 (ep, buffer, count, datatype, cb, flags),
                 ucp_ep_h ep, void *buffer, size_t *count,
                 ucp_datatype_t datatype, ucp_stream_recv_callback_t cb,
                 unsigned flags)
{
    ucp_ep_ext_stream_t *ep_stream;
    ucp_request_t       *req;
    ucs_status_ptr_t    rdata;
    size_t              offset;
    size_t              length;
    ssize_t             unpacked;

    if (UCP_DT_IS_GENERIC(datatype)) {
        ucs_error("ucp_stream_recv_nb doesn't support generic datatype");
        return UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
    }

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    req = ucp_stream_request_recv_get(ep->worker, buffer, count, datatype, cb);
    if (ucs_unlikely(!UCS_PTR_IS_PTR(req))) {
        goto out;
    }

    ep_stream = ep->ext.stream;

    /* if there are already posted requests, need to queue this one */
    if (!ucs_queue_is_empty(&ep_stream->reqs)) {
        ucs_queue_push(&ep_stream->reqs, &req->recv.queue);
        goto out;
    }

    /* OK, lets obtain all arrived data which matches the recv size */
    while (req->recv.state.offset < req->recv.length) {
        /* Firstly, check if there is not completely processed rdesc */
        if (ep_stream->rdesc) {
            rdata  = ucp_stream_rdesc_data(ep_stream->rdesc);
            offset = ep_stream->rdesc_offset;
            length = ep_stream->rdesc_len;
        } else {
            /* Then dequeue new ones */
            rdata  = ucp_stream_recv_data_nb_internal(ep_stream, &length);
            offset = 0;
            ucs_assert(!UCS_PTR_IS_ERR(rdata));
        }

        if (UCS_PTR_IS_PTR(rdata)) {
            unpacked = ucp_stream_data_unpack(UCS_PTR_BYTE_OFFSET(rdata, offset),
                                              length - offset, req);
            if (ucs_unlikely(unpacked < 0)) {
                ucp_request_complete_stream_recv(req, unpacked);
                goto out;
            } else if (unpacked == (length - offset)) {
                if (ucp_stream_rdesc_from_data(rdata) == ep_stream->rdesc) {
                    ep_stream->rdesc = NULL;
                }
                ucp_stream_rdesc_release(ucp_stream_rdesc_from_data(rdata));
            } else {
                if (ep_stream->rdesc == NULL) {
                    ep_stream->rdesc        = ucp_stream_rdesc_from_data(rdata);
                    ep_stream->rdesc_offset = unpacked;
                    ep_stream->rdesc_len    = length;
                } else {
                    ucs_assert(ep_stream->rdesc ==
                               ucp_stream_rdesc_from_data(rdata));
                    ep_stream->rdesc_offset += unpacked;
                }
                ucs_assert(ep_stream->rdesc_offset <= ep_stream->rdesc_len);
                ucs_assert(req->recv.state.offset <= req->recv.length);
            }
        } else {
            /* No data any more */
            ucs_assert(UCS_PTR_STATUS(rdata) == UCS_OK);

            /* Enqueue empty request or complete if some data is filled */
            if (req->recv.state.offset == 0) {
                ucs_queue_push(&ep_stream->reqs, &req->recv.queue);
            } else {
                /* TODO: count != bytes_count */
                ucp_request_complete_stream_recv(req, UCS_OK);
            }
            goto out;
        }
    }

    ucs_assert(req->recv.state.offset == req->recv.length);
    ucp_request_complete_stream_recv(req, UCS_OK);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return UCS_PTR_IS_PTR(req) ? (req + 1) : req;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_ep_rdesc_set(ucp_ep_ext_stream_t *ep_stream, ucp_recv_desc_t *rdesc,
                        size_t recv_len, int is_handled)
{
    ucs_status_ptr_t rdata;
    size_t           len;

    if (ep_stream->rdesc) {
        return UCS_INPROGRESS;
    }

    rdata = ucp_stream_recv_data_nb_internal(ep_stream, &len);
    if (UCS_PTR_IS_PTR(rdata)) {
        /* Firstly, process all arrived data before */
        ep_stream->rdesc        = ucp_stream_rdesc_from_data(rdata);
        ep_stream->rdesc_len    = len;
        ep_stream->rdesc_offset = 0;
        return UCS_INPROGRESS;
    } else if (ucs_unlikely(UCS_PTR_IS_ERR(rdata))){
        return UCS_PTR_STATUS(rdata);
    } else if (!is_handled) {
        ucs_assert(UCS_PTR_STATUS(rdata) == UCS_OK);
        /* No data arrived before, process currently arrived one */
        ep_stream->rdesc        = rdesc;
        ep_stream->rdesc_len    = recv_len;
        ep_stream->rdesc_offset = 0;
        return UCS_OK;
    } else if (ep_stream->rdesc) {
        /* This is current not completely handled descriptor */
        ucs_assert((ep_stream->rdesc == rdesc) &&
                   (ep_stream->rdesc_offset < ep_stream->rdesc_len));
        return UCS_OK;
    } else {
        return UCS_ERR_NO_MESSAGE;
    }
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_stream_am_rdesc_get(ucp_worker_t *worker, void *data, size_t length,
                        unsigned am_flags)
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
            memcpy(rdesc + 1, data, length);
        } else {
            ucs_error("ucp recv descriptor is not allocated");
            return NULL;
        }
    }

    rdesc->length  = length - sizeof(ucp_stream_am_hdr_t);
    rdesc->hdr_len = sizeof(ucp_stream_am_hdr_t);

    return rdesc;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_am_get_ep(ucp_worker_t *worker, uint64_t sender_uuid,
                     ucp_recv_desc_t *rdesc)
{
    ucp_ep_ext_stream_t *ep_stream;
    ucp_ep_h            ep;
    khiter_t            hash_it;

    hash_it = kh_get(ucp_worker_ep_hash, &worker->ep_hash, sender_uuid);
    if (ucs_unlikely(hash_it == kh_end(&worker->ep_hash))) {
        ucs_error("ep is not found by uuid: %lu", sender_uuid);
        return UCS_STATUS_PTR(UCS_ERR_SOME_CONNECTS_FAILED);
    }

    ep        = kh_value(&worker->ep_hash, hash_it);
    ep_stream = ep->ext.stream;

    if (ep->flags & UCP_EP_FLAG_STREAM_IS_QUEUED) {
        ucs_queue_push(&ep_stream->data, &rdesc->stream_queue);
        return UCS_STATUS_PTR(UCS_ERR_NO_PROGRESS);
    } else if (ucs_queue_is_empty(&ep_stream->reqs)) {
        ucs_assert(ucs_queue_is_empty(&ep_stream->reqs));
        ucs_list_add_tail(&worker->stream_eps, &ep_stream->list);
        ep->flags |= UCP_EP_FLAG_STREAM_IS_QUEUED;
        ucs_queue_push(&ep_stream->data, &rdesc->stream_queue);
        return UCS_STATUS_PTR(UCS_ERR_NO_PROGRESS);
    }

    return ep_stream;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_am_handler(void *am_arg, void *am_data, size_t am_length,
                      unsigned am_flags)
{
    ucp_worker_h        worker      = am_arg;
    ucp_stream_am_hdr_t *hdr        = am_data;
    size_t              recv_len    = am_length - sizeof(ucp_stream_am_hdr_t);
    ucp_ep_ext_stream_t *ep_stream  = NULL;
    ucp_recv_desc_t     *rdesc;
    ucp_request_t       *req;
    void                *rdata_ptr;
    size_t              rdata_len;
    ssize_t             unpacked;
    ucs_status_t        status;
    int                 is_handled;

    ucs_assert(am_length >= sizeof(ucp_stream_am_hdr_t));

    rdesc = ucp_stream_am_rdesc_get(worker, am_data, am_length, am_flags);
    if (ucs_unlikely(rdesc == NULL)) {
        goto out;
    }

    ep_stream = ucp_stream_am_get_ep(worker, hdr->sender_uuid, rdesc);
    if (!UCS_PTR_IS_PTR(ep_stream)) {
        goto out;
    }

    is_handled = 0;

    do {
        req = ucs_queue_pull_elem_non_empty(&ep_stream->reqs, ucp_request_t,
                                            recv.queue);
        do {
            status = ucp_stream_ep_rdesc_set(ep_stream, rdesc, recv_len,
                                             is_handled);
            if (status == UCS_OK) {
                is_handled = 1;
            } else if (status == UCS_ERR_NO_MESSAGE) {
                ucp_request_complete_stream_recv(req, UCS_OK);
                goto out;
            }

            rdata_ptr = ucp_stream_rdesc_data_offset(ep_stream->rdesc,
                                                     ep_stream->rdesc_offset);
            rdata_len = ep_stream->rdesc_len - ep_stream->rdesc_offset;
            unpacked  = ucp_stream_data_unpack(rdata_ptr, rdata_len, req);

            if (ucs_unlikely(unpacked < 0)) {
                goto out;
            } else if (unpacked < rdata_len) {
                ep_stream->rdesc_offset += unpacked;
                ucs_assert(ep_stream->rdesc_offset <= ep_stream->rdesc_len);
                /* This request is full, try next one */
                ucp_request_complete_stream_recv(req, UCS_OK);
                break;
            } else {
                /* The descriptor is completely processed, go to next */
                ucs_assert((ep_stream->rdesc_len ==
                           (ep_stream->rdesc_offset += unpacked)));
                if (ep_stream->rdesc != rdesc) {
                    ucp_stream_rdesc_release(ep_stream->rdesc);
                }
                ep_stream->rdesc = NULL;
            }
        } while (ep_stream->rdesc == NULL);
    } while (!ucs_queue_is_empty(&ep_stream->reqs));

out:
    if (UCS_PTR_IS_PTR(ep_stream) && (ep_stream->rdesc == NULL)) {
        if (!(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
            ucs_mpool_put_inline(rdesc);
        }
        return UCS_OK;
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

    snprintf(buffer, max, "STREAM ep uuid %"PRIx64, hdr->sender_uuid);
    p = buffer + strlen(buffer);

    ucp_dump_payload(worker->context, p, buffer + max - p, data + hdr_len,
                     length - hdr_len);
}

UCP_DEFINE_AM(UCP_FEATURE_STREAM, UCP_AM_ID_STREAM_DATA,
              ucp_stream_am_handler, ucp_stream_am_dump,
              UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_STREAM_DATA);
