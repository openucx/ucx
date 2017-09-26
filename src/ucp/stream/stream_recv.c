/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.h>

#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/profile.h>

#include <ucp/stream/stream.h>
#include <ucp/tag/eager.h> /* TODO: remove ucp_eager_sync_hdr_t usage */


#define ucp_stream_rdesc_data(rdesc) \
    (ucs_status_ptr_t)((uintptr_t)((rdesc) + 1) + (rdesc)->hdr_len)

#define ucp_stream_rdesc_from_data(data) \
    ((ucp_recv_desc_t *)((uintptr_t)(data) - sizeof(ucp_stream_am_hdr_t)) - 1)

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_data_nb,
                 (ep, length), ucp_ep_h ep, size_t *length)
{
    ucp_recv_desc_t  *rdesc;
    ucs_status_ptr_t ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    if (ucs_queue_is_empty(&ep->stream_data)) {
        ret = UCS_STATUS_PTR(UCS_OK);
        goto out;
    }

    rdesc = ucs_queue_pull_elem_non_empty(&ep->stream_data, ucp_recv_desc_t,
                                          stream_queue);
    *length = rdesc->length;
    ret = ucp_stream_rdesc_data(rdesc);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);

    return ret;
}

UCS_PROFILE_FUNC_VOID(ucp_stream_data_release, (ep, data),
                      ucp_ep_h ep, void *data)
{
    ucp_recv_desc_t *rdesc = ucp_stream_rdesc_from_data(data);

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        uct_iface_release_desc((char*)rdesc - sizeof(ucp_eager_sync_hdr_t));
    } else {
        ucs_mpool_put_inline(rdesc);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
}

static ucs_status_t ucp_stream_am_handler(void *arg, void *data, size_t length,
                                          unsigned am_flags)
{
    const size_t            hdr_len = sizeof(ucp_stream_am_hdr_t);
    ucp_worker_h            worker  = arg;
    ucp_ep_h                ep;
    ucp_stream_am_hdr_t     *hdr;
    ucp_recv_desc_t         *rdesc;
    size_t                  recv_len;
    khiter_t                hash_it;
    ucs_status_t            status;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    ucs_assert(length >= hdr_len);
    recv_len = length - hdr_len;

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        /* slowpath */
        rdesc        = (ucp_recv_desc_t *)data - 1;
        rdesc->flags = UCP_RECV_DESC_FLAG_UCT_DESC;
        status       = UCS_INPROGRESS;
    } else {
        status = UCS_OK;
        rdesc = (ucp_recv_desc_t*)ucs_mpool_get_inline(&worker->am_mp);
        if (ucs_likely(rdesc != NULL)) {
            rdesc->flags = 0;
            memcpy(rdesc + 1, data, length);
        } else {
            ucs_error("ucp recv descriptor is not allocated");
            goto out;
        }
    }

    rdesc->length  = recv_len;
    rdesc->hdr_len = hdr_len;
    hdr            = data;

    hash_it = kh_get(ucp_worker_ep_hash, &worker->ep_hash, hdr->sender_uuid);
    if (ucs_likely(hash_it != kh_end(&worker->ep_hash))) {
        ep = kh_value(&worker->ep_hash, hash_it);
        ucs_queue_push(&ep->stream_data, &rdesc->stream_queue);
    } else {
        ucs_error("ep is not found by uuid: %lu", hdr->sender_uuid);
        status = UCS_OK;
    }

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return status;
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
