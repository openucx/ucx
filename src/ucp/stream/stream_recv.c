/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.h>

#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/profile.h>

#include "../tag/eager.h"
/*
 * EAGER_ONLY, EAGER_MIDDLE, EAGER_LAST
 */
typedef struct {
    uint64_t             uuid;
} UCS_S_PACKED ucp_stream_eager_hdr_t;

/*
 * EAGER_FIRST
 */
typedef struct {
    ucp_stream_eager_hdr_t    super;
    size_t                    total_len;
} UCS_S_PACKED ucp_stream_eager_first_hdr_t;


UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_data_nb,
                 (ep, length), ucp_ep_h ep, size_t *length)
{
    ucp_recv_desc_t  *rdesc;
    ucs_status_ptr_t ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    if (ucs_list_is_empty(&ep->stream_data)) {
        ret = UCS_STATUS_PTR(UCS_OK);
        goto out;
    }

    rdesc = ucs_list_next(&ep->stream_data, ucp_recv_desc_t,
                          list[UCP_RDESC_HASH_LIST]);
    ucs_list_del(&rdesc->list[UCP_RDESC_HASH_LIST]);
    *length = rdesc->length;
    ret = (ucs_status_ptr_t)((uintptr_t)(rdesc + 1) + rdesc->hdr_len);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);

    return ret;
}

UCS_PROFILE_FUNC_VOID(ucp_stream_data_release, (ep, data),
                      ucp_ep_h ep, void *data)
{
    /* TODO: make const header offset */
    ucp_recv_desc_t *rdesc;
    rdesc = (ucp_recv_desc_t *)((uintptr_t)(data -
                                            sizeof(ucp_stream_eager_hdr_t))) - 1;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        uct_iface_release_desc((char*)rdesc - sizeof(ucp_eager_sync_hdr_t));
    } else {
        ucs_mpool_put_inline(rdesc);
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_handler(void *arg, void *data, size_t length, unsigned am_flags,
                  uint16_t flags, uint16_t hdr_len)
{
    ucp_worker_h            worker  = arg;
    ucp_ep_h                ep;
    ucp_stream_eager_hdr_t  *eager_hdr;
    ucp_recv_desc_t         *rdesc;
    size_t                  recv_len;
    khiter_t                hash_it;

    ucs_status_t status = UCS_ERR_NOT_IMPLEMENTED;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    ucs_assert(length >= hdr_len);
    recv_len = length - hdr_len;

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        /* slowpath */
        rdesc        = (ucp_recv_desc_t *)data - 1;
        rdesc->flags = flags | UCP_RECV_DESC_FLAG_UCT_DESC;
        status       = UCS_INPROGRESS;
    } else {
        rdesc = (ucp_recv_desc_t*)ucs_mpool_get_inline(&worker->am_mp);
        if (rdesc == NULL) {
            ucs_error("ucp recv descriptor is not allocated");
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        rdesc->flags = flags;
        memcpy(rdesc + 1, data, length);
        status = UCS_OK;
    }
    rdesc->length  = recv_len;
    rdesc->hdr_len = hdr_len;
    eager_hdr = data;

    hash_it = kh_get(ucp_worker_ep_hash, &worker->ep_hash, eager_hdr->uuid);
    if (hash_it != kh_end(&worker->ep_hash)) {
        ep = kh_value(&worker->ep_hash, hash_it);
        ucs_list_add_tail(&ep->stream_data, &rdesc->list[UCP_RDESC_HASH_LIST]);
    } else {
        status = UCS_ERR_SOME_CONNECTS_FAILED;
    }

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return status;
}

static ucs_status_t ucp_eager_only_handler(void *arg, void *data, size_t length,
                                           unsigned am_flags)
{
    return ucp_eager_handler(arg, data, length, am_flags,
                             UCP_RECV_DESC_FLAG_STREAM_EAGER|
                             UCP_RECV_DESC_FLAG_STREAM_FIRST|
                             UCP_RECV_DESC_FLAG_STREAM_LAST,
                             sizeof(ucp_stream_eager_hdr_t));
}

static void ucp_eager_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                           uint8_t id, const void *data, size_t length,
                           char *buffer, size_t max)
{
    /* TODO: */
}

UCP_DEFINE_AM(UCP_FEATURE_STREAM, UCP_AM_ID_STREAM_EAGER_ONLY, ucp_eager_only_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
