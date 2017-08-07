/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>

#include <ucp/dt/dt.h>



static inline ucs_status_t
ucp_stream_send_eager_short(ucp_ep_t *ep, const void *buffer, size_t length)
{
    UCS_STATIC_ASSERT(sizeof(ep->dest_uuid) == sizeof(uint64_t));

    return uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_STREAM_EAGER_ONLY,
                           ep->worker->uuid, buffer, length);
}


UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_send_nb,
                 (ep, buffer, count, datatype, cb, flags),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_send_callback_t cb, unsigned flags)
{
    ucs_status_t status;
//    ucp_request_t *req;
    size_t length;
    ucs_status_ptr_t ret = UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

//    ucs_trace_req("send_nb buffer %p count %zu to %s cb %p flags %u",
//                  buffer, count, ucp_ep_peer_name(ep), cb, flags);

    if (ucs_unlikely(flags != 0)) {
        goto out;
    }

    if (ucs_likely(UCP_DT_IS_CONTIG(datatype))) {
        length = ucp_contig_dt_length(datatype, count);
        if (ucs_likely((ssize_t)length <= ucp_ep_config(ep)->am.max_short)) {
            status = UCS_PROFILE_CALL(ucp_stream_send_eager_short, ep, buffer,
                                      length);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                UCP_EP_STAT_TAG_OP(ep, EAGER);
                ret = UCS_STATUS_PTR(status); /* UCS_OK also goes here */
                goto out;
            }
        }
    }

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return ret;
}
