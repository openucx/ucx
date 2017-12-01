/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_STREAM_H_
#define UCP_STREAM_H_

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>

typedef struct {
    uint64_t    sender_uuid;
} UCS_S_PACKED ucp_stream_am_hdr_t;


typedef struct {
    union {
        ucp_stream_am_hdr_t hdr;
        size_t              offset;
    };
    uint8_t                 payload[0];
} UCS_S_PACKED ucp_stream_am_data_t;


static UCS_F_ALWAYS_INLINE void
ucp_stream_ep_enqueue(ucp_ep_ext_stream_t *ep, ucp_worker_h worker)
{
    ucs_assert(!(ep->ucp_ep->flags & UCP_EP_FLAG_STREAM_IS_QUEUED));
    ucs_list_add_tail(&worker->stream_eps, &ep->list);
    ep->ucp_ep->flags |= UCP_EP_FLAG_STREAM_IS_QUEUED;
}

static UCS_F_ALWAYS_INLINE int
ucp_stream_ep_is_queued(ucp_ep_h ep)
{
    return ep->flags & UCP_EP_FLAG_STREAM_IS_QUEUED;
}

static UCS_F_ALWAYS_INLINE void
ucp_stream_ep_dequeue(ucp_ep_ext_stream_t *ep)
{
    ucs_assert(ep->ucp_ep->flags & UCP_EP_FLAG_STREAM_IS_QUEUED);
    ep->ucp_ep->flags &= ~UCP_EP_FLAG_STREAM_IS_QUEUED;
    ucs_list_del(&ep->list);
}

static UCS_F_ALWAYS_INLINE ucp_ep_ext_stream_t *
ucp_stream_worker_dequeue_ep_head(ucp_worker_h worker)
{
    ucp_ep_ext_stream_t *ep = ucs_list_head(&worker->stream_eps,
                                            ucp_ep_ext_stream_t, list);
    ucp_stream_ep_dequeue(ep);
    return ep;
}

#endif /* UCP_STREAM_H_ */
