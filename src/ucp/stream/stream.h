/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_STREAM_H_
#define UCP_STREAM_H_

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>


typedef struct {
    uint64_t                 ep_id;
} UCS_S_PACKED ucp_stream_am_hdr_t;


typedef struct {
    union {
        ucp_stream_am_hdr_t  hdr;
        ucp_recv_desc_t     *rdesc;
    };
} ucp_stream_am_data_t;


void ucp_stream_ep_init(ucp_ep_h ep);

void ucp_stream_ep_cleanup(ucp_ep_h ep, ucs_status_t status);

void ucp_stream_ep_activate(ucp_ep_h ep);


static UCS_F_ALWAYS_INLINE int ucp_stream_ep_is_queued(ucp_ep_ext_t *ep_ext)
{
    return ep_ext->stream.ready_list.next != NULL;
}

static UCS_F_ALWAYS_INLINE int ucp_stream_ep_has_data(ucp_ep_ext_t *ep_ext)
{
    return ep_ext->ep->flags & UCP_EP_FLAG_STREAM_HAS_DATA;
}

static UCS_F_ALWAYS_INLINE
void ucp_stream_ep_enqueue(ucp_ep_ext_t *ep_ext, ucp_worker_h worker)
{
    ucs_assert(!ucp_stream_ep_is_queued(ep_ext));
    ucs_list_add_tail(&worker->stream_ready_eps, &ep_ext->stream.ready_list);
}

static UCS_F_ALWAYS_INLINE void ucp_stream_ep_dequeue(ucp_ep_ext_t *ep_ext)
{
    ucs_list_del(&ep_ext->stream.ready_list);
    ep_ext->stream.ready_list.next = NULL;
}

static UCS_F_ALWAYS_INLINE ucp_ep_ext_t *
ucp_stream_worker_dequeue_ep_head(ucp_worker_h worker)
{
    ucp_ep_ext_t *ep_ext = ucs_list_head(&worker->stream_ready_eps,
                                         ucp_ep_ext_t, stream.ready_list);

    ucs_assert(ep_ext->stream.ready_list.next != NULL);
    ucp_stream_ep_dequeue(ep_ext);
    return ep_ext;
}

#endif /* UCP_STREAM_H_ */
