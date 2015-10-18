/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_EAGER_H_
#define UCP_TAG_EAGER_H_

#include "match.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_ep.h>


/*
 * EAGER_ONLY, EAGER_MIDDLE, EAGER_LAST
 */
typedef struct {
    ucp_tag_hdr_t             super;
    /* TODO offset/sequence number */
} UCS_S_PACKED ucp_eager_hdr_t;


/*
 * EAGER_FIRST
 */
typedef struct {
    ucp_eager_hdr_t           super;
    size_t                    total_len; /* TODO make it 32bit because of rndv */
} UCS_S_PACKED ucp_eager_first_hdr_t;


ucs_status_t ucp_tag_progress_eager_contig(uct_pending_req_t *self);

ucs_status_t ucp_tag_progress_eager_generic(uct_pending_req_t *self);

ucs_status_t ucp_tag_send_eager_only_contig(ucp_ep_t *ep, ucp_tag_t tag,
                                            const void *buffer, size_t length);

static inline ucs_status_t ucp_tag_send_eager_short(ucp_ep_t *ep, ucp_tag_t tag,
                                                    const void *buffer, size_t length)
{
    UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(ucp_eager_hdr_t));
    UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uint64_t));
    return uct_ep_am_short(ep->uct_ep, UCP_AM_ID_EAGER_ONLY, tag,
                           buffer, length);
}

static inline size_t ucp_eager_hdr_len(unsigned flags)
{
    if ((flags & (UCP_RECV_DESC_FLAG_FIRST | UCP_RECV_DESC_FLAG_LAST))
                    == UCP_RECV_DESC_FLAG_FIRST) {
        return sizeof(ucp_eager_first_hdr_t);
    } else {
        return sizeof(ucp_eager_hdr_t);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_unexp_match(ucp_recv_desc_t *rdesc, ucp_tag_t tag, unsigned flags,
                      void *buffer, size_t count, ucp_datatype_t datatype,
                      size_t *offset, ucp_tag_recv_info_t *info)
{
    ucp_eager_first_hdr_t *eager_first_hdr;
    size_t recv_len, hdr_len;
    ucs_status_t status;
    void *data = rdesc + 1;

    hdr_len  = ucp_eager_hdr_len(flags);
    recv_len = rdesc->length - hdr_len;
    status   = ucp_tag_process_recv(buffer, count, datatype, *offset,
                                    data + hdr_len, recv_len);
    *offset += recv_len;

    if (flags & UCP_RECV_DESC_FLAG_FIRST) {
        info->sender_tag = tag;
        if (flags & UCP_RECV_DESC_FLAG_LAST) {
            info->length = recv_len;
        } else {
            ucs_assert(hdr_len == sizeof(*eager_first_hdr));
            eager_first_hdr = data;
            info->length = eager_first_hdr->total_len;
        }
    }

    if (flags & UCP_RECV_DESC_FLAG_LAST) {
        info->length     = *offset;
        return status;
    }

    return UCS_INPROGRESS;
}

#endif
