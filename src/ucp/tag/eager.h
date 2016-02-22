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
#include <ucp/proto/proto.h>


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
    size_t                    total_len;
} UCS_S_PACKED ucp_eager_first_hdr_t;


extern const ucp_proto_t ucp_tag_eager_proto;


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

static UCS_F_ALWAYS_INLINE size_t
ucp_eager_total_len(ucp_eager_hdr_t *hdr, unsigned flags, unsigned payload_length)
{
    ucs_assert(flags & UCP_RECV_DESC_FLAG_FIRST);
    if (flags & UCP_RECV_DESC_FLAG_LAST) {
        return payload_length;
    } else {
        return ucs_container_of(hdr, ucp_eager_first_hdr_t, super)->total_len;
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_unexp_match(ucp_recv_desc_t *rdesc, ucp_tag_t tag, unsigned flags,
                      void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_frag_state_t *state, ucp_tag_recv_info_t *info)
{
    size_t recv_len, hdr_len;
    ucs_status_t status;
    void *data = rdesc + 1;

    hdr_len  = ucp_eager_hdr_len(flags);
    recv_len = rdesc->length - hdr_len;
    status   = ucp_tag_process_recv(buffer, count, datatype, state,
                                    data + hdr_len, recv_len,
                                    flags & UCP_RECV_DESC_FLAG_LAST);
    state->offset += recv_len;

    if (flags & UCP_RECV_DESC_FLAG_FIRST) {
        info->sender_tag = tag;
        info->length     = ucp_eager_total_len(data, flags, recv_len);
    }

    if (flags & UCP_RECV_DESC_FLAG_LAST) {
        info->length     = state->offset;
        return status;
    }

    return UCS_INPROGRESS;
}

#endif
