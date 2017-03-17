/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_MATCH_H_
#define UCP_TAG_MATCH_H_

#include <ucp/core/ucp_request.h>
#include <ucp/dt/dt.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>

#include <inttypes.h>


#define ucp_tag_log_match(_recv_tag, _recv_len,_req, _exp_tag, _exp_tag_mask, \
                          _offset, _title) \
    ucs_trace_req("matched tag %"PRIx64" len %zu to %s request %p offset %zu " \
                  "with tag %"PRIx64"/%"PRIx64, (_recv_tag), (size_t)(_recv_len), \
                  (_title), (_req), (size_t)(_offset), (_exp_tag), (_exp_tag_mask))


/**
 * Tag-match header
 */
typedef struct {
    ucp_tag_t                 tag;
} UCS_S_PACKED ucp_tag_hdr_t;


void ucp_tag_cancel_expected(ucp_context_h context, ucp_request_t *req);


static UCS_F_ALWAYS_INLINE
int ucp_tag_is_match(ucp_tag_t tag, ucp_tag_t exp_tag, ucp_tag_t tag_mask)
{
    /* The bits in which expected and actual tag differ, should not fall
     * inside the mask.
     */
    return ((tag ^ exp_tag) & tag_mask) == 0;
}


static UCS_F_ALWAYS_INLINE
int ucp_tag_recv_is_match(ucp_tag_t recv_tag, unsigned recv_flags,
                          ucp_tag_t exp_tag, ucp_tag_t tag_mask,
                          size_t offset, ucp_tag_t curr_tag)
{
    /*
     * For first fragment, we search a matching request
     * For subsequent fragments, we search for a request with exact same tag,
     * which would also mean it arrives from the same sender.
     */
    return (((offset == 0) && (recv_flags & UCP_RECV_DESC_FLAG_FIRST) &&
              ucp_tag_is_match(recv_tag, exp_tag, tag_mask)) ||
            (!(offset == 0) && !(recv_flags & UCP_RECV_DESC_FLAG_FIRST) &&
              (recv_tag == curr_tag)));
}


#endif
