/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_MATCH_H_
#define UCP_TAG_MATCH_H_

#include <ucp/core/ucp_request.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/dt/dt_generic.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>

#include <string.h>
#include <inttypes.h>


/**
 * Tag-match header
 */
typedef struct {
    ucp_tag_t                 tag;
} UCS_S_PACKED ucp_tag_hdr_t;


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


static inline void ucp_tag_log_match(ucp_tag_t recv_tag, ucp_request_t *req,
                                     ucp_tag_t exp_tag, ucp_tag_t exp_tag_mask)
{
    ucs_trace_req("matched tag %"PRIx64" to request %p offset %zu "
                  "with tag %"PRIx64"/%"PRIx64,
                  recv_tag, req, req->recv.state.offset, exp_tag, exp_tag_mask);
}


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_process_recv(void *buffer, size_t count, ucp_datatype_t datatype,
                     size_t offset, void *recv_data, size_t recv_length)
{
    ucp_dt_generic_t *dt_gen;
    void *state;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (ucs_unlikely(recv_length + offset > ucp_contig_dt_length(datatype, count))) {
            ucs_bug("truncated");
            return UCS_ERR_MESSAGE_TRUNCATED;
        }
        memcpy(buffer + offset, recv_data, recv_length);
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        /* TODO allocate state before
         */
        dt_gen = ucp_dt_generic(datatype);
        state  = dt_gen->ops->start_unpack(dt_gen->context, buffer, count);
        dt_gen->ops->unpack(state, offset, recv_data, recv_length);
        dt_gen->ops->finish(state);
        return UCS_OK;

    default:
        ucs_bug("unexpected datatype");
        return UCS_OK;
    }
}

#endif
