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


static inline void ucp_tag_log_match(ucp_tag_t recv_tag, ucp_request_t *req,
                                     ucp_tag_t exp_tag, ucp_tag_t exp_tag_mask,
                                     size_t offset, const char *title)
{
    ucs_trace_req("matched tag %"PRIx64" to %s request %p offset %zu "
                  "with tag %"PRIx64"/%"PRIx64, recv_tag, title, req,
                  offset, exp_tag, exp_tag_mask);
}


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_process_recv(void *buffer, size_t count, ucp_datatype_t datatype,
                     ucp_frag_state_t *state, void *recv_data, size_t recv_length,
                     int last)
{
    ucp_dt_generic_t *dt_gen;
    size_t offset = state->offset;
    size_t buffer_size;
    ucs_status_t status;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        buffer_size = ucp_contig_dt_length(datatype, count);
        if (ucs_unlikely(recv_length + offset > buffer_size)) {
            return UCS_ERR_MESSAGE_TRUNCATED;
        }
        memcpy(buffer + offset, recv_data, recv_length);
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);

        buffer_size = dt_gen->ops.packed_size(state->dt.generic.state);
        if (ucs_unlikely(recv_length + offset > buffer_size)) {
            status = UCS_ERR_MESSAGE_TRUNCATED;
        } else {
            dt_gen->ops.unpack(state->dt.generic.state, offset, recv_data, recv_length);
            status = UCS_OK;
        }

        if (last) {
            dt_gen->ops.finish(state->dt.generic.state);
        }
        return status;

    default:
        ucs_bug("unexpected datatype");
        return UCS_OK;
    }
}

#endif
