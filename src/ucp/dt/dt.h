/**
 * Copyright (C) Mellanox Technologies Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_H_
#define UCP_DT_H_

#include "dt_contig.h"
#include "dt_iov.h"
#include "dt_generic.h"

#include <uct/api/uct.h>
#include <ucs/debug/profile.h>
#include <string.h>


/**
 * State of progressing sent/receive operation on a datatype.
 */
typedef struct ucp_dt_state {
    size_t                        offset;  /* Total offset in overall payload. */
    union {
        struct {
            uct_mem_h             memh;
        } contig;
        struct {
            size_t                iov_offset;     /* Offset in the IOV item */
            size_t                iovcnt_offset;  /* The IOV item to start copy */
            size_t                iovcnt;         /* Number of IOV buffers */
            uct_mem_h             *memh;          /* Pointer to IOV memh[iovcnt] */
        } iov;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_dt_state_t;


/**
 * Get the total length of the data
 */
static UCS_F_ALWAYS_INLINE
size_t ucp_dt_length(ucp_datatype_t datatype, size_t count,
                     const ucp_dt_iov_t *iov, const ucp_dt_state_t *state)
{
    ucp_dt_generic_t *dt_gen;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        return ucp_contig_dt_length(datatype, count);

    case UCP_DATATYPE_IOV:
        ucs_assert(NULL != iov);
        return ucp_dt_iov_length(iov, count);

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        ucs_assert(NULL != state);
        ucs_assert(NULL != dt_gen);
        return dt_gen->ops.packed_size(state->dt.generic.state);

    default:
        ucs_error("Invalid data type");
    }

    return 0;
}

size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length);

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_dt_unpack(ucp_datatype_t datatype, void *buffer, size_t buffer_size,
              ucp_dt_state_t *state, const void *recv_data,
              size_t recv_length, int last)
{
    ucp_dt_generic_t *dt_gen;
    size_t offset = state->offset;
    ucs_status_t status;

    if (ucs_unlikely((recv_length + offset) > buffer_size)) {
        ucs_trace_req("message truncated: recv_length %zu offset %zu buffer_size %zu",
                      recv_length, offset, buffer_size);
        if (UCP_DT_IS_GENERIC(datatype) && last) {
            ucp_dt_generic(datatype)->ops.finish(state->dt.generic.state);
        }
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, buffer + offset,
                               recv_data, recv_length);
        return UCS_OK;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, (ucp_dt_iov_t*)buffer, state->dt.iov.iovcnt,
                         recv_data, recv_length, &state->dt.iov.iov_offset,
                         &state->dt.iov.iovcnt_offset);
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                        state->dt.generic.state, offset,
                                        recv_data, recv_length);
        if (last) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                        state->dt.generic.state);
        }
        return status;

    default:
        ucs_error("unexpected datatype=%lx", datatype);
        return UCS_ERR_INVALID_PARAM;
    }
}

#endif
