/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/debug/profile.h>

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

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_dt_unpack(ucp_datatype_t datatype, void *buffer, size_t buffer_size,
              ucp_dt_state_t *state, const void *recv_data,
              size_t recv_length, unsigned flags)
{
    ucp_dt_generic_t *dt_gen;
    size_t           offset = state->offset;
    ucs_status_t     status = UCS_OK;

    if (ucs_unlikely((recv_length + offset) > buffer_size)) {
        ucs_debug("message truncated: recv_length %zu offset %zu buffer_size %zu",
                  recv_length, offset, buffer_size);
        if (UCP_DT_IS_GENERIC(datatype) && (flags & UCP_RECV_DESC_FLAG_LAST)) {
            ucp_dt_generic(datatype)->ops.finish(state->dt.generic.state);
        }

        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, buffer + offset,
                               recv_data, recv_length);
        return status;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, buffer, state->dt.iov.iovcnt,
                         recv_data, recv_length, &state->dt.iov.iov_offset,
                         &state->dt.iov.iovcnt_offset);
        return status;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                        state->dt.generic.state, offset,
                                        recv_data, recv_length);
        if (flags & UCP_RECV_DESC_FLAG_LAST) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                        state->dt.generic.state);
        }
        return status;

    default:
        ucs_error("unexpected datatype=%lx", datatype);
        return UCS_ERR_INVALID_PARAM;
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_dt_recv_state_init(ucp_dt_state_t *dt_state, void *buffer,
                       ucp_datatype_t dt, size_t dt_count)
{
    ucp_dt_generic_t *dt_gen;

    dt_state->offset = 0;

    switch (dt & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        dt_state->dt.contig.md_map     = 0;
        break;
   case UCP_DATATYPE_IOV:
        dt_state->dt.iov.iov_offset    = 0;
        dt_state->dt.iov.iovcnt_offset = 0;
        dt_state->dt.iov.iovcnt        = dt_count;
        dt_state->dt.iov.dt_reg        = NULL;
        break;
    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(dt);
        dt_state->dt.generic.state = 
            UCS_PROFILE_NAMED_CALL("dt_start", dt_gen->ops.start_unpack,
                                   dt_gen->context, buffer, dt_count);
        ucs_trace("dt state %p buffer %p count %zu dt_gen state=%p", dt_state,
                  buffer, dt_count, dt_state->dt.generic.state);
        break;
    default:
        break;
    }
}
