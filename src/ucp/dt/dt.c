/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt.h"

#include <ucp/core/ucp_request.h>

size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length)
{
    ucp_dt_generic_t *dt;
    size_t result_len = 0;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_CALL(memcpy, dest, src + state->offset, length);
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, dest, src, length,
                              &state->dt.iov.iov_offset,
                              &state->dt.iov.iovcnt_offset);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt = ucp_dt_generic(datatype);
        result_len = UCS_PROFILE_NAMED_CALL("dt_pack", dt->ops.pack,
                                            state->dt.generic.state,
                                            state->offset, dest, length);
        break;

    default:
        ucs_error("Invalid data type");
    }

    state->offset += result_len;
    return result_len;
}

ucs_status_t
ucp_dt_unpack(ucp_datatype_t datatype, void *buffer, size_t buffer_size,
              ucp_dt_state_t *state, const void *recv_data,
              size_t recv_length, unsigned flags)
{
    ucp_dt_generic_t *dt_gen;
    size_t offset = state->offset;
    ucs_status_t status = UCS_OK;

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
