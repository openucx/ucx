/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DT_INL_
#define UCP_DT_INL_

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
ucp_dt_unpack_only(ucp_worker_h worker, void *buffer, size_t count,
                   ucp_datatype_t datatype, uct_memory_type_t mem_type,
                   const void *data, size_t length, int truncation)
{
    size_t iov_offset, iovcnt_offset;
    ucp_dt_generic_t *dt_gen;
    ucs_status_t status;
    size_t buffer_size;
    void *state;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (truncation &&
            ucs_unlikely(length > (buffer_size = ucp_contig_dt_length(datatype, count)))) {
            goto err_truncated;
        }
        if (ucs_likely(UCP_MEM_IS_HOST(mem_type))) {
            UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, buffer, data, length);
        } else {
            ucp_mem_type_unpack(worker, buffer, data, length, mem_type);
        }
        return UCS_OK;

    case UCP_DATATYPE_IOV:
        if (truncation &&
            ucs_unlikely(length > (buffer_size = ucp_dt_iov_length(buffer, count)))) {
            goto err_truncated;
        }
        iov_offset = iovcnt_offset = 0;
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, buffer, count, data, length,
                         &iov_offset, &iovcnt_offset);
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        state  = UCS_PROFILE_NAMED_CALL("dt_start", dt_gen->ops.start_unpack,
                                        dt_gen->context, buffer, count);
        if (truncation &&
            ucs_unlikely(length > (buffer_size = dt_gen->ops.packed_size(state)))) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish, state);
            goto err_truncated;
        }
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack, state,
                                        0, data, length);
        UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish, state);
        return status;

    default:
        ucs_fatal("unexpected datatype=%lx", datatype);
    }

err_truncated:
    ucs_debug("message truncated: recv_length %zu buffer_size %zu", length,
              buffer_size);
    return UCS_ERR_MESSAGE_TRUNCATED;
}

static UCS_F_ALWAYS_INLINE void
ucp_dt_recv_state_init(ucp_dt_state_t *dt_state, void *buffer,
                       ucp_datatype_t dt, size_t dt_count)
{
    ucp_dt_generic_t *dt_gen;

    switch (dt & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        dt_state->dt.contig.md_map     = 0;
        break;
   case UCP_DATATYPE_IOV:
       /* on receive side, only IOV uses offset field, to allow seeking
        * to different position.
        * TODO remove offset from dt_state, move it inside iov and send state.
        */
        dt_state->offset               = 0;
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

#endif
