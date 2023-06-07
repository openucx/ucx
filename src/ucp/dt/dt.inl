/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DT_INL_
#define UCP_DT_INL_

#include "dt.h"
#include "dt_contig.h"
#include "dt_generic.h"
#include "dt_iov.h"

#include <ucp/core/ucp_mm.h>
#include <ucs/profile/profile.h>
#include <inttypes.h>


/**
 * Get the class of a given datatype
 *
 * @param [in]   datatype Handle to a datatype
 *
 * @return datatype class of a datatype handle
 */
static UCS_F_ALWAYS_INLINE ucp_dt_class_t
ucp_datatype_class(ucp_datatype_t datatype)
{
    return (ucp_dt_class_t)(datatype & UCP_DATATYPE_CLASS_MASK);
}

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
        dt_gen = ucp_dt_to_generic(datatype);
        ucs_assert(NULL != state);
        ucs_assert(NULL != dt_gen);
        return dt_gen->ops.packed_size(state->dt.generic.state);

    default:
        ucs_error("Invalid data type");
    }

    return 0;
}

static UCS_F_ALWAYS_INLINE void
ucp_dt_recv_state_init(ucp_dt_state_t *dt_state, void *buffer,
                       ucp_datatype_t dt, size_t dt_count)
{
    ucp_dt_generic_t *dt_gen;

    switch (dt & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        dt_state->dt.contig.memh       = NULL;
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
        dt_state->dt.iov.memhs         = NULL;
        break;
    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_to_generic(dt);
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
