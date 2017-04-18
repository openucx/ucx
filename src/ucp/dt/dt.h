/**
 * Copyright (C) Mellanox Technologies Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_H_
#define UCP_DT_H_

#include "dt_contig.h"
#include "dt_iov.h"
#include "dt_stride.h"
#include "dt_generic.h"
#include "dt_reusable.h"

#include <uct/api/uct.h>
#include <ucs/debug/profile.h>
#include <string.h>


#define ucp_dt_ptr(datatype) \
    ((ucp_dt_extended_t*)(datatype & ~UCP_DATATYPE_CLASS_MASK))

/**
 * Datatype content, when requiring additional memory allocation.
 */
typedef struct ucp_dt_extended {
    ucp_dt_reusable_t reusable; /* Must be first, for reusable optimization */
    union {
        ucp_dt_stride_t stride;
        ucp_dt_generic_t generic;
    };
} ucp_dt_extended_t;

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
            uct_mem_h             contig_memh;    /* For contiguous read/write  */
        } iov;
        struct {
            size_t                item_offset;    /* Offset within a single item */
            size_t                count;          /* Count total strided objects */
            size_t                dim_index[UCP_DT_STRIDE_MAX_DIMS];
            uct_mem_h             memh;           /* Pointer to inclusive memh */
            uct_mem_h             contig_memh;    /* For contiguous read/write  */
        } stride;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_dt_state_t;

size_t ucp_dt_count_uct_iov(ucp_datatype_t datatype, size_t count,
                            const ucp_dt_iov_t *iov, const ucp_dt_state_t *state);

/**
 * Get the total length of the data
 */
static size_t ucp_dt_length_recursive(ucp_datatype_t datatype, size_t count,
                                      const ucp_dt_iov_t *iov,
                                      const ucp_dt_state_t *state, int is_extent)
{
    ucp_dt_extended_t *dt_ex;
    size_t iov_it, total;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        return ucp_contig_dt_length(datatype, count);

    case UCP_DATATYPE_IOV_R:
        dt_ex = ucp_dt_ptr(datatype);
        if (dt_ex->reusable.iov_memh != UCT_MEM_HANDLE_NULL) {
            return dt_ex->reusable.length;
        }
    case UCP_DATATYPE_IOV:
        total = 0;
        for (iov_it = 0; iov_it < count; ++iov_it) {
            total += ucp_dt_length_recursive(iov[iov_it].dt,
                    iov[iov_it].count, NULL, NULL, is_extent);
        }
        return total;

    case UCP_DATATYPE_STRIDE_R:
        dt_ex = ucp_dt_ptr(datatype);
        if (dt_ex->reusable.stride_memh != UCT_MEM_HANDLE_NULL) {
            return dt_ex->reusable.length;
        }
    case UCP_DATATYPE_STRIDE:
        dt_ex = ucp_dt_ptr(datatype);
        return count * (is_extent ? dt_ex->stride.total_extent :
                                    dt_ex->stride.total_length);

    case UCP_DATATYPE_GENERIC:
        dt_ex = ucp_dt_ptr(datatype);
        ucs_assert(NULL != state);
        ucs_assert(NULL != dt_ex);
        return dt_ex->generic.ops.packed_size(state->dt.generic.state);

    default:
        ucs_error("Invalid data type");
    }

    return 0;
}

static UCS_F_ALWAYS_INLINE
size_t ucp_dt_length(ucp_datatype_t datatype, size_t count,
                     const ucp_dt_iov_t *iov, const ucp_dt_state_t *state)
{
    return ucp_dt_length_recursive(datatype, count, iov, state, 0);
}

static UCS_F_ALWAYS_INLINE
size_t ucp_dt_extent(ucp_datatype_t datatype, size_t count,
                     const ucp_dt_iov_t *iov, const ucp_dt_state_t *state)
{
    return ucp_dt_length_recursive(datatype, count, iov, state, 1);
}

size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length);

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_dt_unpack(ucp_datatype_t datatype, void *buffer, size_t buffer_size,
              ucp_dt_state_t *state, const void *recv_data,
              size_t recv_length, int last)
{
    ucp_dt_extended_t *dt_ex;
    size_t offset = state->offset;
    ucs_status_t status;

    if (ucs_unlikely((recv_length + offset) > buffer_size)) {
        ucs_trace_req("message truncated: recv_length %zu offset %zu buffer_size %zu",
                      recv_length, offset, buffer_size);
        if (UCP_DT_IS_GENERIC(datatype) && last) {
            dt_ex = ucp_dt_ptr(datatype);
            dt_ex->generic.ops.finish(state->dt.generic.state);
        }
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, (char*)buffer + offset,
                               recv_data, recv_length);
        return UCS_OK;

    case UCP_DATATYPE_IOV_R:
    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, (ucp_dt_iov_t*)buffer,
                         state->dt.iov.iovcnt, recv_data, recv_length,
                         &state->dt.iov.iov_offset, &state->dt.iov.iovcnt_offset);
        return UCS_OK;

    case UCP_DATATYPE_STRIDE_R:
    case UCP_DATATYPE_STRIDE:
        dt_ex = ucp_dt_ptr(datatype);
        UCS_PROFILE_CALL(ucp_dt_stride_scatter, &dt_ex->stride, buffer,
                         recv_data, recv_length, &state->dt.stride.item_offset,
                         state->dt.stride.dim_index);
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        dt_ex = ucp_dt_ptr(datatype);
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_ex->generic.ops.unpack,
                                        state->dt.generic.state, offset,
                                        recv_data, recv_length);
        if (last) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_ex->generic.ops.finish,
                                        state->dt.generic.state);
        }
        return status;

    default:
        ucs_error("unexpected datatype=%lx", datatype);
        return UCS_ERR_INVALID_PARAM;
    }
}

#endif
