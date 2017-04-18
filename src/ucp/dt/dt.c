/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt.h"
#include <ucp/api/ucp.h>
#include <ucs/debug/memtrack.h>


size_t ucp_dt_count_uct_iov(ucp_datatype_t datatype, size_t count,
                            const ucp_dt_iov_t *iov, const ucp_dt_state_t *state)
{
    ucp_dt_extended_t *dt_ex;
    size_t iov_it, total;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        return 1;

    case UCP_DATATYPE_IOV_R:
    case UCP_DATATYPE_IOV:
        total = count;
        for (iov_it = 0; iov_it < count; iov_it++) {
            total += ucp_dt_count_uct_iov(iov[iov_it].dt, iov[iov_it].count,
                    iov[iov_it].buffer, NULL);
        }
        return total;

    case UCP_DATATYPE_STRIDE_R:
    case UCP_DATATYPE_STRIDE:
        dt_ex = ucp_dt_ptr(datatype);
        return dt_ex->stride.dim_cnt *
                ucp_dt_count_uct_iov(dt_ex->stride.dt, 1, NULL, NULL);

    default:
        ucs_error("Invalid data type");
    }

    return 0;
}

size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length)
{
    size_t result_len = 0;
    ucp_dt_extended_t *dt_ex;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_CALL(memcpy, dest, src + state->offset, length);
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
    case UCP_DATATYPE_IOV_R:
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, dest, src, length,
                              &state->dt.iov.iov_offset,
                              &state->dt.iov.iovcnt_offset);
        result_len = length;
        break;

    case UCP_DATATYPE_STRIDE:
    case UCP_DATATYPE_STRIDE_R:
        dt_ex = ucp_dt_ptr(datatype);
        UCS_PROFILE_CALL_VOID(ucp_dt_stride_gather, dest, src, length,
                              &dt_ex->stride, &state->dt.stride.item_offset,
                              state->dt.stride.dim_index);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt_ex = ucp_dt_ptr(datatype);
        result_len = UCS_PROFILE_NAMED_CALL("dt_pack", dt_ex->generic.ops.pack,
                                            state->dt.generic.state,
                                            state->offset, dest, length);
        break;

    default:
        ucs_error("Invalid data type");
    }

    state->offset += result_len;
    return result_len;
}

ucp_datatype_t ucp_dt_create(enum ucp_dt_type type, ...)
{
    ucp_dt_extended_t *dt;

    if (type == UCP_DATATYPE_CONTIG) {
        /* Contiguous datatype "pointer" contains length instead */
        return (ucp_datatype_t)type;
    }

    dt = ucs_memalign(UCS_BIT(UCP_DATATYPE_SHIFT), sizeof(*dt), "datatype");
    if (dt == NULL) {
        return 0;
    }

    memset(dt, 0, sizeof(*dt));

    if ((type == UCP_DATATYPE_STRIDE) ||
        (type == UCP_DATATYPE_STRIDE_R)) {
        va_list ap;
        va_start(ap, type);
        ucp_dt_stride_create(&dt->stride, ap);
        va_end(ap);
    } else if (type == UCP_DATATYPE_GENERIC) {
        ucp_generic_dt_ops_t *ops;
        void *context;
        va_list ap;
        va_start(ap, type);
        ops = va_arg(ap, ucp_generic_dt_ops_t*);
        context = va_arg(ap, void*);
        ucp_dt_generic_create(&dt->generic, ops, context);
    }

    return (ucp_datatype_t)((uintptr_t)dt | type);
}

ucs_status_t ucp_dt_create_generic(const ucp_generic_dt_ops_t *ops, void *context,
                                   ucp_datatype_t *datatype_p)
{
    ucp_datatype_t dt = ucp_dt_create(UCP_DATATYPE_GENERIC, ops, context);
    if (!dt) {
        return UCS_ERR_NO_MEMORY;
    }

    *datatype_p = dt;
    return UCS_OK;
}

void ucp_dt_destroy(ucp_datatype_t datatype)
{
    ucp_dt_extended_t *dt_ex;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        break;

    case UCP_DATATYPE_GENERIC:
        dt_ex = ucp_dt_ptr(datatype);
        ucs_free(dt_ex);
        break;

    case UCP_DATATYPE_STRIDE:
        dt_ex = ucp_dt_ptr(datatype);
        ucs_free(dt_ex);
        break;

    case UCP_DATATYPE_IOV_R:
    case UCP_DATATYPE_STRIDE_R:
        dt_ex = ucp_dt_ptr(datatype);
        if (dt_ex->reusable.nc_memh != UCT_MEM_HANDLE_NULL) {
            ucp_dt_reusable_destroy(&dt_ex->reusable);
        }
        break;

    default:
        break;
    }
}
