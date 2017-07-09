/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "proto.h"

UCS_F_ALWAYS_INLINE
size_t ucp_dt_stride_copy_uct(uct_iov_t *iov, size_t *iovcnt, size_t max_dst_iov,
                              ucp_dt_state_t *state, const ucp_dt_iov_t *src_iov,
                              ucp_datatype_t datatype, size_t length_max)
{
    ucp_dt_stride_t *stride = &ucp_dt_ptr(datatype)->stride;
    size_t item_length = ucp_dt_length(stride->dt, 1, NULL, NULL);
    size_t dim_it = stride->dim_cnt;
    size_t segment_length;
    size_t length_it = 1;
    size_t dst_it = 0;
    size_t src_dim_it;
    size_t dim_off_it;
    size_t dim_offset;
    size_t dim_inc;

    ucs_assert(item_length < length_max); // TODO: improve by recursive call

    /*
     * How to complete a partial N-dimensional stride pattern?
     * First - complete the last dimension (interrupted),
     * then the second dimension if interrupted (0 < count < dim[X].count),
     * and so on until the first dimension.
     */
    do {
        /* Find the dimension that requires completion */
        segment_length = item_length;
        do {
            dim_it--;
            dim_inc = stride->dims[dim_it].count -
                    state->dt.stride.dim_index[dim_it];
            segment_length *= dim_inc;
        } while ((dim_it) &&
                 (stride->dim_cnt - dim_it < max_dst_iov - dst_it) &&
                 (state->dt.stride.dim_index[dim_it] == 0) &&
                 (length_it + segment_length > length_max));

        /* If length is the constraint - decrease the last dimension count */
        if (length_it + segment_length > length_max) {
            /* Fall-back to the previous (smaller) dimension */
            segment_length /= dim_inc;
            if (length_it + segment_length > length_max) {
                /* The smaller dimension doesn't fit as well! */
                ucs_assert(length_it);
                break;
            }

            /* Check by how much this dimension can be incremented */
            dim_inc = 0;
            while (length_it + ((dim_inc + 1) * segment_length) <= length_max) {
                dim_inc++;
            }
            dim_it++;
            ucs_assert(dim_inc);
            ucs_assert(dim_it < stride->dim_cnt);
            segment_length *= dim_inc;
        }

        /* Calculate the offset */
        dim_offset = 0;
        for (dim_off_it = 0; dim_off_it < dim_it; dim_off_it++) {
            dim_offset += stride->dims[dim_off_it].extent *
                    state->dt.stride.dim_index[dim_off_it];
        }

        /* Create the "master" (UCT-)IOV entry */
        iov[dst_it].buffer    = (void *)src_iov + dim_offset;
        iov[dst_it].memh      = state->dt.stride.memh;
        iov[dst_it].length    = length_max;
        iov[dst_it].ilv_ratio = stride->ratio; // TODO: assert(==0) for partial sends

        /* Create (UCT-)IOV entries for the incremented dimensions */
        for (src_dim_it = dim_it;
             ((dst_it + src_dim_it < max_dst_iov) &&
              (src_dim_it < stride->dim_cnt));
             src_dim_it++) {
            iov[dst_it + src_dim_it].stride = stride->dims[src_dim_it].extent;
            iov[dst_it + src_dim_it].count  = dim_inc ? dim_inc :
                    stride->dims[src_dim_it].count -
                    state->dt.stride.dim_index[src_dim_it];
            dim_inc = 0; /* Only applied to the first, incremented, dimension */
        }

        /* Apply the changes to the return values */
        length_it += segment_length;
        dst_it += src_dim_it;
    } while (dim_it);

    *iovcnt = dst_it;
    return length_it;
}

UCS_F_ALWAYS_INLINE
size_t ucp_dt_iov_copy_uct(uct_iov_t *iov, size_t *iovcnt, size_t max_dst_iov,
                           ucp_dt_state_t *state, const ucp_dt_iov_t *src_iov,
                           ucp_datatype_t datatype, size_t length_max)
{
    size_t iov_offset, max_src_iov, src_it, dst_it;
    const uct_mem_h *memh;
    size_t length_it = 0;

    memh                        = state->dt.iov.memh;
    iov_offset                  = state->dt.iov.iov_offset;
    max_src_iov                 = state->dt.iov.iovcnt;
    src_it                      = state->dt.iov.iovcnt_offset;
    dst_it                      = 0;
    state->dt.iov.iov_offset    = 0;
    while ((dst_it < max_dst_iov) && (src_it < max_src_iov)) {
        if (src_iov[src_it].count) {
            iov[dst_it].buffer  = src_iov[src_it].buffer + iov_offset;
            iov[dst_it].length  = src_iov[src_it].count - iov_offset;
            iov[dst_it].memh    = memh[src_it];
            iov[dst_it].stride  = 0;
            iov[dst_it].count   = 1;
            length_it          += iov[dst_it].length;

            ++dst_it;
            if (length_it >= length_max) {
                iov[dst_it - 1].length      -= (length_it - length_max);
                length_it                    = length_max;
                state->dt.iov.iov_offset     = iov_offset + iov[dst_it - 1].length;
                break;
            }
        }
        iov_offset = 0;
        ++src_it;
    }

    state->dt.iov.iovcnt_offset = src_it;
    *iovcnt                     = dst_it;

    return length_it;
}

UCS_F_ALWAYS_INLINE
ucs_status_t ucp_dt_reusable_create(uct_ep_h ep, void *buffer, size_t length,
                                    ucp_datatype_t datatype, ucp_dt_state_t *state)
{
    ucp_dt_extended_t *dt_ex = ucp_dt_ptr(datatype);
    const ucp_dt_iov_t *iov_it;
    size_t uct_iovcnt = 0;
    size_t length_it = 0;

    dt_ex->reusable.nc_iov = ucs_malloc(sizeof(uct_iov_t) * uct_iovcnt, "reusable_iov");
    if (!dt_ex->reusable.nc_iov) {
        return UCS_ERR_NO_MEMORY;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_STRIDE_R:
        uct_iovcnt = ucp_dt_count_uct_iov(datatype, 1, buffer, NULL); // TODO: "half-state"?
        dt_ex->reusable.length = ucp_dt_stride_copy_uct(dt_ex->reusable.nc_iov,
                &dt_ex->reusable.nc_iovcnt, uct_iovcnt, state, buffer, datatype, length);
        break;

    case UCP_DATATYPE_IOV_R:
        iov_it = buffer;
        while (length_it < length) {
            length_it += ucp_dt_length(iov_it->dt, iov_it->count, iov_it->buffer, NULL);
            uct_iovcnt += ucp_dt_count_uct_iov(iov_it->dt, iov_it->count, iov_it->buffer, NULL);
            iov_it++;
        }

        dt_ex->reusable.length = ucp_dt_iov_copy_uct(dt_ex->reusable.nc_iov,
                &dt_ex->reusable.nc_iovcnt, uct_iovcnt, state, buffer, datatype, length);
        break;

    default:
        ucs_error("Invalid data type %lx", datatype);
        ucs_free(dt_ex->reusable.nc_iov);
        return UCS_ERR_INVALID_PARAM;
    }

    dt_ex->reusable.nc_comp.func = ucp_dt_reusable_completion;
    return uct_ep_mem_reg_nc(ep, dt_ex->reusable.nc_iov, dt_ex->reusable.nc_iovcnt,
            &dt_ex->reusable.nc_md, &dt_ex->reusable.nc_memh, &dt_ex->reusable.nc_comp);
}

UCS_F_ALWAYS_INLINE
ucs_status_t ucp_dt_reusable_update(uct_ep_h ep, void *buffer, size_t length,
                                    ucp_datatype_t datatype, ucp_dt_state_t *state)
{
    ucp_dt_extended_t *dt_ex = ucp_dt_ptr(datatype);
    unsigned uct_iovcnt = dt_ex->reusable.nc_iovcnt;
    dt_ex->reusable.nc_status = UCS_OK;
    dt_ex->reusable.nc_comp.count = 1;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_STRIDE_R:
        dt_ex->reusable.length = ucp_dt_stride_copy_uct(dt_ex->reusable.nc_iov,
                &dt_ex->reusable.nc_iovcnt, uct_iovcnt, state, buffer, datatype, length);
        break;

    case UCP_DATATYPE_IOV_R:
        dt_ex->reusable.length = ucp_dt_iov_copy_uct(dt_ex->reusable.nc_iov,
                &dt_ex->reusable.nc_iovcnt, uct_iovcnt, state, buffer, datatype, length);
        break;

    default:
        ucs_error("Invalid data type %lx", datatype);
        return UCS_ERR_INVALID_PARAM;
    }

    return uct_ep_mem_reg_nc(ep, dt_ex->reusable.nc_iov, dt_ex->reusable.nc_iovcnt,
            &dt_ex->reusable.nc_md, &dt_ex->reusable.nc_memh, &dt_ex->reusable.nc_comp);
}
