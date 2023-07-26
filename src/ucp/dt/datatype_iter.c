/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "datatype_iter.inl"


#define ucp_datatype_iter_iov_for_each(_iov_index, _length, _dt_iter) \
    for (_iov_index = 0, _length = 0; _length < (_dt_iter)->length; \
         _length += ucp_datatype_iter_iov_at(_dt_iter, _iov_index++)->length)


static UCS_F_ALWAYS_INLINE const ucp_dt_iov_t *
ucp_datatype_iter_iov_at(const ucp_datatype_iter_t *dt_iter, size_t index)
{
    ucs_assertv(index < dt_iter->type.iov.iov_count, "index=%zu count=%zu",
                index, dt_iter->type.iov.iov_count);
    return &dt_iter->type.iov.iov[index];
}

size_t ucp_datatype_iter_iov_count(const ucp_datatype_iter_t *dt_iter)
{
    size_t iov_count, length;

    ucp_datatype_iter_iov_for_each(iov_count, length, dt_iter);

    return iov_count;
}

static void
ucp_datatype_iter_iov_check_memh_mds(const ucp_context_h context,
                                     const ucp_datatype_iter_t *dt_iter,
                                     ucp_md_map_t md_map)
{
    size_t iov_index, length;

    ucp_datatype_iter_iov_for_each(iov_index, length, dt_iter) {
        ucs_assertv(ucs_test_all_flags(
                            dt_iter->type.iov.memh[iov_index]->md_map, md_map),
                    "md_map mismatch: memh: %lu, required: %lu",
                    dt_iter->type.iov.memh[iov_index]->md_map, md_map);
    }
}

ucs_status_t
ucp_datatype_iter_set_iov_memh(ucp_datatype_iter_t *dt_iter, ucp_mem_h memh)
{
    size_t iov_count = ucp_datatype_iter_iov_count(dt_iter);
    size_t iov_index;
    ucs_status_t status;

    status = ucp_datatype_iter_iov_allocate_memh(dt_iter, iov_count);
    if (status != UCS_OK) {
        return status;
    }

    for (iov_index = 0; iov_index < iov_count; ++iov_index) {
        /* All buffers are contained in a single memh */
        dt_iter->type.iov.memh[iov_index] = memh;
    }

    return UCS_OK;
}

ucs_status_t ucp_datatype_iter_iov_mem_reg(ucp_context_h context,
                                           ucp_datatype_iter_t *dt_iter,
                                           ucp_md_map_t md_map,
                                           unsigned uct_flags)
{
    size_t iov_count = ucp_datatype_iter_iov_count(dt_iter);
    const ucp_dt_iov_t *iov;
    ucs_status_t status;
    size_t iov_index;

    if (md_map == 0) {
        return UCS_OK;
    }

    if (dt_iter->type.iov.memh != NULL) {
        /* User memh supplied, verify all required MDs received */
        ucp_datatype_iter_iov_check_memh_mds(context, dt_iter, md_map);
        return UCS_OK;
    }

    status = ucp_datatype_iter_iov_allocate_memh(dt_iter, iov_count);
    if (status != UCS_OK) {
        return status;
    }

    /* For coverity */
    ucs_assert(dt_iter->type.iov.memh != NULL);

    for (iov_index = 0; iov_index < iov_count; ++iov_index) {
        iov    = ucp_datatype_iter_iov_at(dt_iter, iov_index);
        status = ucp_memh_get(context, iov->buffer, iov->length,
                              dt_iter->mem_info.type, md_map, uct_flags,
                              &dt_iter->type.iov.memh[iov_index]);
        if (status != UCS_OK) {
            ucp_datatype_iter_iov_mem_dereg(context, dt_iter);
            return status;
        }
    }

    return UCS_OK;
}

void ucp_datatype_iter_iov_mem_dereg(ucp_context_h context,
                                     ucp_datatype_iter_t *dt_iter)
{
    ucp_mem_h *memh = dt_iter->type.iov.memh;
    size_t iov_index, length;

    if ((memh == NULL) || ucp_memh_is_user_memh(*memh)) {
        return;
    }

    ucp_datatype_iter_iov_for_each(iov_index, length, dt_iter) {
        ucp_datatype_memh_dereg(context, memh + iov_index);
    }

    ucs_free(memh);
    dt_iter->type.iov.memh = NULL;
}

size_t ucp_datatype_iter_iov_next_iov(const ucp_datatype_iter_t *dt_iter,
                                      size_t max_length,
                                      ucp_rsc_index_t memh_index,
                                      ucp_datatype_iter_t *next_iter,
                                      uct_iov_t *iov, size_t max_iov)
{
    ucp_mem_h *iov_memh = dt_iter->type.iov.memh;
    size_t remaining_dst, remaining_src;
    size_t iov_offset, dst_iov_index;
    size_t length, max_iter_length;
    const ucp_dt_iov_t *src_iov;
    ucp_mem_h memh;
    uct_iov_t *dst_iov;

    ucp_datatype_iter_iov_check(dt_iter);

#if UCS_ENABLE_ASSERT
    next_iter->type.iov.iov_count  = dt_iter->type.iov.iov_count;
#endif
    next_iter->type.iov.iov_index  = dt_iter->type.iov.iov_index;
    next_iter->type.iov.iov_offset = dt_iter->type.iov.iov_offset;

    /* Limiting data length by max_iter_length prevents from going outside the
       iov list */
    ucs_assert(dt_iter->offset <= dt_iter->length);
    max_iter_length = ucs_min(max_length, dt_iter->length - dt_iter->offset);

    length        = 0;
    dst_iov_index = 0;
    while ((dst_iov_index < max_iov) && (length < max_iter_length)) {
        ucp_datatype_iter_iov_check(next_iter);

        src_iov = ucp_datatype_iter_iov_at(dt_iter,
                                           next_iter->type.iov.iov_index);
        if (src_iov->length > 0) {
            dst_iov         = &iov[dst_iov_index++];
            memh            = (iov_memh == NULL) ? NULL :
                              iov_memh[next_iter->type.iov.iov_index];
            iov_offset      = next_iter->type.iov.iov_offset;
            dst_iov->buffer = UCS_PTR_BYTE_OFFSET(src_iov->buffer, iov_offset);
            dst_iov->memh   = (memh == NULL) ? UCT_MEM_HANDLE_NULL :
                              ucp_datatype_iter_uct_memh(memh, memh_index);
            dst_iov->stride = 0;
            dst_iov->count  = 1;

            remaining_src = src_iov->length - iov_offset;
            remaining_dst = max_length - length;
            ucs_assert(remaining_src > 0);
            ucs_assert(remaining_dst > 0);

            if (remaining_dst < remaining_src) {
                /* Reached max_length before end of current iov */
                dst_iov->length                 = remaining_dst;
                length                         += remaining_dst;
                next_iter->type.iov.iov_offset += remaining_dst;
                break;
            }

            /* Reached end of current iov */
            dst_iov->length = remaining_src;
            length         += remaining_src;
            ucs_assert(next_iter->type.iov.iov_offset + remaining_src ==
                       src_iov->length);
        }

        /* Start next IOV */
        next_iter->type.iov.iov_offset = 0;
        ++next_iter->type.iov.iov_index;
    }

    /* Check that max length was not exceeded */
    ucs_assertv(length <= max_length, "length=%zu max_length=%zu", length,
                max_length);

    /* Check that if not reached the end, packed data is not empty */
    ucs_assertv((dt_iter->offset == dt_iter->length) || (length > 0),
                "dt_iter->offset=%zu dt_iter->length=%zu length=%zu",
                dt_iter->offset, dt_iter->length, length);

    /* Advance offset by the length of packed data */
    next_iter->offset = dt_iter->offset + length;
    ucs_assert(next_iter->offset <= dt_iter->length);

    return dst_iov_index;
}

void ucp_datatype_iter_str(const ucp_datatype_iter_t *dt_iter,
                           ucs_string_buffer_t *strb)
{
    size_t iov_index, offset;
    const ucp_dt_iov_t *iov;
    const char *sysdev_name;

    if (dt_iter->mem_info.type != UCS_MEMORY_TYPE_HOST) {
        ucs_string_buffer_appendf(
                strb, "%s ", ucs_memory_type_names[dt_iter->mem_info.type]);
    }

    if (dt_iter->mem_info.sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        sysdev_name = ucs_topo_sys_device_get_name(dt_iter->mem_info.sys_dev);
        ucs_string_buffer_appendf(strb, "%s ", sysdev_name);
    }

    ucs_string_buffer_appendf(strb, "%zu/%zu %s", dt_iter->offset,
                              dt_iter->length,
                              ucp_datatype_class_names[dt_iter->dt_class]);

    switch (dt_iter->dt_class) {
    case UCP_DATATYPE_CONTIG:
        ucs_string_buffer_appendf(strb, " buffer:%p",
                                  dt_iter->type.contig.buffer);
        break;
    case UCP_DATATYPE_IOV:
        iov_index = 0;
        offset    = 0;
        while (offset < dt_iter->length) {
            iov = ucp_datatype_iter_iov_at(dt_iter, iov_index);
            ucs_string_buffer_appendf(strb, " [%zu]", iov_index);
            if (iov_index == dt_iter->type.iov.iov_index) {
                ucs_string_buffer_appendf(strb, " *{%p,%zu/%zu}", iov->buffer,
                                          dt_iter->type.iov.iov_offset,
                                          iov->length);
            } else {
                ucs_string_buffer_appendf(strb, " {%p, %zu}", iov->buffer,
                                          iov->length);
            }
            offset += iov->length;
            ++iov_index;
        }
        break;
    case UCP_DATATYPE_GENERIC:
        ucs_string_buffer_appendf(strb, " dt_gen:%p state:%p",
                                  dt_iter->type.generic.dt_gen,
                                  dt_iter->type.generic.state);

        break;
    default:
        break;
    }
}

ucs_status_t
ucp_datatype_iter_is_user_memh_valid(const ucp_datatype_iter_t *dt_iter,
                                     const ucp_mem_h memh)
{
    UCS_STRING_BUFFER_ONSTACK(err_msg, 256);
    size_t iov_count;

    if (memh == NULL) {
        ucs_error("got NULL memory handle");
        return UCS_ERR_INVALID_PARAM;
    }

    switch (dt_iter->dt_class) {
    case UCP_DATATYPE_CONTIG:
        if (!ucp_memh_is_buffer_in_range(memh, dt_iter->type.contig.buffer,
                                         dt_iter->length)) {
            ucs_string_buffer_appendf(&err_msg, "[buffer %p length %zu]",
                                      dt_iter->type.contig.buffer,
                                      dt_iter->length);
            goto err_memh_mismatch;
        }
        break;
    case UCP_DATATYPE_IOV:
        iov_count = ucp_datatype_iter_iov_count(dt_iter);
        if (!ucp_memh_is_iov_buffer_in_range(memh, dt_iter->type.iov.iov,
                                             iov_count, &err_msg)) {
            goto err_memh_mismatch;
        }
        break;
    default:
        ucs_error("unsupported memory handle datatype: [%s]",
                  ucp_datatype_class_names[dt_iter->dt_class]);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;

err_memh_mismatch:
    ucs_error("mismatched memory handle %p [address %p length %zu] for %s",
              memh, ucp_memh_address(memh), ucp_memh_length(memh),
              ucs_string_buffer_cstr(&err_msg));
    return UCS_ERR_INVALID_PARAM;
}
