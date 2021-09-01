/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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


static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_mem_dereg_some(ucp_context_h context,
                                 ucp_md_map_t keep_md_map,
                                 const ucp_dt_reg_t *dt_reg,
                                 uct_mem_h *prev_memh)
{
    ucp_md_index_t md_index, memh_index, memh_index_old;
    ucs_status_t status;
    uct_mem_h uct_memh;

    memh_index_old = 0;
    memh_index     = 0;
    ucs_for_each_bit(md_index, dt_reg->md_map) {
        uct_memh = dt_reg->memh[memh_index++];
        if (keep_md_map & UCS_BIT(md_index)) {
            prev_memh[memh_index_old++] = uct_memh;
        } else if (ucs_likely(uct_memh != UCT_MEM_HANDLE_NULL)) {
            /* memh not needed and registered - deregister it */
            ucs_trace("de-registering memh=%p from md[%d]=%s", uct_memh,
                      md_index, context->tl_mds[md_index].rsc.md_name);
            status = uct_md_mem_dereg(context->tl_mds[md_index].md, uct_memh);
            if (ucs_unlikely(status != UCS_OK)) {
                ucs_warn("failed to dereg from md[%d]=%s: %s", md_index,
                         context->tl_mds[md_index].rsc.md_name,
                         ucs_status_string(status));
            }
        }
    }
}

static void UCS_F_NOINLINE ucp_datatype_iter_mem_dereg_some_noninline(
        ucp_context_h context, ucp_md_map_t keep_md_map,
        const ucp_dt_reg_t *dt_reg, uct_mem_h *prev_memh)
{
    ucp_datatype_iter_mem_dereg_some(context, keep_md_map, dt_reg, prev_memh);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_datatype_iter_mem_reg_internal,
                 (context, address, length, uct_flags, mem_type, md_map,
                  dt_reg),
                 ucp_context_h context, void *address, size_t length,
                 unsigned uct_flags, ucs_memory_type_t mem_type,
                 ucp_md_map_t md_map, ucp_dt_reg_t *dt_reg)
{
    uct_mem_h tmp_reg[UCP_MAX_OP_MDS] = {UCT_MEM_HANDLE_NULL}; /* cppcheck */
    ucp_md_index_t md_index, memh_index, memh_index_old;
    ucs_memory_info_t mem_info;
    ucs_log_level_t log_level;
    ucs_status_t status;
    void *reg_address;
    size_t reg_length;

    if (ucs_unlikely(dt_reg->md_map != 0)) {
        ucp_datatype_iter_mem_dereg_some_noninline(context, md_map, dt_reg,
                                                   tmp_reg);
    }

    if (ucs_unlikely(length == 0)) {
        for (memh_index = 0; UCS_BIT(memh_index) <= md_map; ++memh_index) {
            dt_reg->memh[memh_index] = UCT_MEM_HANDLE_NULL;
        }
        goto out;
    }

    ucs_assert(address != NULL);
    if (ucs_unlikely(context->config.ext.reg_whole_alloc_bitmap &
                     UCS_BIT(mem_type))) {
        ucp_memory_detect_internal(context, address, length, &mem_info);
        reg_address = mem_info.base_address;
        reg_length  = mem_info.alloc_length;
    } else {
        reg_address = address;
        reg_length  = length;
    }

    memh_index_old = 0;
    memh_index     = 0;
    ucs_for_each_bit(md_index, md_map) {
        if (UCS_BIT(md_index) & dt_reg->md_map) {
            /* memh already registered */
            ucs_assert(memh_index_old < UCP_MAX_OP_MDS);
            dt_reg->memh[memh_index++] = tmp_reg[memh_index_old++];
            continue;
        }

        /* MD supports registration, register new memh on it */
        status = uct_md_mem_reg(context->tl_mds[md_index].md, reg_address,
                                reg_length, uct_flags,
                                &dt_reg->memh[memh_index]);
        if (ucs_unlikely(status != UCS_OK)) {
            log_level = (uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ?
                                UCS_LOG_LEVEL_DIAG :
                                UCS_LOG_LEVEL_ERROR;
            ucs_log(log_level,
                    "failed to register %s %p length %zu on md[%d]=%s: %s",
                    ucs_memory_type_names[mem_type], reg_address, reg_length,
                    md_index, context->tl_mds[md_index].rsc.md_name,
                    ucs_status_string(status));
            dt_reg->md_map |= md_map & UCS_MASK(md_index);
            ucp_datatype_iter_mem_dereg_internal(context, dt_reg);
            return status;
        }

        ucs_trace("registered address %p length %zu on md[%d]=%s memh[%d]=%p",
                  reg_address, reg_length, md_index,
                  context->tl_mds[md_index].rsc.md_name, memh_index,
                  dt_reg->memh[memh_index]);
        ++memh_index;
    }

    ucs_assert(memh_index == ucs_popcount(md_map));

out:
    /* We expect the registration to happen on all desired memory domains,
     * since subsequent access to the iterator will use 'memh_index' which
     * assumes the md_map is as expected.
     */
    dt_reg->md_map = md_map;
    return UCS_OK;
}

UCS_PROFILE_FUNC_VOID(ucp_datatype_iter_mem_dereg_internal, (context, dt_reg),
                      ucp_context_h context, ucp_dt_reg_t *dt_reg)
{
    ucp_datatype_iter_mem_dereg_some(context, 0, dt_reg, NULL);
    dt_reg->md_map = 0;
}

static UCS_F_ALWAYS_INLINE const ucp_dt_iov_t *
ucp_datatype_iter_iov_at(const ucp_datatype_iter_t *dt_iter, size_t index)
{
    ucs_assertv(index < dt_iter->type.iov.iov_count, "index=%zu count=%zu",
                index, dt_iter->type.iov.iov_count);
    return &dt_iter->type.iov.iov[index];
}

static size_t ucp_datatype_iter_iov_count(ucp_datatype_iter_t *dt_iter)
{
    size_t iov_count, length;

    ucp_datatype_iter_iov_for_each(iov_count, length, dt_iter);

    return iov_count;
}

ucs_status_t ucp_datatype_iter_iov_mem_reg(ucp_context_h context,
                                           ucp_datatype_iter_t *dt_iter,
                                           ucp_md_map_t md_map,
                                           unsigned uct_flags)
{
    size_t iov_count = ucp_datatype_iter_iov_count(dt_iter);
    const ucp_dt_iov_t *iov;
    ucp_dt_reg_t *dt_reg;
    ucs_status_t status;
    size_t iov_index;

    /* TODO allocate from memory pool */
    dt_reg = ucs_calloc(iov_count, sizeof(*dt_reg), "dt_iov_reg");
    if (dt_reg == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (iov_index = 0; iov_index < iov_count; ++iov_index) {
        iov    = ucp_datatype_iter_iov_at(dt_iter, iov_index);
        status = ucp_datatype_iter_mem_reg_internal(context, iov->buffer,
                                                    iov->length, uct_flags,
                                                    dt_iter->mem_info.type,
                                                    md_map, &dt_reg[iov_index]);
        if (status != UCS_OK) {
            ucp_datatype_iter_iov_mem_dereg(context, dt_iter);
            return status;
        }
    }

    dt_iter->type.iov.reg = dt_reg;
    return UCS_OK;
}

void ucp_datatype_iter_iov_mem_dereg(ucp_context_h context,
                                     ucp_datatype_iter_t *dt_iter)
{
    ucp_dt_reg_t *dt_reg = dt_iter->type.iov.reg;
    size_t iov_index, length;

    ucp_datatype_iter_iov_for_each(iov_index, length, dt_iter) {
        ucp_datatype_iter_mem_dereg_internal(context, &dt_reg[iov_index]);
    }

    ucs_free(dt_reg);
    dt_iter->type.iov.reg = NULL;
}

size_t ucp_datatype_iter_iov_next_iov(const ucp_datatype_iter_t *dt_iter,
                                      size_t max_length,
                                      ucp_rsc_index_t memh_index,
                                      ucp_datatype_iter_t *next_iter,
                                      uct_iov_t *iov, size_t max_iov)
{
    size_t remaining_dst, remaining_src;
    size_t iov_offset, dst_iov_index;
    size_t length, max_iter_length;
    const ucp_dt_iov_t *src_iov;
    const ucp_dt_reg_t *dt_reg;
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
            dst_iov = &iov[dst_iov_index++];
            dt_reg  = &dt_iter->type.iov.reg[next_iter->type.iov.iov_index];

            iov_offset      = next_iter->type.iov.iov_offset;
            dst_iov->buffer = UCS_PTR_BYTE_OFFSET(src_iov->buffer, iov_offset);
            dst_iov->memh   = ucp_datatype_iter_uct_memh(dt_reg, memh_index);
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
    char buffer[32];

    if (dt_iter->mem_info.type != UCS_MEMORY_TYPE_HOST) {
        ucs_string_buffer_appendf(
                strb, "%s ", ucs_memory_type_names[dt_iter->mem_info.type]);
    }

    if (dt_iter->mem_info.sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        sysdev_name = ucs_topo_sys_device_bdf_name(dt_iter->mem_info.sys_dev,
                                                   buffer, sizeof(buffer));
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
