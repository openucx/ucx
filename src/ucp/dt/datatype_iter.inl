/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DATATYPE_ITER_INL_
#define UCP_DATATYPE_ITER_INL_

#include "datatype_iter.h"
#include "dt.inl"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.inl>
#include <ucs/profile/profile.h>


static UCS_F_ALWAYS_INLINE int
ucp_datatype_iter_is_class(const ucp_datatype_iter_t *dt_iter,
                           enum ucp_dt_type dt_class, unsigned dt_mask)
{
    /* The following branch could be eliminated by the compiler if dt_mask and
     * the tested dt_class are constants, and dt_mask does not contain dt_class.
     * We still expect that the actual dt_iter->dt_class will be part of dt_mask,
     * and check for it if assertions are enabled.
     */
    ucs_assertv(UCS_BIT(dt_iter->dt_class) & dt_mask,
                "dt_iter %p type %d (%s) but expected mask is 0x%x", dt_iter,
                dt_iter->dt_class, ucp_datatype_class_names[dt_iter->dt_class],
                dt_mask);
    return (dt_mask & UCS_BIT(dt_class)) && (dt_iter->dt_class == dt_class);
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_contig_iter_init(ucp_context_h context, void *buffer,
                              size_t length, ucp_datatype_iter_t *dt_iter,
                              const ucp_request_param_t *param)
{
    if (!(param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMH)) {
        ucp_memory_detect(context, buffer, length, &dt_iter->mem_info);
    }

    dt_iter->length                 = length;
    dt_iter->type.contig.buffer     = buffer;
    dt_iter->type.contig.memh       = NULL;
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iov_iter_init(ucp_context_h context, void *buffer, size_t count,
                           size_t length, ucp_datatype_iter_t *dt_iter,
                           uint8_t *sg_count, const ucp_request_param_t *param)
{
    const ucp_dt_iov_t *iov = (const ucp_dt_iov_t*)buffer;

    dt_iter->length              = length;
    dt_iter->type.iov.iov        = iov;
#if UCS_ENABLE_ASSERT
    dt_iter->type.iov.iov_count  = count;
#endif
    dt_iter->type.iov.iov_index  = 0;
    dt_iter->type.iov.iov_offset = 0;
    dt_iter->type.iov.memh       = NULL;

    if (ucs_likely(count > 0)) {
        *sg_count = ucs_min(count, (size_t)UINT8_MAX);
        if (!(param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMH)) {
            ucp_memory_detect(context, iov->buffer, iov->length,
                              &dt_iter->mem_info);
        }
    } else {
        *sg_count = 1;
        ucp_memory_info_set_host(&dt_iter->mem_info);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_generic_iter_init(ucp_context_h context, void *buffer,
                               size_t count, ucp_datatype_t datatype,
                               int is_pack, ucp_datatype_iter_t *dt_iter)
{
    ucp_dt_generic_t *dt_gen = ucp_dt_to_generic(datatype);
    void *state;

    if (is_pack) {
        state = dt_gen->ops.start_pack(dt_gen->context, buffer, count);
    } else {
        state = dt_gen->ops.start_unpack(dt_gen->context, buffer, count);
    }

    dt_iter->length              = dt_gen->ops.packed_size(state);
    dt_iter->type.generic.buffer = buffer;
    dt_iter->type.generic.count  = count;
    dt_iter->type.generic.dt_gen = dt_gen;
    dt_iter->type.generic.state  = state;
    ucp_memory_info_set_host(&dt_iter->mem_info);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_datatype_iter_iov_allocate_memh(
        ucp_datatype_iter_t *dt_iter, size_t iov_count)
{
    /* TODO allocate from memory pool */
    ucp_mem_h *iov_memh;

    iov_memh = (ucp_mem_h*)ucs_calloc(iov_count, sizeof(*iov_memh),
                                      "dt_iov_memh");
    if (iov_memh == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dt_iter->type.iov.memh = iov_memh;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_datatype_iter_set_memh(
        ucp_datatype_iter_t *dt_iter, const ucp_request_param_t *param)
{
    if (!(param->op_attr_mask & UCP_OP_ATTR_FIELD_MEMH) ||
        (dt_iter->dt_class == UCP_DATATYPE_GENERIC)) {
        return UCS_OK;
    }

    if (ENABLE_PARAMS_CHECK &&
        ucp_datatype_iter_is_user_memh_valid(dt_iter, param->memh) != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_assert(param->memh != NULL); /* For Coverity */

    if (ucs_likely(dt_iter->dt_class == UCP_DATATYPE_CONTIG)) {
        dt_iter->type.contig.memh = param->memh;
    } else if (dt_iter->dt_class == UCP_DATATYPE_IOV) {
        ucp_datatype_iter_set_iov_memh(dt_iter, param->memh);
    } else {
        /* Type is not supported for user memh */
        return UCS_ERR_UNSUPPORTED;
    }

    dt_iter->mem_info.sys_dev = param->memh->sys_dev;
    dt_iter->mem_info.type    = param->memh->mem_type;

    return UCS_OK;
}

/*
 * Initialize a datatype iterator, also returns number of scatter-gather entries
 * for protocol selection.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_datatype_iter_init(ucp_context_h context, void *buffer, size_t count,
                       ucp_datatype_t datatype, size_t contig_length,
                       int is_pack, ucp_datatype_iter_t *dt_iter,
                       uint8_t *sg_count, const ucp_request_param_t *param)
{
    size_t iov_length;

    dt_iter->offset   = 0;
    dt_iter->dt_class = ucp_datatype_class(datatype);

    if (ucs_likely(dt_iter->dt_class == UCP_DATATYPE_CONTIG)) {
        ucp_datatype_contig_iter_init(context, buffer, contig_length, dt_iter,
                                      param);
        *sg_count = 1;
    } else if (dt_iter->dt_class == UCP_DATATYPE_IOV) {
        iov_length = ucp_dt_iov_length((const ucp_dt_iov_t*)buffer, count);
        ucp_datatype_iov_iter_init(context, buffer, count, iov_length, dt_iter,
                                   sg_count, param);
    } else {
        ucs_assert(dt_iter->dt_class == UCP_DATATYPE_GENERIC);
        ucp_datatype_generic_iter_init(context, buffer, count, datatype,
                                       is_pack, dt_iter);
        *sg_count = 0;
    }

    return ucp_datatype_iter_set_memh(dt_iter, param);
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_init_null(ucp_datatype_iter_t *dt_iter, size_t length,
                            uint8_t *sg_count)
{
    dt_iter->dt_class               = UCP_DATATYPE_CONTIG;
    dt_iter->length                 = length;
    dt_iter->offset                 = 0;
    dt_iter->type.contig.buffer     = NULL;
    dt_iter->type.contig.memh       = NULL;
    *sg_count                       = 1;
    ucp_memory_info_set_host(&dt_iter->mem_info);
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_init_from_dt_state(ucp_context_h context, void *buffer,
                                     size_t length, ucp_datatype_t datatype,
                                     const ucp_dt_state_t *dt_state,
                                     ucp_datatype_iter_t *dt_iter,
                                     uint8_t *sg_count)
{
    static const ucp_request_param_t dummy_param = {0};

    dt_iter->offset   = 0;
    dt_iter->dt_class = ucp_datatype_class(datatype);

    if (ucs_likely(dt_iter->dt_class == UCP_DATATYPE_CONTIG)) {
        ucp_datatype_contig_iter_init(context, buffer, length, dt_iter,
                                      &dummy_param);
        *sg_count = 1;
    } else if (dt_iter->dt_class == UCP_DATATYPE_IOV) {
        ucp_datatype_iov_iter_init(context, buffer, dt_state->dt.iov.iovcnt,
                                   length, dt_iter, sg_count, &dummy_param);
    } else {
        ucs_assert(dt_iter->dt_class == UCP_DATATYPE_GENERIC);
        /* Transfer ownership from dt_state to dt_iter */
        dt_iter->length              = length;
        dt_iter->type.generic.dt_gen = ucp_dt_to_generic(datatype);
        dt_iter->type.generic.state  = dt_state->dt.generic.state;
        ucp_memory_info_set_host(&dt_iter->mem_info);
        *sg_count = 0;
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_slice(const ucp_datatype_iter_t *dt_iter, size_t offset,
                        size_t length, ucp_datatype_iter_t *sliced_dt_iter,
                        uint8_t *sg_count)
{
    ucs_assertv(dt_iter->dt_class == UCP_DATATYPE_CONTIG, "dt=%d (%s)",
                dt_iter->dt_class, ucp_datatype_class_names[dt_iter->dt_class]);

    sliced_dt_iter->dt_class               = dt_iter->dt_class;
    sliced_dt_iter->mem_info               = dt_iter->mem_info;
    sliced_dt_iter->length                 = length;
    sliced_dt_iter->offset                 = 0;
    sliced_dt_iter->type.contig.buffer     = UCS_PTR_BYTE_OFFSET(
                                                    dt_iter->type.contig.buffer,
                                                    offset);
    sliced_dt_iter->type.contig.memh       = NULL;
    *sg_count                              = 1;
}

/*
 * Cleanup datatype iterator. dt_mask is a bitmap of possible datatypes, which
 * could help the compiler eliminate some branches.
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_cleanup(ucp_datatype_iter_t *dt_iter, unsigned dt_mask)
{
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_CONTIG, dt_mask)) {
        dt_iter->type.contig.memh = NULL;
    } else if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_IOV, dt_mask)) {
        ucs_free(dt_iter->type.iov.memh);
        dt_iter->type.iov.memh = NULL;
    } else if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_GENERIC,
                                          dt_mask)) {
        dt_iter->type.generic.dt_gen->ops.finish(dt_iter->type.generic.state);
    }
}

/*
 * Check if the iterator has start position
 */
static UCS_F_ALWAYS_INLINE int
ucp_datatype_iter_is_begin(const ucp_datatype_iter_t *dt_iter)
{
    return dt_iter->offset == 0;
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_iov_check(const ucp_datatype_iter_t *dt_iter)
{
    ucs_assertv((dt_iter->type.iov.iov_count == 0) ||
                (dt_iter->type.iov.iov_index < dt_iter->type.iov.iov_count),
                "index=%zu count=%zu", dt_iter->type.iov.iov_index,
                dt_iter->type.iov.iov_count);
}

/*
 * Pack data and set some fields of next_iter as next iterator state
 */
static UCS_F_ALWAYS_INLINE size_t
ucp_datatype_iter_next_pack(const ucp_datatype_iter_t *dt_iter,
                            ucp_worker_h worker, size_t max_length,
                            ucp_datatype_iter_t *next_iter, void *dest)
{
    ucp_dt_generic_t *dt_gen;
    const void *src;
    size_t length;

    switch (dt_iter->dt_class) {
    case UCP_DATATYPE_CONTIG:
        ucs_assert(dt_iter->mem_info.type < UCS_MEMORY_TYPE_LAST);
        length = ucs_min(dt_iter->length - dt_iter->offset, max_length);
        src    = UCS_PTR_BYTE_OFFSET(dt_iter->type.contig.buffer,
                                     dt_iter->offset);
        ucp_dt_contig_pack(worker, dest, src, length,
                           (ucs_memory_type_t)dt_iter->mem_info.type);
        break;
    case UCP_DATATYPE_IOV:
        ucp_datatype_iter_iov_check(dt_iter);
        length = ucs_min(dt_iter->length - dt_iter->offset, max_length);
        next_iter->type.iov.iov_index  = dt_iter->type.iov.iov_index;
        next_iter->type.iov.iov_offset = dt_iter->type.iov.iov_offset;
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, worker, dest,
                              dt_iter->type.iov.iov, length,
                              &next_iter->type.iov.iov_offset,
                              &next_iter->type.iov.iov_index,
                              (ucs_memory_type_t)dt_iter->mem_info.type);
        break;
    case UCP_DATATYPE_GENERIC:
        if (max_length != 0) {
            dt_gen = dt_iter->type.generic.dt_gen;
            length = UCS_PROFILE_NAMED_CALL("dt_pack", dt_gen->ops.pack,
                                            dt_iter->type.generic.state,
                                            dt_iter->offset, dest, max_length);
        } else {
            length = 0;
        }
        break;
    default:
        ucs_fatal("invalid data type");
    }

    next_iter->offset = dt_iter->offset + length;
    return length;
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_iov_seek(ucp_datatype_iter_t *dt_iter, size_t offset)
{
    const ucp_dt_iov_t *iov = dt_iter->type.iov.iov;
    ssize_t iov_offset;
    size_t length_it;

    ucp_datatype_iter_iov_check(dt_iter);

    if (ucs_likely(offset == dt_iter->offset)) {
        return;
    }

    iov_offset = dt_iter->type.iov.iov_offset + (offset - dt_iter->offset);
    if (iov_offset < 0) {
        /* seek backwards */
        do {
            ucs_assert(dt_iter->type.iov.iov_index > 0);
            --dt_iter->type.iov.iov_index;
            iov_offset += iov[dt_iter->type.iov.iov_index].length;
        } while (iov_offset < 0);
    } else {
        /* seek forward */
        while (iov_offset >=
               (length_it = iov[dt_iter->type.iov.iov_index].length)) {
            iov_offset -= length_it;
            ++dt_iter->type.iov.iov_index;
            ucp_datatype_iter_iov_check(dt_iter);
        }
    }

    dt_iter->offset              = offset;
    dt_iter->type.iov.iov_offset = iov_offset;
}

/*
 * Unpack data at the given offset. Some datatypes, such as iov, can cache
 * offset information in the iterator.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_datatype_iter_unpack(ucp_datatype_iter_t *dt_iter, ucp_worker_h worker,
                         size_t length, size_t offset, const void *src)
{
    ucp_dt_generic_t *dt_gen;
    size_t unpacked_length;
    ucs_status_t status;
    void *dest;

    if (ucs_unlikely(dt_iter->length - offset < length)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (dt_iter->dt_class) {
    case UCP_DATATYPE_CONTIG:
        ucs_assert(dt_iter->mem_info.type < UCS_MEMORY_TYPE_LAST);
        dest = UCS_PTR_BYTE_OFFSET(dt_iter->type.contig.buffer, offset);
        ucp_dt_contig_unpack(worker, dest, src, length,
                             (ucs_memory_type_t)dt_iter->mem_info.type);
        status = UCS_OK;
        break;
    case UCP_DATATYPE_IOV:
        ucp_datatype_iter_iov_seek(dt_iter, offset);
        unpacked_length = UCS_PROFILE_CALL(ucp_dt_iov_scatter,
                                           worker, dt_iter->type.iov.iov, SIZE_MAX, src,
                                           length,
                                           &dt_iter->type.iov.iov_offset,
                                           &dt_iter->type.iov.iov_index,
                                           (ucs_memory_type_t)dt_iter->mem_info.type);
        ucs_assert(unpacked_length <= length);
        dt_iter->offset += unpacked_length;
        status           = UCS_OK;
        break;
    case UCP_DATATYPE_GENERIC:
        if (length != 0) {
            dt_gen = dt_iter->type.generic.dt_gen;
            status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                            dt_iter->type.generic.state, offset,
                                            src, length);
        } else {
            status = UCS_OK;
        }
        break;
    default:
        ucs_fatal("invalid data type");
    }

    return status;
}

/* Advances dt iterator, returns length */
static UCS_F_ALWAYS_INLINE size_t
ucp_datatype_iter_next(const ucp_datatype_iter_t *dt_iter,
                       size_t max_length, ucp_datatype_iter_t *next_iter)
{
    size_t offset, length;

    offset            = dt_iter->offset;
    length            = ucs_min(max_length, dt_iter->length - offset);
    next_iter->offset = offset + length;

    return length;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_datatype_iter_get_ptr(const ucp_datatype_iter_t *dt_iter, size_t max_length,
                          void **ptr)
{
    ucs_assert(dt_iter->dt_class == UCP_DATATYPE_CONTIG);

    *ptr = UCS_PTR_BYTE_OFFSET(dt_iter->type.contig.buffer, dt_iter->offset);

    return ucs_min(max_length, dt_iter->length - dt_iter->offset);
}

/*
 * Returns a pointer to next chunk of data (could be done only on some datatype
 * classes)
 */
static UCS_F_ALWAYS_INLINE size_t
ucp_datatype_iter_next_ptr(const ucp_datatype_iter_t *dt_iter,
                           size_t max_length, ucp_datatype_iter_t *next_iter,
                           void **ptr)
{
    ucs_assert(dt_iter->dt_class == UCP_DATATYPE_CONTIG);
    *ptr = UCS_PTR_BYTE_OFFSET(dt_iter->type.contig.buffer, dt_iter->offset);

    return ucp_datatype_iter_next(dt_iter, max_length, next_iter);
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_next_slice(const ucp_datatype_iter_t *dt_iter,
                             size_t max_length,
                             ucp_datatype_iter_t *sliced_dt_iter,
                             ucp_datatype_iter_t *next_iter, uint8_t *sg_count)
{
    size_t length;

    length = ucp_datatype_iter_next(dt_iter, max_length, next_iter);
    ucp_datatype_iter_slice(dt_iter, dt_iter->offset, length, sliced_dt_iter,
                            sg_count);
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_datatype_iter_uct_memh(const ucp_mem_h memh, ucp_rsc_index_t memh_index)
{
    if (memh_index == UCP_NULL_RESOURCE) {
        return UCT_MEM_HANDLE_NULL;
    }

    ucs_assertv((UCS_BIT(memh_index) & memh->md_map) ||
                ucp_memh_is_zero_length(memh),
                "memh_index=%d md_map=0x%" PRIx64, memh_index, memh->md_map);
    return memh->uct[memh_index];
}

/*
 * Returns a pointer to next chunk of data as IOV entry of registered memory
 * (could be done only on some datatype classes)
 *
 * @param memh_index  Index of UCT memory handle (within the memory domain map
 *                    which was passed to @ref ucp_datatype_iter_mem_reg, or
 *                    UCP_NULL_RESOURCE to pass UCT_MEM_HANDLE_NULL.
 *
 * @return Number of iov elements.
 */
static UCS_F_ALWAYS_INLINE size_t
ucp_datatype_iter_next_iov(const ucp_datatype_iter_t *dt_iter,
                           size_t max_length, ucp_rsc_index_t memh_index,
                           unsigned dt_mask, ucp_datatype_iter_t *next_iter,
                           uct_iov_t *iov, size_t max_iov)
{
    ucs_assert(max_iov >= 1);
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_CONTIG, dt_mask)) {
#ifdef __clang_analyzer__
        /* clang analyzer falsely warns about next_iter being used uninitialized
           in ucp_datatype_iter_copy_position() IOV case */
        next_iter->type.iov.iov_index  = 0;
        next_iter->type.iov.iov_offset = 0;
#endif
        iov[0].length = ucp_datatype_iter_next_ptr(dt_iter, max_length,
                                                   next_iter, &iov[0].buffer);
        iov[0].memh   = ucp_datatype_iter_uct_memh(dt_iter->type.contig.memh,
                                                   memh_index);
        iov[0].stride = 0;
        iov[0].count  = 1;
        return 1;
    } else if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_IOV, dt_mask)) {
        return ucp_datatype_iter_iov_next_iov(dt_iter, max_length, memh_index,
                                              next_iter, iov, max_iov);
    } else {
        /* Silence compiler warning */
        next_iter->offset = dt_iter->offset;
        iov[0].length     = 0;
        iov[0].buffer     = NULL;
        iov[0].memh       = UCT_MEM_HANDLE_NULL;
        ucs_bug("unsupported datatype %s",
                ucp_datatype_class_names[dt_iter->dt_class]);
        return 0;
    }
}

/*
 * Copy iterator position only.
 * 'src_dt_iter' must be initialized from the same datatype object as 'dt_iter',
 * or returned as 'next_iter' from @ref ucp_datatype_iter_next_pack
 * @ref ucp_datatype_iter_next_unpack or @ref ucp_datatype_iter_next_iov that
 * were used on 'dt_iter'.
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_copy_position(ucp_datatype_iter_t *dt_iter,
                                const ucp_datatype_iter_t *src_dt_iter,
                                unsigned dt_mask)
{
    dt_iter->offset = src_dt_iter->offset;
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_IOV, dt_mask)) {
        dt_iter->type.iov.iov_index  = src_dt_iter->type.iov.iov_index;
        dt_iter->type.iov.iov_offset = src_dt_iter->type.iov.iov_offset;
    }
}

/*
 * Check if the next iterator has reached the end
 */
static UCS_F_ALWAYS_INLINE int
ucp_datatype_iter_is_end_position(const ucp_datatype_iter_t *dt_iter,
                                  const ucp_datatype_iter_t *pos_iter)
{
    ucs_assert(dt_iter->offset <= dt_iter->length);
    return pos_iter->offset == dt_iter->length;
}

/*
 * Check if the iterator has reached the end
 */
static UCS_F_ALWAYS_INLINE int
ucp_datatype_iter_is_end(const ucp_datatype_iter_t *dt_iter)
{
    return ucp_datatype_iter_is_end_position(dt_iter, dt_iter);
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_memh_dereg(ucp_context_h context, ucp_mem_h *memh_p)
{
    if (*memh_p == NULL) {
        return;
    }

    /* Do no release user memory handle */
    if (ucp_memh_is_user_memh(*memh_p)) {
        return;
    }

    ucp_memh_put(context, *memh_p);
    *memh_p = NULL;
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_contig_check_memh_mds(ucp_mem_h memh, ucp_md_map_t md_map)
{
    /* md_map is not relevant in case of zero length request */
    if (ucp_memh_is_zero_length(memh)) {
        return;
    }

    /* Verify all required MDs are registered */
    ucs_assertv(ucs_test_all_flags(memh->md_map, md_map),
                "md_map mismatch: memh: 0x%" PRIx64 ", required: 0x%" PRIx64,
                memh->md_map, md_map);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_datatype_iter_contig_mem_reg(
        const ucp_context_h context, ucp_datatype_iter_t *dt_iter,
        ucp_md_map_t md_map, unsigned uct_flags)
{
    ucp_mem_h memh = dt_iter->type.contig.memh;

    if (memh != NULL) {
        ucp_datatype_iter_contig_check_memh_mds(memh, md_map);
        return UCS_OK;
    }

    /* Iterator may work with cacheable MDs only */
    ucs_assertv(ucs_test_all_flags(
                        context->cache_md_map[dt_iter->mem_info.type], md_map),
                "iterator mem_type=%s cache_md_map=0x%" PRIx64
                " md_map=0x%" PRIx64,
                ucs_memory_type_names[dt_iter->mem_info.type],
                context->cache_md_map[dt_iter->mem_info.type], md_map);

    return ucp_memh_get(context, dt_iter->type.contig.buffer, dt_iter->length,
                        (ucs_memory_type_t)dt_iter->mem_info.type, md_map,
                        uct_flags, &dt_iter->type.contig.memh);
}

/*
 * Register memory and update iterator state
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_datatype_iter_mem_reg(ucp_context_h context, ucp_datatype_iter_t *dt_iter,
                          ucp_md_map_t md_map, unsigned uct_flags,
                          unsigned dt_mask)
{
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_CONTIG, dt_mask)) {
        return ucp_datatype_iter_contig_mem_reg(context, dt_iter, md_map,
                                                uct_flags);
    } else if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_IOV, dt_mask)) {
        return ucp_datatype_iter_iov_mem_reg(context, dt_iter, md_map,
                                             uct_flags);
    } else if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_GENERIC,
                                          dt_mask)) {
        return UCS_OK;
    } else {
        ucs_error("datatype %s does not support registration",
                  ucp_datatype_class_names[dt_iter->dt_class]);
        return UCS_ERR_INVALID_PARAM;
    }
}

/*
 * De-register memory and update iterator state
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_mem_dereg(ucp_context_h context, ucp_datatype_iter_t *dt_iter,
                            unsigned dt_mask)
{
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_CONTIG, dt_mask)) {
        ucp_datatype_memh_dereg(context, &dt_iter->type.contig.memh);
    } else if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_IOV, dt_mask)) {
        ucp_datatype_iter_iov_mem_dereg(context, dt_iter);
    }
}

#endif
