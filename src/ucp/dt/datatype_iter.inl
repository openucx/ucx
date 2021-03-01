/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DATATYPE_ITER_INL_
#define UCP_DATATYPE_ITER_INL_

#include "datatype_iter.h"
#include "dt.inl"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
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
    ucs_assert(UCS_BIT(dt_iter->dt_class) & dt_mask);
    return (dt_mask & UCS_BIT(dt_class)) && (dt_iter->dt_class == dt_class);
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_contig_iter_init(ucp_context_h context, void *buffer, size_t length,
                              ucp_datatype_t datatype, ucp_datatype_iter_t *dt_iter)
{
    ucp_memory_detect(context, buffer, length, &dt_iter->mem_info);
    dt_iter->length                 = length;
    dt_iter->type.contig.buffer     = buffer;
    dt_iter->type.contig.reg.md_map = 0;
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_iov_iter_init(ucp_context_h context, void *buffer, size_t count,
                           ucp_datatype_t datatype, ucp_datatype_iter_t *dt_iter,
                           uint8_t *sg_count)
{
    const ucp_dt_iov_t *iov = (const ucp_dt_iov_t*)buffer;

    dt_iter->length              = ucp_dt_iov_length(iov, count);
    dt_iter->type.iov.iov        = iov;
    dt_iter->type.iov.iov_index  = 0;
    dt_iter->type.iov.iov_offset = 0;

    if (ucs_likely(count > 0)) {
        *sg_count         = ucs_min(count, (size_t)UINT8_MAX);
        ucp_memory_detect(context, iov->buffer, iov->length, &dt_iter->mem_info);
    } else {
        *sg_count = 1;
        ucp_memory_info_set_host(&dt_iter->mem_info);
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_datatype_generic_iter_init(ucp_context_h context, void *buffer, size_t count,
                               ucp_datatype_t datatype, ucp_datatype_iter_t *dt_iter)
{
    ucp_dt_generic_t *dt_gen = ucp_dt_to_generic(datatype);
    void *state;

    state                        = dt_gen->ops.start_pack(dt_gen->context,
                                                          buffer, count);
    dt_iter->length              = dt_gen->ops.packed_size(state);
    dt_iter->type.generic.dt_gen = dt_gen;
    dt_iter->type.generic.state  = state;
    ucp_memory_info_set_host(&dt_iter->mem_info);
}

/*
 * Initialize a datatype iterator, also returns number of scatter-gather entries
 * for protocol selection.
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_init(ucp_context_h context, void *buffer, size_t count,
                       ucp_datatype_t datatype, size_t contig_length,
                       ucp_datatype_iter_t *dt_iter, uint8_t *sg_count)
{
    dt_iter->offset   = 0;
    dt_iter->dt_class = ucp_datatype_class(datatype);

    if (ucs_likely(dt_iter->dt_class == UCP_DATATYPE_CONTIG)) {
        ucp_datatype_contig_iter_init(context, buffer, contig_length, datatype,
                                      dt_iter);
        *sg_count = 1;
    } else if (dt_iter->dt_class == UCP_DATATYPE_IOV) {
        ucp_datatype_iov_iter_init(context, buffer, count, datatype, dt_iter,
                                   sg_count);
    } else {
        ucs_assert(dt_iter->dt_class == UCP_DATATYPE_GENERIC);
        ucp_datatype_generic_iter_init(context, buffer, count, datatype, dt_iter);
        *sg_count = 1;
    }
}

/*
 * Cleanup datatype iterator. dt_mask is a bitmap of possible datatypes, which
 * could help the compiler eliminate some branches.
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_cleanup(ucp_datatype_iter_t *dt_iter, unsigned dt_mask)
{
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_GENERIC, dt_mask)) {
        dt_iter->type.generic.dt_gen->ops.finish(dt_iter->type.generic.state);
    }
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
        length = ucs_min(dt_iter->length - dt_iter->offset, max_length);
        next_iter->type.iov.iov_index  = dt_iter->type.iov.iov_index;
        next_iter->type.iov.iov_offset = dt_iter->type.iov.iov_offset;
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, dest, dt_iter->type.iov.iov,
                              length, &next_iter->type.iov.iov_offset,
                              &next_iter->type.iov.iov_index);
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

/*
 * Unpack data and set some fields of next_iter as next iterator state
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_datatype_iter_next_unpack(const ucp_datatype_iter_t *dt_iter,
                              ucp_worker_h worker, size_t length,
                              ucp_datatype_iter_t *next_iter, const void *src)
{
    ucp_dt_generic_t *dt_gen;
    ucs_status_t status;
    void *dest;

    if (ucs_unlikely(dt_iter->length - dt_iter->offset < length)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (dt_iter->dt_class) {
    case UCP_DATATYPE_CONTIG:
        ucs_assert(dt_iter->mem_info.type < UCS_MEMORY_TYPE_LAST);
        dest = UCS_PTR_BYTE_OFFSET(dt_iter->type.contig.buffer, dt_iter->offset);
        ucp_dt_contig_unpack(worker, dest, src, length,
                             (ucs_memory_type_t)dt_iter->mem_info.type);
        status = UCS_OK;
        break;
    case UCP_DATATYPE_IOV:
        next_iter->type.iov.iov_index  = dt_iter->type.iov.iov_index;
        next_iter->type.iov.iov_offset = dt_iter->type.iov.iov_offset;
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_scatter, dt_iter->type.iov.iov,
                              SIZE_MAX, src, length,
                              &next_iter->type.iov.iov_offset,
                              &next_iter->type.iov.iov_index);
        status = UCS_OK;
        break;
    case UCP_DATATYPE_GENERIC:
        if (length != 0) {
            dt_gen = dt_iter->type.generic.dt_gen;
            status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                            dt_iter->type.generic.state,
                                            dt_iter->offset, src, length);
        } else {
            status = UCS_OK;
        }
        break;
    default:
        ucs_fatal("invalid data type");
    }

    next_iter->offset = dt_iter->offset + length;
    return status;
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
    size_t offset, length;

    ucs_assert(dt_iter->dt_class == UCP_DATATYPE_CONTIG);

    offset            = dt_iter->offset;
    length            = ucs_min(max_length, dt_iter->length - offset);
    *ptr              = UCS_PTR_BYTE_OFFSET(dt_iter->type.contig.buffer, offset);
    next_iter->offset = offset + length;

    return length;
}

/*
 * Returns a pointer to next chunk of data as IOV entry of registered memory
 * (could be done only on some datatype classes)
 *
 * @param memh_index  Index of UCT memory handle (within the memory domain map
 *                    which was passed to @ref ucp_datatype_iter_mem_reg, or
 *                    UCP_NULL_RESOURCE to pass UCT_MEM_HANDLE_NULL.
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_next_iov(const ucp_datatype_iter_t *dt_iter,
                           ucp_rsc_index_t memh_index, size_t max_length,
                           ucp_datatype_iter_t *next_iter, uct_iov_t *iov)
{
    /* TODO support IOV datatype */
    ucs_assert(dt_iter->dt_class == UCP_DATATYPE_CONTIG);

    if (memh_index == UCP_NULL_RESOURCE) {
        iov[0].memh = UCT_MEM_HANDLE_NULL;
    } else {
        ucs_assert(memh_index < ucs_popcount(dt_iter->type.contig.reg.md_map));
        iov[0].memh = dt_iter->type.contig.reg.memh[memh_index];
    }

    iov[0].length   = ucp_datatype_iter_next_ptr(dt_iter, max_length, next_iter,
                                                 &iov[0].buffer);
    iov[0].stride   = 0;
    iov[0].count    = 1;
}

/*
 * Copy iterator position
 */
static UCS_F_ALWAYS_INLINE
void ucp_datatype_iter_copy_from_next(ucp_datatype_iter_t *dt_iter,
                                      const ucp_datatype_iter_t *next_iter,
                                      unsigned dt_mask)
{
    dt_iter->offset = next_iter->offset;
    if (ucp_datatype_iter_is_class(dt_iter, UCP_DATATYPE_IOV, dt_mask)) {
        dt_iter->type.iov.iov_index  = next_iter->type.iov.iov_index;
        dt_iter->type.iov.iov_offset = next_iter->type.iov.iov_offset;
    }
}

/*
 * Check if the iterator has reached the end
 */
static UCS_F_ALWAYS_INLINE int
ucp_datatype_iter_is_end(const ucp_datatype_iter_t *dt_iter)
{
    ucs_assert(dt_iter->offset <= dt_iter->length);
    return dt_iter->offset == dt_iter->length;
}

/*
 * Register memory and update iterator state
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_datatype_iter_mem_reg(ucp_context_h context, ucp_datatype_iter_t *dt_iter,
                          ucp_md_map_t md_map, unsigned uct_flags)
{
    /* TODO support IOV datatype */
    ucs_assert(dt_iter->dt_class == UCP_DATATYPE_CONTIG);

    return ucp_mem_rereg_mds(context, md_map, dt_iter->type.contig.buffer,
                             dt_iter->length, uct_flags, NULL,
                             (ucs_memory_type_t)dt_iter->mem_info.type, NULL,
                             dt_iter->type.contig.reg.memh,
                             &dt_iter->type.contig.reg.md_map);
}

/*
 * De-register memory and update iterator state
 */
static UCS_F_ALWAYS_INLINE void
ucp_datatype_iter_mem_dereg(ucp_context_h context, ucp_datatype_iter_t *dt_iter)
{
    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL,
                      (ucs_memory_type_t)dt_iter->mem_info.type, NULL,
                      dt_iter->type.contig.reg.memh,
                      &dt_iter->type.contig.reg.md_map);
}

#endif
