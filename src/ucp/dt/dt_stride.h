/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_STRIDED_H_
#define UCP_DT_STRIDED_H_

#include <ucp/api/ucp.h>

#define UCP_DT_STRIDE_MAX_DIMS (1)

#define UCP_DT_IS_STRIDED(_datatype) \
    (((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_STRIDED)

typedef struct ucp_dt_stride_dim {
    size_t extent;
    size_t count;
} ucp_dt_stride_dim_t;

typedef struct ucp_dt_stride {
    ucp_datatype_t dt;
    size_t item_length;
    size_t total_length;
    size_t total_extent;
    size_t ratio; /* for interleaving */
    unsigned dim_cnt;
    ucp_dt_stride_dim_t dims[UCP_DT_STRIDE_MAX_DIMS];
} ucp_dt_stride_t;


void ucp_dt_stride_create(ucp_dt_stride_t *dt, va_list ap);

/**
 * Copy iov data buffers from @a src to contiguous buffer @a dest with
 * a iov item data @a iov_offset and iov item @a iovcnt_offset
 *
 * @param [in]     dest           Destination contiguous buffer
 *                                (no offset applicable)
 * @param [in]     iov            Source @ref ucp_dt_iov_t buffer
 * @param [in]     length         Total data length to copy in bytes
 * @param [inout]  iov_offset     The offset in bytes to start copying
 *                                from an @a iov item pointed by
 *                                @a iovcnt_offset. The @a iov_offset is not aligned
 *                                by @ref ucp_dt_iov_t items length.
 * @param [inout]  iovcnt_offset  Auxiliary offset to select @a iov item which
 *                                belongs to the @a iov_offset. The point to start
 *                                copying from should be selected as
 *                                iov[iovcnt_offset].buffer + iov_offset
 */
void ucp_dt_stride_gather(void *dest, const void *src, size_t length,
                          const ucp_dt_stride_t *dt,
                          size_t *item_offset, size_t *dim_indexes);

/**
 * Copy contiguous buffer @a src into @ref ucp_dt_iov_t data buffers in @a iov
 * with an iov item data @a iov_offset and iov item @a iovcnt_offset
 *
 * @param [in]     iov            Destination @ref ucp_dt_iov_t buffer
 * @param [in]     iovcnt         Size of the @a iov buffer
 * @param [in]     src            Source contiguous buffer (no offset applicable)
 * @param [in]     length         Total data length to copy in bytes
 * @param [inout]  iov_offset     The offset in bytes to start copying
 *                                to an @a iov item pointed by @a iovcnt_offset.
 *                                The @a iov_offset is not aligned by
 *                                @ref ucp_dt_iov_t items length.
 * @param [inout]  iovcnt_offset  Auxiliary offset to select @a iov item which
 *                                belongs to the @a iov_offset. The point to
 *                                start copying from should be selected as
 *                                iov[iovcnt_offset].buffer + iov_offset
 *
 * @return Size in bytes that is actually copied from @a src to @a iov. It must
 *         be less or equal to @a length.
 */
size_t ucp_dt_stride_scatter(const ucp_dt_stride_t *dt, void *dest,
                             const void *src, size_t length,
                             size_t *item_offset, size_t *dim_indexes);

#endif
