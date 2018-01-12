/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_IOV_H_
#define UCP_DT_IOV_H_

#include <ucp/api/ucp.h>


#define UCP_DT_IS_IOV(_datatype) \
    (((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_IOV)


/**
 * Get the total length of the data contains in IOV buffers
 */
static inline size_t ucp_dt_iov_length(const ucp_dt_iov_t *iov, size_t iovcnt)
{
    size_t iov_it, total_length = 0;

    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        total_length += iov[iov_it].length;
    }

    return total_length;
}

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
void ucp_dt_iov_gather(void *dest, const ucp_dt_iov_t *iov, size_t length,
                       size_t *iov_offset, size_t *iovcnt_offset);

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
size_t ucp_dt_iov_scatter(ucp_dt_iov_t *iov, size_t iovcnt, const void *src,
                          size_t length, size_t *iov_offset, size_t *iovcnt_offset);


/**
 * Seek to a logical offset in the iov
 *
 * @param [in]     iov            @ref ucp_dt_iov_t buffer to seek in
 * @param [in]     iovcnt         Number of entries the @a iov buffer
 * @param [in]     distance       Distance to move, relative to the current
 *                                current location
 * @param [inout]  iov_offset     The offset in bytes from the beginning of the
 *                                current iov entry
 * @param [inout]  iovcnt_offset  Current @a iov item index
 */
void ucp_dt_iov_seek(ucp_dt_iov_t *iov, size_t iovcnt, ptrdiff_t distance,
                     size_t *iov_offset, size_t *iovcnt_offset);


#endif
