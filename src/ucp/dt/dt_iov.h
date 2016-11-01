/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_IOV_H_
#define UCP_DT_IOV_H_

#include <ucp/api/ucp.h>

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
 * a iov item data @a offset and iov item @a iovcnt_offset
 *
 * @param [in]     dest           Destination contiguous buffer
 *                                (no offset applicable)
 * @param [in]     iov            Source @ref ucp_dt_iov_t buffer
 * @param [in]     length         Total data length to copy in bytes
 * @param [inout]  iov_offset     The offset in bytes to start copying
 *                                from an @a iov item pointed by
 *                                @a iovcnt_offset. The @a offset is not aligned
 *                                by @ref ucp_dt_iov_t items length.
 * @param [inout]  iovcnt_offset  Auxiliary offset to select iov item which
 *                                belongs to the @a offset. The point to start
 *                                copying from should be selected as
 *                                iov[iovcnt_offset].buffer + offset
 */
void ucp_dt_iov_memcpy(void *dest, const ucp_dt_iov_t *iov, size_t length,
                       size_t *iov_offset, size_t *iovcnt_offset);


#endif
