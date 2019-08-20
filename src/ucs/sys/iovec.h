/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/debug/assert.h>

#include <stdint.h>
#include <sys/uio.h>


/* A direction for copying a data to/from iovec */
typedef enum ucs_sys_copy_iov_dir {
    UCS_SYS_COPY_IOV_TO_BUF = 0,
    UCS_SYS_COPY_BUF_TO_IOV = 1
} ucs_sys_copy_iov_dir_t;


/**
 * Copy a data from iovec [buffer] to buffer [iovec]
 *
 * @param [in] iov           A pointer to an array of iovec elements
 * @param [in] iov_cnt       A number of elements in a iov array
 * @param [in] iov_offset    An offset in a iov array
 * @param [in] buf           A buffer that should be used for copying a data
 * @param [in] buf_size      A size of a buffer
 * @param [in] buf_offset    An offset in a buffer
 * @param [in] dir           Direction that specifies destination and source
 *
 * @return The amount, in bytes, of the data that was copied
 */
size_t ucs_sys_copy_iov_buf(const struct iovec *iov, size_t iov_cnt,
                            size_t iov_offset, void *buf, size_t buf_size,
                            size_t buf_offset, ucs_sys_copy_iov_dir_t dir);

/**
 * Copy a data from iovec to buffer
 *
 * @param [in] iov           A pointer to an array of iovec elements
 * @param [in] iov_cnt       A number of elements in a iov array
 * @param [in] iov_offset    An offset in a iov array
 * @param [in] buf           A buffer that should be used as a destination
 *                           for copying a data
 * @param [in] buf_size      A size of a buffer
 * @param [in] buf_offset    An offset in a buffer
 *
 * @return The amount, in bytes, of the data that was copied
 */
static inline size_t
ucs_sys_copy_from_iov(void *buf, size_t buf_size, size_t buf_offset,
                      const struct iovec *iov, size_t iov_cnt,
                      size_t iov_offset)
{
    return ucs_sys_copy_iov_buf(iov, iov_cnt, iov_offset,
                                buf, buf_size, buf_offset,
                                UCS_SYS_COPY_IOV_TO_BUF);
}

/**
 * Update an array of iovec elements to consider an already consumed data
 *
 * @param [in] iov                A pointer to an array of iovec elements
 * @param [in] iov_cnt            A number of elements in a iov array
 * @param [in/out] cur_iov_idx    A pointer to an index in a iov array from
 *                                which the operation should be started
 * @param [in] consumed           An amount of data consumed that should be
 *                                considered in a current iov array
 */
static inline void
ucs_sys_consume_iov(struct iovec *iov, size_t iov_cnt,
                    size_t *cur_iov_idx, size_t consumed)
{
    size_t i;

    ucs_assert(*cur_iov_idx <= iov_cnt);

    for (i = *cur_iov_idx; i < iov_cnt; i++) {
        if (consumed < iov[i].iov_len) {
            iov[i].iov_len  -= consumed;
            iov[i].iov_base  = UCS_PTR_BYTE_OFFSET(iov[i].iov_base,
                                                   consumed);
            *cur_iov_idx     = i;
            return;
        }

        consumed -= iov[i].iov_len;
    }

    ucs_assert(!consumed && (i == *cur_iov_idx));
}
