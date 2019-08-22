/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/debug/assert.h>

#include <stdint.h>
#include <sys/uio.h>


/* A direction for copying a data to/from an array of iovec elements */
typedef enum ucs_iov_copy_direction {
    UCS_IOV_COPY_TO_BUF,
    UCS_IOV_COPY_FROM_BUF
} ucs_iov_copy_direction_t;


/**
 * Copy a data from iovec [buffer] to buffer [iovec].
 *
 * @param [in] iov           A pointer to an array of iovec elements.
 * @param [in] iov_cnt       A number of elements in a iov array.
 * @param [in] iov_offset    An offset in a iov array.
 * @param [in] buf           A buffer that should be used for copying a data.
 * @param [in] max_copye     A maximum amount of data that should be copied.
 * @param [in] dir           Direction that specifies destination and source.
 *
 * @return The amount, in bytes, of the data that was copied.
 */
size_t ucs_iov_copy(const struct iovec *iov, size_t iov_cnt,
                    size_t iov_offset, void *buf, size_t max_copy,
                    ucs_iov_copy_direction_t dir);

/**
 * Update an array of iovec elements to consider an already consumed data.
 *
 * @param [in]     iov            A pointer to an array of iovec elements.
 * @param [in]     iov_cnt        A number of elements in a iov array.
 * @param [in/out] cur_iov_idx    A pointer to an index in a iov array from
 *                                which the operation should be started.
 * @param [in]     consumed       An amount of data consumed that should be
 *                                considered in a current iov array.
 */
void ucs_iov_advance(struct iovec *iov, size_t iov_cnt,
                     size_t *cur_iov_idx, size_t consumed);
