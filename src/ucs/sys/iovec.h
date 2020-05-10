/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_IOVEC_H
#define UCS_IOVEC_H

#include <ucs/debug/assert.h>

#include <stdint.h>
#include <sys/uio.h>

BEGIN_C_DECLS

/* A direction for copying a data to/from an array of iovec elements */
typedef enum ucs_iov_copy_direction {
    UCS_IOV_COPY_TO_BUF,
    UCS_IOV_COPY_FROM_BUF
} ucs_iov_copy_direction_t;


/* An iterator that should be used by IOV convertor in order to save
 * information about the current offset in the destination IOV array */
typedef struct ucs_iov_iter {
    size_t     iov_index;     /* The current index in iov array */
    size_t     buffer_offset; /* The current offset in the buffer of the
                               * current iov element */
} ucs_iov_iter_t;


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

/**
 * Returns the maximum possible value for the number of IOVs.
 * It maybe either value from the system configuration or IOV_MAX
 * value or UIO_MAXIOV value or 1024 if nothing is defined.
 *
 * @return The maximum number of IOVs.
 */
size_t ucs_iov_get_max();

END_C_DECLS

#endif
