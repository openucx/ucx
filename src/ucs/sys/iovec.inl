/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_IOVEC_INL_
#define UCS_IOVEC_INL_

#include <ucs/sys/math.h>
#include <ucs/sys/iovec.h>
#include <ucs/debug/assert.h>


/**
 * Fill the destination array of IOVs by data provided in the source
 * array of IOVs.
 * The function avoids copying IOVs with zero length.
 *
 * @param [out]    _dst_iov              Pointer to the resulted array of IOVs.
 * @param [in/out] _dst_iov_cnt_p        Pointer to the variable that holds the number
 *                                       of the elements in the array of IOVs (input:
 *                                       initial, out: result).
 * @param [in]     _dst_iov_set_buffer_f Function that sets the buffer to the IOV element
 *                                       from the destination array.
 * @param [in]     _dst_iov_set_length_f Function that sets the length to the IOV element
 *                                       from the destination array.
 * @param [in]     _src_iov              Pointer to the source array of IOVs.
 * @param [in]     _src_iov_cnt          Number of the elements in the source array of IOVs.
 * @param [in]     _src_iov_get_buffer_f Function that gets the buffer of the IOV element
 *                                       from the destination array.
 * @param [in]     _src_iov_get_length_f Function that gets the length of the IOV element
 *                                       from the destination array.
 * @param [in]     _max_length           Maximal total length of the data that can be
 *                                       placed in the resulted array of IOVs.
 * @param [in]     _dst_iov_iter_p       Pointer to the IOV iterator for the destination
 *                                       array of IOVs.
 *
 * @return The total length of the resulted array of IOVs.
 */
#define ucs_iov_converter(_dst_iov, _dst_iov_cnt_p, \
                          _dst_iov_set_buffer_f, _dst_iov_set_length_f, \
                          _src_iov, _src_iov_cnt, \
                          _src_iov_get_buffer_f, _src_iov_get_length_f, \
                          _max_length, _dst_iov_iter_p) \
   ({ \
        size_t __remain_length = _max_length; \
        size_t __dst_iov_index = 0; \
        size_t __src_iov_index = (_dst_iov_iter_p)->iov_index; \
        size_t __dst_iov_length, __src_iov_length; \
        void *__dst_iov_buffer; \
        \
        while ((__src_iov_index < (_src_iov_cnt)) && (__remain_length != 0) && \
               (__dst_iov_index < *(_dst_iov_cnt_p))) { \
            ucs_assert(_src_iov_get_length_f(&(_src_iov)[__src_iov_index]) >= \
                       (_dst_iov_iter_p)->buffer_offset); \
            __src_iov_length = _src_iov_get_length_f(&(_src_iov)[__src_iov_index]) - \
                               (_dst_iov_iter_p)->buffer_offset; \
            if (__src_iov_length == 0) { \
                /* Avoid zero length elements in resulted IOV */ \
                ++__src_iov_index; \
                continue; \
            } \
            \
            __dst_iov_length = ucs_min(__src_iov_length, __remain_length); \
            \
            _dst_iov_set_length_f(&(_dst_iov)[__dst_iov_index], __dst_iov_length); \
            __dst_iov_buffer = UCS_PTR_BYTE_OFFSET(_src_iov_get_buffer_f( \
                                                       &(_src_iov)[__src_iov_index]), \
                                                   (_dst_iov_iter_p)->buffer_offset); \
            _dst_iov_set_buffer_f(&(_dst_iov)[__dst_iov_index], __dst_iov_buffer); \
            \
            if (__src_iov_length > __remain_length) { \
                (_dst_iov_iter_p)->buffer_offset += __remain_length; \
            } else { \
                ucs_assert(((_dst_iov_iter_p)->buffer_offset == 0) || \
                           (__src_iov_index == (_dst_iov_iter_p)->iov_index)); \
                (_dst_iov_iter_p)->buffer_offset = 0; \
                ++__src_iov_index; \
            } \
            \
            ucs_assert(__remain_length >= __dst_iov_length); \
            __remain_length            -= __dst_iov_length; \
            ++__dst_iov_index; \
            \
        } \
        \
        ucs_assert(__dst_iov_index<= *(_dst_iov_cnt_p)); \
        (_dst_iov_iter_p)->iov_index = __src_iov_index; \
        *(_dst_iov_cnt_p)            = __dst_iov_index; \
        ((_max_length) - __remain_length); \
    })

/**
 * Calculates the total length of the IOV array buffers.
 *
 * @param [in]     iov             Pointer to the array of IOVs.
 * @param [in]     iov_cnt         Number of the elements in the array of IOVs.
 *
 * @return The total length of the array of IOVs.
 */
#define ucs_iov_total_length(_iov, _iov_cnt, _iov_get_length_f) \
    ({ \
        size_t __total_length = 0; \
        size_t __iov_it; \
        \
        for (__iov_it = 0; __iov_it < (_iov_cnt); ++__iov_it) { \
            __total_length += _iov_get_length_f(&(_iov)[__iov_it]); \
        } \
        \
        __total_length; \
    })

/**
 * Calculates the flat offset in the IOV array, which is the total data size
 * before the position of the iterator.
 *
 * @param [in]     iov             Pointer to the array of IOVs.
 * @param [in]     iov_cnt         Number of the elements in the array of IOVs.
 * @param [in]     iov_iter        Pointer to the IOV iterator.
 *
 * @return The flat offset in the IOV array.
 */
#define ucs_iov_iter_flat_offset(_iov, _iov_cnt, _iov_iter, _iov_get_length_f) \
    ({ \
        size_t __offset = 0; \
        size_t __iov_it; \
        \
        for (__iov_it = 0; __iov_it < (_iov_iter)->iov_index; ++__iov_it) { \
            __offset += _iov_get_length_f(&(_iov)[__iov_it]); \
        } \
        \
        if ((_iov_iter)->iov_index < (_iov_cnt)) { \
            __offset += (_iov_iter)->buffer_offset; \
        } \
        \
        __offset; \
    })


/**
 * Initializes the IOV iterator by the initial values.
 *
 * @param [in]     iov_iter        Pointer to the IOV iterator.
 */
static UCS_F_ALWAYS_INLINE
void ucs_iov_iter_init(ucs_iov_iter_t *iov_iter)
{
    iov_iter->iov_index     = 0;
    iov_iter->buffer_offset = 0;
}

/**
 * Sets the particular IOVEC data buffer.
 *
 * @param [in]     iov             Pointer to the IOVEC element.
 * @param [in]     length          Length that needs to be set.
 */
static UCS_F_ALWAYS_INLINE
void ucs_iovec_set_length(struct iovec *iov, size_t length)
{
    iov->iov_len = length;
}

/**
 * Sets the length of the particular IOVEC data buffer.
 *
 * @param [in]     iov             Pointer to the IOVEC element.
 * @param [in]     buffer          Buffer that needs to be set.
 */
static UCS_F_ALWAYS_INLINE
void ucs_iovec_set_buffer(struct iovec *iov, void *buffer)
{
    iov->iov_base = buffer;
}

/**
 * Returns the length of the particular IOVEC data buffer.
 *
 * @param [in]     iov             Pointer to the IOVEC element.
 *
 * @return The length of the IOVEC data buffer.
 */
static UCS_F_ALWAYS_INLINE
size_t ucs_iovec_get_length(const struct iovec *iov)
{
    return iov->iov_len;
}

/**
 * Calculates the total length of the IOVEC array buffers.
 *
 * @param [in]     iov            Pointer to the array of IOVEC elements.
 * @param [in]     iov_cnt        Number of elements in the IOVEC array.
 *
 * @return The amount, in bytes, of the data that is stored in the IOVEC
 *         array buffers.
 */
static UCS_F_ALWAYS_INLINE
size_t ucs_iovec_total_length(const struct iovec *iov, size_t iov_cnt)
{
    return ucs_iov_total_length(iov, iov_cnt, ucs_iovec_get_length);
}

#endif
