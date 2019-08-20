/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/sys/iovec.h>
#include <ucs/sys/math.h>

#include <string.h>


size_t ucs_iov_copy(const struct iovec *iov, size_t iov_cnt,
                    size_t iov_offset, void *buf, size_t max_copy,
                    ucs_iov_copy_direction_t dir)
{
    size_t copied = 0;
    char *iov_buf;
    size_t i, len;

    for (i = 0; (i < iov_cnt) && max_copy; i++) {
        len = iov[i].iov_len;

        if (iov_offset > len) {
            iov_offset -= len;
            continue;
        }

        iov_buf  = UCS_PTR_BYTE_OFFSET(iov[i].iov_base, iov_offset);
        len     -= iov_offset;

        len = ucs_min(len, max_copy);
        if (dir == UCS_IOV_COPY_FROM_BUF) {
            memcpy(iov_buf, UCS_PTR_BYTE_OFFSET(buf, copied), len);
        } else if (dir == UCS_IOV_COPY_TO_BUF) {
            memcpy(UCS_PTR_BYTE_OFFSET(buf, copied), iov_buf, len);
        }

        iov_offset  = 0;
        max_copy   -= len;
        copied     += len;
    }

    return copied;
}

void ucs_iov_advance(struct iovec *iov, size_t iov_cnt,
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
