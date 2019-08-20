/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/sys/iovec.h>
#include <ucs/sys/math.h>

#include <string.h>


size_t ucs_sys_copy_iov_buf(const struct iovec *iov, size_t iov_cnt,
                            size_t iov_offset, void *buf, size_t buf_size,
                            size_t buf_offset, ucs_sys_copy_iov_dir_t dir)
{
    size_t done = buf_offset;
    char *iov_buf;
    size_t i, len;

    buf_size -= buf_offset;

    for (i = 0; (i < iov_cnt) && buf_size; i++) {
        len = iov[i].iov_len;

        if (iov_offset > len) {
            iov_offset -= len;
            continue;
        }

        iov_buf = (char*)iov[i].iov_base + iov_offset;
        len -= iov_offset;

        len = ucs_min(len, buf_size);
        if (dir == UCS_SYS_COPY_BUF_TO_IOV) {
            memcpy(iov_buf, (char*)buf + done, len);
        } else if (dir == UCS_SYS_COPY_IOV_TO_BUF) {
            memcpy((char*)buf + done, iov_buf, len);
        }

        iov_offset = 0;
        buf_size -= len;
        done += len;
    }

    done -= buf_offset;

    return done;
}
