/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eventfd.h"

#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>


typedef ssize_t (*ucs_async_eventfd_cb_t)(int fd, void *buf, size_t count);


static inline ucs_status_t
ucs_async_eventfd_common_io(int fd, int blocking, ucs_async_eventfd_cb_t cb)
{
    uint64_t dummy = 1;
    int ret;

    do {
        ret = cb(fd, &dummy, sizeof(dummy));
        if (ret > 0) {
            return UCS_OK;
        }

        if ((ret < 0) && (errno != EAGAIN) && (errno != EINTR)) {
            ucs_error("eventfd error (fd %d blocking %d): %m", fd, blocking);
            return UCS_ERR_IO_ERROR;
        }
    } while (blocking);

    return UCS_ERR_NO_PROGRESS;
}

ucs_status_t ucs_async_eventfd_create(int *fd_p)
{
    int local_fd;

    local_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (local_fd == -1) {
        ucs_error("failed to create event fd: %m");
        return UCS_ERR_IO_ERROR;
    }

    *fd_p = local_fd;
    return UCS_OK;
}

void ucs_async_eventfd_destroy(int fd)
{
    if (fd != UCS_ASYNC_EVENTFD_INVALID_FD) {
        close(fd);
    }
}

ucs_status_t ucs_async_eventfd_poll(int fd)
{
    return ucs_async_eventfd_common_io(fd, 0, (ucs_async_eventfd_cb_t)read);
}

ucs_status_t ucs_async_eventfd_signal(int fd)
{
    return ucs_async_eventfd_common_io(fd, 1, (ucs_async_eventfd_cb_t)write);
}
