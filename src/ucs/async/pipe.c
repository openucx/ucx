/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "pipe.h"

#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>


ucs_status_t ucs_async_pipe_create(ucs_async_pipe_t *p)
{
    int pipefds[2];
    int ret;

    ret = pipe(pipefds);
    if (ret < 0) {
        ucs_error("pipe() returned %d: %m", ret);
        goto err;
    }

    /* Set pipe to non blocking */
    if (ucs_sys_fcntl_modfl(pipefds[0], O_NONBLOCK, 0) != UCS_OK ||
        ucs_sys_fcntl_modfl(pipefds[1], O_NONBLOCK, 0) != UCS_OK)
    {
        goto err_close_pipe;
    }

    p->read_fd  = pipefds[0];
    p->write_fd = pipefds[1];
    return UCS_OK;

err_close_pipe:
    close(pipefds[0]);
    close(pipefds[1]);
err:
    return UCS_ERR_IO_ERROR;
}

void ucs_async_pipe_destroy(ucs_async_pipe_t *p)
{
    close(p->read_fd);
    close(p->write_fd);
}

void ucs_async_pipe_push(ucs_async_pipe_t *p)
{
    int dummy = 0;
    int ret;

    ret = write(p->write_fd, &dummy, sizeof(dummy));
    if (ret < 0 && errno != EAGAIN) {
        ucs_error("writing to wakeup pipe failed: %m");
    }
}

void ucs_async_pipe_drain(ucs_async_pipe_t *p)
{
    int dummy;
    while (read(p->read_fd, &dummy, sizeof(dummy)) > 0);
}
