/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_PIPE_H
#define UCS_ASYNC_PIPE_H

#include <ucs/type/status.h>

BEGIN_C_DECLS


/**
 * An invalid fd value which represents a closed pipe
 */
#define UCS_ASYNC_PIPE_INVALID_FD (-1)


/**
 * A pipe for event signaling.
 */
typedef struct ucs_async_pipe {
    int   read_fd;
    int   write_fd;
} ucs_async_pipe_t ;


#define UCS_ASYNC_PIPE_INITIALIZER \
    { \
        UCS_ASYNC_PIPE_INVALID_FD, UCS_ASYNC_PIPE_INVALID_FD \
    }

/**
 * Create/destroy a pipe for event signaling.
 */
ucs_status_t ucs_async_pipe_create(ucs_async_pipe_t *p);
void ucs_async_pipe_destroy(ucs_async_pipe_t *p);


/**
 * Set both file descriptors of a pipe to an invalid value.
 */
void ucs_async_pipe_invalidate(ucs_async_pipe_t *p);


/**
 * Push an event to the signaling pipe.
 */
void ucs_async_pipe_push(ucs_async_pipe_t *p);

/**
 * Remove all events from the pipe.
 */
void ucs_async_pipe_drain(ucs_async_pipe_t *p);

/**
 * @return File descriptor which gets the pipe events.
 */
static inline int ucs_async_pipe_rfd(ucs_async_pipe_t *p) {
    return p->read_fd;
}

END_C_DECLS

#endif
