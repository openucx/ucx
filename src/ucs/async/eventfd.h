/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_EVENTFD_H
#define UCS_ASYNC_EVENTFD_H

#include <ucs/type/status.h>
#include <sys/eventfd.h>

BEGIN_C_DECLS


/**
 * Represent either an uninitialized or a closed event file descriptor.
 */
#define UCS_ASYNC_EVENTFD_INVALID_FD (-1)


/**
 * @ingroup UCS_RESOURCE
 *
 * Create an event file descriptor. This file descriptor can later be passed as
 * arguments to poll/signal functions to wait for notifications or to notify
 * pollers.
 *
 * @param fd Pointer to integer which is populated with a file descriptor.
 */
ucs_status_t ucs_async_eventfd_create(int *fd);


/**
 * @ingroup UCS_RESOURCE
 *
 * Destroy an event file descriptor.
 *
 * @param fd File descriptor to be closed.
 */
void ucs_async_eventfd_destroy(int fd);


/**
 * @ingroup UCS_RESOURCE
 *
 * Notify a file descriptor when it is polled. An appropriate error is returned
 * upon failure.
 *
 * @param fd File descriptor which will be notified.
 */
ucs_status_t ucs_async_eventfd_signal(int fd);


/**
 * @ingroup UCS_RESOURCE
 *
 * Poll on a file descriptor for incoming notifications. If no notifications are
 * observed then UCS_ERR_NO_PROGRESS is returned. An appropriate error is
 * returned upon failure.
 *
 * @param fd File descriptor to be polled on.
 */
ucs_status_t ucs_async_eventfd_poll(int fd);

END_C_DECLS

#endif
