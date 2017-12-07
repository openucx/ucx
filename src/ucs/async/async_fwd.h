/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_FWD_H
#define UCS_ASYNC_FWD_H

#include <ucs/config/types.h>
#include <ucs/time/time_def.h>
#include <ucs/type/status.h>

BEGIN_C_DECLS

typedef struct ucs_async_context ucs_async_context_t;


/**
 * @ingroup UCS_RESOURCE
 *
 * Async event callback.
 *
 * @param id           Event id (timer or file descriptor).
 * @param arg          User-defined argument.
 */
typedef void (*ucs_async_event_cb_t)(int id, void *arg);


/**
 * @ingroup UCS_RESOURCE
 *
 * Register a file descriptor for monitoring (call handler upon events).
 * Every fd can have only one handler.
 *
 * @param mode            Thread or signal.
 * @param event_fd        File descriptor to set handler for.
 * @param events          Events to wait on (POLLxx/EPOLLxx bits).
 * @param cb              Callback function to execute.
 * @param arg             Argument to callback.
 * @param async           Async context to which events are delivered.
 *                        If NULL, safety is up to the user.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucs_async_set_event_handler(ucs_async_mode_t mode, int event_fd,
                                         int events, ucs_async_event_cb_t cb,
                                         void *arg, ucs_async_context_t *async);


/**
 * @ingroup UCS_RESOURCE
 *
 * Add timer handler.
 *
 * @param mode            Thread or signal.
 * @param interval        Timer interval.
 * @param cb              Callback function to execute.
 * @param arg             Argument to callback.
 * @param async           Async context to which events are delivered.
 *                        If NULL, safety is up to the user.
 * @param timer_id_p      Filled with timer id.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucs_async_add_timer(ucs_async_mode_t mode, ucs_time_t interval,
                                 ucs_async_event_cb_t cb, void *arg,
                                 ucs_async_context_t *async, int *timer_id_p);


/**
 * @ingroup UCS_RESOURCE
 *
 * Remove an event handler (Timer or event file).
 *
 * @param id        Timer/FD to remove.
 * @param sync      If nonzero, wait until the handler for this event is not
 *                  running anymore. Cannot be used in the context of the event
 *                  handler itself because it would deadlock.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucs_async_remove_handler(int id, int sync);


/**
 * @ingroup UCS_RESOURCE
 *
 * Modify events mask for an existing event handler (event file).
 *
 * @param fd        File descriptor modify events for.
 * @param events    New set of events to wait on (POLLxx/EPOLLxx bits).
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucs_async_modify_handler(int fd, int events);


/**
 * @ingroup UCS_RESOURCE
 * @brief Create an asynchronous execution context
 *
 * Allocate and initialize an asynchronous execution context.
 * This can be used to ensure safe event delivery.
 *
 * @param mode            Either to use signals or epoll threads to wait.
 * @param async_p         Event context pointer to initialize.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucs_async_context_create(ucs_async_mode_t mode,
                                      ucs_async_context_t **async_p);


/**
 * @ingroup UCS_RESOURCE
 * @brief Destroy the asynchronous execution context
 *
 * Clean up the async context, and release system resources if possible.
 * The context memory released.
 *
 * @param async           Asynchronous context to clean up.
 */
void ucs_async_context_destroy(ucs_async_context_t *async);


/**
 * @ingroup UCS_RESOURCE
 *
 * Poll on async context.
 *
 * @param async Async context to poll on. NULL polls on all.
 */
void ucs_async_poll(ucs_async_context_t *async);


void __ucs_async_poll_missed(ucs_async_context_t *async);

END_C_DECLS

#endif
