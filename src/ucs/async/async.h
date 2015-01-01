/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_ASYNC_H_
#define UCS_ASYNC_H_

#include "thread.h"
#include "signal.h"

#include <ucs/config/types.h>
#include <ucs/type/callback.h>
#include <ucs/datastruct/mpmc.h>
#include <ucs/time/time.h>
#include <ucs/debug/log.h>


/**
 * Async event context. Manages timer and fd notifications.
 */
typedef struct ucs_async_context ucs_async_context_t;
struct ucs_async_context {
    union {
        ucs_async_thread_context_t thread;
        ucs_async_signal_context_t signal;
        int                        poll_block;
    };

    ucs_async_mode_t  mode;          /* Event delivery mode */
    volatile uint32_t num_handlers;  /* Number of event and timer handlers */
    ucs_mpmc_queue_t  missed;        /* Miss queue */
};


/**
 * GLobal initialization and cleanup of async event handling.
 */
void ucs_async_global_init();
void ucs_async_global_cleanup();


/**
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
 */
ucs_status_t ucs_async_set_event_handler(ucs_async_mode_t mode, int event_fd,
                                         int events, ucs_notifier_chain_func_t cb,
                                         void *arg, ucs_async_context_t *async);


/**
 * Unregister a given file descriptor from monitoring.
 *
 * @param event_fd        File descriptor to remove handler for.
 */
ucs_status_t ucs_async_unset_event_handler(int event_fd);


/**
 * Add timer handler.
 *
 * @param mode            Thread or signal.
 * @param interval        Timer interval.
 * @param cb              Callback function to execute.
 * @param arg             Argument to callback.
 * @param async           Async context to which events are delivered.
 *                        If NULL, safety is up to the user.
 * @param timer_id_p      Filled with timer id.
 */
ucs_status_t ucs_async_add_timer(ucs_async_mode_t mode, ucs_time_t interval,
                                 ucs_notifier_chain_func_t cb, void *arg,
                                 ucs_async_context_t *async, int *timer_id_p);


/**
 * Remove a timer previously added.
 *
 * @param timer_id        Timer to remove.
 */
ucs_status_t ucs_async_remove_timer(int timer_id);


/**
 * Initialize an asynchronous execution context.
 * This can be used to ensure safe event delivery.
 *
 * @param async           Event context to initialize.
 * @param mode            Either to use signals or epoll threads to wait.
 *
 * @return Success status.
 */
ucs_status_t ucs_async_context_init(ucs_async_context_t *async, ucs_async_mode_t mode);


/**
 * Clean up the async context, and release system resources if possible.
 *
 * @param event           Asynchronous context to clean up.
 */
void ucs_async_context_cleanup(ucs_async_context_t *async);


/**
 * Poll on async context.
 *
 * @param async Async context to poll on. NULL polls on all.
 */
void ucs_async_poll(ucs_async_context_t *async);


void __ucs_async_poll_missed(ucs_async_context_t *async);


/**
 * Check if an async callback was missed because the main thread has blocked
 * the async context. This works as edge-triggered.
 * Should be called with the lock held.
 */
static inline int ucs_async_check_miss(ucs_async_context_t *async)
{
    if (!ucs_mpmc_queue_is_empty(&async->missed)) {
        __ucs_async_poll_missed(async);
        return 1;
    } else if (async->mode == UCS_ASYNC_MODE_POLL) {
        ucs_async_poll(async);
        return 1;
    }
    return 0;
}


/**
 * Block the async handler (if its currently running, wait until it exists and
 * block it then). Used to serialize accesses with the async handler.
 *
 * @param event Event context to block events for.
 * @note This function might wait until a currently running callback returns.
 */
#define UCS_ASYNC_BLOCK(_async) \
    do { \
        if ((_async)->mode == UCS_ASYNC_MODE_THREAD) { \
            UCS_ASYNC_THREAD_BLOCK(_async); \
        } else if ((_async)->mode == UCS_ASYNC_MODE_SIGNAL) { \
            UCS_ASYNC_SIGNAL_BLOCK(_async); \
        } else { \
            ++(_async)->poll_block; \
        } \
    } while(0)


/**
 * Unblock asynchronous event delivery, and invoke pending callbacks.
 *
 * @param event Event context to unblock events for.
 */
#define UCS_ASYNC_UNBLOCK(_async) \
    do { \
        if ((_async)->mode == UCS_ASYNC_MODE_THREAD) { \
             UCS_ASYNC_THREAD_UNBLOCK(_async); \
        } else if ((_async)->mode == UCS_ASYNC_MODE_SIGNAL) { \
            UCS_ASYNC_SIGNAL_UNBLOCK(_async); \
        } else { \
            --(_async)->poll_block; \
        } \
    } while (0)


#endif
