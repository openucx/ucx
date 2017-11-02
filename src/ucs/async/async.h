/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_H_
#define UCS_ASYNC_H_

#include "thread.h"
#include "signal.h"
#include "async_fwd.h"

#include <ucs/sys/compiler_def.h>
#include <ucs/datastruct/mpmc.h>
#include <ucs/time/time.h>
#include <ucs/debug/log.h>

BEGIN_C_DECLS

/**
 * Async event context. Manages timer and fd notifications.
 */
struct ucs_async_context {
    union {
        ucs_async_thread_context_t thread;
        ucs_async_signal_context_t signal;
        int                        poll_block;
    };

    ucs_async_mode_t  mode;          /* Event delivery mode */
    volatile uint32_t num_handlers;  /* Number of event and timer handlers */
    ucs_mpmc_queue_t  missed;        /* Miss queue */
    ucs_time_t        last_wakeup;   /* time of the last wakeup */
};


/**
 * @ingroup UCS_RESOURCE
 *
 * GLobal initialization and cleanup of async event handling.
 */
void ucs_async_global_init();
void ucs_async_global_cleanup();


/**
 * Initialize an asynchronous execution context. The context is not allocated.
 * To allocate the context, please use public version of the
 * function @ref ucs_async_context_create
 * This can be used to ensure safe event delivery.
 *
 * @param async           Event context to initialize.
 * @param mode            Either to use signals or epoll threads to wait.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucs_async_context_init(ucs_async_context_t *async,
                                    ucs_async_mode_t mode);


/**
 * Clean up the async context, and release system resources if possible.
 *
 * @param async           Asynchronous context to clean up.
 */
void ucs_async_context_cleanup(ucs_async_context_t *async);


/**
 * Check if an async callback was missed because the main thread has blocked
 * the async context. This works as edge-triggered.
 * Should be called with the lock held.
 */
static inline int ucs_async_check_miss(ucs_async_context_t *async)
{
    if (ucs_unlikely(!ucs_mpmc_queue_is_empty(&async->missed))) {
        __ucs_async_poll_missed(async);
        return 1;
    } else if (ucs_unlikely(async->mode == UCS_ASYNC_MODE_POLL)) {
        ucs_async_poll(async);
        return 1;
    }
    return 0;
}


/**
 * Block the async handler (if its currently running, wait until it exits and
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


END_C_DECLS

#endif
